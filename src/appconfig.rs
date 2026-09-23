//! AWS AppConfig Agent source for the hot admission policy.
//!
//! The Agent already owns remote polling, deployment, and its local backup.
//! This module only reads the Agent's local HTTP cache, validates the complete
//! document, and hands an immutable candidate to the admission controller.

use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::BytesMut;
use futures_util::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::task::JoinHandle;
use tracing::{info, warn};
use url::Url;

use crate::admission::{AdmissionController, AdmissionPolicy, AppliedPolicy, PolicySource};
use crate::config::AppConfigSettings;

/// The Agent endpoint is local and should return a small JSON document. A
/// limit protects the proxy from accidentally buffering an unbounded response
/// before JSON parsing.
const MAX_DOCUMENT_BYTES: usize = 64 * 1024;
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);
const CONFIGURATION_VERSION_HEADER: &str = "Configuration-Version";
const MAX_INFLIGHT: u32 = 10_000;
const MAX_POLICY_DURATION_SECS: u64 = 86_400;

/// A validated policy returned by the Agent, including the immutable source
/// identity used for reconciliation and rollback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchedAdmissionPolicy {
    pub policy: AdmissionPolicy,
    pub configuration_version: String,
    pub content_sha256: [u8; 32],
}

/// Errors from reading or validating one Agent response.
#[derive(Debug, thiserror::Error)]
pub enum AppConfigFetchError {
    #[error("AppConfig Agent request failed: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("AppConfig Agent returned HTTP {status}")]
    HttpStatus { status: reqwest::StatusCode },
    #[error("invalid AppConfig document: {0}")]
    Invalid(String),
}

impl AppConfigFetchError {
    fn invalid(error: impl Into<String>) -> Self {
        Self::Invalid(error.into())
    }

    fn is_transport(&self) -> bool {
        matches!(self, Self::Transport(_) | Self::HttpStatus { .. })
    }
}

/// The JSON contract served by the AppConfig Agent.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmissionDocument {
    schema_version: u32,
    target: String,
    admission: AdmissionDocumentValues,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmissionDocumentValues {
    max_inflight: u32,
    backpressure_secs: u64,
    retry_after_secs: u64,
}

/// A configured AppConfig Agent HTTP source.
#[derive(Clone)]
pub struct AppConfigSource {
    settings: AppConfigSettings,
    client: reqwest::Client,
}

impl fmt::Debug for AppConfigSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AppConfigSource")
            .field("settings", &self.settings)
            .field("client", &"<reqwest client>")
            .finish()
    }
}

impl AppConfigSource {
    /// Construct a source whose bearer token can never follow a redirect.
    pub fn new(settings: AppConfigSettings) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self { settings, client })
    }

    pub fn settings(&self) -> &AppConfigSettings {
        &self.settings
    }

    /// Read and fully validate one configuration from the Agent cache.
    pub async fn fetch(&self) -> Result<FetchedAdmissionPolicy, AppConfigFetchError> {
        let endpoint = self.endpoint_url()?;
        let mut request = self.client.get(endpoint).timeout(FETCH_TIMEOUT);
        if let Some(token) = self
            .settings
            .access_token
            .as_ref()
            .map(|token| token.expose())
        {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .map_err(AppConfigFetchError::Transport)?;
        if !response.status().is_success() {
            return Err(AppConfigFetchError::HttpStatus {
                status: response.status(),
            });
        }

        let configuration_version = response
            .headers()
            .get(CONFIGURATION_VERSION_HEADER)
            .ok_or_else(|| AppConfigFetchError::invalid("missing Configuration-Version header"))?
            .to_str()
            .map_err(|_| AppConfigFetchError::invalid("Configuration-Version is not valid UTF-8"))?
            .trim()
            .to_owned();
        if configuration_version.is_empty() {
            return Err(AppConfigFetchError::invalid(
                "Configuration-Version header is empty",
            ));
        }

        let (body, content_sha256) = read_body(response).await?;
        let document: AdmissionDocument = serde_json::from_slice(&body)
            .map_err(|error| AppConfigFetchError::invalid(error.to_string()))?;
        let policy = validate_document(document, &self.settings.target)?;

        Ok(FetchedAdmissionPolicy {
            policy,
            configuration_version,
            content_sha256,
        })
    }

    fn endpoint_url(&self) -> Result<Url, AppConfigFetchError> {
        let mut endpoint = self.settings.agent_url.clone();
        let mut segments = endpoint
            .path_segments_mut()
            .map_err(|_| AppConfigFetchError::invalid("Agent URL cannot accept path segments"))?;
        segments
            .push("applications")
            .push(&self.settings.application)
            .push("environments")
            .push(&self.settings.environment)
            .push("configurations")
            .push(&self.settings.profile);
        drop(segments);
        Ok(endpoint)
    }
}

async fn read_body(
    response: reqwest::Response,
) -> Result<(Vec<u8>, [u8; 32]), AppConfigFetchError> {
    let mut body = BytesMut::new();
    let mut digest = Sha256::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(AppConfigFetchError::Transport)?;
        if body.len().saturating_add(chunk.len()) > MAX_DOCUMENT_BYTES {
            return Err(AppConfigFetchError::invalid(format!(
                "document exceeds {MAX_DOCUMENT_BYTES} bytes"
            )));
        }
        digest.update(&chunk);
        body.extend_from_slice(&chunk);
    }
    let digest: [u8; 32] = digest.finalize().into();
    Ok((body.to_vec(), digest))
}

fn validate_document(
    document: AdmissionDocument,
    expected_target: &str,
) -> Result<AdmissionPolicy, AppConfigFetchError> {
    if document.schema_version != 1 {
        return Err(AppConfigFetchError::invalid(format!(
            "unsupported schema_version {}; expected 1",
            document.schema_version
        )));
    }
    if document.target != expected_target {
        return Err(AppConfigFetchError::invalid(format!(
            "target {:?} does not match configured target {:?}",
            document.target, expected_target
        )));
    }
    if document.admission.max_inflight == 0 {
        return Err(AppConfigFetchError::invalid(
            "admission.max_inflight must be positive",
        ));
    }
    if document.admission.max_inflight > MAX_INFLIGHT {
        return Err(AppConfigFetchError::invalid(format!(
            "admission.max_inflight must not exceed {MAX_INFLIGHT}"
        )));
    }
    if document.admission.backpressure_secs == 0 {
        return Err(AppConfigFetchError::invalid(
            "admission.backpressure_secs must be positive",
        ));
    }
    if document.admission.backpressure_secs > MAX_POLICY_DURATION_SECS {
        return Err(AppConfigFetchError::invalid(format!(
            "admission.backpressure_secs must not exceed {MAX_POLICY_DURATION_SECS}"
        )));
    }
    if document.admission.retry_after_secs == 0 {
        return Err(AppConfigFetchError::invalid(
            "admission.retry_after_secs must be positive",
        ));
    }
    if document.admission.retry_after_secs > MAX_POLICY_DURATION_SECS {
        return Err(AppConfigFetchError::invalid(format!(
            "admission.retry_after_secs must not exceed {MAX_POLICY_DURATION_SECS}"
        )));
    }

    Ok(AdmissionPolicy {
        max_inflight: document.admission.max_inflight,
        backpressure_ttl: Duration::from_secs(document.admission.backpressure_secs),
        retry_after: Duration::from_secs(document.admission.retry_after_secs),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ActiveConfiguration {
    configuration_version: String,
    content_sha256: [u8; 32],
}

/// Start the non-fatal local-cache reconciliation loop.
pub fn spawn_admission_policy_refresh(
    source: AppConfigSource,
    admission: std::sync::Arc<AdmissionController>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut active = None;
        let mut interval = tokio::time::interval(source.settings.refresh_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            reconcile_once(&source, &admission, &mut active).await;
        }
    })
}

async fn reconcile_once(
    source: &AppConfigSource,
    admission: &AdmissionController,
    active: &mut Option<ActiveConfiguration>,
) {
    let fetched = match source.fetch().await {
        Ok(fetched) => fetched,
        Err(error) => {
            let outcome = if error.is_transport() {
                "transport_error"
            } else {
                "invalid"
            };
            metrics::counter!("appconfig_fetch_total", "outcome" => outcome).increment(1);
            warn!(
                application = %source.settings.application,
                environment = %source.settings.environment,
                profile = %source.settings.profile,
                target = %source.settings.target,
                error = %error,
                "AppConfig admission policy read failed; retaining last-known-good policy"
            );
            return;
        }
    };

    metrics::gauge!("appconfig_last_successful_read_timestamp_seconds")
        .set(unix_timestamp_seconds());
    let digest = fetched.content_sha256;
    let digest_hex = hex::encode(digest);
    if let Some(previous) = active {
        if previous.configuration_version == fetched.configuration_version {
            if previous.content_sha256 != digest {
                metrics::counter!("appconfig_fetch_total", "outcome" => "invalid").increment(1);
                warn!(
                    application = %source.settings.application,
                    environment = %source.settings.environment,
                    profile = %source.settings.profile,
                    target = %source.settings.target,
                    configuration_version = %fetched.configuration_version,
                    content_sha256 = %digest_hex,
                    "AppConfig reused a version with different content; retaining last-known-good policy"
                );
                return;
            }
            metrics::counter!("appconfig_fetch_total", "outcome" => "unchanged").increment(1);
            return;
        }
    }

    let candidate = AppliedPolicy {
        policy: fetched.policy.clone(),
        source: PolicySource::AppConfig {
            configuration_version: fetched.configuration_version.clone(),
            content_sha256: digest,
        },
    };
    let previous = admission.current_policy();
    let outcome = match admission.apply_policy(candidate) {
        Ok(outcome) => outcome,
        Err(error) => {
            metrics::counter!("appconfig_fetch_total", "outcome" => "invalid").increment(1);
            warn!(
                application = %source.settings.application,
                environment = %source.settings.environment,
                profile = %source.settings.profile,
                target = %source.settings.target,
                configuration_version = %fetched.configuration_version,
                content_sha256 = %digest_hex,
                error = %error,
                "AppConfig admission policy was not adopted; retaining last-known-good policy"
            );
            return;
        }
    };

    *active = Some(ActiveConfiguration {
        configuration_version: fetched.configuration_version.clone(),
        content_sha256: digest,
    });
    metrics::gauge!("appconfig_active_max_inflight").set(f64::from(fetched.policy.max_inflight));
    match outcome {
        crate::admission::PolicyApplyOutcome::Unchanged => {
            metrics::counter!("appconfig_fetch_total", "outcome" => "unchanged").increment(1);
        }
        crate::admission::PolicyApplyOutcome::MetadataUpdated => {
            metrics::counter!("appconfig_fetch_total", "outcome" => "unchanged").increment(1);
            info!(
                application = %source.settings.application,
                environment = %source.settings.environment,
                profile = %source.settings.profile,
                target = %source.settings.target,
                configuration_version = %fetched.configuration_version,
                content_sha256 = %digest_hex,
                max_inflight = fetched.policy.max_inflight,
                backpressure_secs = fetched.policy.backpressure_ttl.as_secs(),
                retry_after_secs = fetched.policy.retry_after.as_secs(),
                "Advanced AppConfig admission policy source metadata"
            );
        }
        crate::admission::PolicyApplyOutcome::Applied => {
            metrics::counter!("appconfig_fetch_total", "outcome" => "applied").increment(1);
            info!(
                application = %source.settings.application,
                environment = %source.settings.environment,
                profile = %source.settings.profile,
                target = %source.settings.target,
                configuration_version = %fetched.configuration_version,
                content_sha256 = %digest_hex,
                old_max_inflight = previous.as_ref().map_or(0, |applied| applied.policy.max_inflight),
                old_backpressure_secs = previous.as_ref().map_or(0, |applied| applied.policy.backpressure_ttl.as_secs()),
                old_retry_after_secs = previous.as_ref().map_or(0, |applied| applied.policy.retry_after.as_secs()),
                max_inflight = fetched.policy.max_inflight,
                backpressure_secs = fetched.policy.backpressure_ttl.as_secs(),
                retry_after_secs = fetched.policy.retry_after.as_secs(),
                "Applied AppConfig admission policy"
            );
        }
    }
}

fn unix_timestamp_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |duration| duration.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn settings(server: &MockServer) -> AppConfigSettings {
        AppConfigSettings {
            application: "app/name".to_string(),
            environment: "prod env".to_string(),
            profile: "profile/name".to_string(),
            target: "gateway-a".to_string(),
            agent_url: Url::parse(&server.uri()).unwrap(),
            refresh_interval: Duration::from_millis(10),
            access_token: None,
        }
    }

    fn document(target: &str) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "target": target,
            "admission": {
                "max_inflight": 32,
                "backpressure_secs": 10,
                "retry_after_secs": 2
            }
        })
    }

    fn controller() -> AdmissionController {
        AdmissionController::new(
            Some(crate::admission::AdmissionBootstrap {
                static_config: crate::admission::AdmissionStaticConfig {
                    tier_borrowing: false,
                    long_max_inflight_per_host: 0,
                    start_inflight: 8,
                    ramp_step: 1,
                    ramp_interval: Duration::from_secs(60),
                    ttft_p95_max: None,
                    queue_saturated_at: 1,
                },
                policy: AdmissionPolicy {
                    max_inflight: 8,
                    backpressure_ttl: Duration::from_secs(10),
                    retry_after: Duration::from_secs(2),
                },
            }),
            1,
            Arc::new(crate::engine_load::EngineLoad::disabled()),
        )
    }

    async fn source_with_response(
        body: serde_json::Value,
        version: Option<&str>,
        status: u16,
    ) -> (MockServer, AppConfigSource) {
        let server = MockServer::start().await;
        let mut response = ResponseTemplate::new(status).set_body_json(body);
        if let Some(version) = version {
            response = response.insert_header(CONFIGURATION_VERSION_HEADER, version);
        }
        Mock::given(method("GET"))
            .and(path(
                "/applications/app%2Fname/environments/prod%20env/configurations/profile%2Fname",
            ))
            .respond_with(response)
            .mount(&server)
            .await;
        let source = AppConfigSource::new(settings(&server)).unwrap();
        (server, source)
    }

    #[tokio::test]
    async fn fetches_valid_document_and_digest() {
        let (_server, source) = source_with_response(document("gateway-a"), Some("v1"), 200).await;
        let fetched = source.fetch().await.unwrap();
        assert_eq!(fetched.policy.max_inflight, 32);
        assert_eq!(fetched.policy.backpressure_ttl, Duration::from_secs(10));
        assert_eq!(fetched.policy.retry_after, Duration::from_secs(2));
        assert_eq!(fetched.configuration_version, "v1");
        let expected_digest: [u8; 32] =
            Sha256::digest(serde_json::to_vec(&document("gateway-a")).unwrap()).into();
        assert_eq!(fetched.content_sha256, expected_digest);
    }

    #[tokio::test]
    async fn rejects_schema_target_zero_unknown_and_missing_version() {
        let cases = [
            (
                serde_json::json!({"schema_version": 2, "target": "gateway-a", "admission": {"max_inflight": 1, "backpressure_secs": 1, "retry_after_secs": 1}}),
                Some("v1"),
            ),
            (document("gateway-b"), Some("v1")),
            (
                serde_json::json!({"schema_version": 1, "target": "gateway-a", "admission": {"max_inflight": 0, "backpressure_secs": 1, "retry_after_secs": 1}}),
                Some("v1"),
            ),
            (
                serde_json::json!({"schema_version": 1, "target": "gateway-a", "admission": {"max_inflight": 1, "backpressure_secs": 1, "retry_after_secs": 1}, "extra": true}),
                Some("v1"),
            ),
            (
                serde_json::json!({"schema_version": 1, "target": "gateway-a"}),
                Some("v1"),
            ),
            (document("gateway-a"), None),
        ];
        for (body, version) in cases {
            let (_server, source) = source_with_response(body, version, 200).await;
            assert!(matches!(
                source.fetch().await,
                Err(AppConfigFetchError::Invalid(_))
            ));
        }
    }

    #[test]
    fn accepts_schema_limit_boundaries() {
        let document = AdmissionDocument {
            schema_version: 1,
            target: "gateway-a".to_string(),
            admission: AdmissionDocumentValues {
                max_inflight: MAX_INFLIGHT,
                backpressure_secs: MAX_POLICY_DURATION_SECS,
                retry_after_secs: MAX_POLICY_DURATION_SECS,
            },
        };

        let policy = validate_document(document, "gateway-a").unwrap();
        assert_eq!(policy.max_inflight, MAX_INFLIGHT);
        assert_eq!(
            policy.backpressure_ttl,
            Duration::from_secs(MAX_POLICY_DURATION_SECS)
        );
        assert_eq!(
            policy.retry_after,
            Duration::from_secs(MAX_POLICY_DURATION_SECS)
        );
    }

    #[test]
    fn rejects_values_above_schema_limits() {
        let cases = [
            AdmissionDocumentValues {
                max_inflight: MAX_INFLIGHT + 1,
                backpressure_secs: 1,
                retry_after_secs: 1,
            },
            AdmissionDocumentValues {
                max_inflight: 1,
                backpressure_secs: MAX_POLICY_DURATION_SECS + 1,
                retry_after_secs: 1,
            },
            AdmissionDocumentValues {
                max_inflight: 1,
                backpressure_secs: 1,
                retry_after_secs: MAX_POLICY_DURATION_SECS + 1,
            },
        ];

        for admission in cases {
            let result = validate_document(
                AdmissionDocument {
                    schema_version: 1,
                    target: "gateway-a".to_string(),
                    admission,
                },
                "gateway-a",
            );
            assert!(matches!(result, Err(AppConfigFetchError::Invalid(_))));
        }
    }

    #[tokio::test]
    async fn rejects_malformed_json() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header(CONFIGURATION_VERSION_HEADER, "v1")
                    .set_body_string("{not-json"),
            )
            .mount(&server)
            .await;
        let source = AppConfigSource::new(settings(&server)).unwrap();
        assert!(matches!(
            source.fetch().await,
            Err(AppConfigFetchError::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn rejects_non_success_and_recovers_after_response_changes() {
        let server = MockServer::start().await;
        let source = AppConfigSource::new(settings(&server)).unwrap();
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        assert!(matches!(
            source.fetch().await,
            Err(AppConfigFetchError::HttpStatus { .. })
        ));

        server.reset().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header(CONFIGURATION_VERSION_HEADER, "v2")
                    .set_body_json(document("gateway-a")),
            )
            .mount(&server)
            .await;
        let recovered = source.fetch().await.unwrap();
        assert_eq!(recovered.configuration_version, "v2");
    }

    #[tokio::test]
    async fn does_not_follow_agent_redirects_with_a_bearer_token() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(header("authorization", "Bearer do-not-forward"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("location", format!("{}/redirect-target", server.uri())),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(path("/redirect-target"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let mut settings = settings(&server);
        settings.access_token = Some(crate::config::SensitiveString::new(
            "do-not-forward".to_string(),
        ));

        let source = AppConfigSource::new(settings).unwrap();
        assert!(matches!(
            source.fetch().await,
            Err(AppConfigFetchError::HttpStatus { status }) if status.as_u16() == 302
        ));
    }

    #[tokio::test]
    async fn timeout_is_a_transport_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header(CONFIGURATION_VERSION_HEADER, "v1")
                    .set_body_json(document("gateway-a"))
                    .set_delay(Duration::from_secs(6)),
            )
            .mount(&server)
            .await;
        let source = AppConfigSource::new(settings(&server)).unwrap();
        assert!(matches!(
            source.fetch().await,
            Err(AppConfigFetchError::Transport(_))
        ));
    }

    #[tokio::test]
    async fn sends_access_token_as_bearer_without_exposing_it_in_debug() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(header("authorization", "Bearer actual-agent-secret"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header(CONFIGURATION_VERSION_HEADER, "v1")
                    .set_body_json(document("gateway-a")),
            )
            .expect(1)
            .mount(&server)
            .await;
        let mut settings = settings(&server);
        settings.access_token = Some(crate::config::SensitiveString::new(
            "actual-agent-secret".to_string(),
        ));
        let source = AppConfigSource::new(settings).unwrap();
        assert!(!format!("{source:?}").contains("actual-agent-secret"));
        source.fetch().await.unwrap();
        server.verify().await;
    }

    #[tokio::test]
    async fn reconciler_updates_metadata_for_new_version_with_identical_content() {
        let server = MockServer::start().await;
        let source = AppConfigSource::new(settings(&server)).unwrap();
        let admission = controller();
        let mut active = None;
        for version in ["v1", "v2"] {
            Mock::given(method("GET"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .insert_header(CONFIGURATION_VERSION_HEADER, version)
                        .set_body_json(document("gateway-a")),
                )
                .mount(&server)
                .await;
            reconcile_once(&source, &admission, &mut active).await;
            server.reset().await;
        }
        assert_eq!(admission.current_policy().unwrap().policy.max_inflight, 32);
        assert!(matches!(
            admission.current_policy().unwrap().source,
            PolicySource::AppConfig {
                configuration_version,
                ..
            } if configuration_version == "v2"
        ));
    }

    #[tokio::test]
    async fn reconciler_rejects_changed_content_under_the_active_version() {
        let server = MockServer::start().await;
        let source = AppConfigSource::new(settings(&server)).unwrap();
        let admission = controller();
        let mut active = None;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header(CONFIGURATION_VERSION_HEADER, "v1")
                    .set_body_json(document("gateway-a")),
            )
            .mount(&server)
            .await;
        reconcile_once(&source, &admission, &mut active).await;
        let accepted = admission.current_policy().unwrap();
        server.reset().await;

        let mut changed = document("gateway-a");
        changed["admission"]["max_inflight"] = serde_json::json!(64);
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header(CONFIGURATION_VERSION_HEADER, "v1")
                    .set_body_json(changed),
            )
            .mount(&server)
            .await;
        reconcile_once(&source, &admission, &mut active).await;
        assert_eq!(admission.current_policy(), Some(accepted));
        assert_eq!(admission.budget(), 32);
    }

    #[test]
    fn source_debug_does_not_include_access_token() {
        let settings = AppConfigSettings {
            application: "app".to_string(),
            environment: "env".to_string(),
            profile: "profile".to_string(),
            target: "target".to_string(),
            agent_url: Url::parse("http://127.0.0.1:2772").unwrap(),
            refresh_interval: Duration::from_secs(1),
            access_token: Some(crate::config::SensitiveString::new(
                "actual-agent-secret".to_string(),
            )),
        };
        assert!(!format!("{settings:?}").contains("actual-agent-secret"));
        let source = AppConfigSource::new(settings).unwrap();
        assert!(!format!("{source:?}").contains("actual-agent-secret"));
    }
}
