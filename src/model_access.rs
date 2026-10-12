//! Per-organization model access (`VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS`,
//! gateway mode).
//!
//! cloud-api can restrict a model to a set of organizations, and applies that
//! on its own inference path. A gateway lane is not on that path: it checks
//! the key and bills the usage itself, so without this a valid key of any
//! organization reaches a restricted model through the lane. cloud-api
//! publishes the rule as one document, and the gateway applies it:
//!
//! ```text
//! GET {CLOUD_API_URL}/v1/internal/model-access
//! Authorization: Bearer <CLOUD_API_USAGE_TOKEN>
//!
//! {"object": "model_access",
//!  "models": [{"id": "<model name>", "organization_ids": ["<uuid>", …]}]}
//! ```
//!
//! The document lists the restricted models, each with the organizations that
//! may use it, possibly none. A model it does not list is open to every valid
//! key. It is read at startup and then on the configured interval, off the
//! request path, and the latest one is kept in memory: a request only ever
//! reads that snapshot.
//!
//! A request made with a cloud-api key for a restricted model, from an
//! organization that is not on the model's list, gets the answer of a model
//! this process does not serve (`AppError::ModelNotFound`), so the two cannot
//! be told apart. A key whose check returned no organization id is refused
//! the same way. A caller that presents the proxy's own config token is not
//! checked: it is trusted to have applied the rule itself.
//!
//! A read that fails leaves the last snapshot in force, however old it is: an
//! outage of cloud-api must not open a restricted model, nor close one to the
//! organizations that have it. Until a first document has been read nothing
//! is restricted. That is the case of a cloud-api that does not have the
//! route yet (a 404) and of any other failure at startup; the static
//! `VLLM_PROXY_ALLOWED_ORG_IDS` list applies throughout.
//!
//! Unset or `0`, the default, none of this runs: no request to cloud-api, no
//! check and no series.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use tracing::{info, warn};

use crate::auth::RequireAuth;
use crate::config::Config;
use crate::error::AppError;
use crate::model_metrics::{model_counter, ModelLabel};
use crate::AppState;

/// cloud-api's route for the document, under `CLOUD_API_URL`.
pub const MODEL_ACCESS_PATH: &str = "/v1/internal/model-access";

/// Timeout of one read of the document.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(5);

/// The restricted models of one document: for each, the organizations that
/// may use it.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    allowed: HashMap<String, HashSet<String>>,
}

#[derive(Deserialize)]
struct Document {
    models: Vec<DocumentModel>,
}

#[derive(Deserialize)]
struct DocumentModel {
    id: String,
    organization_ids: Vec<String>,
}

impl Snapshot {
    /// Parse cloud-api's document. Fields this version does not know are
    /// ignored. A document without `models`, or with an entry that lacks its
    /// `id` or its `organization_ids`, is refused as a whole: half a document
    /// would restrict or open something cloud-api did not say.
    pub fn parse(body: &[u8]) -> Result<Self, serde_json::Error> {
        let document: Document = serde_json::from_slice(body)?;
        let mut allowed: HashMap<String, HashSet<String>> = HashMap::new();
        for model in document.models {
            // An id listed twice is one restricted model with both lists.
            allowed
                .entry(model.id)
                .or_default()
                .extend(model.organization_ids);
        }
        Ok(Self { allowed })
    }

    /// Whether a cloud-api key of organization `org_id` may use `model`: the
    /// model is not restricted, or the organization is on its list. Both are
    /// compared byte for byte, as list mode matches a request's `model` and
    /// the static allow-list matches an organization. A key without an
    /// organization id cannot be on a list.
    pub fn allows(&self, model: &str, org_id: Option<&str>) -> bool {
        match self.allowed.get(model) {
            None => true,
            Some(organizations) => org_id.is_some_and(|org| organizations.contains(org)),
        }
    }

    fn restricted_models(&self) -> usize {
        self.allowed.len()
    }

    /// Organizations on the lists, a model's own counted for each model.
    fn grants(&self) -> usize {
        self.allowed.values().map(HashSet::len).sum()
    }
}

/// The decision for one request: whether it may be served as `model`.
///
/// `snapshot` is `None` until a first document has been read, and always when
/// the setting is off: nothing is restricted then. `customer_key` is false
/// for a caller that authenticated with the proxy's own config token, which
/// is never checked.
fn permits(
    snapshot: Option<&Snapshot>,
    customer_key: bool,
    org_id: Option<&str>,
    model: &str,
) -> bool {
    !customer_key || snapshot.is_none_or(|snapshot| snapshot.allows(model, org_id))
}

/// Why a read of the document did not produce a snapshot. `reason` is the
/// label of `model_access_refresh_failures_total`.
enum Failure {
    /// 404: this cloud-api does not serve the document.
    RouteMissing,
    Status(reqwest::StatusCode),
    Transport(reqwest::Error),
    Malformed(serde_json::Error),
}

impl Failure {
    fn reason(&self) -> &'static str {
        match self {
            Failure::RouteMissing => "route_missing",
            Failure::Status(_) => "status",
            Failure::Transport(_) => "transport",
            Failure::Malformed(_) => "malformed",
        }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::RouteMissing => f.write_str("cloud-api answered 404"),
            Failure::Status(status) => write!(f, "cloud-api answered {status}"),
            Failure::Transport(error) => write!(f, "{error}"),
            Failure::Malformed(error) => write!(f, "not a model access document: {error}"),
        }
    }
}

/// Where the document is read from, and the service token that reads it. Not
/// `Debug`: it holds the token.
struct Source {
    url: String,
    token: String,
}

/// `ModelAccess::last_refresh` after a read that produced a snapshot. After
/// one that did not, it holds the `Failure::reason`.
const READ: &str = "read";

/// The process's view of cloud-api's model access document
/// (`AppState::model_access`). One per process, shared by every model of a
/// list: the document covers all of cloud-api's models.
#[derive(Default)]
pub struct ModelAccess {
    /// `None` = the setting is off: nothing is read and nothing is checked.
    source: Option<Source>,
    /// The last document read. `None` until there is one.
    snapshot: RwLock<Option<Arc<Snapshot>>>,
    /// How the last read ended (`READ` or a `Failure::reason`), `None` before
    /// the first one. A read that ends like the one before it is not logged
    /// again: the log has the changes, not every poll.
    last_refresh: Mutex<Option<&'static str>>,
}

impl ModelAccess {
    /// Off unless `VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS` is set, in which
    /// case startup has required `CLOUD_API_URL` and `CLOUD_API_USAGE_TOKEN`.
    /// Nothing is read here: `refresh` does that, and `start` runs it on the
    /// configured interval.
    pub fn new(config: &Config) -> Self {
        let source = match (&config.cloud_api_url, &config.cloud_api_usage_token) {
            (Some(base), Some(token)) if config.model_access_refresh_secs > 0 => Some(Source {
                url: format!("{}{MODEL_ACCESS_PATH}", base.trim_end_matches('/')),
                token: token.clone(),
            }),
            _ => None,
        };
        Self {
            source,
            ..Self::default()
        }
    }

    /// The setting is on.
    pub fn is_enabled(&self) -> bool {
        self.source.is_some()
    }

    /// A document has been read: restricted models are enforced.
    pub fn has_snapshot(&self) -> bool {
        self.current().is_some()
    }

    fn current(&self) -> Option<Arc<Snapshot>> {
        self.snapshot
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Refuse a request whose organization may not use `model`, with the
    /// answer of a model this process does not serve. `model` is the model
    /// the request would be served as: the configured one, or in list mode
    /// the entry the body's `model` selected. There the name in the answer
    /// is the caller's own string, as it is for an unknown model.
    ///
    /// Called after authentication and model selection, before anything is
    /// counted for the model, admitted, dispatched or billed. Nothing is
    /// logged per request; refusals are counted in
    /// `model_access_rejections_total` (`model` label in list mode).
    pub fn check(
        &self,
        auth: &RequireAuth,
        model: &str,
        label: ModelLabel,
    ) -> Result<(), AppError> {
        if self.source.is_none() {
            return Ok(());
        }
        let snapshot = self.current();
        if permits(
            snapshot.as_deref(),
            auth.cloud_api_key.is_some(),
            auth.org_id.as_deref(),
            model,
        ) {
            return Ok(());
        }
        model_counter!(label, "model_access_rejections_total").increment(1);
        Err(AppError::ModelNotFound {
            model: Some(model.to_string()),
        })
    }

    /// Read the document once and keep it. A failure keeps what was read
    /// before. Does nothing when the setting is off.
    pub async fn refresh(&self, client: &reqwest::Client) {
        let Some(source) = &self.source else {
            return;
        };
        match read_document(client, source).await {
            Ok(snapshot) => self.keep(snapshot),
            Err(failure) => self.note_failure(&failure),
        }
    }

    fn keep(&self, snapshot: Snapshot) {
        let (restricted_models, grants) = (snapshot.restricted_models(), snapshot.grants());
        let changed = {
            let mut held = self.snapshot.write().unwrap_or_else(|e| e.into_inner());
            let changed = held.as_deref() != Some(&snapshot);
            *held = Some(Arc::new(snapshot));
            changed
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |since_epoch| since_epoch.as_secs_f64());
        metrics::gauge!("model_access_last_refresh_timestamp_seconds").set(now);
        // The first document, one that differs from the last, and the first
        // one after a failure: not every poll.
        if self.note(READ) || changed {
            info!(restricted_models, grants, "Model access document read");
        }
    }

    fn note_failure(&self, failure: &Failure) {
        let reason = failure.reason();
        metrics::counter!("model_access_refresh_failures_total", "reason" => reason).increment(1);
        if !self.note(reason) {
            return;
        }
        match (self.has_snapshot(), failure) {
            (true, _) => warn!(
                reason,
                error = %failure,
                "Model access document could not be read, keeping the last one"
            ),
            (false, Failure::RouteMissing) => warn!(
                reason,
                "cloud-api does not serve the model access document: no model is restricted until it does"
            ),
            (false, _) => warn!(
                reason,
                error = %failure,
                "Model access document could not be read: no model is restricted until it can"
            ),
        }
    }

    /// Record how a read ended, and say whether that differs from the read
    /// before it.
    fn note(&self, outcome: &'static str) -> bool {
        self.last_refresh
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .replace(outcome)
            != Some(outcome)
    }
}

/// One read of the document. The service token is attached to this request
/// only, so it goes to `{CLOUD_API_URL}/v1/internal/model-access` and nowhere
/// else: the client drops the header when a redirect leaves that host.
async fn read_document(client: &reqwest::Client, source: &Source) -> Result<Snapshot, Failure> {
    let response = client
        .get(&source.url)
        .bearer_auth(&source.token)
        .timeout(REFRESH_TIMEOUT)
        .send()
        .await
        .map_err(Failure::Transport)?;
    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(Failure::RouteMissing);
    }
    if !status.is_success() {
        return Err(Failure::Status(status));
    }
    let body = response.bytes().await.map_err(Failure::Transport)?;
    Snapshot::parse(&body).map_err(Failure::Malformed)
}

/// Read the document now, and again `interval` after each read has ended, for
/// the life of the process. A slow cloud-api is therefore never asked twice
/// at once, nor again the moment it has answered.
pub fn spawn_refresh(access: Arc<ModelAccess>, client: reqwest::Client, interval: Duration) {
    tokio::spawn(async move {
        loop {
            access.refresh(&client).await;
            tokio::time::sleep(interval).await;
        }
    });
}

/// The process's model access state, with its refresh task running when
/// `VLLM_PROXY_MODEL_ACCESS_REFRESH_SECS` is set. `client` is the general
/// client, the one that talks to cloud-api. Must run inside a Tokio runtime.
pub fn start(config: &Config, client: &reqwest::Client) -> Arc<ModelAccess> {
    let access = Arc::new(ModelAccess::new(config));
    if access.is_enabled() {
        info!(
            refresh_secs = config.model_access_refresh_secs,
            "Per-organization model access enforced from cloud-api's document"
        );
        spawn_refresh(
            access.clone(),
            client.clone(),
            Duration::from_secs(config.model_access_refresh_secs),
        );
    }
    access
}

/// `ModelAccess::check` for a route only a single-model process serves: the
/// model is the configured one.
pub fn check_single_model(state: &AppState, auth: &RequireAuth) -> Result<(), AppError> {
    state
        .model_access
        .check(auth, &state.config.model_name, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(document: serde_json::Value) -> Snapshot {
        Snapshot::parse(document.to_string().as_bytes()).expect("a valid document")
    }

    #[test]
    fn parses_the_document_and_ignores_unknown_fields() {
        let parsed = snapshot(serde_json::json!({
            "object": "model_access",
            "generated_at": "2026-01-01T00:00:00Z",
            "models": [
                {"id": "vendor/alpha", "organization_ids": ["org-a", "org-b"], "note": "x"},
                {"id": "vendor/beta", "organization_ids": []}
            ]
        }));
        assert_eq!(parsed.restricted_models(), 2);
        assert_eq!(parsed.grants(), 2);
        assert!(parsed.allows("vendor/alpha", Some("org-a")));
        assert!(parsed.allows("vendor/alpha", Some("org-b")));
        assert!(!parsed.allows("vendor/alpha", Some("org-c")));
        // Listed with nobody on its list: restricted to no one.
        assert!(!parsed.allows("vendor/beta", Some("org-a")));
        // Not listed: not restricted.
        assert!(parsed.allows("vendor/gamma", Some("org-c")));
        assert!(parsed.allows("vendor/gamma", None));
    }

    #[test]
    fn an_empty_document_restricts_nothing() {
        let parsed = snapshot(serde_json::json!({"object": "model_access", "models": []}));
        assert_eq!(parsed, Snapshot::default());
        assert!(parsed.allows("vendor/alpha", None));
    }

    #[test]
    fn a_model_listed_twice_keeps_both_lists() {
        let parsed = snapshot(serde_json::json!({"models": [
            {"id": "vendor/alpha", "organization_ids": ["org-a"]},
            {"id": "vendor/alpha", "organization_ids": ["org-b"]}
        ]}));
        assert_eq!(parsed.restricted_models(), 1);
        assert!(parsed.allows("vendor/alpha", Some("org-a")));
        assert!(parsed.allows("vendor/alpha", Some("org-b")));
    }

    #[test]
    fn a_malformed_document_is_refused_as_a_whole() {
        for body in [
            "",
            "not json",
            "[]",
            "{}",
            r#"{"object": "model_access"}"#,
            r#"{"models": null}"#,
            r#"{"models": {"vendor/alpha": ["org-a"]}}"#,
            r#"{"models": ["vendor/alpha"]}"#,
            r#"{"models": [{"organization_ids": ["org-a"]}]}"#,
            r#"{"models": [{"id": "vendor/alpha"}]}"#,
            r#"{"models": [{"id": "vendor/alpha", "organization_ids": null}]}"#,
            r#"{"models": [{"id": "vendor/alpha", "organization_ids": "org-a"}]}"#,
            r#"{"models": [{"id": "vendor/alpha", "organization_ids": [1]}]}"#,
            r#"{"models": [{"id": 1, "organization_ids": []}]}"#,
            // One bad entry spoils the document: the good one is not applied.
            r#"{"models": [{"id": "vendor/alpha", "organization_ids": []}, {"id": "vendor/beta"}]}"#,
            r#"{"error": {"message": "unauthorized"}}"#,
        ] {
            assert!(Snapshot::parse(body.as_bytes()).is_err(), "{body}");
        }
    }

    #[test]
    fn matching_is_exact() {
        let parsed = snapshot(serde_json::json!({"models": [
            {"id": "vendor/alpha", "organization_ids": ["org-a"]}
        ]}));
        // Another case or a longer name is another model, which the document
        // does not restrict.
        assert!(parsed.allows("vendor/Alpha", Some("org-z")));
        assert!(parsed.allows("vendor/alpha-2", Some("org-z")));
        // An organization is on the list as written or not at all.
        assert!(!parsed.allows("vendor/alpha", Some("ORG-A")));
        assert!(!parsed.allows("vendor/alpha", Some("org-a ")));
        assert!(!parsed.allows("vendor/alpha", Some("")));
    }

    #[test]
    fn the_decision() {
        let held = snapshot(serde_json::json!({"models": [
            {"id": "vendor/alpha", "organization_ids": ["org-a"]}
        ]}));
        let held = Some(&held);
        // A granted organization, and anyone on a model that is not restricted.
        assert!(permits(held, true, Some("org-a"), "vendor/alpha"));
        assert!(permits(held, true, Some("org-b"), "vendor/beta"));
        assert!(permits(held, true, None, "vendor/beta"));
        // Another organization, and a key without one.
        assert!(!permits(held, true, Some("org-b"), "vendor/alpha"));
        assert!(!permits(held, true, None, "vendor/alpha"));
        // The config token is not checked, with or without an organization.
        assert!(permits(held, false, None, "vendor/alpha"));
        assert!(permits(held, false, Some("org-b"), "vendor/alpha"));
        // No document read yet: nothing is restricted.
        assert!(permits(None, true, Some("org-b"), "vendor/alpha"));
        assert!(permits(None, true, None, "vendor/alpha"));
    }

    #[test]
    fn a_read_is_logged_when_its_outcome_changes() {
        let access = ModelAccess::default();
        assert!(access.note(READ));
        assert!(!access.note(READ));
        assert!(access.note("status"));
        assert!(!access.note("status"));
        assert!(access.note("transport"));
        assert!(access.note(READ));
    }

    #[test]
    fn off_by_default() {
        let access = ModelAccess::default();
        assert!(!access.is_enabled());
        assert!(!access.has_snapshot());
    }
}
