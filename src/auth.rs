use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use serde::Deserialize;
use subtle::ConstantTimeEq;
use tracing::warn;

use crate::config::Config;
use crate::error::AppError;
use crate::request_tracing::TracingIds;
use crate::AppState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthPath {
    TrustedConfigToken,
    CloudApiKey,
}

impl AuthPath {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::TrustedConfigToken => "trusted_config_token",
            Self::CloudApiKey => "cloud_api_key",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngressRouteKind {
    Canonical,
    Indexed,
    Long,
    LongIndexed,
    Other,
    Missing,
}

impl IngressRouteKind {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Indexed => "indexed",
            Self::Long => "long",
            Self::LongIndexed => "long_indexed",
            Self::Other => "other",
            Self::Missing => "missing",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestSource {
    pub auth_path: AuthPath,
    pub ingress_route: IngressRouteKind,
}

fn classify_ingress_host(host: Option<&str>) -> IngressRouteKind {
    let Some(host) = host else {
        return IngressRouteKind::Missing;
    };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if !(host.ends_with(".completions.near.ai") || host.ends_with(".completions-stg.near.ai")) {
        return IngressRouteKind::Other;
    }

    let Some(label) = host.split('.').next() else {
        return IngressRouteKind::Other;
    };
    let (base_label, indexed) = label
        .rsplit_once("-i")
        .filter(|(_, index)| !index.is_empty() && index.chars().all(|c| c.is_ascii_digit()))
        .map_or((label, false), |(base, _)| (base, true));
    let long = base_label.ends_with("-long");

    match (long, indexed) {
        (false, false) => IngressRouteKind::Canonical,
        (false, true) => IngressRouteKind::Indexed,
        (true, false) => IngressRouteKind::Long,
        (true, true) => IngressRouteKind::LongIndexed,
    }
}

fn classify_ingress_route(parts: &Parts) -> IngressRouteKind {
    let authority_host = parts.uri.authority().map(|authority| authority.host());
    let header_host = parts
        .headers
        .get(axum::http::header::HOST)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(':').next());
    classify_ingress_host(authority_host.or(header_host))
}

/// Extractor that validates Bearer token authentication.
/// Use as a handler parameter to require auth on a route.
///
/// If authentication was via a cloud API key (`sk-` prefix),
/// `cloud_api_key` contains the key for downstream usage reporting.
/// `org_id`, `workspace_id`, and `api_key_id` carry the subject identity
/// parsed from the `/v1/check_api_key` response — populated when cloud-api
/// returns them, left as `None` against older cloud-api builds that don't
/// surface those fields yet.
pub struct RequireAuth {
    pub cloud_api_key: Option<String>,
    pub org_id: Option<String>,
    pub workspace_id: Option<String>,
    pub api_key_id: Option<String>,
    /// Correlation ID selected by request middleware. Direct-key usage
    /// reporting forwards this to Cloud API so auth, completion, and billing
    /// handoff logs can be joined without using the raw API key.
    pub request_id: Option<String>,
    pub request_source: RequestSource,
}

/// Subject identity extracted from a successful `/v1/check_api_key` response.
/// Each field is `Option` so we degrade gracefully when paired with a
/// cloud-api version that doesn't surface that field yet.
#[derive(Debug, Default, Clone)]
pub struct AuthSubject {
    pub org_id: Option<String>,
    pub workspace_id: Option<String>,
    pub api_key_id: Option<String>,
}

/// On-the-wire shape of `/v1/check_api_key`'s success body. Cloud-api uses
/// `organization_id` (and PR #635 onward also `workspace_id`); `api_key_id`
/// is a follow-up addition. We deserialize lenient on all of them.
#[derive(Debug, Deserialize)]
struct CheckApiKeyResponse {
    #[serde(default)]
    organization_id: Option<String>,
    #[serde(default)]
    workspace_id: Option<String>,
    #[serde(default)]
    api_key_id: Option<String>,
}

/// Constant-time token comparison to prevent timing attacks.
/// Returns true if `a` and `b` are equal, using a fixed-time algorithm
/// that does not short-circuit on the first mismatched byte.
pub(crate) fn token_eq(a: &str, b: &str) -> bool {
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

/// Classify a `reqwest::Error` into a coarse bucket so failure modes are
/// distinguishable in logs and metrics. The raw `Display` is the same string
/// (`"error sending request for url ..."`) for connect/RST/EOF/etc, which is
/// not enough to diagnose intermittent transport failures.
fn classify_reqwest_error(e: &reqwest::Error) -> &'static str {
    if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connect"
    } else if e.is_body() {
        "body"
    } else if e.is_decode() {
        "decode"
    } else if e.is_request() {
        "request"
    } else {
        "other"
    }
}

/// Render the full error chain (`source()` → `source()` → ...) so we don't
/// lose the underlying hyper / rustls / io::Error message that reqwest hides.
fn error_chain(e: &(dyn std::error::Error + 'static)) -> String {
    let mut out = e.to_string();
    let mut src = e.source();
    while let Some(inner) = src {
        out.push_str(" | ");
        out.push_str(&inner.to_string());
        src = inner.source();
    }
    out
}

/// Validate an `sk-` prefixed API key against the cloud API.
///
/// Retries on transport errors and 5xx responses with exponential backoff and
/// full jitter. Terminal responses (200/402/429 and other 4xx) return
/// immediately. On 200, parses the response body for the subject identity
/// (`organization_id`, `workspace_id`, `api_key_id`) — any missing field
/// degrades to `None` so older cloud-api builds keep working.
async fn check_cloud_api_key(
    http_client: &reqwest::Client,
    config: &Config,
    cloud_api_url: &str,
    token: &str,
    request_id: Option<&str>,
) -> Result<AuthSubject, AppError> {
    let url = format!("{cloud_api_url}/v1/check_api_key");
    let max_attempts = config.cloud_api_auth_max_attempts.max(1);
    let per_attempt_timeout = std::time::Duration::from_secs(config.cloud_api_auth_timeout_secs);
    let initial_backoff_ms = config.cloud_api_auth_initial_backoff_ms;

    for attempt in 1..=max_attempts {
        let mut request = http_client
            .post(&url)
            .header("authorization", format!("Bearer {token}"))
            .timeout(per_attempt_timeout);
        if let Some(request_id) = request_id {
            request = request.header("x-request-id", request_id);
        }
        let result = request.send().await;

        let (status, body_bytes) = match result {
            Ok(response) => {
                let s = response.status().as_u16();
                // Drain the body before dropping the response so reqwest can
                // return the connection to the pool; otherwise hyper closes
                // it. Each forced reconnect adds a fresh SYN to the QEMU
                // hostfwd backlog (the documented backlog-1 saturation), so
                // dropping pool reuse here would actively worsen the problem
                // this PR is meant to mitigate. Errors during the drain
                // don't change the auth outcome — we already have the status.
                let body = response.bytes().await.unwrap_or_default();
                (s, body)
            }
            Err(e) => {
                let kind = classify_reqwest_error(&e);
                let chain = error_chain(&e);
                metrics::counter!(
                    "cloud_api_auth_attempts_total",
                    "outcome" => "transport_error",
                    "kind" => kind,
                )
                .increment(1);
                if attempt < max_attempts {
                    let delay = backoff_delay(initial_backoff_ms, attempt);
                    warn!(
                        error = %e,
                        error_chain = %chain,
                        error_kind = kind,
                        is_connect = e.is_connect(),
                        is_timeout = e.is_timeout(),
                        is_request = e.is_request(),
                        is_body = e.is_body(),
                        attempt,
                        max_attempts,
                        retry_in_ms = delay.as_millis() as u64,
                        "Cloud API key check transport error, retrying"
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                warn!(
                    error = %e,
                    error_chain = %chain,
                    error_kind = kind,
                    is_connect = e.is_connect(),
                    is_timeout = e.is_timeout(),
                    is_request = e.is_request(),
                    is_body = e.is_body(),
                    attempt,
                    "Cloud API key check request failed"
                );
                return Err(AppError::Unauthorized);
            }
        };

        match status {
            200 => {
                metrics::counter!("cloud_api_auth_attempts_total", "outcome" => "ok").increment(1);
                // Older cloud-api builds may return a non-JSON body or omit
                // identity fields. Tolerate both shapes: any parse failure
                // or missing field downgrades to `None` and the caller
                // keeps using the legacy reporting path.
                let subject = serde_json::from_slice::<CheckApiKeyResponse>(&body_bytes)
                    .map(|r| AuthSubject {
                        org_id: r.organization_id,
                        workspace_id: r.workspace_id,
                        api_key_id: r.api_key_id,
                    })
                    .unwrap_or_default();
                return Ok(subject);
            }
            402 => {
                metrics::counter!(
                    "cloud_api_auth_attempts_total",
                    "outcome" => "insufficient_credits"
                )
                .increment(1);
                return Err(AppError::InsufficientCredits);
            }
            429 => {
                metrics::counter!("cloud_api_auth_attempts_total", "outcome" => "rate_limited")
                    .increment(1);
                return Err(AppError::RateLimited);
            }
            s if (500..600).contains(&s) => {
                metrics::counter!("cloud_api_auth_attempts_total", "outcome" => "upstream_5xx")
                    .increment(1);
                if attempt < max_attempts {
                    let delay = backoff_delay(initial_backoff_ms, attempt);
                    warn!(
                        status = s,
                        attempt,
                        max_attempts,
                        retry_in_ms = delay.as_millis() as u64,
                        "Cloud API key check returned 5xx, retrying"
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                warn!(status = s, attempt, "Cloud API key check failed (5xx)");
                return Err(AppError::Unauthorized);
            }
            s => {
                metrics::counter!("cloud_api_auth_attempts_total", "outcome" => "rejected")
                    .increment(1);
                warn!(status = s, "Cloud API key check rejected");
                return Err(AppError::Unauthorized);
            }
        }
    }

    // Unreachable: the loop above either returns or `continue`s, and the final
    // attempt always returns. Defensive fallback.
    Err(AppError::Unauthorized)
}

/// Full-jitter exponential backoff: random uniform between 0 and
/// `initial_ms * 2^(attempt-1)`, capped at 5s.
///
/// Full jitter (rather than fixed exponential) avoids retry stampedes when
/// many in-flight requests fail simultaneously due to a shared upstream blip.
fn backoff_delay(initial_ms: u64, attempt: usize) -> std::time::Duration {
    if initial_ms == 0 {
        return std::time::Duration::from_millis(0);
    }
    let exp = (attempt as u32).saturating_sub(1).min(10);
    let upper = initial_ms.saturating_mul(1u64 << exp).min(5_000);
    let jitter = rand::random_range(0..=upper);
    std::time::Duration::from_millis(jitter)
}

impl FromRequestParts<AppState> for RequireAuth {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let ingress_route = classify_ingress_route(parts);
        // request_id_middleware has already normalized this to a UUID. Reuse it
        // for the Cloud API auth check so direct traffic is joinable across the
        // check-api-key and completion logs.
        let request_id = parts
            .extensions
            .get::<TracingIds>()
            .map(|ids| ids.request_id.clone());
        let auth_header = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok());

        match auth_header {
            Some(header) if header.starts_with("Bearer ") => {
                let token = &header[7..];
                // Constant-time comparison against each configured admin token
                // so multiple tokens can be active simultaneously (rotation).
                if state.config.tokens.iter().any(|t| token_eq(token, t)) {
                    return Ok(RequireAuth {
                        cloud_api_key: None,
                        org_id: None,
                        workspace_id: None,
                        api_key_id: None,
                        request_id,
                        request_source: RequestSource {
                            auth_path: AuthPath::TrustedConfigToken,
                            ingress_route,
                        },
                    });
                }

                // Fallback: validate sk- tokens against cloud API
                if token.starts_with("sk-") {
                    if let Some(cloud_api_url) = &state.config.cloud_api_url {
                        let subject = check_cloud_api_key(
                            &state.http_client,
                            &state.config,
                            cloud_api_url,
                            token,
                            request_id.as_deref(),
                        )
                        .await?;
                        return Ok(RequireAuth {
                            cloud_api_key: Some(token.to_string()),
                            org_id: subject.org_id,
                            workspace_id: subject.workspace_id,
                            api_key_id: subject.api_key_id,
                            request_id,
                            request_source: RequestSource {
                                auth_path: AuthPath::CloudApiKey,
                                ingress_route,
                            },
                        });
                    }
                }

                Err(AppError::Unauthorized)
            }
            _ => Err(AppError::Unauthorized),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    fn classifies_bounded_ingress_route_kinds() {
        assert_eq!(
            classify_ingress_host(Some("glm-5-2.completions.near.ai")),
            IngressRouteKind::Canonical
        );
        assert_eq!(
            classify_ingress_host(Some("glm-5-2-i7.completions.near.ai")),
            IngressRouteKind::Indexed
        );
        assert_eq!(
            classify_ingress_host(Some("glm-5-2-long.completions.near.ai")),
            IngressRouteKind::Long
        );
        assert_eq!(
            classify_ingress_host(Some("glm-5-2-long-i2.completions-stg.near.ai")),
            IngressRouteKind::LongIndexed
        );
        assert_eq!(
            classify_ingress_host(Some("127.0.0.1")),
            IngressRouteKind::Other
        );
        assert_eq!(classify_ingress_host(None), IngressRouteKind::Missing);
    }

    /// Measure the median duration (in nanoseconds) of `iterations` calls to `compare_fn(a, b)`.
    fn median_nanos(
        a: &str,
        b: &str,
        compare_fn: fn(&str, &str) -> bool,
        iterations: usize,
    ) -> u64 {
        let mut durations = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let a = black_box(a);
            let b = black_box(b);
            let start = Instant::now();
            let _ = black_box(compare_fn(a, b));
            durations.push(start.elapsed().as_nanos() as u64);
        }
        durations.sort_unstable();
        durations[durations.len() / 2]
    }

    #[test]
    fn test_error_chain_walks_source_chain() {
        #[derive(Debug)]
        struct Wrap {
            msg: &'static str,
            inner: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
        }
        impl std::fmt::Display for Wrap {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.msg)
            }
        }
        impl std::error::Error for Wrap {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.inner.as_deref().map(|e| e as _)
            }
        }

        let inner = Wrap {
            msg: "peer reset",
            inner: None,
        };
        let middle = Wrap {
            msg: "tls write",
            inner: Some(Box::new(inner)),
        };
        let outer = Wrap {
            msg: "send failed",
            inner: Some(Box::new(middle)),
        };

        let s = error_chain(&outer);
        assert!(s.contains("send failed"), "must include outer: {s}");
        assert!(s.contains("tls write"), "must include middle: {s}");
        assert!(s.contains("peer reset"), "must include innermost: {s}");
        assert_eq!(
            s.matches('|').count(),
            2,
            "two separators for three frames: {s}"
        );
    }

    #[tokio::test]
    async fn test_classify_reqwest_error_connect() {
        // Hitting a closed loopback port produces a connect error.
        let client = reqwest::Client::new();
        let err = client
            .get("http://127.0.0.1:1") // reserved, nothing listens
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .expect_err("must fail to connect");
        assert_eq!(classify_reqwest_error(&err), "connect");
        assert!(err.is_connect());
    }

    #[test]
    fn test_backoff_delay_zero_initial_returns_zero() {
        for attempt in 1..=5 {
            assert_eq!(backoff_delay(0, attempt), std::time::Duration::ZERO);
        }
    }

    #[test]
    fn test_backoff_delay_bounded_by_exponential_cap() {
        // Full jitter draws uniformly in [0, initial * 2^(attempt-1)], capped at 5s.
        // Sample many times to confirm the upper bound holds and that the
        // distribution actually grows with attempt (not always 0).
        let initial = 100u64;
        for attempt in 1..=4 {
            let cap_ms = initial * (1u64 << (attempt as u32 - 1));
            let mut max_seen = 0u64;
            for _ in 0..200 {
                let d = backoff_delay(initial, attempt).as_millis() as u64;
                assert!(
                    d <= cap_ms,
                    "attempt {attempt}: {d}ms exceeds cap {cap_ms}ms"
                );
                max_seen = max_seen.max(d);
            }
            // With 200 draws, the max should land somewhere in the upper half
            // of the window for any non-trivial cap. Use a loose bound to
            // avoid flakiness while still catching a stuck-at-zero regression.
            if cap_ms >= 4 {
                assert!(
                    max_seen > 0,
                    "attempt {attempt}: backoff never produced a non-zero delay (cap {cap_ms}ms)"
                );
            }
        }
    }

    #[test]
    fn test_backoff_delay_overall_cap() {
        // A huge initial value must still saturate at 5s.
        for _ in 0..50 {
            assert!(backoff_delay(10_000, 5).as_millis() as u64 <= 5_000);
        }
    }

    #[test]
    fn test_check_api_key_response_full_shape() {
        // Cloud-api with both #635 and the future api_key_id addition shipped.
        let body = br#"{"valid":true,"organization_id":"org-uuid",
            "workspace_id":"ws-uuid","api_key_id":"key-uuid"}"#;
        let parsed: CheckApiKeyResponse = serde_json::from_slice(body).unwrap();
        assert_eq!(parsed.organization_id.as_deref(), Some("org-uuid"));
        assert_eq!(parsed.workspace_id.as_deref(), Some("ws-uuid"));
        assert_eq!(parsed.api_key_id.as_deref(), Some("key-uuid"));
    }

    #[test]
    fn test_check_api_key_response_pre_635_shape() {
        // Older cloud-api built before #635: only organization_id is surfaced.
        // workspace_id and api_key_id must degrade to None so the reporter
        // falls back to the legacy sk- path.
        let body = br#"{"valid":true,"organization_id":"org-uuid"}"#;
        let parsed: CheckApiKeyResponse = serde_json::from_slice(body).unwrap();
        assert_eq!(parsed.organization_id.as_deref(), Some("org-uuid"));
        assert!(parsed.workspace_id.is_none());
        assert!(parsed.api_key_id.is_none());
    }

    #[test]
    fn test_check_api_key_response_completely_missing() {
        // Hypothetical future cloud-api or a malformed body — must not
        // panic, must yield all-None.
        let body = br#"{"valid":true}"#;
        let parsed: CheckApiKeyResponse = serde_json::from_slice(body).unwrap();
        assert!(parsed.organization_id.is_none());
        assert!(parsed.workspace_id.is_none());
        assert!(parsed.api_key_id.is_none());
    }

    #[test]
    fn test_token_eq_correctness() {
        assert!(token_eq("secret-token-123", "secret-token-123"));
        assert!(!token_eq("secret-token-123", "wrong-token-456"));
        assert!(!token_eq("secret-token-123", "secret-token-124"));
        assert!(!token_eq("short", "short-but-longer"));
        assert!(!token_eq("", "notempty"));
        assert!(token_eq("", ""));
    }

    /// Verifies that `token_eq` (constant-time) takes the same time regardless of
    /// where the mismatch occurs. The ratio of late_mismatch / early_mismatch
    /// should be close to 1.0, indicating no timing leak.
    #[test]
    fn test_constant_time_comparison_no_timing_discrepancy() {
        let secret = "a]9$kL2#mP7!xR4&wQ8*nJ5^tY1+hF3@vB6%cD0".repeat(8);
        let early_mismatch = format!("X{}", &secret[1..]);
        let late_mismatch = format!("{}X", &secret[..secret.len() - 1]);

        let iterations = 50_000;

        // Warm up
        median_nanos(&secret, &early_mismatch, token_eq, 1_000);
        median_nanos(&secret, &late_mismatch, token_eq, 1_000);

        let t_early = median_nanos(&secret, &early_mismatch, token_eq, iterations);
        let t_late = median_nanos(&secret, &late_mismatch, token_eq, iterations);

        let ratio = t_late as f64 / t_early.max(1) as f64;
        eprintln!("Constant-time:  early={t_early}ns  late={t_late}ns  ratio={ratio:.2}");

        // Constant-time comparison should have a ratio very close to 1.0.
        // A real timing leak (using ==) would show ratios of 5–50×.
        // We use a generous threshold to tolerate noise on shared CI runners
        // while still catching real timing side-channels.
        assert!(
            ratio < 2.0,
            "Constant-time comparison should not leak timing (ratio {ratio:.2} >= 2.0)"
        );
    }
}
