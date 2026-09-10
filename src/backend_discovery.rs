//! Dynamic backend discovery from the model-proxy registry.
//!
//! In gateway deployments the proxy fronts a whole model fleet instead of the
//! engines of one host, and the fleet changes without a proxy restart (hosts
//! are drained, redeployed, added). model-proxy already knows the live set: its
//! authenticated `GET /backends/list?domain=<model domain>` returns one stable
//! *handle* per registered backend, and `<label>-b<handle>.<base>` is an SNI
//! that routes to exactly that backend. This module polls that listing and
//! keeps the `BackendPool` membership in sync, so the existing least-connections
//! selection, per-backend health checks, and conversation affinity all operate
//! over the live fleet.
//!
//! Failure policy is fail-safe for serving: a poll that errors, returns a
//! malformed body, or returns an empty list keeps the last known membership
//! (the health checker still removes backends that are actually gone). Only a
//! well-formed non-empty listing changes membership.

use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use tracing::{info, warn};

use crate::backend_pool::{BackendPool, MembershipChange};

/// Placeholder in `url_template` replaced by each backend handle.
pub const HANDLE_PLACEHOLDER: &str = "{handle}";

#[derive(Debug, Clone)]
pub struct BackendDiscoveryConfig {
    /// Full listing URL, e.g.
    /// `https://completions.near.ai/backends/list?domain=glm-5-3-flash.completions.near.ai`.
    pub url: String,
    /// Bearer token for the listing endpoint (model-proxy admin token).
    pub token: Option<String>,
    /// Backend base URL template containing `{handle}`, e.g.
    /// `https://glm-5-3-flash-b{handle}.completions.near.ai`.
    pub url_template: String,
    pub interval_secs: u64,
    pub timeout_secs: u64,
}

#[derive(Debug, Deserialize)]
struct ListedBackend {
    handle: String,
    #[serde(default = "default_true")]
    healthy: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct Listing {
    backends: Vec<ListedBackend>,
}

/// A handle is substituted into a hostname label, so only accept what a
/// hostname label can carry: 1–63 ASCII alphanumerics/hyphens, no leading or
/// trailing hyphen. Lowercased for a stable identity.
fn normalize_handle(raw: &str) -> Result<String, String> {
    let handle = raw.trim().to_ascii_lowercase();
    if handle.is_empty() || handle.len() > 63 {
        return Err("handle length must be 1..=63".to_string());
    }
    if handle.starts_with('-') || handle.ends_with('-') {
        return Err("handle must not start or end with '-'".to_string());
    }
    if !handle
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err("handle must be ASCII alphanumeric or '-'".to_string());
    }
    Ok(handle)
}

/// Parse a registry listing into `(base_url, healthy)` members, sorted by URL
/// for a stable membership order. Errors on malformed JSON, a bad handle, or a
/// template without the placeholder. An empty listing is `Ok(vec![])`; the
/// caller decides whether to honor it.
pub fn members_from_listing(
    body: &[u8],
    url_template: &str,
) -> Result<Vec<(String, bool)>, String> {
    if !url_template.contains(HANDLE_PLACEHOLDER) {
        return Err(format!("url template must contain {HANDLE_PLACEHOLDER}"));
    }
    let listing: Listing =
        serde_json::from_slice(body).map_err(|e| format!("invalid listing JSON: {e}"))?;
    let mut members: Vec<(String, bool)> = Vec::with_capacity(listing.backends.len());
    for backend in listing.backends {
        let handle = normalize_handle(&backend.handle)?;
        let url = url_template.replace(HANDLE_PLACEHOLDER, &handle);
        if !members.iter().any(|(u, _)| *u == url) {
            members.push((url, backend.healthy));
        }
    }
    members.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(members)
}

/// One discovery round: fetch the listing and reconcile the pool. Returns the
/// membership delta on success. On any failure the pool is left untouched and
/// the reason is returned for logging/metrics.
pub async fn poll_once(
    client: &reqwest::Client,
    config: &BackendDiscoveryConfig,
    pool: &BackendPool,
) -> Result<MembershipChange, String> {
    let mut request = client
        .get(&config.url)
        .timeout(Duration::from_secs(config.timeout_secs.max(1)));
    if let Some(token) = &config.token {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.map_err(|e| {
        let kind = if e.is_timeout() {
            "timeout"
        } else if e.is_connect() {
            "connect"
        } else {
            "transport"
        };
        format!("registry request failed ({kind})")
    })?;
    let status = response.status();
    if !status.is_success() {
        // Drain so the connection returns to the pool; body content is not logged.
        let _ = response.bytes().await;
        return Err(format!("registry returned HTTP {}", status.as_u16()));
    }
    let body = response
        .bytes()
        .await
        .map_err(|_| "registry body read failed".to_string())?;
    if body.len() > 1024 * 1024 {
        return Err("registry listing exceeds 1 MiB".to_string());
    }
    let members = members_from_listing(&body, &config.url_template)?;
    if members.is_empty() {
        return Err("registry listing is empty; keeping current membership".to_string());
    }
    Ok(pool.set_backends(members))
}

/// Run discovery forever at the configured interval. Membership changes are
/// logged with backend URLs (infrastructure addresses, never customer data).
pub fn spawn_discovery(
    pool: Arc<BackendPool>,
    client: reqwest::Client,
    config: BackendDiscoveryConfig,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(config.interval_secs.max(1)));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await; // the initial poll ran synchronously at startup
        let mut consecutive_failures: u32 = 0;
        loop {
            tick.tick().await;
            match poll_once(&client, &config, &pool).await {
                Ok(change) => {
                    consecutive_failures = 0;
                    metrics::counter!("backend_discovery_polls_total", "outcome" => "ok")
                        .increment(1);
                    if !change.is_empty() {
                        info!(
                            added = ?change.added,
                            removed = ?change.removed,
                            backends = pool.len(),
                            "Backend discovery updated pool membership"
                        );
                    }
                }
                Err(reason) => {
                    consecutive_failures = consecutive_failures.saturating_add(1);
                    metrics::counter!("backend_discovery_polls_total", "outcome" => "error")
                        .increment(1);
                    warn!(
                        reason = %reason,
                        consecutive_failures,
                        backends = pool.len(),
                        "Backend discovery poll failed; keeping last known membership"
                    );
                }
            }
            metrics::gauge!("backend_discovery_consecutive_failures")
                .set(consecutive_failures as f64);
            metrics::gauge!("backend_pool_size").set(pool.len() as f64);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE: &str = "https://glm-b{handle}.example.test";

    #[test]
    fn parses_listing_into_sorted_deduplicated_members() {
        let body = br#"{"domain":"glm.example.test","requested_domain":"glm.example.test",
            "backends":[{"handle":"B2B2B2B2B2B2","healthy":false},
                        {"handle":"a1a1a1a1a1a1","healthy":true},
                        {"handle":"a1a1a1a1a1a1","healthy":false}]}"#;
        let members = members_from_listing(body, TEMPLATE).unwrap();
        assert_eq!(
            members,
            vec![
                ("https://glm-ba1a1a1a1a1a1.example.test".to_string(), true),
                ("https://glm-bb2b2b2b2b2b2.example.test".to_string(), false),
            ]
        );
    }

    #[test]
    fn healthy_defaults_to_true_when_absent() {
        let body = br#"{"backends":[{"handle":"abc"}]}"#;
        let members = members_from_listing(body, TEMPLATE).unwrap();
        assert_eq!(
            members,
            vec![("https://glm-babc.example.test".to_string(), true)]
        );
    }

    #[test]
    fn empty_listing_is_ok_but_empty() {
        let body = br#"{"backends":[]}"#;
        assert!(members_from_listing(body, TEMPLATE).unwrap().is_empty());
    }

    #[test]
    fn rejects_unsafe_handles_and_malformed_bodies() {
        for bad in [
            r#"{"backends":[{"handle":"../../etc"}]}"#,
            r#"{"backends":[{"handle":"a.b"}]}"#,
            r#"{"backends":[{"handle":"-abc"}]}"#,
            r#"{"backends":[{"handle":""}]}"#,
            r#"{"backends":[{"handle":"abc","healthy":"yes"}]}"#,
            r#"{"backends":"nope"}"#,
            r#"not json"#,
        ] {
            assert!(
                members_from_listing(bad.as_bytes(), TEMPLATE).is_err(),
                "should reject: {bad}"
            );
        }
        let too_long = format!(r#"{{"backends":[{{"handle":"{}"}}]}}"#, "a".repeat(64));
        assert!(members_from_listing(too_long.as_bytes(), TEMPLATE).is_err());
    }

    #[test]
    fn rejects_template_without_placeholder() {
        let body = br#"{"backends":[{"handle":"abc"}]}"#;
        assert!(members_from_listing(body, "https://fixed.example.test").is_err());
    }
}
