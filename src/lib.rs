use std::sync::Arc;

use axum::http::{HeaderMap, HeaderValue, Request};
use axum::middleware::Next;
use axum::response::Response;
use tracing::Instrument;

pub mod agent_loop;
pub mod attestation;
pub mod attestation_sdk;
pub mod auth;
pub mod backend_pool;
pub mod cache;
pub mod config;
pub mod encryption;
pub mod error;
pub mod gpu_evidence_delegate;
pub mod image_validation;
pub mod metrics_middleware;
pub mod ohttp_gateway;
pub mod proxy;
pub mod rate_limit;
pub mod routes;
pub mod signing;
pub mod startup_checks;
pub mod types;

/// Shared application state available to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<config::Config>,
    pub signing: Arc<signing::SigningPair>,
    pub cache: Arc<cache::ChatCache>,
    pub attestation_cache: Arc<attestation::AttestationCache>,
    pub http_client: reqwest::Client,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    /// Live SHA-256 hash of the TLS certificate's SPKI (Subject Public Key
    /// Info). The tracker re-stats the cert on every attestation-cache
    /// refresh tick and re-hashes when `mtime` advances, so a certbot-driven
    /// rotation in the co-located ingress sidecar is picked up automatically
    /// without restarting the proxy. Routes call `.current()` on the hot path.
    pub tls_cert_fingerprint: Arc<attestation::TlsCertTracker>,
    pub backend_pool: Arc<backend_pool::BackendPool>,
    pub ohttp_gateway: Option<Arc<ohttp_gateway::OhttpGateway>>,
    pub ohttp_attestation_ed25519: Option<types::OhttpAttestation>,
}

/// Tracing correlation IDs for a request, parsed once by `request_id_middleware`
/// from `X-Request-Id` / `X-Org-Id` / `X-Workspace-Id` headers (propagated by
/// cloud-api) and inserted into the request extensions. Handlers extract via
/// `axum::Extension<TracingIds>` and pass it through to `ProxyOpts` so the same
/// IDs are forwarded to vLLM/SGLang and emitted on the per-request log line.
#[derive(Clone, Debug)]
pub struct TracingIds {
    /// The request ID used in spans, logs, and the response header. Either the
    /// inbound `X-Request-Id` (when `request_id_inbound` is true) or a freshly
    /// generated UUID.
    pub request_id: String,
    /// Whether `request_id` came from the inbound header. When false, the
    /// upstream call does **not** receive `X-Request-Id` (we don't forward
    /// our own generated UUID — it would just be noise to vLLM).
    pub request_id_inbound: bool,
    /// Inbound `X-Org-Id`, if present and ASCII-valid.
    pub org_id: Option<String>,
    /// Inbound `X-Workspace-Id`, if present and ASCII-valid.
    pub workspace_id: Option<String>,
}

impl TracingIds {
    /// Parse from inbound request headers. Generates a fresh UUID for
    /// `request_id` when the header is absent.
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let inbound_request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let request_id_inbound = inbound_request_id.is_some();
        let request_id = inbound_request_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let org_id = headers
            .get("x-org-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let workspace_id = headers
            .get("x-workspace-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        Self {
            request_id,
            request_id_inbound,
            org_id,
            workspace_id,
        }
    }

    /// `(name, value)` pairs to forward to the upstream backend. Only includes
    /// headers that were actually present on the inbound request — we don't
    /// fabricate values upstream when the caller didn't supply them.
    pub fn upstream_headers(&self) -> impl Iterator<Item = (&'static str, &str)> + '_ {
        let req = self
            .request_id_inbound
            .then_some(("x-request-id", self.request_id.as_str()));
        let org = self.org_id.as_deref().map(|v| ("x-org-id", v));
        let ws = self.workspace_id.as_deref().map(|v| ("x-workspace-id", v));
        req.into_iter().chain(org).chain(ws)
    }

    /// Return `org_id` for log fields, defaulting to `""` when absent.
    /// Preserves pre-refactor behavior of recording an empty string rather
    /// than omitting the field entirely.
    pub fn org_id_or_empty(&self) -> &str {
        self.org_id.as_deref().unwrap_or("")
    }

    /// Return `workspace_id` for log fields, defaulting to `""` when absent.
    pub fn workspace_id_or_empty(&self) -> &str {
        self.workspace_id.as_deref().unwrap_or("")
    }
}

/// Request ID middleware: parses tracing correlation headers, stores them in
/// the request extensions for handlers, attaches them to the tracing span so
/// every log line carries them, and echoes `X-Request-Id` back to the caller.
pub async fn request_id_middleware(mut request: Request<axum::body::Body>, next: Next) -> Response {
    let tracing_ids = TracingIds::from_headers(request.headers());

    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let span = tracing::info_span!(
        "request",
        request_id = %tracing_ids.request_id,
        org_id = %tracing_ids.org_id_or_empty(),
        workspace_id = %tracing_ids.workspace_id_or_empty(),
        method = %method,
        path = %path,
    );

    let echo_request_id = tracing_ids.request_id.clone();
    request.extensions_mut().insert(tracing_ids);

    let mut response = next.run(request).instrument(span).await;
    if let Ok(val) = HeaderValue::from_str(&echo_request_id) {
        response.headers_mut().insert("x-request-id", val);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderName, HeaderValue};

    fn headers_with(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut m = HeaderMap::new();
        for (k, v) in pairs {
            m.insert(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        m
    }

    #[test]
    fn tracing_ids_from_headers_all_present() {
        let h = headers_with(&[
            ("x-request-id", "550e8400-e29b-41d4-a716-446655440000"),
            ("x-org-id", "org-uuid-123"),
            ("x-workspace-id", "ws-uuid-456"),
            ("authorization", "Bearer sk-something"),
        ]);
        let ids = TracingIds::from_headers(&h);
        assert_eq!(ids.request_id, "550e8400-e29b-41d4-a716-446655440000");
        assert!(ids.request_id_inbound);
        assert_eq!(ids.org_id.as_deref(), Some("org-uuid-123"));
        assert_eq!(ids.workspace_id.as_deref(), Some("ws-uuid-456"));

        let forwarded: Vec<_> = ids.upstream_headers().collect();
        assert_eq!(forwarded.len(), 3);
        assert!(forwarded.contains(&("x-request-id", "550e8400-e29b-41d4-a716-446655440000")));
        assert!(forwarded.contains(&("x-org-id", "org-uuid-123")));
        assert!(forwarded.contains(&("x-workspace-id", "ws-uuid-456")));
        // authorization is not a tracing header — must not leak into upstream_headers
        assert!(!forwarded.iter().any(|(k, _)| *k == "authorization"));
    }

    #[test]
    fn tracing_ids_from_headers_partial() {
        let h = headers_with(&[("x-request-id", "req-abc")]);
        let ids = TracingIds::from_headers(&h);
        assert_eq!(ids.request_id, "req-abc");
        assert!(ids.request_id_inbound);
        assert!(ids.org_id.is_none());
        assert!(ids.workspace_id.is_none());

        let forwarded: Vec<_> = ids.upstream_headers().collect();
        assert_eq!(forwarded, vec![("x-request-id", "req-abc")]);
    }

    #[test]
    fn tracing_ids_from_headers_none_present() {
        let h = headers_with(&[("authorization", "Bearer sk-x")]);
        let ids = TracingIds::from_headers(&h);

        // request_id is generated when absent — must be a non-empty UUID-ish string.
        assert!(!ids.request_id.is_empty());
        assert!(!ids.request_id_inbound);
        assert!(ids.org_id.is_none());
        assert!(ids.workspace_id.is_none());

        // Generated request_id must NOT be forwarded upstream — we don't fabricate
        // values vLLM didn't ask for; only forward what the caller actually sent.
        assert_eq!(ids.upstream_headers().count(), 0);
    }
}
