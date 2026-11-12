use std::sync::Arc;

use axum::extract::MatchedPath;
use axum::http::{HeaderMap, HeaderValue, Request};
use axum::middleware::Next;
use axum::response::Response;
use tracing::Instrument;

use crate::request_log::{RequestGuard, RequestIdentity, RequestRecord};

use crate::auth::{AuthPath, RequestSource, RequireAuth};

/// Tracing correlation IDs for a request, parsed once by `request_id_middleware`
/// from `X-Request-Id` and inserted into the request extensions. Handlers that
/// have authenticated a trusted config token may opt in to tenant propagation
/// from `X-Org-Id` / `X-Workspace-Id`. Public Cloud API key requests cannot
/// inject those fields; their log identity comes only from the server-side
/// `/v1/check_api_key` response and is not forwarded to the model backend.
#[derive(Clone, Debug)]
pub struct TracingIds {
    /// The request ID used in spans, logs, and the response header. Either the
    /// inbound `X-Request-Id` (when `request_id_inbound` is true) or a freshly
    /// generated UUID.
    pub request_id: String,
    /// The request ID forwarded to the model backend. For the trusted
    /// gateway/config-token path this mirrors `request_id` so engine logs
    /// correlate. For a public direct `sk-` request it is a fresh server UUID.
    pub(crate) upstream_request_id: String,
    /// Whether `request_id` came from a valid inbound UUID header.
    pub request_id_inbound: bool,
    /// Trusted `X-Org-Id`, if present and ASCII-valid.
    pub org_id: Option<String>,
    /// Trusted `X-Workspace-Id`, if present and ASCII-valid.
    pub workspace_id: Option<String>,
    /// Bounded authentication and ingress-route classification attached only
    /// after the request has passed authentication.
    pub request_source: Option<RequestSource>,
    /// Whether tenant IDs may be forwarded to the model backend. This is true
    /// only for the trusted config-token/gateway path. Verified direct-key IDs
    /// are retained for logs but do not change the upstream wire contract.
    pub(crate) forward_tenant_headers: bool,
    /// Shared with the request guard: the identity once authenticated, and
    /// whether a completion line was written (`request_log`).
    pub(crate) record: Arc<RequestRecord>,
}

impl TracingIds {
    /// Parse from inbound request headers. Reuses `X-Request-Id` only when it
    /// is a UUID; otherwise generates a fresh UUID. Tenant headers are not
    /// accepted at this public boundary.
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let inbound_request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .filter(|s| uuid::Uuid::parse_str(s).is_ok())
            .map(str::to_string);
        let request_id_inbound = inbound_request_id.is_some();
        let request_id = inbound_request_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        Self {
            // Until auth classifies the path, mirror the inbound id upstream.
            // `with_request_source` replaces this with a fresh id for direct keys.
            upstream_request_id: request_id.clone(),
            request_id,
            request_id_inbound,
            org_id: None,
            workspace_id: None,
            request_source: None,
            forward_tenant_headers: false,
            record: Arc::default(),
        }
    }

    /// Return a copy with trusted tenant headers attached. Call this only after
    /// auth has proven the request is from the internal gateway/config-token
    /// path, never for public Cloud API key requests.
    pub fn with_trusted_tenant_headers(&self, headers: &HeaderMap, trusted: bool) -> Self {
        if !trusted {
            return self.clone();
        }

        let org_id = non_empty_header(headers, "x-org-id");
        let workspace_id = non_empty_header(headers, "x-workspace-id");

        Self {
            request_id: self.request_id.clone(),
            upstream_request_id: self.upstream_request_id.clone(),
            request_id_inbound: self.request_id_inbound,
            org_id,
            workspace_id,
            request_source: self.request_source,
            forward_tenant_headers: true,
            record: Arc::clone(&self.record),
        }
    }

    /// Return a copy with tenant identity obtained from the authenticated
    /// Cloud API response. These values are authoritative for observability,
    /// but remain log-only so direct traffic cannot alter backend headers.
    fn with_verified_tenant(&self, org_id: Option<&str>, workspace_id: Option<&str>) -> Self {
        Self {
            request_id: self.request_id.clone(),
            upstream_request_id: self.upstream_request_id.clone(),
            request_id_inbound: self.request_id_inbound,
            org_id: non_empty(org_id),
            workspace_id: non_empty(workspace_id),
            request_source: self.request_source,
            forward_tenant_headers: false,
            record: Arc::clone(&self.record),
        }
    }

    /// Attach the only tenant context authorized for the authentication path:
    /// trusted gateway headers for the config token, or server-verified IDs
    /// returned by Cloud API for a direct `sk-` key. Caller-supplied tenant
    /// headers are therefore ignored on the public direct-key path.
    pub fn with_authenticated_context(&self, headers: &HeaderMap, auth: &RequireAuth) -> Self {
        let ids = match auth.request_source.auth_path {
            AuthPath::CloudApiKey => {
                self.with_verified_tenant(auth.org_id.as_deref(), auth.workspace_id.as_deref())
            }
            AuthPath::TrustedConfigToken => self.with_trusted_tenant_headers(headers, true),
        };
        ids.with_request_source(auth.request_source)
    }

    pub fn with_request_source(&self, request_source: RequestSource) -> Self {
        let mut ids = self.clone();
        ids.request_source = Some(request_source);
        // The backend id on the direct `sk-` path is the server's. The trusted
        // gateway path keeps the shared id so engine logs correlate.
        ids.upstream_request_id = match request_source.auth_path {
            AuthPath::CloudApiKey => uuid::Uuid::new_v4().to_string(),
            AuthPath::TrustedConfigToken => ids.request_id.clone(),
        };
        ids
    }

    /// `(name, value)` pairs to forward to the upstream backend. Always forwards
    /// the middleware-selected UUID request ID; tenant IDs appear only when a
    /// handler explicitly attached trusted tenant metadata.
    pub fn upstream_headers(&self) -> impl Iterator<Item = (&'static str, &str)> + '_ {
        let req = Some(("x-request-id", self.upstream_request_id.as_str()));
        let org = self
            .forward_tenant_headers
            .then_some(self.org_id.as_deref())
            .flatten()
            .map(|v| ("x-org-id", v));
        let ws = self
            .forward_tenant_headers
            .then_some(self.workspace_id.as_deref())
            .flatten()
            .map(|v| ("x-workspace-id", v));
        req.into_iter().chain(org).chain(ws)
    }

    /// Record this request's authenticated identity for the request span and
    /// the request's finish or cancel line. Called once auth has passed.
    pub(crate) fn publish_identity(&self) {
        let labels = crate::proxy::request_metric_labels(Some(self));
        self.record.publish_identity(
            &self.request_id,
            RequestIdentity {
                org_id: self.org_id.clone(),
                workspace_id: self.workspace_id.clone(),
                auth_path: labels.auth_path,
                ingress_route: labels.ingress_route,
                tenant_context: labels.tenant_context,
                caller: labels.caller,
            },
        );
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

/// A tenant id header's value, or `None` when it is missing, not ASCII, or
/// blank. A blank id names no tenant and must not count as one.
fn non_empty_header(headers: &HeaderMap, name: &str) -> Option<String> {
    non_empty(headers.get(name).and_then(|v| v.to_str().ok()))
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

/// When the request reached the proxy, put in the request extensions by
/// `request_id_middleware`. A deadline that has to cover authentication (a
/// cloud-api round trip, sometimes retried) and pre-dispatch validation reads
/// its clock origin here instead of starting it in the route handler.
#[derive(Clone, Copy, Debug)]
pub struct RequestStart(pub std::time::Instant);

impl RequestStart {
    /// The arrival instant a handler extracted, or now when the extension is
    /// missing. The fallback is taken at dispatch, so the only thing it
    /// guarantees is that a missing start instant cannot shorten a budget
    /// measured from it — it does not make a route work without this
    /// middleware, which the handlers still need for their tracing IDs.
    pub fn or_now(extracted: Option<axum::Extension<Self>>) -> std::time::Instant {
        extracted.map_or_else(std::time::Instant::now, |axum::Extension(start)| start.0)
    }
}

/// Request ID middleware: parses tracing correlation headers, stores them and
/// the arrival instant in the request extensions for handlers, attaches the
/// IDs to the tracing span so every log line carries them, and echoes
/// `X-Request-Id` back to the caller.
///
/// The span declares the tenant and caller fields empty; the auth extractor
/// records them once the caller is known. A [`RequestGuard`] follows the
/// request to the end of its response body and writes its finish or cancel
/// line (`request_log`).
pub async fn request_id_middleware(mut request: Request<axum::body::Body>, next: Next) -> Response {
    let mut tracing_ids = TracingIds::from_headers(request.headers());
    let start = std::time::Instant::now();
    request.extensions_mut().insert(RequestStart(start));

    let http_method = request.method().clone();
    let method = http_method.to_string();
    let path = request.uri().path().to_string();
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|matched| matched.as_str().to_string());
    let request_bytes = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let span = tracing::info_span!(
        "request",
        request_id = %tracing_ids.request_id,
        auth_path = tracing::field::Empty,
        tenant_context = tracing::field::Empty,
        caller = tracing::field::Empty,
        method = %method,
        path = %path,
    );

    let record = Arc::new(RequestRecord::new(span.clone(), route));
    tracing_ids.record = Arc::clone(&record);
    let echo_request_id = tracing_ids.request_id.clone();
    let guard = RequestGuard::new(
        record,
        echo_request_id.clone(),
        &http_method,
        request_bytes,
        start,
    );
    request.extensions_mut().insert(tracing_ids);

    // If this future is dropped before the response exists, `guard` drops
    // with it and writes the cancel line.
    let mut response = next.run(request).instrument(span).await;
    if let Ok(val) = HeaderValue::from_str(&echo_request_id) {
        response.headers_mut().insert("x-request-id", val);
    }
    guard.attach(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderName, HeaderValue};
    use axum::{body::Body, middleware::from_fn, routing::post, Router};
    use std::io;
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt;
    use tracing::Level;

    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

    struct CapturedLogsWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for CapturedLogsWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut logs = self
                .0
                .lock()
                .expect("captured logs mutex should not poison");
            logs.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl CapturedLogs {
        fn writer(&self) -> CapturedLogsWriter {
            CapturedLogsWriter(Arc::clone(&self.0))
        }

        fn contents(&self) -> String {
            let logs = self
                .0
                .lock()
                .expect("captured logs mutex should not poison");
            String::from_utf8_lossy(&logs).into_owned()
        }
    }

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
        assert!(ids.org_id.is_none());
        assert!(ids.workspace_id.is_none());

        let forwarded: Vec<_> = ids.upstream_headers().collect();
        assert_eq!(
            forwarded,
            vec![("x-request-id", "550e8400-e29b-41d4-a716-446655440000")]
        );
        assert!(!forwarded.iter().any(|(k, _)| *k == "authorization"));

        let trusted_ids = ids.with_trusted_tenant_headers(&h, true);
        let trusted_forwarded: Vec<_> = trusted_ids.upstream_headers().collect();
        assert_eq!(trusted_forwarded.len(), 3);
        assert!(
            trusted_forwarded.contains(&("x-request-id", "550e8400-e29b-41d4-a716-446655440000"))
        );
        assert!(trusted_forwarded.contains(&("x-org-id", "org-uuid-123")));
        assert!(trusted_forwarded.contains(&("x-workspace-id", "ws-uuid-456")));

        let classified = trusted_ids.with_request_source(RequestSource {
            auth_path: crate::auth::AuthPath::TrustedConfigToken,
            ingress_route: crate::auth::IngressRouteKind::LongIndexed,
            caller: crate::auth::CALLER_UNLABELED,
        });
        assert_eq!(
            classified.request_source,
            Some(RequestSource {
                auth_path: crate::auth::AuthPath::TrustedConfigToken,
                ingress_route: crate::auth::IngressRouteKind::LongIndexed,
                caller: crate::auth::CALLER_UNLABELED,
            })
        );

        let direct_auth = RequireAuth {
            cloud_api_key: Some("sk-test-redacted".to_string()),
            org_id: Some("org-authenticated".to_string()),
            workspace_id: Some("ws-authenticated".to_string()),
            api_key_id: Some("key-authenticated".to_string()),
            request_id: Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
            request_source: RequestSource {
                auth_path: crate::auth::AuthPath::CloudApiKey,
                ingress_route: crate::auth::IngressRouteKind::Canonical,
                caller: crate::auth::CALLER_API_KEY,
            },
        };
        let verified_ids = ids.with_authenticated_context(&h, &direct_auth);
        assert_eq!(verified_ids.org_id.as_deref(), Some("org-authenticated"));
        assert_eq!(
            verified_ids.workspace_id.as_deref(),
            Some("ws-authenticated")
        );
        assert_eq!(
            verified_ids.request_source,
            Some(direct_auth.request_source)
        );
        // The inbound request id stays the log/response id; the direct path
        // forwards a fresh server id to the backend. No tenant header leaks either.
        assert_eq!(
            verified_ids.request_id,
            "550e8400-e29b-41d4-a716-446655440000"
        );
        let direct_forwarded = verified_ids.upstream_headers().collect::<Vec<_>>();
        assert_eq!(direct_forwarded.len(), 1);
        let (name, forwarded_id) = direct_forwarded[0];
        assert_eq!(name, "x-request-id");
        assert!(
            uuid::Uuid::parse_str(forwarded_id).is_ok(),
            "upstream request id is a UUID"
        );
        assert_ne!(
            forwarded_id, "550e8400-e29b-41d4-a716-446655440000",
            "a direct-key request must not pin the backend's request id to its inbound value"
        );

        // The explicit auth-path classification is the authorization boundary;
        // incidental population of cloud_api_key must not override it.
        let trusted_auth = RequireAuth {
            cloud_api_key: Some("implementation-detail-only".to_string()),
            org_id: None,
            workspace_id: None,
            api_key_id: None,
            request_id: None,
            request_source: RequestSource {
                auth_path: crate::auth::AuthPath::TrustedConfigToken,
                ingress_route: crate::auth::IngressRouteKind::Canonical,
                caller: crate::auth::CALLER_UNLABELED,
            },
        };
        let trusted_ids = ids.with_authenticated_context(&h, &trusted_auth);
        let trusted_headers = trusted_ids.upstream_headers().collect::<Vec<_>>();
        assert!(trusted_headers.contains(&("x-org-id", "org-uuid-123")));
        assert!(trusted_headers.contains(&("x-workspace-id", "ws-uuid-456")));
    }

    #[test]
    fn blank_tenant_headers_count_as_absent() {
        let h = headers_with(&[("x-org-id", ""), ("x-workspace-id", "   ")]);
        let ids = TracingIds::from_headers(&h).with_trusted_tenant_headers(&h, true);
        assert!(ids.org_id.is_none());
        assert!(ids.workspace_id.is_none());
        let forwarded: Vec<_> = ids.upstream_headers().collect();
        assert_eq!(forwarded, vec![("x-request-id", ids.request_id.as_str())]);

        // One blank, one set: only the set one is kept and forwarded.
        let h = headers_with(&[("x-org-id", "org-1"), ("x-workspace-id", "")]);
        let ids = TracingIds::from_headers(&h).with_trusted_tenant_headers(&h, true);
        assert_eq!(ids.org_id.as_deref(), Some("org-1"));
        assert!(ids.workspace_id.is_none());
        let forwarded: Vec<_> = ids.upstream_headers().collect();
        assert!(forwarded.contains(&("x-org-id", "org-1")));
        assert!(!forwarded.iter().any(|(name, _)| *name == "x-workspace-id"));

        // A verified identity with blank fields is no identity either.
        let direct_auth = RequireAuth {
            cloud_api_key: Some("sk-test-redacted".to_string()),
            org_id: Some(String::new()),
            workspace_id: Some(" ".to_string()),
            api_key_id: None,
            request_id: None,
            request_source: RequestSource {
                auth_path: crate::auth::AuthPath::CloudApiKey,
                ingress_route: crate::auth::IngressRouteKind::Canonical,
                caller: crate::auth::CALLER_API_KEY,
            },
        };
        let ids = TracingIds::from_headers(&HeaderMap::new())
            .with_authenticated_context(&HeaderMap::new(), &direct_auth);
        assert!(ids.org_id.is_none());
        assert!(ids.workspace_id.is_none());
    }

    #[test]
    fn tracing_ids_replaces_invalid_inbound_request_id() {
        let h = headers_with(&[("x-request-id", "req-abc")]);
        let ids = TracingIds::from_headers(&h);
        assert!(uuid::Uuid::parse_str(&ids.request_id).is_ok());
        assert_ne!(ids.request_id, "req-abc");
        assert!(!ids.request_id_inbound);
        assert!(ids.org_id.is_none());
        assert!(ids.workspace_id.is_none());

        let forwarded: Vec<_> = ids.upstream_headers().collect();
        assert_eq!(forwarded, vec![("x-request-id", ids.request_id.as_str())]);
    }

    #[test]
    fn tracing_ids_from_headers_none_present() {
        let h = headers_with(&[("authorization", "Bearer sk-x")]);
        let ids = TracingIds::from_headers(&h);

        assert!(!ids.request_id.is_empty());
        assert!(!ids.request_id_inbound);
        assert!(ids.org_id.is_none());
        assert!(ids.workspace_id.is_none());

        let forwarded: Vec<_> = ids.upstream_headers().collect();
        assert_eq!(forwarded, vec![("x-request-id", ids.request_id.as_str())]);
    }

    #[tokio::test]
    async fn proxy_tracing_logs_strip_public_tenant_spoofing() {
        let logs = CapturedLogs::default();
        let writer_logs = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .with_ansi(false)
            .with_writer(move || writer_logs.writer())
            .finish();
        let request_id = "550e8400-e29b-41d4-a716-446655440000";
        let body = Body::from(
            r#"{"messages":[{"role":"user","content":"PROXY_PROMPT_SENTINEL"}],"api_key":"sk-proxy-sentinel"}"#,
        );
        let app = Router::new()
            .route("/v1/chat/completions", post(|| async { "ok" }))
            .layer(from_fn(request_id_middleware));

        let _subscriber_guard = tracing::subscriber::set_default(subscriber);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("x-request-id", request_id)
                    .header("x-org-id", "org-spoof-log-sentinel")
                    .header("x-workspace-id", "ws-spoof-log-sentinel")
                    .header("authorization", "Bearer sk-proxy-sentinel")
                    .body(body)
                    .expect("test request should build"),
            )
            .await
            .expect("test request should complete");

        assert_eq!(response.status().as_u16(), 200);
        // The request's line is written once its response body has been sent.
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("test body should read");
        let captured = logs.contents();
        assert!(captured.contains(request_id));
        assert!(captured.contains("method=\"POST\""), "{captured}");
        assert!(
            captured.contains("route=/v1/chat/completions"),
            "{captured}"
        );
        assert!(captured.contains("status=200"), "{captured}");
        for forbidden in [
            "PROXY_PROMPT_SENTINEL",
            "sk-proxy-sentinel",
            "org-spoof-log-sentinel",
            "ws-spoof-log-sentinel",
        ] {
            assert!(
                !captured.contains(forbidden),
                "captured logs must not contain {forbidden}; logs: {captured}"
            );
        }
    }
}
