use axum::http::{HeaderMap, HeaderValue, Request};
use axum::middleware::Next;
use axum::response::Response;
use tracing::Instrument;

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
            request_id,
            request_id_inbound,
            org_id: None,
            workspace_id: None,
            request_source: None,
            forward_tenant_headers: false,
        }
    }

    /// Return a copy with trusted tenant headers attached. Call this only after
    /// auth has proven the request is from the internal gateway/config-token
    /// path, never for public Cloud API key requests.
    pub fn with_trusted_tenant_headers(&self, headers: &HeaderMap, trusted: bool) -> Self {
        if !trusted {
            return self.clone();
        }

        let org_id = headers
            .get("x-org-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        let workspace_id = headers
            .get("x-workspace-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        Self {
            request_id: self.request_id.clone(),
            request_id_inbound: self.request_id_inbound,
            org_id,
            workspace_id,
            request_source: self.request_source,
            forward_tenant_headers: true,
        }
    }

    /// Return a copy with tenant identity obtained from the authenticated
    /// Cloud API response. These values are authoritative for observability,
    /// but remain log-only so direct traffic cannot alter backend headers.
    fn with_verified_tenant(&self, org_id: Option<&str>, workspace_id: Option<&str>) -> Self {
        Self {
            request_id: self.request_id.clone(),
            request_id_inbound: self.request_id_inbound,
            org_id: org_id.map(str::to_string),
            workspace_id: workspace_id.map(str::to_string),
            request_source: self.request_source,
            forward_tenant_headers: false,
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
        ids
    }

    /// `(name, value)` pairs to forward to the upstream backend. Always forwards
    /// the middleware-selected UUID request ID; tenant IDs appear only when a
    /// handler explicitly attached trusted tenant metadata.
    pub fn upstream_headers(&self) -> impl Iterator<Item = (&'static str, &str)> + '_ {
        let req = Some(("x-request-id", self.request_id.as_str()));
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
    tracing::debug!(
        request_id = %echo_request_id,
        method = %method,
        path = %path,
        status = response.status().as_u16(),
        "request completed"
    );
    response
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
        });
        assert_eq!(
            classified.request_source,
            Some(RequestSource {
                auth_path: crate::auth::AuthPath::TrustedConfigToken,
                ingress_route: crate::auth::IngressRouteKind::LongIndexed,
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
        assert_eq!(
            verified_ids.upstream_headers().collect::<Vec<_>>(),
            vec![("x-request-id", "550e8400-e29b-41d4-a716-446655440000")],
            "verified direct-key identity is log-only; spoofed public tenant headers stay off wire"
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
            },
        };
        let trusted_ids = ids.with_authenticated_context(&h, &trusted_auth);
        let trusted_headers = trusted_ids.upstream_headers().collect::<Vec<_>>();
        assert!(trusted_headers.contains(&("x-org-id", "org-uuid-123")));
        assert!(trusted_headers.contains(&("x-workspace-id", "ws-uuid-456")));
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
        let captured = logs.contents();
        assert!(captured.contains(request_id));
        assert!(captured.contains("method=POST"));
        assert!(captured.contains("path=/v1/chat/completions"));
        assert!(captured.contains("status=200"));
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
