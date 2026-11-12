//! One log line per request that otherwise leaves none.
//!
//! `request_id_middleware` creates a [`RequestRecord`] for every request and a
//! [`RequestGuard`] that lives until the response body has been sent or
//! dropped. The record is shared through [`crate::TracingIds`]: the auth
//! extractor publishes the request's identity into it, and the completion
//! logger marks it once a `request completed` line has been written.
//!
//! When the guard drops it writes at most one line:
//! - `request cancelled`, when the client went away before the response was
//!   returned (`phase="before_response"`) or while its body was being sent
//!   (`phase="during_response"`);
//! - `request finished`, when the response was sent and no `request completed`
//!   line was written for it (errors, routes without a completion line).
//!
//! A request whose `request completed` line was written gets no extra line, so
//! successful inference adds no volume. A streamed chat or text completion
//! logs its completion line from its own task, possibly after a client that
//! left on the last chunk dropped the body; its cancel line therefore waits
//! until every holder of the record is done, and is dropped if the completion
//! line came. Health, metrics, version and model-list
//! routes, and unmatched paths, never log here. The guard also owns the
//! `active_connections` gauge, so a dropped request always decrements it.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes, HttpBody};
use http_body::{Frame, SizeHint};
use tracing::info;

/// Which lines a route writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinePolicy {
    /// Health, metrics, version, model list, OHTTP key config and unmatched
    /// paths: polled often, never per-tenant, and an unmatched path is
    /// attacker-chosen text. Nothing is logged.
    Silent,
    /// Signature and attestation lookups: a finish line only for an error
    /// status, a cancel line always.
    ErrorsOnly,
    /// Every other route: a finish line unless a completion line was written,
    /// a cancel line always.
    Always,
}

impl LinePolicy {
    pub(crate) fn for_route(route: Option<&str>) -> Self {
        use crate::routes::*;
        match route {
            None => Self::Silent,
            Some(
                ROUTE_ROOT
                | ROUTE_VERSION
                | ROUTE_HEALTHZ
                | ROUTE_METRICS
                | ROUTE_V1_METRICS
                | ROUTE_V1_MODELS
                | ROUTE_OHTTP_WELL_KNOWN
                | ROUTE_OHTTP_CONFIG,
            ) => Self::Silent,
            Some(ROUTE_SIGNATURE | ROUTE_ATTESTATION_REPORT) => Self::ErrorsOnly,
            Some(_) => Self::Always,
        }
    }
}

/// Who the request is from, as established by authentication. Only bounded
/// labels and the tenant ids already logged on the completion line.
#[derive(Clone, Debug)]
pub(crate) struct RequestIdentity {
    pub(crate) org_id: Option<String>,
    pub(crate) workspace_id: Option<String>,
    pub(crate) auth_path: &'static str,
    pub(crate) ingress_route: &'static str,
    pub(crate) tenant_context: &'static str,
    pub(crate) caller: &'static str,
}

impl RequestIdentity {
    const UNKNOWN: &'static str = "unknown";

    fn unauthenticated() -> Self {
        Self {
            org_id: None,
            workspace_id: None,
            auth_path: Self::UNKNOWN,
            ingress_route: Self::UNKNOWN,
            tenant_context: Self::UNKNOWN,
            caller: crate::auth::CALLER_UNKNOWN,
        }
    }
}

/// State shared by everything that handles one request.
#[derive(Debug)]
pub struct RequestRecord {
    span: tracing::Span,
    route: Option<String>,
    policy: Option<LinePolicy>,
    identity: Mutex<Option<RequestIdentity>>,
    /// A cancel line held until the last holder is done (see `RequestGuard`).
    deferred: Mutex<Option<Outcome>>,
    completion_logged: AtomicBool,
}

impl Default for RequestRecord {
    /// A record outside `request_id_middleware` (tests, internal callers):
    /// no span, no route, writes nothing.
    fn default() -> Self {
        Self {
            span: tracing::Span::none(),
            route: None,
            policy: None,
            identity: Mutex::new(None),
            deferred: Mutex::new(None),
            completion_logged: AtomicBool::new(false),
        }
    }
}

impl RequestRecord {
    pub(crate) fn new(span: tracing::Span, route: Option<String>) -> Self {
        let policy = LinePolicy::for_route(route.as_deref());
        Self {
            span,
            route,
            policy: Some(policy),
            identity: Mutex::new(None),
            deferred: Mutex::new(None),
            completion_logged: AtomicBool::new(false),
        }
    }

    /// The matched route (`MatchedPath`), never the raw path.
    pub(crate) fn route(&self) -> Option<&str> {
        self.route.as_deref()
    }

    /// Store the authenticated identity and record its bounded labels on the
    /// request span. Tenant ids stay off the span, so no other line of the
    /// request gains them; they are on this request's own lines only.
    pub(crate) fn publish_identity(&self, identity: RequestIdentity) {
        self.span.record("auth_path", identity.auth_path);
        self.span.record("tenant_context", identity.tenant_context);
        self.span.record("caller", identity.caller);
        if let Ok(mut slot) = self.identity.lock() {
            *slot = Some(identity);
        }
    }

    fn identity(&self) -> RequestIdentity {
        self.identity
            .lock()
            .ok()
            .and_then(|slot| slot.clone())
            .unwrap_or_else(RequestIdentity::unauthenticated)
    }

    /// A `request completed` line was written for this request.
    pub(crate) fn mark_completion_logged(&self) {
        self.completion_logged.store(true, Ordering::Release);
    }

    fn completion_logged(&self) -> bool {
        self.completion_logged.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BodyEnd {
    /// Not sent to the end (or no response yet).
    Pending,
    /// The whole body was sent.
    Complete,
    /// The body failed while it was being sent.
    Error,
}

/// Writes the finish or cancel line when the request is over. Created by
/// `request_id_middleware`, moved into the response body once there is one.
pub(crate) struct RequestGuard {
    record: Arc<RequestRecord>,
    request_id: String,
    method: &'static str,
    request_bytes: Option<u64>,
    start: Instant,
    status: Option<u16>,
    end: BodyEnd,
}

impl RequestGuard {
    pub(crate) fn new(
        record: Arc<RequestRecord>,
        request_id: String,
        method: &axum::http::Method,
        request_bytes: Option<u64>,
        start: Instant,
    ) -> Self {
        let method = method_label(method);
        metrics::gauge!("active_connections").increment(1);
        Self {
            record,
            request_id,
            method,
            request_bytes,
            start,
            status: None,
            end: BodyEnd::Pending,
        }
    }

    /// Attach the guard to the response, so it ends with the response body.
    pub(crate) fn attach(mut self, response: axum::response::Response) -> axum::response::Response {
        self.status = Some(response.status().as_u16());
        response.map(|inner| {
            Body::new(GuardedBody {
                inner,
                guard: Some(self),
            })
        })
    }

    /// What the request's line would say, as of now.
    fn outcome(&self) -> Outcome {
        Outcome {
            request_id: self.request_id.clone(),
            method: self.method,
            request_bytes: self.request_bytes,
            elapsed_ms: self.start.elapsed().as_millis() as u64,
            status: self.status,
            end: self.end,
        }
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        metrics::gauge!("active_connections").decrement(1);
        let outcome = self.outcome();
        // A streamed completion is logged by a task that can still be signing
        // after a client left on the last chunk. Leave that cancel to the
        // record, which writes it once every holder is done and only if no
        // completion line came.
        let late_completion_possible = matches!(outcome.status, Some(status) if status < 400)
            && outcome.end == BodyEnd::Pending
            && self.record.route().is_some_and(completes_asynchronously);
        if late_completion_possible {
            self.record.defer(outcome);
        } else {
            self.record.write(&outcome);
        }
    }
}

/// The request method as a bounded label: a method outside the standard set
/// is client-chosen text and becomes `other`.
fn method_label(method: &axum::http::Method) -> &'static str {
    use axum::http::Method;
    match *method {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        Method::PATCH => "PATCH",
        _ => "other",
    }
}

/// Routes whose `request completed` line can be written after the response
/// body has been dropped (the streaming path logs it from its own task).
fn completes_asynchronously(route: &str) -> bool {
    route == crate::routes::ROUTE_CHAT_COMPLETIONS || route == crate::routes::ROUTE_COMPLETIONS
}

/// The facts the guard knows when the request ends.
#[derive(Debug)]
struct Outcome {
    request_id: String,
    method: &'static str,
    request_bytes: Option<u64>,
    elapsed_ms: u64,
    status: Option<u16>,
    end: BodyEnd,
}

impl RequestRecord {
    /// Keep a cancel line until the last holder of this record is gone.
    fn defer(&self, outcome: Outcome) {
        if let Ok(mut slot) = self.deferred.lock() {
            *slot = Some(outcome);
        }
    }

    /// Write the request's finish or cancel line, unless the route is quiet
    /// or a `request completed` line already stands for the request.
    fn write(&self, outcome: &Outcome) {
        let policy = self.policy.unwrap_or(LinePolicy::Silent);
        if policy == LinePolicy::Silent || self.completion_logged() {
            return;
        }
        let identity = self.identity();
        let route = self.route().unwrap_or("");
        let org_id = identity.org_id.as_deref().unwrap_or("");
        let workspace_id = identity.workspace_id.as_deref().unwrap_or("");
        let elapsed_ms = outcome.elapsed_ms;
        // Root events: they carry exactly the fields below and none of the
        // request span's (whose raw path can hold a client-chosen id).
        match (outcome.status, outcome.end) {
            (None, _) | (Some(_), BodyEnd::Pending) => info!(
                parent: None,
                request_id = %outcome.request_id,
                org_id = %org_id,
                workspace_id = %workspace_id,
                auth_path = identity.auth_path,
                ingress_route = identity.ingress_route,
                tenant_context = identity.tenant_context,
                caller = identity.caller,
                method = outcome.method,
                route = %route,
                status = outcome.status,
                request_bytes = outcome.request_bytes,
                elapsed_ms,
                phase = if outcome.status.is_none() {
                    "before_response"
                } else {
                    "during_response"
                },
                "request cancelled"
            ),
            (Some(status), end) => {
                if policy == LinePolicy::ErrorsOnly && status < 400 && end == BodyEnd::Complete {
                    return;
                }
                info!(
                    parent: None,
                    request_id = %outcome.request_id,
                    org_id = %org_id,
                    workspace_id = %workspace_id,
                    auth_path = identity.auth_path,
                    ingress_route = identity.ingress_route,
                    tenant_context = identity.tenant_context,
                    caller = identity.caller,
                    method = outcome.method,
                    route = %route,
                    status,
                    request_bytes = outcome.request_bytes,
                    duration_ms = elapsed_ms,
                    outcome = if end == BodyEnd::Complete {
                        "complete"
                    } else {
                        "body_error"
                    },
                    "request finished"
                );
            }
        }
    }
}

impl Drop for RequestRecord {
    fn drop(&mut self) {
        let deferred = self.deferred.get_mut().ok().and_then(Option::take);
        if let Some(outcome) = deferred {
            self.write(&outcome);
        }
    }
}

/// The response body with the request's guard: the guard ends when the body
/// is done, fails, or is dropped by a client that went away.
struct GuardedBody {
    inner: Body,
    guard: Option<RequestGuard>,
}

impl GuardedBody {
    fn end(&mut self, end: BodyEnd) {
        if let Some(mut guard) = self.guard.take() {
            guard.end = end;
        }
    }
}

impl HttpBody for GuardedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = &mut *self;
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(None) => this.end(BodyEnd::Complete),
            Poll::Ready(Some(Err(_))) => this.end(BodyEnd::Error),
            _ => {}
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for GuardedBody {
    fn drop(&mut self) {
        // A body with a known length can be written out without a final poll
        // that returns `None`; it is complete when nothing is left in it.
        if self.inner.is_end_stream() {
            self.end(BodyEnd::Complete);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Captured {
        fn messages(&self) -> Vec<serde_json::Value> {
            String::from_utf8(self.0.lock().unwrap().clone())
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                .map(|line| line["fields"].clone())
                .collect()
        }
    }

    fn capture() -> (Captured, tracing::subscriber::DefaultGuard) {
        let captured = Captured::default();
        let writer = captured.clone();
        let guard = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .json()
                .with_max_level(tracing::Level::INFO)
                .with_writer(move || writer.clone())
                .finish(),
        );
        (captured, guard)
    }

    fn guard_for(route: &str) -> (Arc<RequestRecord>, RequestGuard) {
        let record = Arc::new(RequestRecord::new(
            tracing::Span::none(),
            Some(route.to_string()),
        ));
        let guard = RequestGuard::new(
            Arc::clone(&record),
            "req-1".to_string(),
            &axum::http::Method::POST,
            Some(12),
            Instant::now(),
        );
        (record, guard)
    }

    fn identity() -> RequestIdentity {
        RequestIdentity {
            org_id: Some("org-1".to_string()),
            workspace_id: Some("ws-1".to_string()),
            auth_path: "trusted_config_token",
            ingress_route: "canonical",
            tenant_context: "trusted_headers",
            caller: "cloud-api",
        }
    }

    #[tokio::test]
    async fn body_dropped_mid_stream_logs_a_cancel_line() {
        let (captured, _subscriber) = capture();
        let (record, guard) = guard_for(crate::routes::ROUTE_PRIVACY_CLASSIFY);
        record.publish_identity(identity());
        use futures_util::StreamExt;
        let stream = futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from("a"))])
            .chain(futures_util::stream::pending());
        let response = guard.attach(axum::response::Response::new(Body::from_stream(stream)));
        let mut body = response.into_body();
        let first = body.frame().await.unwrap().unwrap();
        assert_eq!(first.into_data().unwrap(), Bytes::from("a"));
        assert!(captured.messages().is_empty());
        drop(body);

        let lines = captured.messages();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["message"], "request cancelled");
        assert_eq!(lines[0]["phase"], "during_response");
        assert_eq!(lines[0]["status"], 200);
        assert_eq!(lines[0]["org_id"], "org-1");
        assert_eq!(lines[0]["workspace_id"], "ws-1");
        assert_eq!(lines[0]["caller"], "cloud-api");
        assert_eq!(lines[0]["request_bytes"], 12);
    }

    /// A streamed chat whose client leaves on the last chunk: the completion
    /// task may still log `request completed`, so the cancel line waits for
    /// it and is dropped if it comes.
    #[tokio::test]
    async fn streamed_completion_after_the_client_left_wins_over_the_cancel() {
        use futures_util::StreamExt;
        let (captured, _subscriber) = capture();
        for completes in [true, false] {
            let (record, guard) = guard_for(crate::routes::ROUTE_CHAT_COMPLETIONS);
            record.publish_identity(identity());
            // The streaming task's own handle on the record.
            let producer = Arc::clone(&record);
            drop(record);
            let stream = futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from("a"))])
                .chain(futures_util::stream::pending());
            let response = guard.attach(axum::response::Response::new(Body::from_stream(stream)));
            let mut body = response.into_body();
            body.frame().await.unwrap().unwrap();
            drop(body);
            assert!(
                captured.messages().is_empty(),
                "nothing before the task ends"
            );
            if completes {
                producer.mark_completion_logged();
            }
            drop(producer);
        }
        let lines = captured.messages();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["message"], "request cancelled");
        assert_eq!(lines[0]["phase"], "during_response");
        assert_eq!(lines[0]["org_id"], "org-1");
    }

    #[tokio::test]
    async fn sent_body_logs_a_finish_line_unless_a_completion_line_was_written() {
        let (captured, _subscriber) = capture();
        let (record, guard) = guard_for(crate::routes::ROUTE_EMBEDDINGS);
        record.publish_identity(identity());
        let response = guard.attach(axum::response::Response::new(Body::from("done")));
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes, Bytes::from("done"));
        let lines = captured.messages();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["message"], "request finished");
        assert_eq!(lines[0]["outcome"], "complete");
        assert_eq!(lines[0]["route"], crate::routes::ROUTE_EMBEDDINGS);

        // A request with a completion line gets nothing more, finished or not.
        let (record, guard) = guard_for(crate::routes::ROUTE_CHAT_COMPLETIONS);
        record.mark_completion_logged();
        drop(guard.attach(axum::response::Response::new(Body::from("x"))));
        let (record, guard) = guard_for(crate::routes::ROUTE_CHAT_COMPLETIONS);
        record.mark_completion_logged();
        drop(guard);
        // Health and lookup routes stay quiet on success.
        let (_, guard) = guard_for(crate::routes::ROUTE_HEALTHZ);
        drop(guard);
        let (_, guard) = guard_for(crate::routes::ROUTE_SIGNATURE);
        let response = guard.attach(axum::response::Response::new(Body::from("sig")));
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(captured.messages().len(), 1);

        // A lookup that failed is logged.
        let (_, guard) = guard_for(crate::routes::ROUTE_SIGNATURE);
        let mut response = axum::response::Response::new(Body::from("missing"));
        *response.status_mut() = axum::http::StatusCode::NOT_FOUND;
        let response = guard.attach(response);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let lines = captured.messages();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert_eq!(lines[1]["status"], 404);
        assert_eq!(lines[1]["caller"], "unknown");
    }

    #[test]
    fn line_policy_by_route() {
        use crate::routes::*;
        for route in [
            ROUTE_ROOT,
            ROUTE_VERSION,
            ROUTE_HEALTHZ,
            ROUTE_METRICS,
            ROUTE_V1_METRICS,
            ROUTE_V1_MODELS,
            ROUTE_OHTTP_WELL_KNOWN,
            ROUTE_OHTTP_CONFIG,
        ] {
            assert_eq!(LinePolicy::for_route(Some(route)), LinePolicy::Silent);
        }
        assert_eq!(LinePolicy::for_route(None), LinePolicy::Silent);
        for route in [ROUTE_SIGNATURE, ROUTE_ATTESTATION_REPORT] {
            assert_eq!(LinePolicy::for_route(Some(route)), LinePolicy::ErrorsOnly);
        }
        for route in [
            ROUTE_CHAT_COMPLETIONS,
            ROUTE_COMPLETIONS,
            ROUTE_PRIVACY_CLASSIFY,
            ROUTE_EMBEDDINGS,
            ROUTE_RERANK,
            ROUTE_SCORE,
            ROUTE_TOKENIZE,
            ROUTE_IMAGES_GENERATIONS,
            ROUTE_IMAGES_EDITS,
            ROUTE_AUDIO_TRANSCRIPTIONS,
            ROUTE_OHTTP_RELAY,
            ROUTE_INTERNAL_GPU_EVIDENCE,
        ] {
            assert_eq!(LinePolicy::for_route(Some(route)), LinePolicy::Always);
        }
    }
}
