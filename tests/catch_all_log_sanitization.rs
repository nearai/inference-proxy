use std::io;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use tracing::Level;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[path = "common/auth_header.rs"]
mod auth_header;
mod common;

use auth_header::*;
use common::*;

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

#[tokio::test(flavor = "current_thread")]
async fn catch_all_logs_sanitized_upstream_url() {
    let logs = CapturedLogs::default();
    let writer_logs = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_ansi(false)
        .with_writer(move || writer_logs.writer())
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("catch_all_log_sanitization installs one global subscriber");

    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/log-sanitization"))
        .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
            "message": "upstream rejected request",
            "type": "BadRequestError"
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri(), TestAppOptions::default());
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/log-sanitization?api_key=TOKEN_QUERY_SENTINEL&email=person%40example.invalid&api_key=second&encoded=a%2Bb")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let received = mock_server
        .received_requests()
        .await
        .expect("wiremock records upstream request");
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].url.path(), "/v1/log-sanitization");
    let query = received[0].url.query().expect("query must be forwarded");
    assert!(query.contains("api_key=TOKEN_QUERY_SENTINEL"));
    assert!(query.contains("email=person%40example.invalid"));
    assert!(query.contains("api_key=second"));
    assert!(query.contains("encoded=a%2Bb"));

    let captured = logs.contents();
    assert!(captured.contains("Catch-all passthrough"));
    assert!(captured.contains("Backend returned non-success status"));
    assert!(captured.contains("upstream_url="));
    for forbidden in [
        "TOKEN_QUERY_SENTINEL",
        "api_key=",
        "person@example.invalid",
        "person%40example.invalid",
        "encoded=a%2Bb",
    ] {
        assert!(
            !captured.contains(forbidden),
            "captured warning/debug logs must not contain {forbidden}; logs: {captured}"
        );
    }
    eprintln!(
        "manual-qa: catch_all_logs_sanitized_upstream_url forwarded path={} query_present={} sentinels_absent_from_logs=true",
        received[0].url.path(),
        received[0].url.query().is_some()
    );
}
