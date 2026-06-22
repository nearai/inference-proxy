use super::*;
use std::io;
use std::sync::{Arc, Mutex};
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

fn eff(status: u16, body: &[u8]) -> StatusCode {
    effective_error_status(status, parse_upstream_error(body).as_ref())
}

fn assert_missing(value: &str, needles: &[&str]) {
    for needle in needles {
        assert!(!value.contains(needle), "{needle} leaked: {value}");
    }
}

#[test]
fn test_parse_upstream_error_vllm_flat_format() {
    let body = br#"{"object":"error","message":"This model's maximum context length is 2048 tokens","type":"BadRequestError","param":null,"code":400}"#;
    let info = parse_upstream_error(body).unwrap();
    assert_eq!(
        info.message,
        "This model's maximum context length is 2048 tokens"
    );
    assert_eq!(info.error_type, "BadRequestError");
}

#[test]
fn test_upstream_error_log_excludes_parsed_body_content() {
    let logs = CapturedLogs::default();
    let writer_logs = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::WARN)
        .with_ansi(false)
        .with_writer(move || writer_logs.writer())
        .finish();
    let body = br#"{"message":"403, message='Forbidden', url='https://private.example/media.jpg?token=SECRET_TOKEN_SENTINEL&email=alice@example.com'","type":"InternalServerError"}"#;

    let _subscriber_guard = tracing::subscriber::set_default(subscriber);
    let info = log_upstream_error(
        reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        "http://backend.invalid/v1/chat/completions",
        body,
    )
    .expect("upstream error body should parse");

    assert!(info.message.contains("SECRET_TOKEN_SENTINEL"));
    let captured = logs.contents();
    assert!(captured.contains("upstream_status=500"));
    assert!(captured.contains("upstream_error_parseable=true"));
    assert!(captured.contains("upstream_error_body_bytes="));
    assert_missing(
        &captured,
        &[
            "SECRET_TOKEN_SENTINEL",
            "alice@example.com",
            "private.example/media.jpg",
            "Forbidden",
            "InternalServerError",
            "error_message",
            "error_type",
        ],
    );
}

#[test]
fn test_effective_status_downgrades_client_fetch_4xx_to_400() {
    for body in [
        br#"{"message":"403, message='Forbidden', url='https://upload.wikimedia.org/x.jpg'","type":"InternalServerError"}"#.as_slice(),
        br#"{"message":"400, message='Bad Request', url='https://host/x.jpg'"}"#.as_slice(),
        br#"{"message":"404, message='Not Found', url='https://host/x.jpg'"}"#.as_slice(),
        br#"{"message":"403 Client Error: Forbidden for url: https://host/x.png"}"#.as_slice(),
        br#"{"error":{"message":"ClientResponseError, status=403, message='Forbidden', url='https://host/x'"}}"#.as_slice(),
    ] {
        assert_eq!(
            eff(500, body),
            StatusCode::BAD_REQUEST,
            "expected 5xx->400 for client-fetch 4xx body: {}",
            String::from_utf8_lossy(body)
        );
    }
}

#[test]
fn test_effective_status_keeps_5xx_when_not_a_client_4xx() {
    let f503 = br#"{"message":"503, message='Service Unavailable', url='https://host/x.jpg'"}"#;
    assert_eq!(eff(500, f503), StatusCode::INTERNAL_SERVER_ERROR);
    let oom = br#"{"message":"CUDA out of memory","type":"InternalServerError"}"#;
    assert_eq!(eff(500, oom), StatusCode::INTERNAL_SERVER_ERROR);
    let noturl = br#"{"message":"requested 450 message tokens exceed the limit"}"#;
    assert_eq!(eff(500, noturl), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        effective_error_status(500, None),
        StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[test]
fn test_effective_status_passes_through_non_5xx() {
    let body = br#"{"message":"400, message='Bad Request', url='https://host/x'"}"#;
    assert_eq!(eff(400, body), StatusCode::BAD_REQUEST);
    assert_eq!(
        eff(404, br#"{"message":"not found"}"#),
        StatusCode::NOT_FOUND
    );
}

#[test]
fn test_parse_upstream_error_nested_formats() {
    let body = br#"{"error":{"message":"model not found","type":"not_found"}}"#;
    let info = parse_upstream_error(body).unwrap();
    assert_eq!(info.message, "model not found");
    assert_eq!(info.error_type, "not_found");

    let body = br#"{"error":{"message":"something went wrong"}}"#;
    let info = parse_upstream_error(body).unwrap();
    assert_eq!(info.message, "something went wrong");
    assert_eq!(info.error_type, "unknown");
}

#[test]
fn test_parse_upstream_error_unparseable_inputs() {
    for body in [
        b"internal secret error details".as_slice(),
        br#"{"type":"BadRequestError","code":400}"#.as_slice(),
        b"".as_slice(),
        b"{}".as_slice(),
    ] {
        assert!(parse_upstream_error(body).is_none());
    }
}
