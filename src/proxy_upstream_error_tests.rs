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
    effective_error_status(status, parse_upstream_error(body).as_ref(), false)
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
    let tracing_ids = TracingIds {
        request_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        request_id_inbound: true,
        org_id: Some("org-test".to_string()),
        workspace_id: Some("workspace-test".to_string()),
        request_source: None,
        forward_tenant_headers: false,
    };

    let _subscriber_guard = tracing::subscriber::set_default(subscriber);
    let info = log_upstream_error(
        reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        "http://backend.invalid/v1/chat/completions",
        body,
        Some(&tracing_ids),
    )
    .expect("upstream error body should parse");

    assert!(info.message.contains("SECRET_TOKEN_SENTINEL"));
    let captured = logs.contents();
    assert!(captured.contains("upstream_status=500"));
    assert!(captured.contains("request_id=550e8400-e29b-41d4-a716-446655440000"));
    assert!(captured.contains("org_id=org-test"));
    assert!(captured.contains("workspace_id=workspace-test"));
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
fn test_effective_status_downgrades_allowed_media_domain_errors_to_400() {
    let body = br#"{"message":"The URL must be from one of the allowed domains: ['prod-files-secure.s3.us-west-2.amazonaws.com']. Input URL domain: cdn.generalcontext.com","type":"InternalServerError"}"#;
    assert_eq!(eff(500, body), StatusCode::BAD_REQUEST);
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
        effective_error_status(500, None, false),
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

#[test]
fn test_queue_full_maps_to_429_only_when_enabled() {
    let body =
        br#"{"object":"error","message":"The request queue is full.","type":"abort","code":503}"#;
    let info = parse_upstream_error(body);
    assert_eq!(
        effective_error_status(503, info.as_ref(), true),
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(
        effective_error_status(503, info.as_ref(), false),
        StatusCode::SERVICE_UNAVAILABLE
    );
    // Only the admission rejection is back-pressure; other 503s stay.
    let other = parse_upstream_error(br#"{"message":"Model is loading","type":"x","code":503}"#);
    assert_eq!(
        effective_error_status(503, other.as_ref(), true),
        StatusCode::SERVICE_UNAVAILABLE
    );
    // And a queue-full message on a non-503 status is never reinterpreted.
    assert_eq!(
        effective_error_status(500, info.as_ref(), true),
        StatusCode::INTERNAL_SERVER_ERROR
    );
}

#[test]
fn test_first_sse_error_event_detection() {
    let queue_full = b"data: {\"error\":{\"object\":\"error\",\"message\":\"The request queue is full.\",\"type\":\"abort\",\"code\":503}}\n\ndata: [DONE]\n\n";
    let err = first_sse_error_event(queue_full).expect("error event");
    assert_eq!(err["code"], 503);
    assert_eq!(err["message"], "The request queue is full.");

    // Leading keep-alive comments and blank lines are skipped.
    let with_comment = b": keep-alive\n\ndata: {\"error\":{\"message\":\"nope\",\"code\":429}}\n\n";
    assert_eq!(first_sse_error_event(with_comment).unwrap()["code"], 429);

    // Ordinary first events are not errors.
    let role =
        b"data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n";
    assert!(first_sse_error_event(role).is_none());
    assert!(first_sse_error_event(b"data: [DONE]\n\n").is_none());
    assert!(first_sse_error_event(b"data: {\"error\":null}\n\n").is_none());
    assert!(first_sse_error_event(b"garbage").is_none());
    assert!(first_sse_error_event(&[0xff, 0xfe]).is_none());
}
