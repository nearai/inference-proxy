use super::*;

use std::io;
use std::sync::{Arc, Mutex};

use tracing::Level;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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

#[tokio::test]
async fn startup_checks_logs_redact_response_bodies_and_request_payloads() {
    let logs = CapturedLogs::default();
    let writer_logs = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_ansi(false)
        .with_writer(move || writer_logs.writer())
        .finish();
    let _subscriber_guard = tracing::subscriber::set_default(subscriber);

    let models_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(500).set_body_string("MODELS_RESPONSE_BODY_SENTINEL"))
        .mount(&models_server)
        .await;
    let client = reqwest::Client::new();
    let models_url = format!("{}/v1/models", models_server.uri());
    let models_result =
        check_models(&client, &models_url, "test-model", Duration::from_secs(5)).await;
    assert!(matches!(
        models_result,
        Err(StartupCheckError::UnexpectedStatus { .. })
    ));

    let chat_error_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("CHAT_RESPONSE_BODY_SENTINEL"))
        .mount(&chat_error_server)
        .await;
    let chat_error_url = format!("{}/v1/chat/completions", chat_error_server.uri());
    let chat_error_result = check_chat_completions_with_tools(
        &client,
        &chat_error_url,
        "test-model",
        false,
        Duration::from_secs(5),
    )
    .await;
    assert!(matches!(
        chat_error_result,
        Err(StartupCheckError::UnexpectedStatus { .. })
    ));

    let chat_text_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-text",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "MODEL_TEXT_RESPONSE_SENTINEL"
                },
                "finish_reason": "stop"
            }]
        })))
        .mount(&chat_text_server)
        .await;
    let chat_text_url = format!("{}/v1/chat/completions", chat_text_server.uri());
    let chat_text_result = check_chat_completions_with_tools(
        &client,
        &chat_text_url,
        "test-model",
        false,
        Duration::from_secs(5),
    )
    .await;
    assert!(chat_text_result.is_ok());

    let captured = logs.contents();
    assert!(captured.contains("body_len="));
    assert!(captured.contains("request_message_count=1"));
    assert!(captured.contains("tool_count=1"));
    assert!(captured.contains("content_len="));
    for forbidden in [
        "MODELS_RESPONSE_BODY_SENTINEL",
        "CHAT_RESPONSE_BODY_SENTINEL",
        "MODEL_TEXT_RESPONSE_SENTINEL",
        "What is the weather in San Francisco?",
    ] {
        assert!(
            !captured.contains(forbidden),
            "captured startup-check logs must not contain {forbidden}; logs: {captured}"
        );
    }
}

#[tokio::test]
async fn startup_checks_logs_redact_streaming_tool_arguments() {
    let logs = CapturedLogs::default();
    let writer_logs = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_ansi(false)
        .with_writer(move || writer_logs.writer())
        .finish();
    let _subscriber_guard = tracing::subscriber::set_default(subscriber);

    let mock_server = MockServer::start().await;
    let sse_body = "\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"secret\\\"\"}}]}}]}\n\n\
data: {\"id\":\"chatcmpl-stream\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":\\\"STREAM_TOOL_ARGUMENT_SENTINEL\\\"}\"}}]}}]}\n\n\
data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let client = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", mock_server.uri());
    let result = check_chat_completions_with_tools(
        &client,
        &url,
        "test-model",
        true,
        Duration::from_secs(5),
    )
    .await;
    assert!(result.is_ok());

    let captured = logs.contents();
    assert!(captured.contains("tool_call_count=1"));
    assert!(captured.contains("argument_len="));
    assert!(captured.contains("data_len="));
    for forbidden in [
        "STREAM_TOOL_ARGUMENT_SENTINEL",
        "{\"secret\"",
        "What is the weather in San Francisco?",
    ] {
        assert!(
            !captured.contains(forbidden),
            "captured startup-check logs must not contain {forbidden}; logs: {captured}"
        );
    }
}
