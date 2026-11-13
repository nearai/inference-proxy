//! On the direct key path the backend gets a server-chosen request id. The
//! proxy's completion, error and cancel lines carry it as
//! `upstream_request_id`, so they join the backend's lines.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;

use common::*;

const REQUEST_ID: &str = "8c1e2f3a-4b5c-4d6e-8f70-81a2b3c4d5e6";

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Logs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Logs {
    fn take(&self) -> Vec<serde_json::Value> {
        let bytes = std::mem::take(&mut *self.0.lock().unwrap());
        String::from_utf8(bytes)
            .unwrap()
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .map(|line| line["fields"].clone())
            .collect()
    }
}

fn line<'a>(lines: &'a [serde_json::Value], message: &str) -> &'a serde_json::Value {
    let found: Vec<_> = lines
        .iter()
        .filter(|line| line["message"] == message)
        .collect();
    assert_eq!(found.len(), 1, "{message}: {lines:#?}");
    found[0]
}

async fn sent_request_ids(backend: &MockServer) -> Vec<String> {
    backend
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| {
            request
                .headers
                .get("x-request-id")
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
        })
        .collect()
}

fn chat(api_key: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .header("x-request-id", REQUEST_ID)
        .body(Body::from(
            r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
        ))
        .unwrap()
}

#[tokio::test]
async fn direct_requests_log_the_backend_request_id() {
    let cloud_api = MockServer::start().await;
    budget_fixture::mount(&cloud_api).await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "organization_id": "org-verified",
            "workspace_id": "ws-verified",
            "api_key_id": "key-verified"
        })))
        .mount(&cloud_api)
        .await;
    let logs = Logs::default();
    let writer = logs.clone();
    let _subscriber = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(tracing_subscriber::EnvFilter::new("vllm_proxy_rs=info"))
            .with_writer(move || writer.clone())
            .finish(),
    );
    let _registered = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let app = |backend: &MockServer| {
        build_test_app(
            &backend.uri(),
            TestAppOptions {
                cloud_api_url: Some(cloud_api.uri()),
                ..Default::default()
            },
        )
    };

    // Completed: the completion line.
    let ok = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-joined",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        })))
        .mount(&ok)
        .await;
    let response = app(&ok)
        .oneshot(chat("sk-test-key-completed-123456"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let sent = sent_request_ids(&ok).await;
    assert_eq!(sent.len(), 1);
    assert_ne!(sent[0], REQUEST_ID);
    let lines = logs.take();
    let completed = line(&lines, "request completed");
    assert_eq!(completed["request_id"], REQUEST_ID);
    assert_eq!(completed["upstream_request_id"], sent[0].as_str());
    assert_eq!(completed["org_id"], "org-verified");

    // Failed: the upstream error line and the finish line.
    let failing = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
            "error": {"message": "engine error", "type": "InternalServerError"}
        })))
        .mount(&failing)
        .await;
    let response = app(&failing)
        .oneshot(chat("sk-test-key-failed-1234567890"))
        .await
        .unwrap();
    assert!(response.status().is_server_error());
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let sent = sent_request_ids(&failing).await;
    assert!(!sent.is_empty());
    let lines = logs.take();
    let error = lines
        .iter()
        .find(|line| line["message"] == "Backend returned non-success status")
        .unwrap_or_else(|| panic!("{lines:#?}"));
    assert_eq!(error["upstream_request_id"], sent[0].as_str());
    let finished = line(&lines, "request finished");
    assert_eq!(finished["upstream_request_id"], sent[0].as_str());
    assert_eq!(finished["org_id"], "org-verified");

    // Cancelled while the backend works: the cancel line.
    let slow = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(30))
                .set_body_json(serde_json::json!({})),
        )
        .mount(&slow)
        .await;
    let outcome = tokio::time::timeout(
        Duration::from_millis(500),
        app(&slow).oneshot(chat("sk-test-key-cancelled-12345678")),
    )
    .await;
    assert!(outcome.is_err());
    let sent = sent_request_ids(&slow).await;
    assert_eq!(sent.len(), 1);
    let lines = logs.take();
    let cancelled = line(&lines, "request cancelled");
    assert_eq!(cancelled["upstream_request_id"], sent[0].as_str());
    assert_eq!(cancelled["request_id"], REQUEST_ID);

    // The trusted path sends the request's own id: no separate field.
    let response = app(&ok).oneshot(chat("test-token")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let lines = logs.take();
    let completed = line(&lines, "request completed");
    assert!(
        completed.get("upstream_request_id").is_none(),
        "{completed}"
    );
}
