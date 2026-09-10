use super::*;
use http_body_util::BodyExt;
use serde_json::{json, Value};

mod json_reports;
mod streams;

const ID: &str = "chatcmpl-direct-cache";

async fn cloud_fixture() -> MockServer {
    let cloud = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "organization_id": "org-cache", "workspace_id": "ws-cache", "api_key_id": "key-cache"
        })))
        .mount(&cloud)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .and(header("authorization", "Bearer test-usage-token"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&cloud)
        .await;
    cloud
}

fn request(route: &str, streaming: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(route)
        .header("content-type", "application/json")
        .header("authorization", "Bearer sk-test-valid-key-12345678901")
        .body(Body::from(
            json!({"model": "test-model", "messages": [{"role": "user", "content": "hello"}],
            "prompt": "hello", "stream": streaming})
            .to_string(),
        ))
        .unwrap()
}

async fn assert_report(cloud: &MockServer, cached: i64, output: i64) {
    wait_for_usage_request(cloud, 1).await;
    let reports = get_usage_requests(cloud).await;
    assert_eq!(reports.len(), 1, "exactly one terminal usage report");
    assert_eq!(
        reports[0],
        json!({"type": "chat_completion", "model": "test-model",
        "id": ID, "input_tokens": 1858, "output_tokens": output, "cache_read_tokens": cached,
        "organization_id": "org-cache", "workspace_id": "ws-cache", "api_key_id": "key-cache"})
    );
    cloud.verify().await;
}

fn frame(usage: Value) -> String {
    format!(
        "data: {}\n\n",
        json!({"id": ID, "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "text": "hi"}], "usage": usage})
    )
}

fn first_frame() -> String {
    frame(json!({"prompt_tokens": 1858, "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 1856}}))
}

async fn assert_no_signature(app: axum::Router) {
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/signature/{ID}?signing_algo=ecdsa"))
                .header("authorization", "Bearer test-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
