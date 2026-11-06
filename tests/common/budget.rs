//! Explicit monetary admission fixture for tests focused on transport/auth behavior.
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};
pub async fn mount(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/internal/budget"))
        .respond_with(|request: &Request| {
            let metadata: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            assert!(metadata.get("messages").is_none());
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id":metadata["id"],"model":metadata["model"],
                "max_output_tokens":metadata["max_output_tokens"].as_i64().unwrap_or(4096)
            }))
        })
        .mount(server)
        .await;
}

/// Tokenizer fixture for transport tests; budget-focused tests override its response.
#[allow(dead_code)]
pub async fn mount_counter(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"count":128})))
        .with_priority(10)
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/internal/input-tokens"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"input_tokens":128})),
        )
        .with_priority(10)
        .mount(server)
        .await;
}
