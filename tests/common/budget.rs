//! Explicit monetary admission fixture for tests focused on transport/auth behavior.
#![allow(dead_code)] // Shared fixture modules use only the helpers needed by each test binary.
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};
pub async fn mount(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/internal/budget"))
        .respond_with(|request: &Request| {
            let metadata: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            assert!(metadata.get("messages").is_none());
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id":metadata["id"],"model":metadata["model"]
            }))
        })
        .mount(server)
        .await;
}

/// Each test process uses an isolated spool, just as a deployed replica does.
pub fn delivery(
    config: &vllm_proxy_rs::config::Config,
) -> std::sync::Arc<vllm_proxy_rs::usage_report::UsageReportDelivery> {
    let delivery =
        vllm_proxy_rs::usage_report::UsageReportDelivery::new(config.usage_report.clone());
    if config.cloud_api_url.is_some() {
        let dir = tempfile::Builder::new()
            .prefix("nearai-estimated-admission-test-")
            .tempdir()
            .unwrap()
            .keep();
        delivery
            .enable_outbox(
                dir,
                reqwest::Client::new(),
                config.cloud_api_url.clone(),
                config.cloud_api_usage_token.clone(),
            )
            .unwrap();
    }
    delivery
}
