use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;
use common::*;

#[tokio::test]
async fn direct_privacy_classify_reports_nested_input_usage() {
    let backend = MockServer::start().await;
    let cloud_api = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/privacy/classify"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "model": "test-model",
            "data": [
                {"index": 0, "spans": [], "usage": {"input_tokens": 11}},
                {"index": 1, "spans": [], "usage": {"input_tokens": 7}}
            ]
        })))
        .mount(&backend)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "organization_id": "org-test",
            "workspace_id": "ws-test",
            "api_key_id": "key-test"
        })))
        .mount(&cloud_api)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .and(header("authorization", "Bearer test-usage-token"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&cloud_api)
        .await;

    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            cloud_api_url: Some(cloud_api.uri()),
            dstack_socket_path: None,
        },
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/privacy/classify")
                .header("authorization", "Bearer sk-test-customer")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"input":["one","two"]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_body: serde_json::Value =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(response_body["data"][0]["usage"]["input_tokens"], 11);

    let usage_request = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Some(request) = cloud_api
                .received_requests()
                .await
                .unwrap_or_default()
                .into_iter()
                .find(|request| request.url.path() == "/v1/internal/usage")
            {
                break request;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("usage report should arrive");
    let usage: serde_json::Value = serde_json::from_slice(&usage_request.body).unwrap();
    assert_eq!(usage["type"], "privacy_classify");
    assert_eq!(usage["input_tokens"], 18);
    assert_eq!(usage["organization_id"], "org-test");
    assert_eq!(usage["workspace_id"], "ws-test");
    assert_eq!(usage["api_key_id"], "key-test");
}
