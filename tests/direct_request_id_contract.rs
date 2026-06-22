use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;
#[path = "common/response_request_id.rs"]
mod response_request_id;
#[path = "common/valid_request_id.rs"]
mod valid_request_id;

use common::*;
use response_request_id::*;
use valid_request_id::*;

#[tokio::test]
async fn direct_privacy_classify_observable_request_id_contract() {
    let backend = MockServer::start().await;
    let cloud_api = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "organization_id": "org-authenticated",
            "workspace_id": "ws-authenticated",
            "api_key_id": "key-authenticated"
        })))
        .mount(&cloud_api)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/privacy/classify"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/plain")
                .set_body_string("classify ok"),
        )
        .mount(&backend)
        .await;

    let app = build_test_app(
        &backend.uri(),
        TestAppOptions {
            cloud_api_url: Some(cloud_api.uri()),
            dstack_socket_path: None,
        },
    );
    let valid = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/privacy/classify?case=valid")
                .header("authorization", "Bearer sk-test-public-key-1234567890123")
                .header("content-type", "application/json")
                .header("x-request-id", VALID_REQUEST_ID)
                .header("x-org-id", "org-spoofed")
                .header("x-workspace-id", "ws-spoofed")
                .body(Body::from(
                    r#"{"model":"openai/privacy-filter","input":"hello"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(valid.status(), StatusCode::OK);
    assert_eq!(
        valid.headers().get("x-request-id").unwrap(),
        VALID_REQUEST_ID
    );

    let invalid = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/privacy/classify?case=invalid")
                .header("authorization", "Bearer sk-test-public-key-1234567890123")
                .header("content-type", "application/json")
                .header("x-request-id", "not-a-uuid")
                .header("x-org-id", "org-spoofed")
                .header("x-workspace-id", "ws-spoofed")
                .body(Body::from(
                    r#"{"model":"openai/privacy-filter","input":"hello"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::OK);
    let replacement_id = request_id_from_response(&invalid);
    assert_uuid(&replacement_id);
    assert_ne!(replacement_id, "not-a-uuid");

    let received = backend
        .received_requests()
        .await
        .expect("backend records requests");
    assert_eq!(
        received.len(),
        2,
        "expected valid and invalid classify requests"
    );

    let valid_upstream = received
        .iter()
        .find(|request| request.url.query() == Some("case=valid"))
        .expect("valid classify request reached backend");
    assert_eq!(valid_upstream.url.path(), "/v1/privacy/classify");
    assert_eq!(
        valid_upstream
            .headers
            .get("x-request-id")
            .and_then(|value| value.to_str().ok()),
        Some(VALID_REQUEST_ID)
    );
    assert!(!valid_upstream.headers.contains_key("x-org-id"));
    assert!(!valid_upstream.headers.contains_key("x-workspace-id"));

    let invalid_upstream = received
        .iter()
        .find(|request| request.url.query() == Some("case=invalid"))
        .expect("invalid classify request reached backend");
    assert_eq!(invalid_upstream.url.path(), "/v1/privacy/classify");
    assert_eq!(
        invalid_upstream
            .headers
            .get("x-request-id")
            .and_then(|value| value.to_str().ok()),
        Some(replacement_id.as_str())
    );
    assert!(!invalid_upstream.headers.contains_key("x-org-id"));
    assert!(!invalid_upstream.headers.contains_key("x-workspace-id"));

    eprintln!(
        "manual-qa: POST /v1/privacy/classify forwarded status=200 valid_x_request_id={} invalid_replacement_x_request_id={} upstream_valid_query={} tenant_headers_present=false",
        VALID_REQUEST_ID,
        replacement_id,
        valid_upstream.url.query().unwrap_or("<missing>")
    );
}
