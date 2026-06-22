use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[path = "common/auth_header.rs"]
mod auth_header;
#[path = "common/body_to_bytes.rs"]
mod body_to_bytes;
mod common;
#[path = "common/minimal_sse_response.rs"]
mod minimal_sse_response;
#[path = "common/response_request_id.rs"]
mod response_request_id;
#[path = "common/streaming_chat_body.rs"]
mod streaming_chat_body;
#[path = "common/upstream_request_id.rs"]
mod upstream_request_id;
#[path = "common/valid_request_id.rs"]
mod valid_request_id;

use auth_header::*;
use body_to_bytes::*;
use common::*;
use minimal_sse_response::*;
use response_request_id::*;
use streaming_chat_body::*;
use upstream_request_id::*;
use valid_request_id::*;

#[tokio::test]
async fn generated_request_id_is_forwarded_to_tokenize_upstream() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "tokens": [1],
            "count": 1
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri(), TestAppOptions::default());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/tokenize")
                .header(auth_header().0, auth_header().1)
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let selected_request_id = request_id_from_response(&response);
    assert_uuid(&selected_request_id);
    eprintln!(
        "manual-qa: tokenize status={} selected x-request-id={selected_request_id}",
        response.status()
    );
    assert_backend_saw_request_id(&mock_server, &selected_request_id).await;
}

#[tokio::test]
async fn invalid_request_id_is_replaced_and_forwarded_to_score_upstream() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/score"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "score-contract",
            "results": [{"score": 0.8}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri(), TestAppOptions::default());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/score")
                .header(auth_header().0, auth_header().1)
                .header("content-type", "application/json")
                .header("x-request-id", "not-a-uuid")
                .body(Body::from(r#"{"model":"m","text_1":"a","text_2":"b"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let selected_request_id = request_id_from_response(&response);
    assert_uuid(&selected_request_id);
    assert_ne!(selected_request_id, "not-a-uuid");
    eprintln!("manual-qa: score invalid input replaced with x-request-id={selected_request_id}");
    assert_backend_saw_request_id(&mock_server, &selected_request_id).await;
}

#[tokio::test]
async fn public_cloud_api_key_tenant_headers_are_not_forwarded() {
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
        .and(path("/v1/internal/usage"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&cloud_api)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    minimal_sse_response("chatcmpl-public-tenant-strip"),
                    "text/event-stream",
                ),
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
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-test-public-key-1234567890123")
                .header("content-type", "application/json")
                .header("x-request-id", VALID_REQUEST_ID)
                .header("x-org-id", "org-spoofed")
                .header("x-workspace-id", "ws-spoofed")
                .body(streaming_chat_body())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("x-request-id").unwrap(),
        VALID_REQUEST_ID
    );

    let received = backend
        .received_requests()
        .await
        .expect("backend records requests");
    assert_eq!(received.len(), 1, "expected one backend request");
    assert_eq!(
        received[0].headers.get("x-request-id").unwrap(),
        VALID_REQUEST_ID
    );
    assert!(!received[0].headers.contains_key("x-org-id"));
    assert!(!received[0].headers.contains_key("x-workspace-id"));
    eprintln!(
        "manual-qa: public tenant strip wiremock x-request-id={} x-org-id_present={} x-workspace-id_present={}",
        VALID_REQUEST_ID,
        received[0].headers.contains_key("x-org-id"),
        received[0].headers.contains_key("x-workspace-id")
    );
}

#[tokio::test]
async fn streaming_response_has_request_id_header_before_body_is_read() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    minimal_sse_response("chatcmpl-streaming-header"),
                    "text/event-stream",
                ),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri(), TestAppOptions::default());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(auth_header().0, auth_header().1)
                .header("content-type", "application/json")
                .body(streaming_chat_body())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let streaming_request_id = request_id_from_response(&response);
    assert_uuid(&streaming_request_id);
    eprintln!(
        "manual-qa: streaming headers observed before body status={} x-request-id={streaming_request_id}",
        response.status()
    );
    let body = body_to_bytes(response).await;
    assert!(
        !body.is_empty(),
        "stream body should remain readable after header assertion"
    );
}
