use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[path = "common/auth_header.rs"]
mod auth_header;
mod common;
#[path = "common/response_request_id.rs"]
mod response_request_id;
#[path = "common/second_valid_request_id.rs"]
mod second_valid_request_id;
#[path = "common/streaming_chat_body.rs"]
mod streaming_chat_body;
#[path = "common/valid_request_id.rs"]
mod valid_request_id;

use auth_header::*;
use common::*;
use response_request_id::*;
use second_valid_request_id::*;
use streaming_chat_body::*;
use valid_request_id::*;

#[tokio::test]
async fn operational_public_routes_echo_uuid_request_id_headers() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock_server)
        .await;
    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_string("backend_metric 1\n"))
        .mount(&mock_server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "test-model", "object": "model"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(
        &mock_server.uri(),
        TestAppOptions {
            cloud_api_url: None,
            dstack_socket_path: Some("/missing/dstack.sock".to_string()),
        },
    );
    let probes = [
        ("GET", "/healthz", None, StatusCode::SERVICE_UNAVAILABLE),
        ("GET", "/metrics", None, StatusCode::OK),
        ("GET", "/v1/metrics", None, StatusCode::OK),
        ("GET", "/v1/models", None, StatusCode::OK),
        (
            "GET",
            "/v1/attestation/report?signing_algo=rsa",
            None,
            StatusCode::BAD_REQUEST,
        ),
        (
            "GET",
            "/v1/signature/missing-chat-id",
            Some(auth_header()),
            StatusCode::NOT_FOUND,
        ),
        (
            "POST",
            "/internal/gpu_evidence",
            Some(auth_header()),
            StatusCode::BAD_REQUEST,
        ),
        (
            "GET",
            "/.well-known/ohttp-gateway",
            None,
            StatusCode::NOT_FOUND,
        ),
        ("GET", "/v1/ohttp/config", None, StatusCode::NOT_FOUND),
    ];

    for (method_name, route, auth, expected_status) in probes {
        let body = if route == "/internal/gpu_evidence" {
            Body::from(r#"{"nonce":"not-hex","no_gpu_mode":true}"#)
        } else {
            Body::empty()
        };
        let mut request = Request::builder()
            .method(method_name)
            .uri(route)
            .header("x-request-id", SECOND_VALID_REQUEST_ID);
        if let Some((name, value)) = auth {
            request = request.header(name, value);
        }
        if route == "/internal/gpu_evidence" {
            request = request.header("content-type", "application/json");
        }

        let response = app
            .clone()
            .oneshot(request.body(body).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), expected_status, "{route}");
        assert_eq!(
            response.headers().get("x-request-id").unwrap(),
            SECOND_VALID_REQUEST_ID
        );
        eprintln!(
            "manual-qa: operational/public route {method_name} {route} status={} x-request-id={}",
            response.status(),
            SECOND_VALID_REQUEST_ID
        );
    }
}

#[tokio::test]
async fn route_matrix_responses_include_uuid_request_id_headers() {
    let mock_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/not-a-real-route"))
        .respond_with(ResponseTemplate::new(200).set_body_string("fallback ok"))
        .mount(&mock_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/images/edits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "img-edit-contract",
            "data": [{"url": "https://example.com/edited.png"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri(), TestAppOptions::default());
    let generated = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(generated.status(), StatusCode::OK);
    let generated_request_id = request_id_from_response(&generated);
    assert_uuid(&generated_request_id);
    eprintln!(
        "manual-qa: root operational status={} generated x-request-id={generated_request_id}",
        generated.status()
    );

    let valid_reuse = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/version")
                .header("x-request-id", SECOND_VALID_REQUEST_ID)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(valid_reuse.status(), StatusCode::OK);
    assert_eq!(
        valid_reuse.headers().get("x-request-id").unwrap(),
        SECOND_VALID_REQUEST_ID
    );
    eprintln!(
        "manual-qa: version operational status={} reused x-request-id={}",
        valid_reuse.status(),
        SECOND_VALID_REQUEST_ID
    );

    let auth_failure = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(streaming_chat_body())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(auth_failure.status(), StatusCode::UNAUTHORIZED);
    let auth_failure_request_id = request_id_from_response(&auth_failure);
    assert_uuid(&auth_failure_request_id);
    eprintln!(
        "manual-qa: auth failure status={} generated x-request-id={auth_failure_request_id}",
        auth_failure.status()
    );

    let fallback = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/not-a-real-route")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(fallback.status(), StatusCode::OK);
    let fallback_request_id = request_id_from_response(&fallback);
    assert_uuid(&fallback_request_id);
    eprintln!(
        "manual-qa: fallback status={} generated x-request-id={fallback_request_id}",
        fallback.status()
    );

    let boundary = "----RequestIdContractBoundary";
    let multipart_body = format!(
        "--{boundary}\r\n\
         Content-Disposition: form-data; name=\"prompt\"\r\n\r\n\
         a cat\r\n\
         --{boundary}\r\n\
         Content-Disposition: form-data; name=\"image\"; filename=\"test.png\"\r\n\
         Content-Type: image/png\r\n\r\n\
         fakepngdata\r\n\
         --{boundary}--\r\n"
    );
    let images_edits = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header(auth_header().0, auth_header().1)
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .header("x-request-id", VALID_REQUEST_ID)
                .body(Body::from(multipart_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(images_edits.status(), StatusCode::OK);
    assert_eq!(
        images_edits.headers().get("x-request-id").unwrap(),
        VALID_REQUEST_ID
    );
    eprintln!(
        "manual-qa: images edits status={} response x-request-id={}",
        images_edits.status(),
        VALID_REQUEST_ID
    );
}
