use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;
use common::*;

fn rerank_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/rerank")
        .header("authorization", "Bearer test-token")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"test-model","query":"q","documents":["a"]}"#,
        ))
        .unwrap()
}

async fn response_id(response: axum::response::Response) -> serde_json::Value {
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

#[tokio::test]
async fn pool_passthrough_uses_backend_client_and_bearer() {
    let pool = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .and(header("authorization", "Bearer backend-secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "rerank-pool",
            "results": []
        })))
        .expect(1)
        .mount(&pool)
        .await;
    let app = build_test_app(
        &pool.uri(),
        TestAppOptions {
            backend_token: Some("backend-secret".to_string()),
            ..Default::default()
        },
    );

    let response = app.oneshot(rerank_request()).await.unwrap();

    assert_eq!(response_id(response).await["id"], "rerank-pool");
    pool.verify().await;
}

#[tokio::test]
async fn override_passthrough_uses_plain_client_without_backend_bearer() {
    let pool = MockServer::start().await;
    let override_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/custom/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "rerank-override",
            "results": []
        })))
        .expect(1)
        .mount(&override_server)
        .await;
    let app = build_test_app(
        &pool.uri(),
        TestAppOptions {
            backend_token: Some("backend-secret".to_string()),
            rerank_url_override: Some(format!("{}/custom/rerank", override_server.uri())),
            ..Default::default()
        },
    );

    let response = app.oneshot(rerank_request()).await.unwrap();

    assert_eq!(response_id(response).await["id"], "rerank-override");
    assert!(pool.received_requests().await.unwrap().is_empty());
    let requests = override_server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].headers.get("authorization").is_none());
}
