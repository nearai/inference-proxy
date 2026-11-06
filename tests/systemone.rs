use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sha2::{Digest, Sha256};
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;
use common::*;

const UPSTREAM_MODEL: &str = "levanto-sage-v1.4";
const CLIENT_HASH: &str = "8d3f6c1a2b4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8";

fn decision_app(backend: &MockServer) -> axum::Router {
    build_test_app(
        &backend.uri(),
        TestAppOptions {
            backend_token: Some("backend-secret".to_string()),
            systemone_upstream_model: Some(UPSTREAM_MODEL.to_string()),
            ..Default::default()
        },
    )
}

fn decision_request(authorization: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("authorization", authorization)
        .header("content-type", "application/json")
        .header("x-request-hash", CLIENT_HASH)
        .header("x-custom-client-header", "must-not-reach-backend")
        .body(Body::from(
            r#"{"model":"test-model","state":"I was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this about billing?"}}}"#,
        ))
        .unwrap()
}

fn sage_answer(id: Option<&str>) -> serde_json::Value {
    let mut answer = serde_json::json!({
        "model": UPSTREAM_MODEL,
        "answers": {"billing": {"type": "noul", "noul": 1.0}},
        "usage": {"input_tokens": 42, "output_tokens": 0}
    });
    if let Some(id) = id {
        answer["id"] = serde_json::Value::String(id.to_string());
    }
    answer
}

async fn body_bytes(response: axum::response::Response) -> bytes::Bytes {
    response.into_body().collect().await.unwrap().to_bytes()
}

#[tokio::test]
async fn decision_is_signed_under_a_generated_id() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .and(header("authorization", "Bearer backend-secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(sage_answer(Some("reused"))))
        .expect(1)
        .mount(&backend)
        .await;
    let app = decision_app(&backend);

    let response = app
        .clone()
        .oneshot(decision_request("Bearer test-token"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_bytes(response).await;
    let answer: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let id = answer["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("systemone-"), "generated id, got {id}");
    assert_ne!(id, "reused", "an upstream id is never the receipt key");
    assert_eq!(answer["answers"]["billing"]["noul"], 1.0);

    // The backend got its own model name, the backend bearer, and no client header.
    let sent = &backend.received_requests().await.unwrap()[0];
    let sent_body: serde_json::Value = serde_json::from_slice(&sent.body).unwrap();
    assert_eq!(sent_body["model"], UPSTREAM_MODEL);
    assert_eq!(sent_body["questions"]["billing"]["type"], "noul");
    assert!(sent.headers.get("x-custom-client-header").is_none());
    assert!(sent.headers.get("x-request-hash").is_none());

    // The receipt covers the client's hash and the exact bytes returned.
    for algo in ["ecdsa", "ed25519"] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/signature/{id}?signing_algo={algo}"))
                    .header("authorization", "Bearer test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let receipt: serde_json::Value =
            serde_json::from_slice(&body_bytes(response).await).unwrap();
        let expected = format!(
            "test-model:{CLIENT_HASH}:{}",
            hex::encode(Sha256::digest(&bytes))
        );
        assert_eq!(receipt["text"], expected, "{algo}");
        assert_eq!(receipt["signing_algo"], algo);
    }
}

#[tokio::test]
async fn customer_keys_and_encryption_are_refused_before_the_backend() {
    let backend = MockServer::start().await;
    let app = decision_app(&backend);

    let response = app
        .clone()
        .oneshot(decision_request("Bearer sk-test-customer"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let mut encrypted = decision_request("Bearer test-token");
    encrypted
        .headers_mut()
        .insert("x-signing-algo", "ed25519".parse().unwrap());
    encrypted
        .headers_mut()
        .insert("x-client-pub-key", "aa".repeat(32).parse().unwrap());
    let response = app.clone().oneshot(encrypted).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    assert!(backend.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn an_answer_that_is_not_a_json_object_is_never_signed() {
    for (content_type, body) in [
        ("text/plain", r#"{"model":"x"}"#),
        ("application/json", "not json"),
        ("application/json", "[1,2]"),
    ] {
        let backend = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", content_type)
                    .set_body_string(body),
            )
            .mount(&backend)
            .await;
        let response = decision_app(&backend)
            .oneshot(decision_request("Bearer test-token"))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_GATEWAY,
            "{content_type} {body}"
        );
    }
}

#[tokio::test]
async fn backend_errors_keep_their_status() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(422).set_body_json(serde_json::json!({
            "detail": "unknown model"
        })))
        .mount(&backend)
        .await;
    let response = decision_app(&backend)
        .oneshot(decision_request("Bearer test-token"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn model_list_names_the_catalog_model_while_the_backend_is_ready() {
    for (backend_status, expected) in [
        (200, StatusCode::OK),
        (503, StatusCode::SERVICE_UNAVAILABLE),
    ] {
        let backend = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(backend_status))
            .expect(1)
            .mount(&backend)
            .await;
        let response = decision_app(&backend)
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
        if expected == StatusCode::OK {
            let list: serde_json::Value =
                serde_json::from_slice(&body_bytes(response).await).unwrap();
            assert_eq!(list["data"][0]["id"], "test-model");
            assert_eq!(list["data"].as_array().unwrap().len(), 1);
        }
    }
}

#[tokio::test]
async fn route_is_absent_without_a_decision_model() {
    let backend = MockServer::start().await;
    let app = build_test_app(&backend.uri(), TestAppOptions::default());
    for authorization in ["Bearer test-token", "Bearer sk-test-customer"] {
        let response = app
            .clone()
            .oneshot(decision_request(authorization))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{authorization}");
        let body: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
        assert_eq!(body["error"]["type"], "not_found");
    }
    assert!(backend.received_requests().await.unwrap().is_empty());
}
