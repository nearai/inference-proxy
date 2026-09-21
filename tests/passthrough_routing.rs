use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sha2::{Digest, Sha256};
use tower::ServiceExt;
use vllm_proxy_rs::encryption;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

mod common;
use common::*;

#[derive(Clone, Copy)]
struct MultipartCase {
    route: &'static str,
    pool_path: &'static str,
    override_path: &'static str,
    boundary: &'static str,
    body: &'static str,
    response_id: &'static str,
}

const MULTIPART_CASES: [MultipartCase; 2] = [
    MultipartCase {
        route: "/v1/images/edits",
        pool_path: "/v1/images/edits",
        override_path: "/custom/images/edits",
        boundary: "ImageBoundary",
        body: concat!(
            "--ImageBoundary\r\n",
            "Content-Disposition: form-data; name=\"prompt\"\r\n\r\n",
            "edit this\r\n",
            "--ImageBoundary\r\n",
            "Content-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\n",
            "Content-Type: image/png\r\n\r\n",
            "fakepngdata\r\n",
            "--ImageBoundary--\r\n",
        ),
        response_id: "image-edit",
    },
    MultipartCase {
        route: "/v1/audio/transcriptions",
        pool_path: "/v1/audio/transcriptions",
        override_path: "/custom/audio/transcriptions",
        boundary: "AudioBoundary",
        body: concat!(
            "--AudioBoundary\r\n",
            "Content-Disposition: form-data; name=\"model\"\r\n\r\n",
            "whisper-1\r\n",
            "--AudioBoundary\r\n",
            "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
            "Content-Type: audio/wav\r\n\r\n",
            "fakeaudiodata\r\n",
            "--AudioBoundary--\r\n",
        ),
        response_id: "transcription",
    },
];

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

fn multipart_request(route: &str, boundary: &str, body: String) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(route)
        .header("authorization", "Bearer test-token")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap()
}

async fn response_json(response: axum::response::Response) -> serde_json::Value {
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

    assert_eq!(response_json(response).await["id"], "rerank-pool");
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

    assert_eq!(response_json(response).await["id"], "rerank-override");
    assert!(pool.received_requests().await.unwrap().is_empty());
    let requests = override_server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].headers.get("authorization").is_none());
}

#[tokio::test]
async fn pool_multipart_passthrough_uses_backend_client_and_bearer() {
    let pool = MockServer::start().await;
    for case in MULTIPART_CASES {
        Mock::given(method("POST"))
            .and(path(case.pool_path))
            .and(header("authorization", "Bearer backend-secret"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id": case.response_id})),
            )
            .expect(1)
            .mount(&pool)
            .await;
    }
    let app = build_test_app(
        &pool.uri(),
        TestAppOptions {
            backend_token: Some("backend-secret".to_string()),
            ..Default::default()
        },
    );

    for case in MULTIPART_CASES {
        let response = app
            .clone()
            .oneshot(multipart_request(
                case.route,
                case.boundary,
                case.body.to_string(),
            ))
            .await
            .unwrap();
        assert_eq!(response_json(response).await["id"], case.response_id);
    }
    pool.verify().await;
}

#[tokio::test]
async fn override_multipart_passthrough_uses_plain_client_without_backend_bearer() {
    let pool = MockServer::start().await;
    let overrides = MockServer::start().await;
    for case in MULTIPART_CASES {
        Mock::given(method("POST"))
            .and(path(case.override_path))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id": case.response_id})),
            )
            .expect(1)
            .mount(&overrides)
            .await;
    }
    let app = build_test_app(
        &pool.uri(),
        TestAppOptions {
            backend_token: Some("backend-secret".to_string()),
            images_edits_url_override: Some(format!(
                "{}{}",
                overrides.uri(),
                MULTIPART_CASES[0].override_path
            )),
            transcriptions_url_override: Some(format!(
                "{}{}",
                overrides.uri(),
                MULTIPART_CASES[1].override_path
            )),
            ..Default::default()
        },
    );

    for case in MULTIPART_CASES {
        let response = app
            .clone()
            .oneshot(multipart_request(
                case.route,
                case.boundary,
                case.body.to_string(),
            ))
            .await
            .unwrap();
        assert_eq!(response_json(response).await["id"], case.response_id);
    }

    assert!(pool.received_requests().await.unwrap().is_empty());
    let requests = overrides.received_requests().await.unwrap();
    assert_eq!(requests.len(), MULTIPART_CASES.len());
    for case in MULTIPART_CASES {
        let request = requests
            .iter()
            .find(|request| request.url.path() == case.override_path)
            .expect("override route must receive its multipart request");
        assert!(request.headers.get("authorization").is_none());
    }
}

#[tokio::test]
async fn encrypted_image_edit_signature_hashes_raw_request_and_response() {
    let backend = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/images/edits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "img-hash",
            "data": [{"b64_json": "encoded-image"}]
        })))
        .expect(1)
        .mount(&backend)
        .await;
    let app = build_test_app(&backend.uri(), TestAppOptions::default());
    let signing = test_signing_pair();
    let public_key = signing.ed25519.signing_public_key.clone();
    let encryption_context = encryption::EncryptionContext {
        algo: encryption::EncryptionAlgo::Ed25519,
        client_pub_key: hex::decode(&public_key).unwrap(),
        version: 1,
        encrypt_all_fields: false,
    };
    let encrypted_prompt =
        encryption::encrypt_string("edit this", &encryption_context, &signing).unwrap();
    let boundary = "EncryptedImageBoundary";
    let body = format!(
        "--{boundary}\r\n\
         Content-Disposition: form-data; name=\"prompt\"\r\n\r\n\
         {encrypted_prompt}\r\n\
         --{boundary}\r\n\
         Content-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\n\
         Content-Type: image/png\r\n\r\n\
         fakepngdata\r\n\
         --{boundary}--\r\n"
    );
    let mut request = multipart_request("/v1/images/edits", boundary, body);
    request
        .headers_mut()
        .insert("x-signing-algo", "ed25519".parse().unwrap());
    request
        .headers_mut()
        .insert("x-client-pub-key", public_key.parse().unwrap());

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let signature = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/img-hash?signing_algo=ed25519")
                .header("authorization", "Bearer test-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let signed_text = response_json(signature).await["text"]
        .as_str()
        .unwrap()
        .to_string();
    let request_bytes = format!("{encrypted_prompt}fakepngdata");
    assert_eq!(
        signed_text,
        format!(
            "test-model:{}:{}",
            hex::encode(Sha256::digest(request_bytes)),
            hex::encode(Sha256::digest(response_bytes))
        )
    );
}
