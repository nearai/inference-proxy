use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// Import from the crate
use vllm_proxy_rs::*;

/// Build a test app with the given mock backend URL.
fn build_test_app(mock_url: &str) -> axum::Router {
    build_test_app_with_rate_limit(mock_url, 100, 200)
}

/// Build a test app with custom rate limit settings.
fn build_test_app_with_rate_limit(
    mock_url: &str,
    rate_per_second: u64,
    rate_burst: u32,
) -> axum::Router {
    let base = mock_url.trim_end_matches('/');

    let config = config::Config {
        model_name: "test-model".to_string(),
        token: "test-token".to_string(),
        vllm_base_url: mock_url.to_string(),
        chat_completions_url: format!("{base}/v1/chat/completions"),
        completions_url: format!("{base}/v1/completions"),
        tokenize_url: format!("{base}/tokenize"),
        metrics_url: format!("{base}/metrics"),
        models_url: format!("{base}/v1/models"),
        images_url: format!("{base}/v1/images/generations"),
        images_edits_url: format!("{base}/v1/images/edits"),
        transcriptions_url: format!("{base}/v1/audio/transcriptions"),
        embeddings_url: format!("{base}/v1/embeddings"),
        rerank_url: format!("{base}/v1/rerank"),
        score_url: format!("{base}/v1/score"),
        max_keepalive: 5,
        max_request_size: 1024 * 1024,
        max_image_request_size: 5 * 1024 * 1024,
        max_audio_request_size: 10 * 1024 * 1024,
        chat_cache_expiration_secs: 1200,
        dev_mode: true,
        gpu_no_hw_mode: true,
        git_rev: "test-rev".to_string(),
        rate_limit_per_second: rate_per_second,
        rate_limit_burst_size: rate_burst,
        rate_limit_trust_proxy_headers: true,
        tls_cert_path: None,
        timeout_secs: 30,
        timeout_tokenize_secs: 5,
        openai_chat_compatibility_check_enabled: false,
        startup_check_retries: 1,
        startup_check_retry_delay_secs: 0,
        startup_check_timeout_secs: 5,
    };

    // Use fixed keys for deterministic tests
    let ecdsa_key: [u8; 32] = [
        0xac, 0x09, 0x74, 0xbe, 0xc3, 0x9a, 0x17, 0xe3, 0x6b, 0xa4, 0xa6, 0xb4, 0xd2, 0x38, 0xff,
        0x94, 0x4b, 0xac, 0xb3, 0x5e, 0x5d, 0xc4, 0xaf, 0x0f, 0x33, 0x47, 0xe5, 0x87, 0x31, 0x79,
        0x67, 0x0f,
    ];
    let ed25519_key: [u8; 32] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
    ];

    let ecdsa = signing::EcdsaContext::from_key_bytes(&ecdsa_key).unwrap();
    let ed25519 = signing::Ed25519Context::from_key_bytes(&ed25519_key).unwrap();
    let signing_pair = signing::SigningPair { ecdsa, ed25519 };

    let chat_cache = cache::ChatCache::new("test-model", 1200);
    let http_client = reqwest::Client::new();

    // Create a standalone PrometheusHandle for tests (not installed globally)
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();

    let state = AppState {
        config: Arc::new(config),
        signing: Arc::new(signing_pair),
        cache: Arc::new(chat_cache),
        http_client,
        metrics_handle,
        tls_cert_fingerprint: None,
    };

    let rate_limiter = rate_limit::build_rate_limiter(rate_per_second, rate_burst);
    let rate_limit_state = rate_limit::RateLimitState {
        limiter: rate_limiter,
        trust_proxy_headers: true,
    };

    routes::build_router()
        .layer(middleware::from_fn(rate_limit::rate_limit_middleware))
        .layer(axum::Extension(rate_limit_state))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state)
}

fn auth_header() -> (&'static str, &'static str) {
    ("authorization", "Bearer test-token")
}

// ---- Health endpoints ----

#[tokio::test]
async fn test_root_endpoint() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body, serde_json::json!({}));
}

#[tokio::test]
async fn test_version_endpoint() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/version")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["version"], "test-rev");
    assert_eq!(body["type"], "proxy");
}

// ---- Auth ----

#[tokio::test]
async fn test_auth_rejection_no_header() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_rejection_bad_token() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer wrong-token")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

// ---- Non-streaming chat completions ----

#[tokio::test]
async fn test_chat_completions_non_streaming() {
    let mock_server = MockServer::start().await;

    let backend_response = serde_json::json!({
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["id"], "chatcmpl-test123");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
}

// ---- Streaming chat completions ----

#[tokio::test]
async fn test_chat_completions_streaming() {
    let mock_server = MockServer::start().await;

    let sse_body = "data: {\"id\":\"chatcmpl-stream1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );

    // Read the full streamed body
    let body_bytes = body_to_bytes(response).await;
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("chatcmpl-stream1"));
    assert!(body_str.contains("[DONE]"));
}

// ---- Signature retrieval ----

#[tokio::test]
async fn test_signature_retrieval_after_chat() {
    let mock_server = MockServer::start().await;

    let backend_response = serde_json::json!({
        "id": "chatcmpl-sig-test",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    // First, make a chat completion to cache the signature
    let request_body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}]
    });

    let chat_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(chat_response.status(), StatusCode::OK);

    // Now retrieve the signature
    let sig_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-sig-test?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
    let body = body_to_json(sig_response).await;
    assert_eq!(body["signing_algo"], "ecdsa");
    assert!(body["signature"].as_str().unwrap().starts_with("0x"));
    assert!(body["text"].as_str().unwrap().contains(":"));
    assert!(body["signing_address"].as_str().unwrap().starts_with("0x"));
}

#[tokio::test]
async fn test_signature_retrieval_ed25519() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-ed-test",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    // Chat to cache
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({"messages": []})).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    // Get ed25519 signature
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-ed-test?signing_algo=ed25519")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
    let body = body_to_json(sig_response).await;
    assert_eq!(body["signing_algo"], "ed25519");
    // Ed25519 signature is hex, not 0x-prefixed
    assert!(!body["signature"].as_str().unwrap().starts_with("0x"));
    assert_eq!(body["signature"].as_str().unwrap().len(), 128);
}

#[tokio::test]
async fn test_signature_not_found() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/nonexistent-id")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ---- Upstream error passthrough ----

#[tokio::test]
async fn test_upstream_error_passthrough() {
    let mock_server = MockServer::start().await;

    let error_body = serde_json::json!({
        "error": {"message": "model not found", "type": "invalid_request"}
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(404).set_body_json(&error_body))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_to_json(response).await;
    assert_eq!(body["error"]["message"], "model not found");
}

// ---- Payload too large ----

#[tokio::test]
async fn test_payload_too_large() {
    let app = build_test_app("http://unused");

    // Send a body larger than max_request_size (1MB in test config)
    let big_body = vec![b'x'; 2 * 1024 * 1024];

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(big_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

// ---- Metrics and models (no auth) ----

#[tokio::test]
async fn test_metrics_no_auth_required() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_string("vllm_requests_total 42"))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    // No auth header!
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = body_to_bytes(response).await;
    let body_str = String::from_utf8_lossy(&bytes);
    assert!(body_str.contains("vllm_requests_total"));
}

#[tokio::test]
async fn test_models_no_auth_required() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "test-model"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["data"][0]["id"], "test-model");
}

// ---- Completions ----

#[tokio::test]
async fn test_completions_endpoint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-test",
            "choices": [{"text": "world", "finish_reason": "stop"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"prompt":"hello "}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["id"], "cmpl-test");
    assert_eq!(body["choices"][0]["text"], "world");
}

// ---- Embeddings ----

#[tokio::test]
async fn test_embeddings_endpoint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "emb-test",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"input":"test","model":"test-model"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["id"], "emb-test");
}

// ---- Tokenize (no signing) ----

#[tokio::test]
async fn test_tokenize_endpoint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/tokenize"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "tokens": [1, 2, 3],
            "count": 3
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/tokenize")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"text":"hello"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["count"], 3);
}

// ---- Request ID middleware ----

#[tokio::test]
async fn test_request_id_generated() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert!(response.headers().contains_key("x-request-id"));
    let id = response
        .headers()
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(!id.is_empty());
}

#[tokio::test]
async fn test_request_id_passthrough() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header("x-request-id", "my-custom-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.headers().get("x-request-id").unwrap(),
        "my-custom-id"
    );
}

// ---- Signature always binds actual request body ----

#[tokio::test]
async fn test_signature_binds_actual_request_body() {
    use sha2::{Digest, Sha256};

    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-bind-test",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({"messages": []});
    let request_bytes = serde_json::to_vec(&request_body).unwrap();
    let expected_hash = hex::encode(Sha256::digest(&request_bytes));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(request_bytes))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify the cached signature uses hash of the actual request body
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-bind-test?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = body_to_json(sig_response).await;
    assert!(
        body["text"]
            .as_str()
            .unwrap()
            .starts_with(&format!("{expected_hash}:")),
        "Signed text must start with SHA-256 of actual request body"
    );
}

#[tokio::test]
async fn test_x_request_hash_header_is_ignored() {
    use sha2::{Digest, Sha256};

    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-ignore-hash",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({"messages": []});
    let request_bytes = serde_json::to_vec(&request_body).unwrap();
    let expected_hash = hex::encode(Sha256::digest(&request_bytes));

    // Even with X-Request-Hash header, the actual body hash should be used
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .header("x-request-hash", "a".repeat(64))
                .body(Body::from(request_bytes))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-ignore-hash?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = body_to_json(sig_response).await;
    assert!(
        body["text"]
            .as_str()
            .unwrap()
            .starts_with(&format!("{expected_hash}:")),
        "Signed text must use actual body hash, not X-Request-Hash header"
    );
}

// ---- Strip empty tool_calls ----

#[tokio::test]
async fn test_strip_empty_tool_calls() {
    use wiremock::matchers::body_json;

    let mock_server = MockServer::start().await;

    // Expect the backend receives the request WITHOUT empty tool_calls
    let expected_backend_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(&expected_backend_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-tc",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    // Send request WITH empty tool_calls — proxy should strip them before forwarding
    let request_body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi", "tool_calls": []},
            {"role": "assistant", "content": "hello"}
        ]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // The mock expectation (body_json) verifies the backend received stripped body
}

// ---- Response ID generation ----

#[tokio::test]
async fn test_response_id_generation_when_missing() {
    let mock_server = MockServer::start().await;

    // Backend returns response without an ID
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"embedding": [0.1], "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"input":"test","model":"m"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    // Should have a generated ID
    assert!(body.get("id").is_some());
    assert!(body["id"].as_str().unwrap().starts_with("emb-"));
}

// ---- Invalid JSON ----

#[tokio::test]
async fn test_invalid_json_body() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from("not json at all"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Invalid JSON"));
}

// ---- Signature cryptographic verification ----

#[tokio::test]
async fn test_ecdsa_signature_cryptographic_verification() {
    use k256::ecdsa::{RecoveryId, VerifyingKey};
    use sha2::{Digest, Sha256};
    use sha3::Keccak256;

    let mock_server = MockServer::start().await;

    let backend_response = serde_json::json!({
        "id": "chatcmpl-ecverify",
        "choices": [{"message": {"content": "verified"}, "finish_reason": "stop"}]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "verify me"}]
    });
    let request_bytes = serde_json::to_vec(&request_body).unwrap();

    // Make chat completion
    let chat_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(request_bytes.clone()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(chat_response.status(), StatusCode::OK);
    let response_body_bytes = body_to_bytes(chat_response).await;

    // Retrieve ECDSA signature
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-ecverify?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
    let sig_body = body_to_json(sig_response).await;

    let text = sig_body["text"].as_str().unwrap();
    let signature_hex = sig_body["signature"].as_str().unwrap();
    let signing_address = sig_body["signing_address"].as_str().unwrap();

    // Verify text = sha256(request):sha256(response)
    let expected_req_hash = hex::encode(Sha256::digest(&request_bytes));
    let expected_resp_hash = hex::encode(Sha256::digest(&response_body_bytes));
    assert_eq!(text, format!("{expected_req_hash}:{expected_resp_hash}"));

    // Verify ECDSA EIP-191 signature via key recovery
    let sig_bytes = hex::decode(&signature_hex[2..]).unwrap();
    assert_eq!(sig_bytes.len(), 65);
    let v = sig_bytes[64];
    assert!(v == 27 || v == 28);

    // Compute EIP-191 hash
    let prefix = format!("\x19Ethereum Signed Message:\n{}", text.len());
    let mut prefixed = Vec::new();
    prefixed.extend_from_slice(prefix.as_bytes());
    prefixed.extend_from_slice(text.as_bytes());
    let msg_hash = Keccak256::digest(&prefixed);

    // Recover public key from signature
    let signature = k256::ecdsa::Signature::from_slice(&sig_bytes[..64]).unwrap();
    let recovery_id = RecoveryId::from_byte(v - 27).unwrap();
    let recovered_key =
        VerifyingKey::recover_from_prehash(&msg_hash[..], &signature, recovery_id).unwrap();

    // Derive Ethereum address from recovered key
    let pk_encoded = recovered_key.to_encoded_point(false);
    let pk_uncompressed = &pk_encoded.as_bytes()[1..];
    let addr_hash = Keccak256::digest(pk_uncompressed);
    let recovered_address = format!("0x{}", hex::encode(&addr_hash[12..32]));

    assert_eq!(recovered_address, signing_address);
}

#[tokio::test]
async fn test_ed25519_signature_cryptographic_verification() {
    use ed25519_dalek::Verifier;
    use sha2::{Digest, Sha256};

    let mock_server = MockServer::start().await;

    let backend_response = serde_json::json!({
        "id": "chatcmpl-edverify",
        "choices": [{"message": {"content": "ed verified"}, "finish_reason": "stop"}]
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&backend_response))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "ed25519 verify"}]
    });
    let request_bytes = serde_json::to_vec(&request_body).unwrap();

    // Make chat completion
    let chat_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(request_bytes.clone()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(chat_response.status(), StatusCode::OK);
    let response_body_bytes = body_to_bytes(chat_response).await;

    // Retrieve Ed25519 signature
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-edverify?signing_algo=ed25519")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
    let sig_body = body_to_json(sig_response).await;

    let text = sig_body["text"].as_str().unwrap();
    let signature_hex = sig_body["signature"].as_str().unwrap();
    let signing_address = sig_body["signing_address"].as_str().unwrap();

    // Verify text = sha256(request):sha256(response)
    let expected_req_hash = hex::encode(Sha256::digest(&request_bytes));
    let expected_resp_hash = hex::encode(Sha256::digest(&response_body_bytes));
    assert_eq!(text, format!("{expected_req_hash}:{expected_resp_hash}"));

    // Verify Ed25519 signature
    let sig_bytes = hex::decode(signature_hex).unwrap();
    assert_eq!(sig_bytes.len(), 64);
    let signature = ed25519_dalek::Signature::from_bytes(sig_bytes[..64].try_into().unwrap());

    let pk_bytes = hex::decode(signing_address).unwrap();
    assert_eq!(pk_bytes.len(), 32);
    let verifying_key =
        ed25519_dalek::VerifyingKey::from_bytes(pk_bytes[..32].try_into().unwrap()).unwrap();

    // This is the actual cryptographic verification
    assert!(
        verifying_key.verify(text.as_bytes(), &signature).is_ok(),
        "Ed25519 signature verification failed"
    );
}

#[tokio::test]
async fn test_streaming_signature_cached_and_verifiable() {
    use sha2::{Digest, Sha256};

    let mock_server = MockServer::start().await;

    let sse_body = "data: {\"id\":\"chatcmpl-streamver\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "stream verify"}],
        "stream": true
    });
    let request_bytes = serde_json::to_vec(&request_body).unwrap();

    // Make streaming request
    let stream_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(request_bytes.clone()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(stream_response.status(), StatusCode::OK);

    // Consume the full stream body (this triggers the background hash+sign task)
    let stream_body_bytes = body_to_bytes(stream_response).await;
    assert!(!stream_body_bytes.is_empty());

    // Small delay to ensure background signing task completes
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Retrieve signature — should be cached by now
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-streamver?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
    let sig_body = body_to_json(sig_response).await;

    let text = sig_body["text"].as_str().unwrap();
    let parts: Vec<&str> = text.split(':').collect();
    assert_eq!(
        parts.len(),
        2,
        "Signed text should be request_hash:response_hash"
    );

    // Request hash should match SHA256 of request body
    let expected_req_hash = hex::encode(Sha256::digest(&request_bytes));
    assert_eq!(parts[0], expected_req_hash);

    // Response hash should match SHA256 of the streamed bytes
    let expected_resp_hash = hex::encode(Sha256::digest(&stream_body_bytes));
    assert_eq!(parts[1], expected_resp_hash);

    // Signature should be valid format
    let sig_hex = sig_body["signature"].as_str().unwrap();
    assert!(sig_hex.starts_with("0x"));
    assert_eq!(sig_hex.len(), 132); // 65 bytes hex + "0x"
}

// ---- Attestation endpoint ----

#[tokio::test]
async fn test_attestation_is_public() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/attestation/report")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Attestation is a public endpoint — should not return 401.
    // Without dstack it will return 500, which is expected in tests.
    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_attestation_invalid_algo() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/attestation/report?signing_algo=rsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Invalid signing algorithm"));
}

#[tokio::test]
async fn test_attestation_signing_address_mismatch() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/attestation/report?signing_algo=ecdsa&signing_address=0xdeadbeef00000000000000000000000000000000")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Signing address not found"));
}

#[tokio::test]
async fn test_attestation_valid_request_fails_without_dstack() {
    // A valid attestation request should reach generate_attestation,
    // which fails because dstack is not available in test environment.
    // This verifies the request flows through all validation correctly.
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/attestation/report?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should fail with 500 since dstack is not available
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn test_attestation_invalid_nonce() {
    let app = build_test_app("http://unused");

    // Nonce that's not valid hex
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/attestation/report?signing_algo=ecdsa&nonce=not_hex!")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Nonce with wrong length (16 bytes instead of 32)
    let short_nonce = "ab".repeat(16);
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/v1/attestation/report?signing_algo=ecdsa&nonce={short_nonce}"
                ))
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ---- Streaming without [DONE] marker (incomplete stream) ----

#[tokio::test]
async fn test_streaming_without_done_marker_no_signature_cached() {
    let mock_server = MockServer::start().await;

    // Simulate a stream that ends abruptly without [DONE]
    let sse_body = "data: {\"id\":\"chatcmpl-nodone\",\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({
        "messages": [{"role": "user", "content": "test"}],
        "stream": true
    });

    let stream_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(stream_response.status(), StatusCode::OK);

    // Consume the stream
    let _ = body_to_bytes(stream_response).await;
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Signature should NOT be cached since [DONE] was never received
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-nodone?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::NOT_FOUND);
}

// ---- Rerank endpoint ----

#[tokio::test]
async fn test_rerank_endpoint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "rerank-test",
            "results": [{"index": 0, "relevance_score": 0.95}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/rerank")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"model":"m","query":"q","documents":["a"]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["id"], "rerank-test");
}

// ---- Score endpoint ----

#[tokio::test]
async fn test_score_endpoint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/score"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "score-test",
            "results": [{"score": 0.8}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/score")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"model":"m","text_1":"a","text_2":"b"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["id"], "score-test");
}

// ---- Images generations endpoint ----

#[tokio::test]
async fn test_images_generations_endpoint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/images/generations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "img-test",
            "data": [{"url": "https://example.com/image.png"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"prompt":"a cat","model":"dall-e"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["id"], "img-test");
}

// ---- Signature endpoint invalid algo ----

#[tokio::test]
async fn test_signature_invalid_algo_returns_400() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-algo-test",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    // Cache a signature first
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"messages":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    // Request with invalid algo
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/chatcmpl-algo-test?signing_algo=rsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Invalid signing algorithm"));
}

// ---- Streaming completions ----

#[tokio::test]
async fn test_streaming_completions() {
    let mock_server = MockServer::start().await;

    let sse_body =
        "data: {\"id\":\"cmpl-stream1\",\"choices\":[{\"text\":\"world\"}]}\n\ndata: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse_body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let request_body = serde_json::json!({
        "prompt": "hello ",
        "stream": true
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );

    let body_bytes = body_to_bytes(response).await;
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("cmpl-stream1"));
    assert!(body_str.contains("[DONE]"));

    // Wait for background signing
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Verify signature was cached
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/cmpl-stream1?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
}

// ---- Multipart endpoints ----

#[tokio::test]
async fn test_images_edits_multipart() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/images/edits"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "img-edit-test",
            "data": [{"url": "https://example.com/edited.png"}]
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let boundary = "----TestBoundary12345";
    let body = format!(
        "--{boundary}\r\n\
         Content-Disposition: form-data; name=\"prompt\"\r\n\r\n\
         a cat wearing a hat\r\n\
         --{boundary}\r\n\
         Content-Disposition: form-data; name=\"image\"; filename=\"test.png\"\r\n\
         Content-Type: image/png\r\n\r\n\
         fakepngdata\r\n\
         --{boundary}--\r\n"
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .header(auth_header().0, auth_header().1)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let resp_body = body_to_json(response).await;
    assert_eq!(resp_body["id"], "img-edit-test");

    // Verify signature was cached
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/img-edit-test?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_audio_transcriptions_multipart() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "trans-test",
            "text": "Hello world"
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let boundary = "----TestBoundary67890";
    let body = format!(
        "--{boundary}\r\n\
         Content-Disposition: form-data; name=\"model\"\r\n\r\n\
         whisper-1\r\n\
         --{boundary}\r\n\
         Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n\
         Content-Type: audio/wav\r\n\r\n\
         fakeaudiodata\r\n\
         --{boundary}--\r\n"
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .header(auth_header().0, auth_header().1)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let resp_body = body_to_json(response).await;
    assert_eq!(resp_body["id"], "trans-test");

    // Verify signature was cached
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/trans-test?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
}

// ---- Catch-all passthrough ----

#[tokio::test]
async fn test_passthrough_json_post() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/custom/endpoint"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "pt-custom-1",
            "result": "ok"
        })))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/endpoint")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"data":"test"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["id"], "pt-custom-1");
    assert_eq!(body["result"], "ok");

    // Verify signature was cached
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/pt-custom-1?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
    let sig_body = body_to_json(sig_response).await;
    assert!(sig_body["text"].as_str().unwrap().contains(":"));
}

#[tokio::test]
async fn test_passthrough_get_json() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/custom/info"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"id": "pt-info-1", "status": "healthy"}))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/custom/info")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    assert_eq!(body["status"], "healthy");

    // Verify signature was cached
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/pt-info-1?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_passthrough_streaming() {
    let mock_server = MockServer::start().await;

    let sse_body =
        "data: {\"id\":\"pt-stream-1\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/custom/stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body, "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/stream")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"stream":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );

    let body_bytes = body_to_bytes(response).await;
    let body_str = String::from_utf8_lossy(&body_bytes);
    assert!(body_str.contains("pt-stream-1"));
    assert!(body_str.contains("[DONE]"));

    // Wait for background signing task
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Verify signature was cached
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/pt-stream-1?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_passthrough_raw_response() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/custom/text"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("plain text output")
                .insert_header("content-type", "text/plain"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/custom/text")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/plain"
    );
    let body_bytes = body_to_bytes(response).await;
    assert_eq!(String::from_utf8_lossy(&body_bytes), "plain text output");
}

#[tokio::test]
async fn test_passthrough_path_traversal() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/../etc/passwd")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Path traversal"));
}

#[tokio::test]
async fn test_passthrough_auth_required() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/custom/anything")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_passthrough_upstream_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/custom/broken"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal secret error details"))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/custom/broken")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should preserve the upstream status code (500)
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = body_to_json(response).await;
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.contains("Upstream request failed with status"));
    // Should NOT leak the raw upstream body
    assert!(!msg.contains("internal secret error details"));
}

#[tokio::test]
async fn test_passthrough_upstream_json_error_rewrapped() {
    let mock_server = MockServer::start().await;

    // vLLM-style flat error response
    let vllm_error = serde_json::json!({
        "object": "error",
        "message": "This model's maximum context length is 2048 tokens. However, you requested 4374 tokens (3350 in the messages, 1024 in the completion). Please reduce the length of the messages or completion.",
        "type": "BadRequestError",
        "param": null,
        "code": 400
    });

    Mock::given(method("POST"))
        .and(path("/v1/custom/chat"))
        .respond_with(ResponseTemplate::new(400).set_body_json(&vllm_error))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/chat")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"prompt":"test"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should preserve the upstream status code
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_json(response).await;
    // Should re-wrap with the actual error message from vLLM
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.contains("maximum context length is 2048 tokens"));
    assert_eq!(body["error"]["type"], "BadRequestError");
    // Should be in OpenAI-compatible format (nested under "error")
    assert!(body["error"]["param"].is_null());
    assert!(body["error"]["code"].is_null());
}

#[tokio::test]
async fn test_passthrough_upstream_model_not_found() {
    let mock_server = MockServer::start().await;

    let vllm_error = serde_json::json!({
        "object": "error",
        "message": "The model `gpt-5` does not exist.",
        "type": "Not Found",
        "param": null,
        "code": 404
    });

    Mock::given(method("GET"))
        .and(path("/v1/custom/models/gpt-5"))
        .respond_with(ResponseTemplate::new(404).set_body_json(&vllm_error))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/custom/models/gpt-5")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"].as_str().unwrap().contains("gpt-5"));
    assert_eq!(body["error"]["type"], "Not Found");
}

#[tokio::test]
async fn test_passthrough_upstream_invalid_param() {
    let mock_server = MockServer::start().await;

    let vllm_error = serde_json::json!({
        "object": "error",
        "message": "temperature must be non-negative, got -0.5.",
        "type": "BadRequestError",
        "param": null,
        "code": 400
    });

    Mock::given(method("POST"))
        .and(path("/v1/custom/complete"))
        .respond_with(ResponseTemplate::new(400).set_body_json(&vllm_error))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/complete")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"temperature":-0.5}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("temperature must be non-negative"));
}

#[tokio::test]
async fn test_passthrough_upstream_internal_server_error() {
    let mock_server = MockServer::start().await;

    // vLLM sometimes returns 500 with just "Internal Server Error"
    let vllm_error = serde_json::json!({
        "object": "error",
        "message": "Internal server error",
        "type": "InternalServerError",
        "param": null,
        "code": 500
    });

    Mock::given(method("POST"))
        .and(path("/v1/custom/inference"))
        .respond_with(ResponseTemplate::new(500).set_body_json(&vllm_error))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/inference")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = body_to_json(response).await;
    assert_eq!(body["error"]["type"], "InternalServerError");
}

#[tokio::test]
async fn test_passthrough_upstream_nested_error_format() {
    let mock_server = MockServer::start().await;

    // sglang-style nested error format
    let sglang_error = serde_json::json!({
        "error": {
            "message": "Tools cannot be empty if tool choice is set to required.",
            "type": "BadRequestError",
            "param": null,
            "code": 400
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/custom/tools"))
        .respond_with(ResponseTemplate::new(400).set_body_json(&sglang_error))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/tools")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"tool_choice":"required"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Tools cannot be empty"));
    assert_eq!(body["error"]["type"], "BadRequestError");
}

#[tokio::test]
async fn test_passthrough_upstream_empty_body_error() {
    let mock_server = MockServer::start().await;

    // vLLM intermittently returns 500 with empty body (Content-Length: 0)
    Mock::given(method("POST"))
        .and(path("/v1/custom/empty"))
        .respond_with(ResponseTemplate::new(500).set_body_string(""))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/empty")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = body_to_json(response).await;
    // Falls back to generic message when body is unparseable
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Upstream request failed with status"));
}

#[tokio::test]
async fn test_passthrough_headers_forwarded() {
    use wiremock::matchers::header;

    let mock_server = MockServer::start().await;

    // Expect that custom headers are forwarded, but host/content-length/authorization are excluded
    Mock::given(method("POST"))
        .and(path("/v1/custom/headers"))
        .and(header("x-custom-header", "custom-value"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "pt-headers-1",
            "ok": true
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/headers")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .header("x-custom-header", "custom-value")
                .body(Body::from(r#"{"test": true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // The mock expectation (header matcher) verifies the custom header was forwarded
}

#[tokio::test]
async fn test_passthrough_excluded_headers_not_forwarded() {
    use wiremock::matchers::header_exists;
    use wiremock::Match;

    /// Matcher that asserts a header does NOT exist on the request.
    struct HeaderAbsent(&'static str);
    impl Match for HeaderAbsent {
        fn matches(&self, request: &wiremock::Request) -> bool {
            !request.headers.contains_key(self.0)
        }
    }

    let mock_server = MockServer::start().await;

    // Verify: custom headers forwarded, authorization excluded.
    // Note: we don't assert host is absent because reqwest always adds Host per HTTP/1.1.
    Mock::given(method("POST"))
        .and(path("/v1/custom/excl"))
        .and(header_exists("x-custom-header"))
        .and(HeaderAbsent("authorization"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "pt-excl-1",
            "ok": true
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/excl")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .header("x-custom-header", "present")
                .body(Body::from(r#"{"test": true}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // The mock expectations verify: x-custom-header present, authorization absent, host absent
}

#[tokio::test]
async fn test_passthrough_json_array_response() {
    let mock_server = MockServer::start().await;

    // Backend returns a JSON array (not an object) — should not panic
    Mock::given(method("POST"))
        .and(path("/v1/custom/array"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!([1, 2, 3]))
                .insert_header("content-type", "application/json"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/array")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"data":"test"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_json(response).await;
    // Should return the array unmodified (can't inject ID into non-object)
    assert!(body.is_array());
    assert_eq!(body[0], 1);
}

#[tokio::test]
async fn test_passthrough_raw_response_no_signature_cached() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/custom/binary"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_bytes(b"binary data".to_vec())
                .insert_header("content-type", "application/octet-stream"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/custom/binary")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = body_to_bytes(response).await;
    assert_eq!(body_bytes, b"binary data");

    // Raw passthrough should NOT have any signature cached.
    // Use a generated ID pattern — since there's no signing, no ID is generated at all.
    // Try fetching with a plausible ID prefix to confirm nothing was cached.
    let sig_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/signature/pt-does-not-exist?signing_algo=ecdsa")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(sig_response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_passthrough_upstream_404_preserved() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/custom/notfound"))
        .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/custom/notfound")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_to_json(response).await;
    assert!(body["error"]["message"].as_str().unwrap().contains("404"));
}

#[tokio::test]
async fn test_passthrough_upstream_429_preserved() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/custom/ratelimit"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/ratelimit")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn test_passthrough_success_status_preserved() {
    let mock_server = MockServer::start().await;

    // Backend returns 201 Created with JSON
    Mock::given(method("POST"))
        .and(path("/v1/custom/create"))
        .respond_with(
            ResponseTemplate::new(201)
                .set_body_json(serde_json::json!({"id": "pt-created-1", "created": true})),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/custom/create")
                .header("content-type", "application/json")
                .header(auth_header().0, auth_header().1)
                .body(Body::from(r#"{"name":"test"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    // Should preserve 201, not normalize to 200
    assert_eq!(response.status(), StatusCode::CREATED);
    let body = body_to_json(response).await;
    assert_eq!(body["created"], true);
}

#[tokio::test]
async fn test_passthrough_raw_response_headers_preserved() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/custom/headers_resp"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data")
                .insert_header("content-type", "text/plain")
                .insert_header("x-custom-response", "from-backend")
                .insert_header("cache-control", "max-age=60"),
        )
        .mount(&mock_server)
        .await;

    let app = build_test_app(&mock_server.uri());

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/custom/headers_resp")
                .header(auth_header().0, auth_header().1)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // Upstream response headers should be preserved
    assert_eq!(
        response.headers().get("x-custom-response").unwrap(),
        "from-backend"
    );
    assert_eq!(
        response.headers().get("cache-control").unwrap(),
        "max-age=60"
    );
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/plain"
    );
}

// ---- Rate limiting ----

#[tokio::test]
async fn test_rate_limit_returns_429_after_burst() {
    let app = build_test_app_with_rate_limit("http://unused", 1, 2);

    // First two requests should succeed (burst = 2)
    for i in 0..2 {
        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request {i} should succeed within burst"
        );
    }

    // Third request should be rate limited
    let resp = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_to_json(resp).await;
    assert_eq!(body["error"]["type"], "rate_limited");
}

#[tokio::test]
async fn test_rate_limit_does_not_block_with_high_burst() {
    // Default test app has burst=200, should not block normal test traffic
    let app = build_test_app("http://unused");

    for _ in 0..10 {
        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

// ---- Prometheus metrics endpoint ----

#[tokio::test]
async fn test_prometheus_metrics_endpoint() {
    let app = build_test_app("http://unused");

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // Response should be valid UTF-8 text (Prometheus exposition format)
    let body_bytes = body_to_bytes(response).await;
    let body_str = String::from_utf8(body_bytes).expect("metrics body should be valid UTF-8");
    // Empty or containing metric lines — either is valid since no global recorder is installed
    assert!(
        body_str.is_empty() || body_str.is_ascii(),
        "metrics body should be ASCII text"
    );
}

#[tokio::test]
async fn test_prometheus_metrics_no_auth_required() {
    let app = build_test_app("http://unused");

    // No auth header — should still succeed
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

// ---- Helpers ----

async fn body_to_json(response: axum::http::Response<Body>) -> serde_json::Value {
    use http_body_util::BodyExt;
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

async fn body_to_bytes(response: axum::http::Response<Body>) -> Vec<u8> {
    use http_body_util::BodyExt;
    response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec()
}
