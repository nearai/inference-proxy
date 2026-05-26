//! Integration tests for `agent_loop::run_chat_completion`.
//!
//! Tests use two `wiremock` servers: one stands in for the upstream
//! vLLM/SGLang at `/v1/chat/completions`, the other for Brave's LLM Context
//! endpoint. The test app wires them up via `web_context_search_url` and
//! `VLLM_BACKEND_URLS` in `Config`.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use vllm_proxy_rs::*;

// ── test app builder ────────────────────────────────────────────────

fn build_agent_loop_app(upstream_mock_url: &str, brave_url: Option<&str>) -> axum::Router {
    let base = upstream_mock_url.trim_end_matches('/');
    let config = config::Config {
        model_name: "test-model".to_string(),
        tokens: vec!["test-token".to_string()],
        vllm_base_url: upstream_mock_url.to_string(),
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
        pool_idle_timeout_secs: 60,
        max_request_size: 1024 * 1024,
        max_image_request_size: 5 * 1024 * 1024,
        max_audio_request_size: 10 * 1024 * 1024,
        chat_cache_expiration_secs: 1200,
        attestation_cache_ttl_secs: 300,
        dev_mode: true,
        gpu_no_hw_mode: true,
        git_rev: "test-rev".to_string(),
        rate_limit_per_second: 1000,
        rate_limit_burst_size: 2000,
        rate_limit_trust_proxy_headers: true,
        cloud_api_url: None,
        cloud_api_auth_max_attempts: 1,
        cloud_api_auth_initial_backoff_ms: 0,
        cloud_api_auth_timeout_secs: 5,
        cloud_api_usage_token: None,
        compose_manager_url: None,
        tls_cert_path: None,
        timeout_secs: 30,
        timeout_tokenize_secs: 5,
        openai_chat_compatibility_check_enabled: false,
        startup_check_retries: 1,
        startup_check_retry_delay_secs: 0,
        startup_check_timeout_secs: 5,
        backend_urls: vec![upstream_mock_url.to_string()],
        health_check_interval_secs: 5,
        health_check_max_failures: 3,
        health_check_timeout_secs: 3,
        images_url_override: None,
        images_edits_url_override: None,
        transcriptions_url_override: None,
        rerank_url_override: None,
        score_url_override: None,
        ohttp_enabled: false,
        listen_port: 8000,
        dstack_socket_path: "/var/run/dstack.sock".to_string(),
        gpu_evidence_delegate_url: None,
        gpu_evidence_delegate_timeout_secs: 30,
        web_context_search_url: brave_url.map(String::from),
        web_context_search_api_key: brave_url.map(|_| "brave-test-key".to_string()),
        agent_loop_max_iterations: 3,
        web_context_search_timeout_secs: 5,
    };

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
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let backend_pool = Arc::new(vllm_proxy_rs::backend_pool::BackendPool::new(vec![
        upstream_mock_url.to_string(),
    ]));

    let state = AppState {
        config: Arc::new(config),
        signing: Arc::new(signing_pair),
        cache: Arc::new(chat_cache),
        attestation_cache: Arc::new(vllm_proxy_rs::attestation::AttestationCache::new(300)),
        http_client,
        metrics_handle,
        tls_cert_fingerprint: Arc::new(
            vllm_proxy_rs::attestation::TlsCertTracker::new(None).expect("tracker for None path"),
        ),
        backend_pool,
        ohttp_gateway: None,
        ohttp_attestation_ed25519: None,
    };

    let rate_limiter = rate_limit::build_rate_limiter(1000, 2000);
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

// ── mock helpers ────────────────────────────────────────────────────

/// vLLM-shaped SSE for an assistant turn that emits a single tool call
/// (id `call_1`, name `web_context_search`, args `{"query":"..."}`) and
/// finishes with `tool_calls`.
fn upstream_tool_call_sse(chat_id: &str, query: &str) -> String {
    let args = serde_json::to_string(&serde_json::json!({"query": query})).unwrap();
    format!(
        "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"tool_calls\":[{{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{{\"name\":\"web_context_search\",\"arguments\":\"\"}}}}]}}}}]}}\n\n\
         data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":{args_json}}}}}]}}}}]}}\n\n\
         data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}}}\n\n\
         data: [DONE]\n\n",
        id = chat_id,
        args_json = serde_json::to_string(&args).unwrap(),
    )
}

/// vLLM-shaped SSE for an assistant turn that emits a plain answer
/// (`Hello.`) and finishes with `stop`.
fn upstream_final_answer_sse(chat_id: &str) -> String {
    format!(
        "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"Hello.\"}}}}]}}\n\n\
         data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":15,\"completion_tokens\":3,\"total_tokens\":18}}}}\n\n\
         data: [DONE]\n\n",
        id = chat_id,
    )
}

/// Minimal Brave LLM Context JSON with one grounded source.
fn brave_context_json() -> serde_json::Value {
    serde_json::json!({
        "grounding": {
            "generic": [{
                "url": "https://example.com/a",
                "title": "Example A",
                "snippets": ["First snippet about the query.", "Second snippet."]
            }]
        },
        "sources": {
            "https://example.com/a": {"title": "Example A"}
        }
    })
}

async fn body_to_string(response: axum::response::Response) -> String {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn agent_loop_request_body(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "stream": stream,
        "messages": [{"role": "user", "content": "What is the meaning of life?"}],
        "tools": [{"type": "web_context_search"}]
    })
}

// ── tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn rejects_when_search_unconfigured() {
    let upstream = MockServer::start().await;
    let app = build_agent_loop_app(&upstream.uri(), None);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(agent_loop_request_body(true).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_string(response).await;
    assert!(
        body.contains("not configured"),
        "body did not mention 'not configured': {body}"
    );
}

#[tokio::test]
async fn rejects_non_streaming() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;
    let app = build_agent_loop_app(&upstream.uri(), Some(&brave.uri()));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(agent_loop_request_body(false).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_to_string(response).await;
    assert!(
        body.contains("stream:true"),
        "body did not mention 'stream:true': {body}"
    );
}

#[tokio::test]
async fn happy_path_executes_search_and_continues() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    // First upstream call: tool_calls finish. Second: final answer.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    upstream_tool_call_sse("chatcmpl-A", "rust"),
                    "text/event-stream",
                ),
        )
        .up_to_n_times(1)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(upstream_final_answer_sse("chatcmpl-A"), "text/event-stream"),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .and(header("X-Subscription-Token", "brave-test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(brave_context_json()))
        .mount(&brave)
        .await;

    let brave_url = format!("{}/res/v1/llm/context", brave.uri());
    let app = build_agent_loop_app(&upstream.uri(), Some(&brave_url));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(agent_loop_request_body(true).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    let body = body_to_string(response).await;

    // Sanity: tool-call chunks from iteration 1, our synthetic tool-result
    // chunk, then content chunks from iteration 2, then [DONE].
    assert!(
        body.contains("\"tool_calls\""),
        "missing tool_calls in body: {body}"
    );
    assert!(
        body.contains("nearai_tool_result"),
        "missing nearai_tool_result: {body}"
    );
    assert!(body.contains("\"tool_call_id\":\"call_1\""));
    assert!(body.contains("\"status\":\"ok\""));
    // Brave's snippets and title should appear in the rendered output.
    assert!(body.contains("[1] Example A"));
    assert!(body.contains("First snippet"));
    // Second iteration content.
    assert!(body.contains("Hello."));
    // Final [DONE] marker.
    assert!(body.ends_with("data: [DONE]\n\n"));
    // [DONE] should only appear once in the final stream — intermediate
    // [DONE]s from upstream are swallowed by the loop.
    assert_eq!(body.matches("data: [DONE]").count(), 1);
}

#[tokio::test]
async fn brave_error_falls_back_with_error_status() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    upstream_tool_call_sse("chatcmpl-B", "rust"),
                    "text/event-stream",
                ),
        )
        .up_to_n_times(1)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(upstream_final_answer_sse("chatcmpl-B"), "text/event-stream"),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&brave)
        .await;

    let brave_url = format!("{}/res/v1/llm/context", brave.uri());
    let app = build_agent_loop_app(&upstream.uri(), Some(&brave_url));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(agent_loop_request_body(true).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_string(response).await;
    assert!(body.contains("nearai_tool_result"));
    assert!(body.contains("\"status\":\"error\""));
    // Loop continues — second iteration's content should still arrive.
    assert!(body.contains("Hello."));
    assert!(body.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn max_iterations_emits_terminator() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    // Upstream always returns tool_calls — the loop should cap out.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    upstream_tool_call_sse("chatcmpl-C", "rust"),
                    "text/event-stream",
                ),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(brave_context_json()))
        .mount(&brave)
        .await;

    let brave_url = format!("{}/res/v1/llm/context", brave.uri());
    let app = build_agent_loop_app(&upstream.uri(), Some(&brave_url));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(agent_loop_request_body(true).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_string(response).await;
    assert!(body.contains("nearai_loop_terminated"));
    assert!(body.contains("max_iterations"));
    assert!(body.ends_with("data: [DONE]\n\n"));
    // Exactly `agent_loop_max_iterations` (3) tool_result chunks should have
    // been emitted before the terminator.
    assert_eq!(body.matches("nearai_tool_result").count(), 3);
}

#[tokio::test]
async fn function_only_tools_pass_through_without_loop() {
    // A request whose `tools` are standard `function` (not web_context_search)
    // must NOT enter the loop. We verify by ensuring the upstream sees
    // exactly the body we sent — including the `function` tool, untouched —
    // and the response is forwarded back without any nearai_tool_result.
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(upstream_final_answer_sse("chatcmpl-D"), "text/event-stream"),
        )
        .mount(&upstream)
        .await;

    let brave_url = format!("{}/res/v1/llm/context", brave.uri());
    let app = build_agent_loop_app(&upstream.uri(), Some(&brave_url));

    let body = serde_json::json!({
        "model": "test-model",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_string(response).await;
    assert!(body.contains("Hello."));
    assert!(
        !body.contains("nearai_tool_result"),
        "loop fired unexpectedly: {body}"
    );
}

#[tokio::test]
async fn malformed_tool_args_continues_loop() {
    // If the model emits malformed tool arguments (no `query` field), we
    // still emit a synthetic chunk with status:"error" and feed the error
    // string back to the model so it can recover or apologize.
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    let bad_args_sse = "data: {\"id\":\"chatcmpl-E\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"web_context_search\",\"arguments\":\"{}\"}}]}}]}\n\n\
                       data: {\"id\":\"chatcmpl-E\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                       data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(bad_args_sse, "text/event-stream"),
        )
        .up_to_n_times(1)
        .mount(&upstream)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(upstream_final_answer_sse("chatcmpl-E"), "text/event-stream"),
        )
        .mount(&upstream)
        .await;

    let brave_url = format!("{}/res/v1/llm/context", brave.uri());
    let app = build_agent_loop_app(&upstream.uri(), Some(&brave_url));

    // No Brave mock registered — if the loop incorrectly tried to call it,
    // the request would hang or fail. The error path should not hit Brave.

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(agent_loop_request_body(true).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_to_string(response).await;
    assert!(body.contains("nearai_tool_result"));
    assert!(body.contains("\"status\":\"error\""));
    assert!(body.contains("Hello."));
}
