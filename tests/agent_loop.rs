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
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use vllm_proxy_rs::*;

// ── test app builder ────────────────────────────────────────────────

fn build_agent_loop_app(upstream_mock_url: &str, brave_url: Option<&str>) -> axum::Router {
    build_agent_loop_app_with_cloud(upstream_mock_url, brave_url, None)
}

fn build_agent_loop_app_with_cloud(
    upstream_mock_url: &str,
    brave_url: Option<&str>,
    cloud_api_url: Option<&str>,
) -> axum::Router {
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
        image_validation_enabled: false,
        image_validation_timeout_secs: 5,
        image_validation_max_bytes: 8192,
        image_validation_max_concurrency: 8,
        image_validation_allow_private_hosts: false,
        chat_cache_expiration_secs: 1200,
        attestation_cache_ttl_secs: 300,
        dev_mode: true,
        gpu_no_hw_mode: true,
        git_rev: "test-rev".to_string(),
        rate_limit_per_second: 1000,
        rate_limit_burst_size: 2000,
        rate_limit_trust_proxy_headers: true,
        cloud_api_url: cloud_api_url.map(String::from),
        cloud_api_auth_max_attempts: 1,
        cloud_api_auth_initial_backoff_ms: 0,
        cloud_api_auth_timeout_secs: 5,
        cloud_api_usage_token: Some("test-usage-token".to_string()),
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

/// vLLM-shaped SSE for a final answer turn that carries *cumulative* usage on
/// every chunk (continuous_usage_stats) but is cut off before `[DONE]` — i.e. an
/// interrupted/aborted stream. The completion count climbs 1→2→3 across chunks;
/// a correct biller keeps the latest (3), a buggy one that sums would report 6.
fn upstream_final_answer_sse_no_done(chat_id: &str) -> String {
    format!(
        "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"Hel\"}}}}],\"usage\":{{\"prompt_tokens\":15,\"completion_tokens\":1,\"total_tokens\":16}}}}\n\n\
         data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"lo.\"}}}}],\"usage\":{{\"prompt_tokens\":15,\"completion_tokens\":2,\"total_tokens\":17}}}}\n\n\
         data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":15,\"completion_tokens\":3,\"total_tokens\":18}}}}\n\n",
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

// ── review regression tests (PR #144) ───────────────────────────────

/// Reviewer #1: a 4xx/5xx from the first upstream call must surface as a
/// real HTTP error, NOT as 200 text/event-stream with a half-broken body.
#[tokio::test]
async fn upstream_5xx_on_first_call_propagates_status() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("content-type", "application/json")
                .set_body_string(r#"{"error":{"message":"out of capacity","type":"overloaded"}}"#),
        )
        .mount(&upstream)
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

    // Status must NOT be 200; the proxy must not pretend the stream
    // started successfully when upstream rejected the very first call.
    assert_ne!(response.status(), StatusCode::OK);
    assert!(response.status().is_server_error() || response.status().is_client_error());
    let ct = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        !ct.contains("text/event-stream"),
        "expected non-SSE content-type, got {ct}"
    );
}

/// Reviewer #2: when the upstream emits more than one tool_call in a
/// single iteration, the loop must NOT execute any of them (Phase 1 cap = 1).
/// The model's tool_call chunks are still forwarded as-is so the client can
/// see what the model wanted; but no synthetic `nearai_tool_result` chunk
/// is emitted and Brave is never called.
#[tokio::test]
async fn multiple_tool_calls_in_one_iteration_skips_loop() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    // SSE that emits TWO tool_calls (indices 0 and 1) and finishes with
    // `tool_calls`. Phase 1 caps execution at one — neither should run.
    let two_tool_calls_sse = "data: {\"id\":\"chatcmpl-MULTI\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[\
        {\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"web_context_search\",\"arguments\":\"{\\\"query\\\":\\\"a\\\"}\"}},\
        {\"index\":1,\"id\":\"call_b\",\"type\":\"function\",\"function\":{\"name\":\"web_context_search\",\"arguments\":\"{\\\"query\\\":\\\"b\\\"}\"}}\
    ]}}]}\n\n\
    data: {\"id\":\"chatcmpl-MULTI\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
    data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(two_tool_calls_sse, "text/event-stream"),
        )
        .mount(&upstream)
        .await;

    // Brave mock: track that it's never called by failing loudly.
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(ResponseTemplate::new(500).set_body_string("brave should not be called"))
        .expect(0)
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
    assert!(body.contains("call_a"));
    assert!(body.contains("call_b"));
    assert!(!body.contains("nearai_tool_result"));
    assert!(body.ends_with("data: [DONE]\n\n"));
    // Verify Brave was never called (wiremock asserts on drop via `.expect(0)`).
    drop(brave);
}

/// Reviewer #3: when the client disconnects while we're waiting on Brave,
/// the loop must abort promptly — no synthetic `[DONE]`, no signature
/// cached. We trigger this by dropping the response body partway through;
/// Brave is configured with a delay long enough that the disconnect lands
/// while we're awaiting it.
#[tokio::test]
async fn client_disconnect_during_brave_aborts_without_signature() {
    use http_body_util::BodyExt as _;

    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    upstream_tool_call_sse("chatcmpl-DISC", "rust"),
                    "text/event-stream",
                ),
        )
        .up_to_n_times(1)
        .mount(&upstream)
        .await;

    // Brave takes 3 seconds — long enough for the client to disconnect.
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(brave_context_json())
                .set_delay(std::time::Duration::from_secs(3)),
        )
        .mount(&brave)
        .await;

    let brave_url = format!("{}/res/v1/llm/context", brave.uri());
    // Clone the router so the signature-endpoint check below hits the
    // SAME `AppState` (and therefore the SAME `ChatCache`) that the loop
    // task would have written to on a clean completion.
    let app = build_agent_loop_app(&upstream.uri(), Some(&brave_url));
    let app_for_sig = app.clone();

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

    // Read the first iteration's chunks (which arrive before Brave is even
    // called) then drop the body to simulate the client going away.
    let mut body = response.into_body();
    let mut saw_finish = false;
    for _ in 0..16 {
        match body.frame().await {
            Some(Ok(frame)) => {
                if let Ok(data) = frame.into_data() {
                    if String::from_utf8_lossy(&data).contains("\"finish_reason\":\"tool_calls\"") {
                        saw_finish = true;
                        break;
                    }
                }
            }
            _ => break,
        }
    }
    assert!(saw_finish, "did not see the tool_calls finish chunk");
    drop(body); // → closes the mpsc receiver → tx.closed() resolves in the loop task

    // Give the spawned loop task a moment to notice the disconnect and bail
    // out of the Brave call. We don't need to wait for Brave's full 3s delay.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // Hit the signature endpoint on the SAME app instance (cloned above);
    // its cache is the same one the loop task would have used if it had
    // completed cleanly. A 404 proves the loop did NOT cache anything
    // for this chat_id — i.e., it aborted on disconnect before signing.
    let sig_response = app_for_sig
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/signature/chatcmpl-DISC")
                .header("authorization", "Bearer test-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sig_response.status(), StatusCode::NOT_FOUND);
}

/// Reviewer #4: a top-level `{"error": {...}}` SSE chunk mid-stream means
/// the upstream aborted. The loop must NOT forward `[DONE]` or cache a
/// signature; the client sees the error chunk and a stream that ends
/// without `[DONE]`.
#[tokio::test]
async fn mid_stream_error_chunk_skips_done_and_signature() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    // SGLang-shaped abort: a content chunk, then a top-level error chunk,
    // then `[DONE]`. The loop should swallow the upstream `[DONE]` and
    // refuse to forward one of its own.
    let aborted_sse = "data: {\"id\":\"chatcmpl-ABORT\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"test-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"partial\"}}]}\n\n\
                       data: {\"error\":{\"object\":\"error\",\"message\":\"request was aborted\",\"type\":\"BadRequestError\",\"code\":400}}\n\n\
                       data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(aborted_sse, "text/event-stream"),
        )
        .mount(&upstream)
        .await;

    // Brave should not be called; the iteration never finishes with
    // tool_calls because the error short-circuits.
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
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
    // We surface a sanitized error chunk (controlled message text) so the
    // client knows something failed, but we do NOT pass through
    // upstream-provided strings — `error.message` could carry user data
    // an upstream backend echoed back.
    assert!(
        body.contains("upstream emitted an error chunk"),
        "expected sanitized error chunk in body: {body}"
    );
    // The upstream's own error message and type must NOT appear on the wire.
    assert!(
        !body.contains("BadRequestError") && !body.contains("request was aborted"),
        "upstream error.message/type leaked verbatim to the client: {body}"
    );
    // And NO `[DONE]` — aborted generation is not a clean completion.
    assert!(
        !body.contains("data: [DONE]"),
        "expected no [DONE] after upstream error, got: {body}"
    );
}

/// Reviewer #5: when E2EE is active, `delta.nearai_tool_result.output`
/// must be encrypted even without `X-Encrypt-All-Fields`. The agent loop
/// is the privacy-sensitive path; the search output must travel encrypted
/// alongside the rest of the stream.
#[tokio::test]
async fn e2ee_without_encrypt_all_fields_encrypts_tool_result() {
    use vllm_proxy_rs::encryption;

    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    upstream_tool_call_sse("chatcmpl-E2EE", "rust"),
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
                .set_body_raw(
                    upstream_final_answer_sse("chatcmpl-E2EE"),
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

    // Test client keypair — different from the server keys baked into
    // build_agent_loop_app so the directionality is meaningful.
    let client_ecdsa: [u8; 32] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x20,
    ];
    let client_ed25519: [u8; 32] = [
        0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
        0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e,
        0x3f, 0x40,
    ];
    let client_pair = signing::SigningPair {
        ecdsa: signing::EcdsaContext::from_key_bytes(&client_ecdsa).unwrap(),
        ed25519: signing::Ed25519Context::from_key_bytes(&client_ed25519).unwrap(),
    };
    let client_pub_hex = client_pair.ed25519.signing_public_key.clone();
    let client_pub_bytes = hex::decode(&client_pub_hex).unwrap();

    // Server public key (matches the fixed keys in build_agent_loop_app).
    let server_ed25519: [u8; 32] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
    ];
    let server_ed25519_ctx = signing::Ed25519Context::from_key_bytes(&server_ed25519).unwrap();
    let server_pub_bytes = hex::decode(&server_ed25519_ctx.signing_public_key).unwrap();

    // Client encrypts the user message with the server's pub key — without
    // X-Encrypt-All-Fields, so the encryption flag should not be required
    // for the agent loop's tool result to be encrypted.
    let enc_for_request = encryption::EncryptionContext {
        algo: encryption::EncryptionAlgo::Ed25519,
        client_pub_key: server_pub_bytes,
        version: 1,
        encrypt_all_fields: false,
    };
    let dec_for_response = encryption::EncryptionContext {
        algo: encryption::EncryptionAlgo::Ed25519,
        client_pub_key: client_pub_bytes,
        version: 1,
        encrypt_all_fields: false,
    };
    let encrypted_content =
        encryption::encrypt_string("hello", &enc_for_request, &client_pair).unwrap();

    let request_body = serde_json::json!({
        "model": "test-model",
        "stream": true,
        "messages": [{"role": "user", "content": encrypted_content}],
        "tools": [{"type": "web_context_search"}],
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .header("x-signing-algo", "ed25519")
                .header("x-client-pub-key", &client_pub_hex)
                // NOTE: deliberately NOT setting `x-encrypt-all-fields`.
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_to_string(response).await;

    // ── nearai_tool_result.output must be encrypted ──────────────────
    let tool_result_chunk = body
        .lines()
        .find(|l| l.contains("nearai_tool_result"))
        .expect("expected a nearai_tool_result chunk in the stream");
    let data = tool_result_chunk
        .strip_prefix("data: ")
        .expect("data: prefix");
    let parsed: serde_json::Value = serde_json::from_str(data).expect("parse chunk JSON");
    let output = parsed["choices"][0]["delta"]["nearai_tool_result"]["output"]
        .as_str()
        .expect("output field is a string");
    assert!(
        !output.contains("Example A") && !output.contains("First snippet"),
        "tool result output was sent plaintext under E2EE: {output}"
    );
    let decrypted_output =
        encryption::decrypt_string(output, &dec_for_response, &client_pair).expect("decrypt");
    assert!(decrypted_output.contains("Example A"));
    assert!(decrypted_output.contains("First snippet"));

    // ── tool_calls[].function.{name,arguments} must also be encrypted
    // even though the client never sent X-Encrypt-All-Fields. The
    // arguments field holds the model-generated search query, which is the
    // same privacy class as the user's original prompt.
    //
    // (Note: the synthetic `nearai_tool_result` envelope legitimately
    // contains a plaintext `name` field identifying which server-side tool
    // ran. That's metadata, not user data — its value is a fixed string
    // controlled by the proxy. Sensitive fields on the envelope —
    // `output` — are encrypted, checked above.)

    // Find a chunk carrying the assembled tool_call function args. The
    // upstream emits the args delta on its own chunk; the encrypt path
    // replaces the string in-place so the chunk still has the field shape
    // but its value is ciphertext.
    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|s| *s != "[DONE]")
        .filter_map(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .collect();

    let args_ciphertext = chunks
        .iter()
        .find_map(|chunk| {
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
        })
        .expect("expected a tool_calls function.arguments chunk");
    assert!(
        !args_ciphertext.contains("query") && !args_ciphertext.contains("rust"),
        "function.arguments was sent plaintext under E2EE: {args_ciphertext}"
    );
    let decrypted_args =
        encryption::decrypt_string(&args_ciphertext, &dec_for_response, &client_pair)
            .expect("decrypt arguments");
    assert!(
        decrypted_args.contains(r#""query":"rust""#),
        "decrypted args did not match expected query: {decrypted_args}"
    );

    // function.name on the model-generated tool_call should also be
    // ciphertext (round-trips to "web_context_search").
    let name_ciphertext = chunks
        .iter()
        .find_map(|chunk| {
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
                .as_str()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
        })
        .expect("expected a tool_calls function.name chunk");
    assert_ne!(name_ciphertext, "web_context_search");
    let decrypted_name =
        encryption::decrypt_string(&name_ciphertext, &dec_for_response, &client_pair)
            .expect("decrypt name");
    assert_eq!(decrypted_name, "web_context_search");
}

// ── second-round review regression tests ────────────────────────────

/// PR #144 round 3: top-level upstream `data: {"error": ...}` chunks must
/// NOT be forwarded verbatim — backends sometimes echo input/prompt
/// fragments in `error.message`, and under E2EE that's data we decrypted
/// inside the CVM. The loop must replace the upstream chunk with a
/// sanitized synthetic error chunk whose text is controlled by the proxy.
#[tokio::test]
async fn upstream_error_message_does_not_leak_prompt_or_query() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    // Upstream emits an error chunk whose `message` echoes a sensitive
    // marker string from the user's prompt — simulating a backend that
    // includes the validation input in error messages.
    let secret = "TOPSECRET_USER_PROMPT_MARKER_42";
    let leaky_error_sse = format!(
        "data: {{\"error\":{{\"object\":\"error\",\"message\":\"validation failed for prompt: '{secret}'\",\"type\":\"BadRequestError\",\"code\":400}}}}\n\n\
         data: [DONE]\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(leaky_error_sse, "text/event-stream"),
        )
        .mount(&upstream)
        .await;
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&brave)
        .await;

    let brave_url = format!("{}/res/v1/llm/context", brave.uri());
    let app = build_agent_loop_app(&upstream.uri(), Some(&brave_url));

    let request_body = serde_json::json!({
        "model": "test-model",
        "stream": true,
        "messages": [{"role": "user", "content": secret}],
        "tools": [{"type": "web_context_search"}],
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer test-token")
                .header("content-type", "application/json")
                .body(Body::from(request_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_to_string(response).await;
    // Sanitized synthetic error chunk reached the client.
    assert!(
        body.contains("upstream emitted an error chunk"),
        "expected sanitized error chunk in body: {body}"
    );
    // The upstream error.message that included the prompt MUST NOT appear
    // anywhere on the wire.
    assert!(
        !body.contains(secret),
        "prompt fragment leaked via upstream error.message: {body}"
    );
    // And no [DONE] — aborted generation is not a clean completion.
    assert!(
        !body.contains("data: [DONE]"),
        "expected no [DONE] after upstream error chunk: {body}"
    );
}

/// PR #144 round 3 (P2): Brave responses are size-capped. A misconfigured
/// or compromised search backend returning a multi-megabyte body must
/// not allocate unbounded memory in the proxy. Instead we surface it as
/// a tool error and let the loop continue.
#[tokio::test]
async fn brave_response_oversized_body_is_rejected() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    upstream_tool_call_sse("chatcmpl-OVERSIZE", "rust"),
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
                .set_body_raw(
                    upstream_final_answer_sse("chatcmpl-OVERSIZE"),
                    "text/event-stream",
                ),
        )
        .mount(&upstream)
        .await;

    // Brave returns 3 MiB of valid JSON — above the 2 MiB cap.
    let huge_url = format!("https://example.com/{}", "a".repeat(3 * 1024 * 1024));
    let oversized = serde_json::json!({
        "grounding": {
            "generic": [{
                "url": huge_url,
                "title": "huge",
                "snippets": ["x"]
            }]
        }
    });
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(oversized))
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
    // The oversized response should be surfaced as a tool error result,
    // not a successful tool call. The loop continues and the model gets
    // a chance to respond.
    assert!(body.contains("nearai_tool_result"));
    assert!(body.contains("\"status\":\"error\""));
    // The leaked huge URL must NOT appear on the wire — we rejected the
    // response before parsing/forwarding.
    assert!(
        !body.contains(&"a".repeat(2048)),
        "oversized response payload leaked into stream"
    );
    // Loop still completes normally on the next iteration.
    assert!(body.contains("Hello."));
    assert!(body.ends_with("data: [DONE]\n\n"));
}

/// PR #144 round 3 (P2): the formatted tool output emitted to the client
/// AND fed back to the model is capped, even if Brave returns a valid
/// but very large response that respects raw-body limits.
#[tokio::test]
async fn brave_formatted_output_is_truncated() {
    let upstream = MockServer::start().await;
    let brave = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    upstream_tool_call_sse("chatcmpl-TRUNC", "rust"),
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
                .set_body_raw(
                    upstream_final_answer_sse("chatcmpl-TRUNC"),
                    "text/event-stream",
                ),
        )
        .mount(&upstream)
        .await;

    // 200 entries × ~600 bytes snippet → ~120 KB formatted output, well
    // above the 32 KiB cap. Body itself is ~150 KB JSON — under the 2 MiB
    // body cap so the body read succeeds.
    let mut entries = Vec::with_capacity(200);
    for i in 0..200 {
        entries.push(serde_json::json!({
            "url": format!("https://example.com/page-{i}"),
            "title": format!("Title {i}"),
            "snippets": [format!("snippet-{i}-{}", "x".repeat(500))],
        }));
    }
    let big = serde_json::json!({
        "grounding": {"generic": entries}
    });
    Mock::given(method("GET"))
        .and(path("/res/v1/llm/context"))
        .respond_with(ResponseTemplate::new(200).set_body_json(big))
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
    let tool_result_chunk = body
        .lines()
        .find(|l| l.contains("nearai_tool_result"))
        .expect("expected a nearai_tool_result chunk");
    let parsed: serde_json::Value =
        serde_json::from_str(tool_result_chunk.strip_prefix("data: ").unwrap())
            .expect("parse chunk JSON");
    let output = parsed["choices"][0]["delta"]["nearai_tool_result"]["output"]
        .as_str()
        .expect("output field is a string");

    // Cap is 32 KiB; allow some slack for the truncation marker.
    assert!(
        output.len() <= 32 * 1024 + 64,
        "tool output not truncated to cap: {} bytes",
        output.len()
    );
    assert!(
        output.contains("[truncated]"),
        "expected truncation marker in capped output"
    );
}

/// Regression test for nearai/infra#98: an interrupted agent loop (the final
/// upstream turn carries usage but is cut off before `[DONE]`) must still bill
/// the tokens already produced — while still withholding the signature, since
/// the response is incomplete and cannot be verified.
#[tokio::test]
async fn interrupted_agent_loop_reports_usage_without_signature() {
    let upstream = MockServer::start().await;
    let cloud_api = MockServer::start().await;

    // sk- key validation succeeds → usage reporter is active for this request.
    // Identity fields enable the service-token /v1/internal/usage reporting path.
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
        .respond_with(ResponseTemplate::new(200))
        .mount(&cloud_api)
        .await;

    // Single turn: a final answer carrying cumulative usage on every chunk
    // (1→2→3 completion tokens) but no [DONE]. The mock only matches if the loop
    // forces continuous_usage_stats — otherwise a real backend wouldn't emit the
    // per-chunk usage this interrupted case depends on. `expect(1)` enforces it.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"continuous_usage_stats\":true"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    upstream_final_answer_sse_no_done("chatcmpl-INT"),
                    "text/event-stream",
                ),
        )
        .expect(1)
        .mount(&upstream)
        .await;

    // Brave configured (so the agent-loop path engages) but never actually
    // called — the single turn finishes with "stop", not a tool call.
    let app = build_agent_loop_app_with_cloud(
        &upstream.uri(),
        Some("http://brave.invalid/res/v1/llm/context"),
        Some(&cloud_api.uri()),
    );
    let app_for_sig = app.clone();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-test-interrupted-loop-key")
                .header("content-type", "application/json")
                .body(Body::from(agent_loop_request_body(true).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Drain the stream so the spawned loop task runs to completion.
    let _ = response.into_body().collect().await;

    // Usage is reported (fire-and-forget) even though the stream never saw [DONE].
    let mut usage_body = None;
    for _ in 0..50 {
        let reqs = cloud_api.received_requests().await.unwrap();
        if let Some(req) = reqs.iter().find(|r| r.url.path() == "/v1/internal/usage") {
            usage_body = Some(serde_json::from_slice::<serde_json::Value>(&req.body).unwrap());
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let usage = usage_body.expect("interrupted agent loop must still report usage");
    assert_eq!(usage["type"], "chat_completion");
    // Latest cumulative usage, NOT the sum of the per-chunk cumulative values
    // (summing the 3 chunks would overbill to 45 in / 6 out).
    assert_eq!(usage["input_tokens"], 15);
    assert_eq!(usage["output_tokens"], 3);
    assert_eq!(usage["id"], "chatcmpl-INT");

    // ...but no signature is cached over the incomplete response.
    let sig_response = app_for_sig
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/signature/chatcmpl-INT")
                .header("authorization", "Bearer test-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sig_response.status(), StatusCode::NOT_FOUND);
}
