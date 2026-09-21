//! Gateway-mode behaviors: the proxy fronting a fleet of inference-proxies
//! (a backend-only bearer, modality policy, non-TEE route hiding, queue-full
//! back-pressure, the first-event peek, keep-alives).

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use vllm_proxy_rs::*;

#[derive(Default)]
struct GatewayOptions {
    backend_token: Option<String>,
    backend_priority: Option<i64>,
    /// Mock cloud-api base URL, for `sk-` key requests.
    cloud_api_url: Option<String>,
    backend_health_path: Option<String>,
    non_tee_deployment: bool,
    map_queue_full_to_429: bool,
    stream_error_peek_ms: u64,
    stream_commit_ms: u64,
    first_token_deadline_ms: u64,
    first_token_deadline_per_1k_tokens_ms: u64,
    first_token_deadline_max_ms: u64,
    rejected_content_part_types: Vec<String>,
    allowed_org_ids: Vec<String>,
    sse_keepalive_secs: u64,
    stream_idle_timeout_secs: u64,
    /// Backend base URLs; empty = just the mock server.
    backend_urls: Vec<String>,
    backend_conversation_affinity: bool,
    admission_max_inflight: u32,
    admission_tier_borrowing: bool,
    admission_long_max_inflight_per_host: u32,

    admission_start_inflight: Option<u32>,
    admission_ttft_p95_max_ms: Option<u64>,
    admission_backpressure_secs: Option<u64>,
    backend_connect_failover: bool,
    /// Engine metrics probe base URLs, one per backend (polled every 100 ms here).
    backend_probe_urls: Vec<String>,
    /// Long-context tier: backends appended after `backend_urls`, their probe
    /// URLs, and the estimated-input threshold above which requests go there.
    backend_long_context_urls: Vec<String>,
    backend_long_context_probe_urls: Vec<String>,
    long_context_above_tokens: u64,
    /// Source of the models document (a mock cloud-api `/v1/models`).
    models_document_url: Option<String>,
    capacity_requests_per_minute: u64,
    /// `VLLM_PROXY_REASONING_OFF_EFFORT` (default `none`).
    reasoning_off_effort: Option<String>,
}

fn build_gateway(mock_url: &str, options: GatewayOptions) -> axum::Router {
    build_gateway_with_state(mock_url, options).0
}

fn build_gateway_with_state(mock_url: &str, options: GatewayOptions) -> (axum::Router, AppState) {
    let base = mock_url.trim_end_matches('/');
    let backend_urls = if options.backend_urls.is_empty() {
        vec![mock_url.to_string()]
    } else {
        options.backend_urls.clone()
    };
    let config = config::Config {
        model_name: "test-model".to_string(),
        tokens: vec!["test-token".to_string()],
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
        pool_idle_timeout_secs: 60,
        max_request_size: 1024 * 1024,
        max_image_request_size: 5 * 1024 * 1024,
        max_audio_request_size: 10 * 1024 * 1024,
        image_validation_enabled: false,
        image_validation_timeout_secs: 5,
        image_validation_max_bytes: 8192,
        image_validation_max_concurrency: 8,
        image_validation_allow_private_hosts: false,
        image_validation_allowed_domains: Vec::new(),
        image_validation_reject_non_rgb_images: false,
        image_validation_reject_single_channel_images: false,
        chat_cache_expiration_secs: 1200,
        attestation_cache_ttl_secs: 300,
        dev_mode: true,
        gpu_no_hw_mode: true,
        git_rev: "test-rev".to_string(),
        rate_limit_per_second: 100,
        rate_limit_burst_size: 200,
        rate_limit_trust_proxy_headers: true,
        cloud_api_url: options.cloud_api_url.clone(),
        cloud_api_auth_max_attempts: 1,
        cloud_api_auth_initial_backoff_ms: 0,
        cloud_api_auth_timeout_secs: 5,
        cloud_api_usage_token: None,
        compose_manager_url: None,
        tls_cert_path: None,
        timeout_secs: 30,
        stream_idle_timeout_secs: options.stream_idle_timeout_secs,
        timeout_tokenize_secs: 5,
        openai_chat_compatibility_check_enabled: false,
        startup_check_retries: 1,
        startup_check_retry_delay_secs: 0,
        startup_check_timeout_secs: 5,
        backend_urls: backend_urls.clone(),
        vllm_data_parallel_size: None,
        backend_conversation_affinity: options.backend_conversation_affinity,
        backend_affinity_max_imbalance: 8,
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
        listen_addr: "127.0.0.1".to_string(),
        backend_token: options.backend_token.clone(),
        backend_priority: options.backend_priority,
        backend_health_path: options
            .backend_health_path
            .unwrap_or_else(|| "/health".to_string()),
        non_tee_deployment: options.non_tee_deployment,
        map_queue_full_to_429: options.map_queue_full_to_429,
        stream_error_peek_ms: options.stream_error_peek_ms,
        stream_commit_ms: options.stream_commit_ms,
        first_token_deadline_ms: options.first_token_deadline_ms,
        first_token_deadline_per_1k_tokens_ms: options.first_token_deadline_per_1k_tokens_ms,
        first_token_deadline_max_ms: options.first_token_deadline_max_ms,
        rejected_content_part_types: options.rejected_content_part_types,
        models_document_url: options.models_document_url.clone(),
        capacity_requests_per_minute: options.capacity_requests_per_minute,
        reasoning_off_effort: options
            .reasoning_off_effort
            .clone()
            .unwrap_or_else(|| "none".to_string()),
        allowed_org_ids: options.allowed_org_ids,
        sse_keepalive_secs: options.sse_keepalive_secs,
        admission_max_inflight: options.admission_max_inflight,
        admission_tier_borrowing: options.admission_tier_borrowing,
        admission_long_max_inflight_per_host: options.admission_long_max_inflight_per_host,
        admission_start_inflight: options
            .admission_start_inflight
            .unwrap_or(options.admission_max_inflight),
        admission_ramp_step: 8,
        admission_ramp_interval_secs: 1800,
        admission_ttft_p95_max_ms: options.admission_ttft_p95_max_ms.unwrap_or(30_000),
        admission_backpressure_secs: options.admission_backpressure_secs.unwrap_or(10),
        admission_retry_after_secs: 2,
        backend_connect_failover: options.backend_connect_failover,
        backend_probe_urls: options.backend_probe_urls.clone(),
        backend_probe_interval_secs: 2,
        backend_long_context_urls: options.backend_long_context_urls.clone(),
        backend_long_context_probe_urls: options.backend_long_context_probe_urls.clone(),
        long_context_above_tokens: options.long_context_above_tokens,
        dstack_socket_path: "/nonexistent/dstack.sock".to_string(),
        gpu_evidence_delegate_url: None,
        gpu_evidence_delegate_timeout_secs: 30,
        web_context_search_api_key: None,
        web_context_search_url: None,
        agent_loop_max_iterations: 5,
        web_context_search_timeout_secs: 30,
        fusion_enabled: false,
        fusion_endpoints_url: "https://completions.near.ai/endpoints".to_string(),
        fusion_endpoints_ttl_secs: 300,
        fusion_internal_bearer_token: None,
        fusion_default_analysis_models: Vec::new(),
        fusion_max_panel_models: 8,
        fusion_max_depth: 1,
        fusion_panel_timeout_secs: 120,
        fusion_max_response_bytes: 10 * 1024 * 1024,
        fusion_internal_max_attempts: 2,
        fusion_internal_retry_initial_backoff_ms: 1,
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
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();

    // Mirror main.rs: the backend bearer and priority header are default
    // headers on a dedicated client, never on the general-purpose one.
    let http_client = reqwest::Client::new();
    let mut backend_headers = reqwest::header::HeaderMap::new();
    if let Some(token) = &options.backend_token {
        backend_headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
    }
    if let Some(priority) = options.backend_priority {
        backend_headers.insert(
            priority::PRIORITY_HEADER,
            reqwest::header::HeaderValue::from(priority),
        );
    }
    let backend_client = if backend_headers.is_empty() {
        http_client.clone()
    } else {
        reqwest::Client::builder()
            .default_headers(backend_headers)
            .build()
            .unwrap()
    };

    let backend_pool = Arc::new(backend_pool::BackendPool::with_long_context(
        backend_urls,
        options.backend_long_context_urls.clone(),
    ));
    let engine_load = Arc::new(engine_load::EngineLoad::new(
        backend_pool.len(),
        Duration::from_secs(5),
    ));
    let probe_urls = config.pool_probe_urls();
    if !probe_urls.is_empty() {
        engine_load::spawn_engine_load_poller(
            engine_load.clone(),
            reqwest::Client::new(),
            probe_urls,
            Duration::from_millis(100),
        );
    }
    let admission = Arc::new(admission::AdmissionController::new(
        config.admission(),
        backend_pool.len(),
        engine_load,
    ));
    let backend_affinity = Arc::new(backend_affinity::BackendConversationAffinity::new(
        config.backend_conversation_affinity,
        backend_pool.len(),
        config.backend_affinity_max_imbalance,
        1_200,
    ));
    let state = AppState {
        config: Arc::new(config),
        signing: Arc::new(signing_pair),
        cache: Arc::new(cache::ChatCache::new("test-model", 1200)),
        attestation_cache: Arc::new(attestation::AttestationCache::new(300)),
        http_client,
        backend_client,
        metrics_handle,
        tls_cert_fingerprint: Arc::new(
            attestation::TlsCertTracker::new(None).expect("tracker for None path"),
        ),
        backend_pool,
        ohttp_gateway: None,
        ohttp_attestation_ed25519: None,
        fusion_caches: Arc::new(fusion::FusionCaches::default()),
        vllm_dp_affinity: Arc::new(vllm_dp_affinity::VllmDpAffinity::new(None, 1_200)),
        backend_affinity,
        admission,
    };
    let rate_limit_state = rate_limit::RateLimitState {
        limiter: rate_limit::build_rate_limiter(100, 200),
        trust_proxy_headers: true,
    };
    let router = routes::build_router()
        .layer(middleware::from_fn(rate_limit::rate_limit_middleware))
        .layer(axum::Extension(rate_limit_state))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state.clone());
    (router, state)
}

fn chat_request(body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer test-token")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn chat_completion_json() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-gw-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    })
}

async fn json_body(response: axum::response::Response) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| panic!("non-JSON body: {}", String::from_utf8_lossy(&bytes)))
}

// ---------------------------------------------------------------------------
// VLLM_BACKEND_TOKEN: sent to backends, customer token never forwarded
// ---------------------------------------------------------------------------

#[tokio::test]
async fn backend_token_is_attached_to_backend_requests() {
    let mock = MockServer::start().await;
    // Only a request carrying the backend bearer is answered; a request that
    // still carried the client's token (or none) would fall through to 404.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer backend-secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-token"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&mock)
        .await;

    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            backend_token: Some("backend-secret".to_string()),
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], "hi");
}

#[tokio::test]
async fn without_backend_token_no_authorization_reaches_backend() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-token"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
        .mount(&mock)
        .await;

    let app = build_gateway(&mock.uri(), GatewayOptions::default());
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Request priority: trusted callers set it by header, everyone else gets 0
// ---------------------------------------------------------------------------

fn chat_request_with(token: &str, priority_header: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {token}"))
        .header("content-type", "application/json");
    if let Some(value) = priority_header {
        builder = builder.header(priority::PRIORITY_HEADER, value);
    }
    // A client-supplied priority is always overwritten.
    builder
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "test-model",
                "priority": 999,
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap(),
        ))
        .unwrap()
}

fn expect_priority(value: i64) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_partial_json(
            serde_json::json!({"priority": value}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
}

#[tokio::test]
async fn gateway_sends_priority_header_next_to_backend_token() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer backend-secret"))
        .and(header(priority::PRIORITY_HEADER, "-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            backend_token: Some("backend-secret".to_string()),
            backend_priority: Some(-1),
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn trusted_caller_sets_priority_by_header_everything_else_is_zero() {
    let mock = MockServer::start().await;
    // Requests below use the proxy's config token, i.e. a trusted caller
    // (cloud-api, or the gateway on the CVM hop).
    for (header_value, expected) in [
        (Some("-1"), -1),  // the OpenRouter gateway
        (None, 0),         // cloud-api: no header, default
        (Some("abc"), 0),  // garbage falls back to the default
        (Some("5000"), 0), // out of bounds too
    ] {
        expect_priority(expected).mount(&mock).await;
        let app = build_gateway(&mock.uri(), GatewayOptions::default());
        let response = app
            .oneshot(chat_request_with("test-token", header_value))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "header={header_value:?}");
        mock.verify().await;
        mock.reset().await;
    }
}

#[tokio::test]
async fn customer_key_cannot_set_priority() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "valid": true, "organization_id": "org", "workspace_id": "ws", "api_key_id": "k"
        })))
        .mount(&mock)
        .await;
    for header_value in [Some("5"), Some("-1"), None] {
        expect_priority(0).mount(&mock).await;
        let app = build_gateway(
            &mock.uri(),
            GatewayOptions {
                cloud_api_url: Some(mock.uri()),
                ..Default::default()
            },
        );
        let response = app
            .oneshot(chat_request_with("sk-live-customer", header_value))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "header={header_value:?}");
        mock.verify().await;
        mock.reset().await;
        Mock::given(method("POST"))
            .and(path("/v1/check_api_key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "valid": true, "organization_id": "org", "workspace_id": "ws", "api_key_id": "k"
            })))
            .mount(&mock)
            .await;
    }
}

#[tokio::test]
async fn completions_route_gets_priority_too() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .and(wiremock::matchers::body_partial_json(
            serde_json::json!({"priority": -1}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-1", "object": "text_completion", "model": "test-model",
            "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&mock)
        .await;
    let app = build_gateway(&mock.uri(), GatewayOptions::default());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("authorization", "Bearer test-token")
        .header("content-type", "application/json")
        .header(priority::PRIORITY_HEADER, "-1")
        .body(Body::from(
            serde_json::to_vec(
                &serde_json::json!({"model": "test-model", "prompt": "hi", "priority": 3}),
            )
            .unwrap(),
        ))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    mock.verify().await;
}

// ---------------------------------------------------------------------------
// VLLM_PROXY_ALLOWED_ORG_IDS: only the partner's keys may use the lane
// ---------------------------------------------------------------------------

async fn mount_key_check(mock: &MockServer, key: &str, org: Option<&str>) {
    let mut body = serde_json::json!({"valid": true, "workspace_id": "ws", "api_key_id": "k"});
    if let Some(org) = org {
        body["organization_id"] = serde_json::json!(org);
    }
    // The proxy presents the customer key as the bearer of the check call.
    let bearer = format!("Bearer {key}");
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .and(header("authorization", bearer.as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(mock)
        .await;
}

#[tokio::test]
async fn org_allowlist_admits_partner_keys_and_refuses_others() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(2) // the partner key and the config token
        .mount(&mock)
        .await;
    mount_key_check(&mock, "sk-live-partner", Some("org-partner")).await;
    mount_key_check(&mock, "sk-live-other", Some("org-other")).await;
    mount_key_check(&mock, "sk-live-noorg", None).await;

    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            cloud_api_url: Some(mock.uri()),
            allowed_org_ids: vec!["org-partner".to_string()],
            ..Default::default()
        },
    );
    for (token, expected) in [
        ("sk-live-partner", StatusCode::OK),
        ("sk-live-other", StatusCode::FORBIDDEN),
        ("sk-live-noorg", StatusCode::FORBIDDEN),
        ("test-token", StatusCode::OK), // config token: operators are not gated
    ] {
        let response = app
            .clone()
            .oneshot(chat_request_with(token, None))
            .await
            .unwrap();
        assert_eq!(response.status(), expected, "token={token}");
        if expected == StatusCode::FORBIDDEN {
            let json = json_body(response).await;
            assert_eq!(json["error"]["type"], "forbidden");
        }
    }
    mock.verify().await;
}

#[tokio::test]
async fn empty_org_allowlist_admits_every_valid_key() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
        .mount(&mock)
        .await;
    mount_key_check(&mock, "sk-live-other", Some("org-other")).await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            cloud_api_url: Some(mock.uri()),
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request_with("sk-live-other", None))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    mock.verify().await;
}

// ---------------------------------------------------------------------------
// VLLM_PROXY_REJECTED_CONTENT_PART_TYPES
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rejected_content_part_types_get_400_without_dispatch() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1) // exactly the text+image request below
        .mount(&mock)
        .await;

    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            rejected_content_part_types: vec!["video_url".to_string(), "input_audio".to_string()],
            ..Default::default()
        },
    );

    let video = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "earlier turn"},
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}}
            ]}
        ]
    });
    let response = app
        .clone()
        .oneshot(chat_request(video.clone()))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = json_body(response).await;
    assert_eq!(body["error"]["type"], "bad_request");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("'video_url' is not supported"),
        "{body}"
    );

    // Streaming variant is refused identically (before any upstream call).
    let mut streaming = video.clone();
    streaming["stream"] = serde_json::json!(true);
    let response = app.clone().oneshot(chat_request(streaming)).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Allowed modalities still dispatch.
    let ok = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "a text part mentioning video_url"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
        ]}]
    });
    let response = app.oneshot(chat_request(ok)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// /healthz in a non-TEE deployment
// ---------------------------------------------------------------------------

#[tokio::test]
async fn healthz_can_skip_dstack_and_probe_a_custom_backend_path() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/healthz"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{\"status\":\"ok\"}"))
        .expect(1)
        .mount(&mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&mock)
        .await;

    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            non_tee_deployment: true,
            backend_health_path: Some("/healthz".to_string()),
            ..Default::default()
        },
    );
    let response = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["checks"]["dstack"], "skipped");
    assert_eq!(body["checks"]["backend"], "ok");
}

#[tokio::test]
async fn healthz_reports_unhealthy_without_dstack_by_default() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;
    let app = build_gateway(&mock.uri(), GatewayOptions::default());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = json_body(response).await;
    assert_eq!(body["checks"]["dstack"], "unreachable");
    assert_eq!(body["checks"]["backend"], "ok");
}

// ---------------------------------------------------------------------------
// SSE keep-alive comments
// ---------------------------------------------------------------------------

/// Backend that sends response headers immediately, stays silent for `delay`,
/// then emits `chunks` and closes — the shape of a long prefill.
async fn spawn_silent_then_stream_backend(
    delay: Duration,
    chunks: Vec<&'static str>,
) -> (String, tokio::task::JoinHandle<()>) {
    use axum::routing::post;
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let chunks = chunks.clone();
            async move {
                let (tx, rx) =
                    tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(8);
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    for c in chunks {
                        if tx
                            .send(Ok(axum::body::Bytes::from_static(c.as_bytes())))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                });
                axum::response::Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(axum::body::Body::from_stream(
                        tokio_stream::wrappers::ReceiverStream::new(rx),
                    ))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}"), handle)
}

#[tokio::test]
async fn sse_keepalive_comments_bridge_a_silent_upstream() {
    let (backend, handle) = spawn_silent_then_stream_backend(
        Duration::from_millis(2600),
        vec![
            "data: {\"id\":\"chatcmpl-ka\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-ka\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
            "data: [DONE]\n\n",
        ],
    )
    .await;

    let app = build_gateway(
        &backend,
        GatewayOptions {
            sse_keepalive_secs: 1,
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // Collect frames in arrival order.
    let mut body = response.into_body();
    let mut frames: Vec<String> = Vec::new();
    while let Some(frame) = body.frame().await {
        if let Ok(data) = frame.unwrap().into_data() {
            frames.push(String::from_utf8_lossy(&data).to_string());
        }
    }
    let joined = frames.concat();
    assert!(
        joined.contains("data: [DONE]"),
        "stream must complete: {joined}"
    );
    assert!(joined.contains("\"content\":\"hi\""), "{joined}");

    let first_data = frames
        .iter()
        .position(|f| f.starts_with("data:"))
        .expect("a data frame");
    let keepalives_before_data = frames[..first_data]
        .iter()
        .filter(|f| f.as_str() == ": keep-alive\n\n")
        .count();
    assert!(
        keepalives_before_data >= 2,
        "expected ≥2 keep-alive comments during a 2.6s silent prefill, got {keepalives_before_data}: {frames:?}"
    );
    assert!(
        !frames[first_data..]
            .iter()
            .any(|f| f.as_str() == ": keep-alive\n\n"),
        "no keep-alives once the upstream is streaming quickly: {frames:?}"
    );
    handle.abort();
}

/// Backend that sends headers immediately, emits `first` after `delay`, then
/// keeps the stream open but silent for `hang` — a stalled engine.
async fn spawn_stream_then_hang_backend(
    delay: Duration,
    first: &'static str,
    hang: Duration,
) -> (String, tokio::task::JoinHandle<()>) {
    use axum::routing::post;
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        post(move || async move {
            let (tx, rx) =
                tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(8);
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                let _ = tx
                    .send(Ok(axum::body::Bytes::from_static(first.as_bytes())))
                    .await;
                tokio::time::sleep(hang).await;
                drop(tx);
            });
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(axum::body::Body::from_stream(
                    tokio_stream::wrappers::ReceiverStream::new(rx),
                ))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}"), handle)
}

#[tokio::test]
async fn idle_timeout_is_not_reset_by_keepalive_ticks() {
    // 1 s keep-alives must not push back a 2 s idle watchdog: the deadline is
    // measured from the last upstream chunk, not from the last loop wake-up.
    let (backend, handle) = spawn_stream_then_hang_backend(
        Duration::from_millis(100),
        "data: {\"id\":\"chatcmpl-idle\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"start\"},\"finish_reason\":null}]}\n\n",
        Duration::from_secs(20),
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            sse_keepalive_secs: 1,
            stream_idle_timeout_secs: 2,
            ..Default::default()
        },
    );
    let started = std::time::Instant::now();
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut text = String::new();
    let mut keepalives = 0;
    let mut errored = false;
    while let Some(frame) = body.frame().await {
        match frame {
            Ok(frame) => {
                if let Ok(data) = frame.into_data() {
                    let s = String::from_utf8_lossy(&data).to_string();
                    if s == ": keep-alive\n\n" {
                        keepalives += 1;
                    }
                    text.push_str(&s);
                }
            }
            Err(_) => {
                errored = true;
                break;
            }
        }
    }
    let elapsed = started.elapsed();
    assert!(text.contains("\"content\":\"start\""), "{text}");
    assert!(!text.contains("[DONE]"), "{text}");
    assert!(
        errored,
        "the idle timeout must end the stream with an error"
    );
    assert!(
        elapsed < Duration::from_secs(8),
        "2 s idle timeout must fire despite 1 s keep-alives; took {elapsed:?}"
    );
    assert!(keepalives >= 1, "expected keep-alives before the timeout");
    handle.abort();
}

#[tokio::test]
async fn sse_keepalive_is_off_by_default() {
    let (backend, handle) = spawn_silent_then_stream_backend(
        Duration::from_millis(1300),
        vec![
            "data: {\"id\":\"chatcmpl-nk\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ],
    )
    .await;
    let app = build_gateway(&backend, GatewayOptions::default());
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(!text.contains(": keep-alive"), "{text}");
    assert!(text.contains("data: [DONE]"), "{text}");
    handle.abort();
}

// ---------------------------------------------------------------------------
// NON_TEE_DEPLOYMENT: attestation surface is not advertised
// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_tee_deployment_hides_attestation_signature_and_gpu_evidence() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            non_tee_deployment: true,
            ..Default::default()
        },
    );

    // A completion still works and is (dev-)signed internally...
    let response = app
        .clone()
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // ...but nothing unverifiable is offered.
    let cases = [
        ("GET", "/v1/attestation/report", false, ""),
        (
            "GET",
            "/v1/attestation/report?signing_algo=ed25519",
            false,
            "",
        ),
        ("GET", "/v1/signature/chatcmpl-gw-1", true, ""),
        (
            "POST",
            "/internal/gpu_evidence",
            true,
            r#"{"nonce":"0000000000000000000000000000000000000000000000000000000000000000"}"#,
        ),
    ];
    for (m, uri, auth, body) in cases {
        let mut req = Request::builder()
            .method(m)
            .uri(uri)
            .header("content-type", "application/json");
        if auth {
            req = req.header("authorization", "Bearer test-token");
        }
        let response = app
            .clone()
            .oneshot(req.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{m} {uri}");
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "not_found", "{m} {uri}");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("does not run inside a TEE"),
            "{m} {uri}: {json}"
        );
    }
}

#[tokio::test]
async fn tee_deployment_keeps_signature_route() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .mount(&mock)
        .await;
    let app = build_gateway(&mock.uri(), GatewayOptions::default());
    let response = app
        .clone()
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    let id = body["id"].as_str().unwrap().to_string();
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/signature/{id}"))
                .header("authorization", "Bearer test-token")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Queue-full → 429 and the bounded first-event peek
// ---------------------------------------------------------------------------

const QUEUE_FULL_BODY: &str =
    r#"{"object":"error","message":"The request queue is full.","type":"abort","code":503}"#;

#[tokio::test]
async fn queue_full_503_becomes_429_only_when_enabled() {
    for (enabled, expected) in [
        (true, StatusCode::TOO_MANY_REQUESTS),
        (false, StatusCode::SERVICE_UNAVAILABLE),
    ] {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(503).set_body_raw(QUEUE_FULL_BODY, "application/json"),
            )
            .mount(&mock)
            .await;
        let app = build_gateway(
            &mock.uri(),
            GatewayOptions {
                map_queue_full_to_429: enabled,
                ..Default::default()
            },
        );
        let response = app
            .oneshot(chat_request(serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), expected, "enabled={enabled}");
        let retry_after = response
            .headers()
            .get("retry-after")
            .map(|v| v.to_str().unwrap().to_string());
        let body = json_body(response).await;
        assert_eq!(body["error"]["message"], "The request queue is full.");
        if enabled {
            // Same shape as the gateway's own refusals: clients back off once.
            assert_eq!(retry_after.as_deref(), Some("2"));
            assert_eq!(body["error"]["type"], "overloaded");
        } else {
            assert!(retry_after.is_none());
            assert_eq!(body["error"]["type"], "abort");
        }
    }
}

#[tokio::test]
async fn other_503s_are_not_rewritten() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_raw(
            r#"{"object":"error","message":"Model is loading","type":"ServiceUnavailable","code":503}"#,
            "application/json",
        ))
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            map_queue_full_to_429: true,
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

/// SGLang rejects at admission on an HTTP 200 SSE stream: the first event is
/// `data: {"error": …}` followed by a clean `[DONE]`.
fn queue_full_sse_stream() -> String {
    format!("data: {{\"error\":{QUEUE_FULL_BODY}}}\n\ndata: [DONE]\n\n")
}

#[tokio::test]
async fn streaming_queue_full_first_event_becomes_429_with_peek() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(queue_full_sse_stream()),
        )
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            map_queue_full_to_429: true,
            stream_error_peek_ms: 1000,
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("2")
    );
    let body = json_body(response).await;
    assert_eq!(body["error"]["message"], "The request queue is full.");
    assert_eq!(body["error"]["type"], "overloaded");
}

#[tokio::test]
async fn streaming_first_event_error_keeps_engine_status_without_mapping() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(queue_full_sse_stream()),
        )
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            stream_error_peek_ms: 1000,
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn streaming_without_peek_forwards_the_error_event_on_200() {
    // Existing behavior, preserved for CVM deployments (peek off).
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(queue_full_sse_stream()),
        )
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            map_queue_full_to_429: true,
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("The request queue is full."), "{text}");
    assert!(text.contains("data: [DONE]"), "{text}");
}

#[tokio::test]
async fn stream_peek_timeout_leaves_a_slow_stream_intact() {
    let (backend, handle) = spawn_silent_then_stream_backend(
        Duration::from_millis(700),
        vec![
            "data: {\"id\":\"chatcmpl-pk\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"slow\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-pk\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" hello\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ],
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            stream_error_peek_ms: 200,
            map_queue_full_to_429: true,
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("\"content\":\"slow\""), "{text}");
    assert!(text.contains("\"content\":\" hello\""), "{text}");
    assert!(text.contains("data: [DONE]"), "{text}");
    handle.abort();
}

#[tokio::test]
async fn stream_peek_keeps_a_fast_first_chunk() {
    // The peeked chunk must be re-attached: nothing is lost when the first
    // event is ordinary content that arrives inside the peek window.
    let (backend, handle) = spawn_silent_then_stream_backend(
        Duration::from_millis(50),
        vec![
            "data: {\"id\":\"chatcmpl-fast\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-fast\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" second\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ],
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            stream_error_peek_ms: 2000,
            ..Default::default()
        },
    );
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("\"content\":\"first\""), "{text}");
    assert!(text.contains("\"content\":\" second\""), "{text}");
    assert!(text.contains("data: [DONE]"), "{text}");
    handle.abort();
}

// ---------------------------------------------------------------------------
// Lane admission: budget, observed back-pressure, connection fail-over
// ---------------------------------------------------------------------------

fn hello_body() -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}]
    })
}

/// A URL nothing listens on (the port was bound and released).
fn unreachable_backend_url() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    format!("http://127.0.0.1:{port}")
}

async fn assert_overloaded(response: axum::response::Response) {
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("2")
    );
    let json = json_body(response).await;
    assert_eq!(json["error"]["type"], "overloaded");
}

#[tokio::test]
async fn admission_budget_refuses_with_429_before_dispatch() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_completion_json())
                .set_delay(Duration::from_millis(700)),
        )
        .expect(2) // the first and the third request; the second never dispatches
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            admission_max_inflight: 1,
            ..Default::default()
        },
    );

    let first = tokio::spawn({
        let app = app.clone();
        async move { app.oneshot(chat_request(hello_body())).await.unwrap() }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    let started = std::time::Instant::now();
    let second = app
        .clone()
        .oneshot(chat_request(hello_body()))
        .await
        .unwrap();
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "refusal must not wait for the in-flight request"
    );
    assert_overloaded(second).await;

    assert_eq!(first.await.unwrap().status(), StatusCode::OK);
    // The slot was released with the response.
    let third = app.oneshot(chat_request(hello_body())).await.unwrap();
    assert_eq!(third.status(), StatusCode::OK);
    mock.verify().await;
}

#[tokio::test]
async fn admission_is_inert_when_not_configured() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(3)
        .mount(&mock)
        .await;
    let app = build_gateway(&mock.uri(), GatewayOptions::default());
    for _ in 0..3 {
        let response = app
            .clone()
            .oneshot(chat_request(hello_body()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get("retry-after").is_none());
    }
    mock.verify().await;
}

#[tokio::test]
async fn engine_queue_full_on_every_backend_refuses_new_work_without_dispatch() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string(QUEUE_FULL_BODY))
        .expect(1)
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            admission_max_inflight: 8,
            map_queue_full_to_429: true,
            ..Default::default()
        },
    );
    // The engine's rejection reaches the client as 429 (mapped), with the
    // same Retry-After as a gateway refusal...
    let first = app
        .clone()
        .oneshot(chat_request(hello_body()))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        first
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("2")
    );
    // ...and the only backend is now known to be saturated: the next request
    // is refused by the gateway itself, with Retry-After, and never dispatched.
    let second = app
        .clone()
        .oneshot(chat_request(hello_body()))
        .await
        .unwrap();
    assert_overloaded(second).await;
    mock.verify().await;
}

#[tokio::test]
async fn connect_error_fails_over_to_another_backend_when_enabled() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
        .mount(&mock)
        .await;
    // Least-connections picks the first (dead) backend for the first request.
    let dead = unreachable_backend_url();
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            backend_urls: vec![dead, mock.uri()],
            backend_connect_failover: true,
            admission_max_inflight: 4,
            ..Default::default()
        },
    );
    let response = app.oneshot(chat_request(hello_body())).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    mock.verify().await;
}

#[tokio::test]
async fn connect_error_is_not_retried_without_failover() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(0)
        .mount(&mock)
        .await;
    let dead = unreachable_backend_url();
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            backend_urls: vec![dead, mock.uri()],
            ..Default::default()
        },
    );
    let response = app.oneshot(chat_request(hello_body())).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let json = json_body(response).await;
    assert_eq!(json["error"]["type"], "upstream_unreachable");
    mock.verify().await;
}

#[tokio::test]
async fn failover_never_retries_an_engine_rejection() {
    let saturated = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string(QUEUE_FULL_BODY))
        .expect(1)
        .mount(&saturated)
        .await;
    let idle = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(0)
        .mount(&idle)
        .await;
    let app = build_gateway(
        &saturated.uri(),
        GatewayOptions {
            backend_urls: vec![saturated.uri(), idle.uri()],
            backend_connect_failover: true,
            map_queue_full_to_429: true,
            ..Default::default()
        },
    );
    let response = app.oneshot(chat_request(hello_body())).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    saturated.verify().await;
    idle.verify().await;
}

#[tokio::test]
async fn failover_with_no_other_backend_is_a_typed_502() {
    let dead = unreachable_backend_url();
    let app = build_gateway(
        &dead,
        GatewayOptions {
            backend_urls: vec![dead.clone()],
            backend_connect_failover: true,
            ..Default::default()
        },
    );
    let response = app.oneshot(chat_request(hello_body())).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let json = json_body(response).await;
    assert_eq!(json["error"]["type"], "upstream_unreachable");
}

#[tokio::test]
async fn streaming_request_holds_its_budget_slot_until_the_stream_ends() {
    let (backend, handle) = spawn_silent_then_stream_backend(
        Duration::from_millis(900),
        vec![
            "data: {\"id\":\"chatcmpl-adm\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl-adm\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
            "data: [DONE]\n\n",
        ],
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            admission_max_inflight: 1,
            ..Default::default()
        },
    );
    let stream_body = serde_json::json!({
        "model": "test-model",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}]
    });

    // Headers of the first stream arrive at once; its slot stays taken while
    // the upstream is still silent.
    let first = app
        .clone()
        .oneshot(chat_request(stream_body.clone()))
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let second = app
        .clone()
        .oneshot(chat_request(stream_body.clone()))
        .await
        .unwrap();
    assert_overloaded(second).await;

    // Drain the first stream; the slot is released when it ends.
    let mut body = first.into_body();
    let mut seen = String::new();
    while let Some(frame) = body.frame().await {
        if let Ok(data) = frame.unwrap().into_data() {
            seen.push_str(&String::from_utf8_lossy(&data));
        }
    }
    assert!(seen.contains("data: [DONE]"), "{seen}");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let third = app.oneshot(chat_request(stream_body)).await.unwrap();
    assert_eq!(third.status(), StatusCode::OK);
    handle.abort();
}

#[tokio::test]
async fn engine_rejection_after_the_peek_window_still_reaches_admission() {
    // The engine takes longer than the peek window to answer, then rejects
    // with an error event on the committed 200 stream.
    let (backend, handle) = spawn_silent_then_stream_backend(
        Duration::from_millis(1300),
        vec!["data: {\"error\":{\"object\":\"error\",\"message\":\"The request queue is full.\",\"type\":\"abort\",\"code\":503}}\n\n"],
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            admission_max_inflight: 8,
            map_queue_full_to_429: true,
            stream_error_peek_ms: 1000,
            ..Default::default()
        },
    );
    let stream_body = serde_json::json!({
        "model": "test-model",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}]
    });
    let first = app
        .clone()
        .oneshot(chat_request(stream_body.clone()))
        .await
        .unwrap();
    assert_eq!(
        first.status(),
        StatusCode::OK,
        "peek timed out: status is committed"
    );
    let mut body = first.into_body();
    let mut seen = String::new();
    while let Some(frame) = body.frame().await {
        if let Ok(data) = frame.unwrap().into_data() {
            seen.push_str(&String::from_utf8_lossy(&data));
        }
    }
    assert!(seen.contains("queue is full"), "{seen}");
    tokio::time::sleep(Duration::from_millis(100)).await;

    // The only backend is now known to be saturated: refused before dispatch.
    let second = app.oneshot(chat_request(stream_body)).await.unwrap();
    assert_overloaded(second).await;
    handle.abort();
}

// ---------------------------------------------------------------------------
// Engine load view: placement and admission follow the engines' queues
// ---------------------------------------------------------------------------

fn metrics_body(running: u32, queued: u32) -> String {
    format!(
        "sglang:num_running_reqs{{model_name=\"m\"}} {running}.0\nsglang:num_queue_reqs{{model_name=\"m\"}} {queued}.0\n"
    )
}

async fn mount_engine(mock: &MockServer, running: u32, queued: u32, expected_chats: u64) {
    Mock::given(method("GET"))
        .and(path("/v1/metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_string(metrics_body(running, queued)))
        .mount(mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(expected_chats)
        .mount(mock)
        .await;
}

#[tokio::test]
async fn a_queueing_engine_is_steered_around_and_a_fleet_wide_queue_refuses() {
    let busy = MockServer::start().await;
    let idle = MockServer::start().await;
    // busy is first in the pool, so least-connections alone would pick it.
    mount_engine(&busy, 20, 3, 0).await;
    mount_engine(&idle, 2, 0, 2).await;
    let app = build_gateway(
        &busy.uri(),
        GatewayOptions {
            backend_urls: vec![busy.uri(), idle.uri()],
            backend_probe_urls: vec![busy.uri(), idle.uri()],
            admission_max_inflight: 8,
            ..Default::default()
        },
    );
    tokio::time::sleep(Duration::from_millis(400)).await; // a few polls

    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(chat_request(hello_body()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    busy.verify().await;
    idle.verify().await;

    // Now the idle host queues too: nothing has room, refuse before dispatch.
    idle.reset().await;
    mount_engine(&idle, 20, 1, 0).await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    let response = app.oneshot(chat_request(hello_body())).await.unwrap();
    assert_overloaded(response).await;
    idle.verify().await;
}

// ---------------------------------------------------------------------------
// Long-context tier: oversized prompts are placed on their own backends
// ---------------------------------------------------------------------------

/// Estimated-input threshold used below. A body of `bytes` text estimates
/// `bytes / 4` tokens, which the 1.2 safety factor compares against this, so
/// 4000 bytes (1200) crosses it and 400 bytes (120) does not.
const ABOVE_TOKENS: u64 = 1_000;

async fn mount_chat(mock: &MockServer, expected: u64) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(expected)
        .mount(mock)
        .await;
}

fn sized_body(bytes: usize) -> serde_json::Value {
    serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "x".repeat(bytes)}]
    })
}

/// Turn `n` of one append-only conversation: the same leading messages (the
/// affinity key) with a history that is far above the threshold from the
/// first follow-up on.
fn conversation(turn: usize) -> serde_json::Value {
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": "be brief"}),
        serde_json::json!({"role": "user", "content": "start"}),
    ];
    for i in 0..turn {
        messages.push(serde_json::json!({"role": "assistant", "content": "y".repeat(4_000)}));
        messages.push(serde_json::json!({"role": "user", "content": format!("more {i}")}));
    }
    serde_json::json!({"model": "test-model", "messages": messages})
}

#[tokio::test]
async fn oversized_requests_go_to_the_long_tier_and_the_rest_to_the_base_fleet() {
    let base = MockServer::start().await;
    let long = MockServer::start().await;
    mount_chat(&base, 1).await;
    mount_chat(&long, 1).await;
    let app = build_gateway(
        &base.uri(),
        GatewayOptions {
            backend_urls: vec![base.uri()],
            backend_long_context_urls: vec![long.uri()],
            long_context_above_tokens: ABOVE_TOKENS,
            ..Default::default()
        },
    );
    for bytes in [4_000, 400] {
        let response = app
            .clone()
            .oneshot(chat_request(sized_body(bytes)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    base.verify().await;
    long.verify().await;
}

#[tokio::test]
async fn a_conversation_that_grows_past_the_threshold_moves_and_stays_there() {
    let base = MockServer::start().await;
    let long = MockServer::start().await;
    let long_peer = MockServer::start().await;
    mount_chat(&base, 1).await;
    // One slow unrelated request parks on the long host the conversation
    // moves onto, so plain least-connections would send the turn after it to
    // the idle peer instead.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_partial_json(
            serde_json::json!({"user": "holder"}),
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_completion_json())
                .set_delay(Duration::from_millis(700)),
        )
        .expect(1)
        .mount(&long)
        .await;
    mount_chat(&long, 2).await;
    mount_chat(&long_peer, 0).await;
    let app = build_gateway(
        &base.uri(),
        GatewayOptions {
            backend_urls: vec![base.uri()],
            backend_long_context_urls: vec![long.uri(), long_peer.uri()],
            long_context_above_tokens: ABOVE_TOKENS,
            backend_conversation_affinity: true,
            ..Default::default()
        },
    );
    // The first turn is short: it goes to the base fleet and pins there. The
    // second is above the threshold, so the pin is outside its tier — it is
    // placed on the first long host and re-pinned onto it.
    for turn in 0..2 {
        let response = app
            .clone()
            .oneshot(chat_request(conversation(turn)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let holder = tokio::spawn({
        let app = app.clone();
        async move {
            let mut body = sized_body(4_000);
            body["user"] = "holder".into();
            app.oneshot(chat_request(body)).await.unwrap()
        }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    // The third turn follows the pin onto the busy host rather than the idle
    // peer: its prefix cache is there and a move would re-prefill it.
    let response = app
        .clone()
        .oneshot(chat_request(conversation(2)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(holder.await.unwrap().status(), StatusCode::OK);
    base.verify().await;
    long.verify().await;
    long_peer.verify().await;
}

#[tokio::test]
async fn a_tier_without_a_healthy_backend_falls_back_to_the_other_one() {
    let live = MockServer::start().await;
    let dead = unreachable_backend_url();
    // The long tier is down...
    let app = build_gateway(
        &live.uri(),
        GatewayOptions {
            backend_urls: vec![live.uri()],
            backend_long_context_urls: vec![dead],
            long_context_above_tokens: ABOVE_TOKENS,
            backend_connect_failover: true,
            admission_max_inflight: 4,
            ..Default::default()
        },
    );
    mount_chat(&live, 2).await;
    // ...the first oversized request fails over out of its tier as soon as
    // that host leaves the rotation, and the next one is placed on the base
    // fleet straight away. Neither is refused.
    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(chat_request(sized_body(4_000)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    live.verify().await;
    live.reset().await;

    // And the other way around: with the base fleet gone, a short request is
    // served by the long-context host.
    let dead = unreachable_backend_url();
    let app = build_gateway(
        &live.uri(),
        GatewayOptions {
            backend_urls: vec![dead],
            backend_long_context_urls: vec![live.uri()],
            long_context_above_tokens: ABOVE_TOKENS,
            backend_connect_failover: true,
            admission_max_inflight: 4,
            ..Default::default()
        },
    );
    mount_chat(&live, 2).await;
    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(chat_request(sized_body(400)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    live.verify().await;
}

#[tokio::test]
async fn a_full_long_tier_refuses_while_the_base_fleet_keeps_serving() {
    let base = MockServer::start().await;
    let long = MockServer::start().await;
    mount_chat(&base, 1).await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_completion_json())
                .set_delay(Duration::from_millis(700)),
        )
        .expect(1)
        .mount(&long)
        .await;
    // Budget 2 over two hosts: one lane request each.
    let app = build_gateway(
        &base.uri(),
        GatewayOptions {
            backend_urls: vec![base.uri()],
            backend_long_context_urls: vec![long.uri()],
            long_context_above_tokens: ABOVE_TOKENS,
            admission_max_inflight: 2,
            ..Default::default()
        },
    );
    let first = tokio::spawn({
        let app = app.clone();
        async move { app.oneshot(chat_request(sized_body(4_000))).await.unwrap() }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    // The long host holds its whole share: the next oversized request is
    // refused instead of spilling its prefill onto the base fleet. (The 429
    // body carries only the generic `overloaded` type; which refusal it was
    // lives in the log line and in `admission_rejections_total{reason}`,
    // neither observable from here.)
    let second = app
        .clone()
        .oneshot(chat_request(sized_body(4_000)))
        .await
        .unwrap();
    assert_overloaded(second).await;
    // ...which keeps serving short requests throughout.
    let third = app.oneshot(chat_request(sized_body(400))).await.unwrap();
    assert_eq!(third.status(), StatusCode::OK);
    assert_eq!(first.await.unwrap().status(), StatusCode::OK);
    base.verify().await;
    long.verify().await;
}

#[tokio::test]
async fn a_token_id_prompt_is_routed_by_its_exact_length() {
    let base = MockServer::start().await;
    let long = MockServer::start().await;
    for mock in [&base, &long] {
        Mock::given(method("POST"))
            .and(path("/v1/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-1", "object": "text_completion", "model": "test-model",
                "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(mock)
            .await;
    }
    let app = build_gateway(
        &base.uri(),
        GatewayOptions {
            backend_urls: vec![base.uri()],
            backend_long_context_urls: vec![long.uri()],
            long_context_above_tokens: ABOVE_TOKENS,
            ..Default::default()
        },
    );
    // Token ids count exactly and carry no safety factor, and the bound is
    // strict: 1001 ids is the long tier, 1000 is not.
    for ids in [1_001u32, 1_000] {
        let body = serde_json::json!({
            "model": "test-model",
            "prompt": (0..ids).collect::<Vec<u32>>()
        });
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("authorization", "Bearer test-token")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    base.verify().await;
    long.verify().await;
}

// ---- Models document ----

async fn get_models(app: axum::Router) -> (StatusCode, serde_json::Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn models_document_is_reduced_to_this_model_and_carries_capacity() {
    let engine = MockServer::start().await;
    let source = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [
                {"id": "other-model", "name": "Other", "is_ready": true},
                {"id": "test-model", "name": "Test Model", "is_ready": true,
                 "pricing": {"prompt": "0.00000015"}, "openrouter": {"slug": "test-model"}}
            ]
        })))
        .expect(1)
        .mount(&source)
        .await;

    let app = build_gateway(
        &engine.uri(),
        GatewayOptions {
            models_document_url: Some(format!("{}/v1/models", source.uri())),
            capacity_requests_per_minute: 150,
            admission_max_inflight: 48,
            admission_start_inflight: Some(32),
            ..Default::default()
        },
    );
    let (status, body) = get_models(app).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1, "{body}");
    assert_eq!(data[0]["id"], "test-model");
    assert_eq!(data[0]["name"], "Test Model");
    assert_eq!(data[0]["pricing"]["prompt"], "0.00000015");
    assert_eq!(data[0]["openrouter"]["slug"], "test-model");
    assert_eq!(
        data[0]["capacity"],
        serde_json::json!([
            {"type": "concurrency", "unit": "request", "value": 48},
            {"type": "request", "unit": "request", "per": "minute", "value": 150}
        ])
    );
}

#[tokio::test]
async fn models_document_without_a_budget_or_rate_declares_no_capacity() {
    let engine = MockServer::start().await;
    let source = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [{"id": "test-model", "name": "Test Model"}]
        })))
        .mount(&source)
        .await;
    let app = build_gateway(
        &engine.uri(),
        GatewayOptions {
            models_document_url: Some(format!("{}/v1/models", source.uri())),
            ..Default::default()
        },
    );
    let (status, body) = get_models(app).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"][0].get("capacity").is_none(), "{body}");
}

#[tokio::test]
async fn models_document_falls_back_to_the_engine_list_when_the_source_fails() {
    let engine = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [{"id": "test-model", "object": "model", "owned_by": "sglang"}]
        })))
        .expect(2)
        .mount(&engine)
        .await;
    let source = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&source)
        .await;

    // Source down: the engine list is served.
    let app = build_gateway(
        &engine.uri(),
        GatewayOptions {
            models_document_url: Some(format!("{}/v1/models", source.uri())),
            admission_max_inflight: 48,
            ..Default::default()
        },
    );
    let (status, body) = get_models(app).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"][0]["owned_by"], "sglang");
    assert!(body["data"][0].get("capacity").is_none());

    // Source serving a document that lacks this model: same fallback.
    let empty = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [{"id": "other-model"}]
        })))
        .mount(&empty)
        .await;
    let app = build_gateway(
        &engine.uri(),
        GatewayOptions {
            models_document_url: Some(format!("{}/v1/models", empty.uri())),
            ..Default::default()
        },
    );
    let (status, body) = get_models(app).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"][0]["owned_by"], "sglang");
}

// ---- Reasoning switch ----

/// Matches a chat body that carries no `reasoning_effort` at all.
struct NoReasoningEffort;

impl wiremock::Match for NoReasoningEffort {
    fn matches(&self, request: &wiremock::Request) -> bool {
        serde_json::from_slice::<serde_json::Value>(&request.body)
            .map(|body| body.get("reasoning_effort").is_none())
            .unwrap_or(false)
    }
}

#[tokio::test]
async fn reasoning_object_becomes_reasoning_effort_in_gateway_mode() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_partial_json(serde_json::json!({
            "reasoning_effort": "low",
            "reasoning": {"enabled": false, "effort": "low"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(2)
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            backend_token: Some("backend-secret".to_string()),
            reasoning_off_effort: Some("low".to_string()),
            ..Default::default()
        },
    );
    // The aggregator's object, and the same intent as an explicit value the
    // model cannot honour cleanly: both become the configured off effort.
    for body in [
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning": {"enabled": false}
        }),
        serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "none",
            "reasoning": {"enabled": false, "effort": "none"}
        }),
    ] {
        let response = app.clone().oneshot(chat_request(body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    mock.verify().await;
}

#[tokio::test]
async fn reasoning_object_is_left_alone_outside_gateway_mode() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(NoReasoningEffort)
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
        .mount(&mock)
        .await;
    let app = build_gateway(&mock.uri(), GatewayOptions::default());
    let response = app
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning": {"enabled": false}
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    mock.verify().await;
}

// ---------------------------------------------------------------------------
// Committing the stream before the upstream answers (long prefill)
// ---------------------------------------------------------------------------

/// Backend that withholds its response *headers* for `delay` — the shape of an
/// engine prefilling a long prompt, which sends nothing at all until it has its
/// first token — and then answers with `status` and `body`.
async fn spawn_delayed_response_backend(
    delay: Duration,
    status: u16,
    content_type: &'static str,
    body: &'static str,
) -> (String, tokio::task::JoinHandle<()>) {
    use axum::routing::post;
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        post(move || async move {
            tokio::time::sleep(delay).await;
            axum::response::Response::builder()
                .status(status)
                .header("content-type", content_type)
                .body(axum::body::Body::from(body))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}"), handle)
}

fn stream_request() -> Request<Body> {
    chat_request(serde_json::json!({
        "model": "test-model",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}]
    }))
}

const ONE_TOKEN_STREAM: &str = concat!(
    "data: {\"id\":\"chatcmpl-c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
    "data: {\"id\":\"chatcmpl-c\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
    "data: [DONE]\n\n",
);

/// Collect the frames of a committed stream in arrival order.
async fn stream_frames(response: axum::response::Response) -> Vec<String> {
    let mut body = response.into_body();
    let mut frames = Vec::new();
    while let Some(frame) = body.frame().await {
        if let Ok(data) = frame.unwrap().into_data() {
            frames.push(String::from_utf8_lossy(&data).to_string());
        }
    }
    frames
}

#[tokio::test]
async fn stream_is_committed_with_keepalives_while_the_engine_prefills() {
    let (backend, handle) = spawn_delayed_response_backend(
        Duration::from_millis(2500),
        200,
        "text/event-stream",
        ONE_TOKEN_STREAM,
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            stream_commit_ms: 300,
            sse_keepalive_secs: 1,
            stream_error_peek_ms: 1000,
            ..Default::default()
        },
    );

    let started = std::time::Instant::now();
    let response = app.oneshot(stream_request()).await.unwrap();
    let committed_after = started.elapsed();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    // Committed on the window, not on the engine: headers are out long before
    // the backend has anything to say.
    assert!(
        committed_after < Duration::from_millis(1500),
        "expected the response before the backend answered, took {committed_after:?}"
    );

    let frames = stream_frames(response).await;
    let joined = frames.concat();
    assert!(joined.contains("data: [DONE]"), "{joined}");
    assert!(joined.contains("\"content\":\"hi\""), "{joined}");
    let first_data = frames
        .iter()
        .position(|f| f.starts_with("data:"))
        .expect("a data frame");
    let keepalives = frames[..first_data]
        .iter()
        .filter(|f| f.as_str() == ": keep-alive\n\n")
        .count();
    assert!(
        keepalives >= 1,
        "expected keep-alives during the 2.5s prefill, got {keepalives}: {frames:?}"
    );
    handle.abort();
}

#[tokio::test]
async fn an_upstream_failure_after_the_commit_becomes_a_terminal_stream_event() {
    // The trade the commit window makes: past it, the status is already 200,
    // so the engine's rejection has to travel as an SSE error event.
    let (backend, handle) = spawn_delayed_response_backend(
        Duration::from_millis(1200),
        503,
        "application/json",
        QUEUE_FULL_BODY,
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            stream_commit_ms: 200,
            map_queue_full_to_429: true,
            sse_keepalive_secs: 1,
            ..Default::default()
        },
    );

    let response = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let joined = stream_frames(response).await.concat();
    let event = joined
        .lines()
        .find(|l| l.starts_with("data: {"))
        .expect("an error event");
    let parsed: serde_json::Value =
        serde_json::from_str(event.trim_start_matches("data: ")).unwrap();
    assert_eq!(parsed["error"]["message"], "The request queue is full.");
    assert_eq!(parsed["error"]["type"], "overloaded");
    assert!(joined.ends_with("data: [DONE]\n\n"), "{joined}");
    handle.abort();
}

#[tokio::test]
async fn an_upstream_failure_inside_the_window_is_still_a_status_code() {
    let (backend, handle) = spawn_delayed_response_backend(
        Duration::from_millis(50),
        503,
        "application/json",
        QUEUE_FULL_BODY,
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            stream_commit_ms: 3000,
            map_queue_full_to_429: true,
            ..Default::default()
        },
    );

    let response = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("2")
    );
    handle.abort();
}

#[tokio::test]
async fn without_a_commit_window_a_slow_upstream_still_decides_the_status() {
    // Every in-CVM deployment: nothing is sent until the upstream has answered,
    // however long that takes.
    let (backend, handle) = spawn_delayed_response_backend(
        Duration::from_millis(700),
        503,
        "application/json",
        QUEUE_FULL_BODY,
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            map_queue_full_to_429: true,
            sse_keepalive_secs: 1,
            ..Default::default()
        },
    );

    let started = std::time::Instant::now();
    let response = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        started.elapsed() >= Duration::from_millis(600),
        "the handler must wait for the upstream when no window is set"
    );
    handle.abort();
}

// ---------------------------------------------------------------------------
// First-token deadline
// ---------------------------------------------------------------------------

/// Backend that never answers inside a test's lifetime, reporting when it
/// accepts a request and when its handler future is dropped — which here only
/// happens when the gateway hangs up on the in-flight request.
struct NeverAnsweringBackend {
    url: String,
    arrivals: tokio::sync::mpsc::UnboundedReceiver<()>,
    hang_ups: tokio::sync::mpsc::UnboundedReceiver<()>,
    handle: tokio::task::JoinHandle<()>,
}

impl NeverAnsweringBackend {
    async fn wait(what: &str, signal: &mut tokio::sync::mpsc::UnboundedReceiver<()>) {
        tokio::time::timeout(Duration::from_secs(5), signal.recv())
            .await
            .unwrap_or_else(|_| panic!("the backend never reported {what}"))
            .expect("the reporting channel stays open");
    }
}

async fn spawn_never_answering_backend() -> NeverAnsweringBackend {
    use axum::routing::post;
    let (arrived, arrivals) = tokio::sync::mpsc::unbounded_channel();
    let (hung_up, hang_ups) = tokio::sync::mpsc::unbounded_channel();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let arrived = arrived.clone();
            let hung_up = hung_up.clone();
            async move {
                struct ReportOnDrop(tokio::sync::mpsc::UnboundedSender<()>);
                impl Drop for ReportOnDrop {
                    fn drop(&mut self) {
                        let _ = self.0.send(());
                    }
                }
                let _report = ReportOnDrop(hung_up);
                let _ = arrived.send(());
                tokio::time::sleep(Duration::from_secs(300)).await;
                axum::response::Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(axum::body::Body::from(ONE_TOKEN_STREAM))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    NeverAnsweringBackend {
        url: format!("http://{addr}"),
        arrivals,
        hang_ups,
        handle,
    }
}

/// Backend that answers its `200 text/event-stream` headers at once and sends
/// its first SSE event only `delay` later — what an engine that opens the
/// stream before it has generated anything looks like on the wire (vLLM), and
/// what defeats a deadline measured on the response headers alone.
async fn spawn_late_first_event_backend(delay: Duration) -> (String, tokio::task::JoinHandle<()>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut request = [0_u8; 8192];
                let _ = socket.read(&mut request).await;
                // Headers now, body later: the two are separate writes, so the
                // gateway sees the 200 long before the first event.
                if socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
                    )
                    .await
                    .is_err()
                {
                    return;
                }
                tokio::time::sleep(delay).await;
                let chunked = format!(
                    "{:X}\r\n{}\r\n0\r\n\r\n",
                    ONE_TOKEN_STREAM.len(),
                    ONE_TOKEN_STREAM
                );
                let _ = socket.write_all(chunked.as_bytes()).await;
            });
        }
    });
    (format!("http://{addr}"), handle)
}

/// A prompt long enough to be estimated at about 1,000 tokens (bytes / 4).
fn long_stream_request() -> Request<Body> {
    chat_request(serde_json::json!({
        "model": "test-model",
        "stream": true,
        "messages": [{"role": "user", "content": "x".repeat(4_000)}]
    }))
}

#[tokio::test]
async fn a_missed_first_token_deadline_is_a_429_instead_of_a_committed_200() {
    let (backend, handle) = spawn_delayed_response_backend(
        Duration::from_millis(2500),
        200,
        "text/event-stream",
        ONE_TOKEN_STREAM,
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            first_token_deadline_ms: 400,
            // A request under a deadline is not committed early, so this
            // window never fires for it.
            stream_commit_ms: 300,
            sse_keepalive_secs: 1,
            stream_error_peek_ms: 1_000,
            ..Default::default()
        },
    );

    let started = std::time::Instant::now();
    let response = app.oneshot(stream_request()).await.unwrap();
    let answered = started.elapsed();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("2")
    );
    // Answered on the deadline, not on the backend.
    assert!(
        answered >= Duration::from_millis(350) && answered < Duration::from_millis(1500),
        "expected a refusal around the deadline, took {answered:?}"
    );
    let body = json_body(response).await;
    assert_eq!(body["error"]["type"], "overloaded");
    handle.abort();
}

#[tokio::test]
async fn a_deadline_refusal_drops_the_upstream_and_releases_the_admission_slot() {
    let mut backend = spawn_never_answering_backend().await;
    let app = build_gateway(
        &backend.url,
        GatewayOptions {
            first_token_deadline_ms: 400,
            admission_max_inflight: 1,
            ..Default::default()
        },
    );

    let response = app.clone().oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    // The engine is not left generating for nobody: the upstream request is
    // dropped, which closes the connection the backend is serving.
    NeverAnsweringBackend::wait("the request", &mut backend.arrivals).await;
    NeverAnsweringBackend::wait("a hang-up", &mut backend.hang_ups).await;

    // And the budget slot came back: with a budget of one, the next request is
    // admitted and waits for its own deadline instead of being refused on the
    // spot by admission.
    let started = std::time::Instant::now();
    let response = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        started.elapsed() >= Duration::from_millis(350),
        "the second request was refused by admission, not by its deadline"
    );
    backend.handle.abort();
}

#[tokio::test]
async fn a_first_token_inside_the_deadline_streams_as_usual() {
    let (backend, handle) = spawn_delayed_response_backend(
        Duration::from_millis(100),
        200,
        "text/event-stream",
        ONE_TOKEN_STREAM,
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            first_token_deadline_ms: 2_000,
            stream_error_peek_ms: 1_000,
            ..Default::default()
        },
    );

    let response = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let joined = stream_frames(response).await.concat();
    assert!(joined.contains("\"content\":\"hi\""), "{joined}");
    assert!(joined.ends_with("data: [DONE]\n\n"), "{joined}");
    handle.abort();
}

#[tokio::test]
async fn the_deadline_grows_with_the_estimated_prompt() {
    // The long-context tier is off here, so this also covers the estimate
    // being computed for the deadline alone.
    let (backend, handle) = spawn_delayed_response_backend(
        Duration::from_millis(900),
        200,
        "text/event-stream",
        ONE_TOKEN_STREAM,
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            first_token_deadline_ms: 300,
            first_token_deadline_per_1k_tokens_ms: 4_000,
            ..Default::default()
        },
    );

    // A short prompt is refused: 300 ms, and the engine takes 900.
    let response = app.clone().oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    // The same backend, with a prompt estimated at ~1,000 tokens: 4.3 s.
    let response = app.oneshot(long_stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    handle.abort();
}

#[tokio::test]
async fn a_deadline_above_the_cap_keeps_the_commit_window() {
    let (backend, handle) = spawn_delayed_response_backend(
        Duration::from_millis(2500),
        200,
        "text/event-stream",
        ONE_TOKEN_STREAM,
    )
    .await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            first_token_deadline_ms: 300,
            first_token_deadline_per_1k_tokens_ms: 4_000,
            first_token_deadline_max_ms: 2_000,
            stream_commit_ms: 300,
            sse_keepalive_secs: 1,
            ..Default::default()
        },
    );

    // The control, same backend: a short prompt stays under the cap, so its
    // 300 ms deadline applies and the 2.5 s engine misses it.
    let response = app.clone().oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    // ~1,000 estimated tokens → a 4.3 s deadline, above the cap: no deadline,
    // so this request is committed on the window and kept alive as before.
    let started = std::time::Instant::now();
    let response = app.oneshot(long_stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        started.elapsed() < Duration::from_millis(1500),
        "expected the commit window, not the engine"
    );
    let frames = stream_frames(response).await;
    assert!(frames.iter().any(|f| f == ": keep-alive\n\n"), "{frames:?}");
    assert!(frames.concat().contains("\"content\":\"hi\""), "{frames:?}");
    handle.abort();
}

#[tokio::test]
async fn early_headers_do_not_satisfy_the_deadline() {
    // The deadline is about the first generated event, not the status line.
    for peek_ms in [0, 1_000] {
        let (backend, handle) = spawn_late_first_event_backend(Duration::from_millis(2500)).await;
        let app = build_gateway(
            &backend,
            GatewayOptions {
                first_token_deadline_ms: 400,
                stream_error_peek_ms: peek_ms,
                ..Default::default()
            },
        );

        let started = std::time::Instant::now();
        let response = app.oneshot(stream_request()).await.unwrap();
        let answered = started.elapsed();
        assert_eq!(
            response.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "peek {peek_ms}: the 200 headers arrived at once, the event did not"
        );
        assert!(
            answered < Duration::from_millis(1500),
            "peek {peek_ms}: refused after {answered:?}, expected the deadline"
        );
        handle.abort();
    }
}

#[tokio::test]
async fn a_first_event_inside_the_deadline_is_delivered_once() {
    let (backend, handle) = spawn_late_first_event_backend(Duration::from_millis(200)).await;
    let app = build_gateway(
        &backend,
        GatewayOptions {
            first_token_deadline_ms: 3_000,
            ..Default::default()
        },
    );

    let response = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let joined = stream_frames(response).await.concat();
    // Re-attached exactly once, and the stream is complete.
    assert_eq!(joined.matches("\"content\":\"hi\"").count(), 1, "{joined}");
    assert!(joined.ends_with("data: [DONE]\n\n"), "{joined}");
    handle.abort();
}

/// Control for the two tests above: without a deadline the headers alone do
/// end the wait, which is exactly what made them defeat it.
#[tokio::test]
async fn without_a_deadline_early_headers_commit_the_stream_at_once() {
    let (backend, handle) = spawn_late_first_event_backend(Duration::from_millis(2500)).await;
    let app = build_gateway(&backend, GatewayOptions::default());

    let started = std::time::Instant::now();
    let response = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        started.elapsed() < Duration::from_millis(1500),
        "expected the status on the headers, took {:?}",
        started.elapsed()
    );
    handle.abort();
}

#[tokio::test]
async fn a_deadline_spent_before_dispatch_refuses_without_asking_an_engine() {
    let mock = MockServer::start().await;
    // Authentication counts against the deadline: a slow cloud-api round trip
    // can use the whole budget before the request would be dispatched.
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(700))
                .set_body_json(serde_json::json!({
                    "valid": true, "organization_id": "org", "workspace_id": "ws", "api_key_id": "k"
                })),
        )
        .mount(&mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(0)
        .mount(&mock)
        .await;
    let app = build_gateway(
        &mock.uri(),
        GatewayOptions {
            cloud_api_url: Some(mock.uri()),
            first_token_deadline_ms: 300,
            ..Default::default()
        },
    );

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer sk-live-customer")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "test-model",
                "stream": true,
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .unwrap(),
        ))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    mock.verify().await;
}

#[tokio::test]
async fn a_client_that_disconnects_before_the_deadline_is_still_a_disconnect() {
    let mut backend = spawn_never_answering_backend().await;
    let app = build_gateway(
        &backend.url,
        GatewayOptions {
            first_token_deadline_ms: 5_000,
            ..Default::default()
        },
    );

    // Wait until the engine actually holds the request, then let the client go
    // away — still well inside the 5 s deadline. The disconnect ends the
    // request there and then: the upstream is dropped at once instead of being
    // held until the deadline would have fired. (Which of the two paths logged
    // it is not visible from here.)
    let waiting = tokio::spawn(async move { app.oneshot(stream_request()).await });
    NeverAnsweringBackend::wait("the request", &mut backend.arrivals).await;
    waiting.abort();

    let gone = std::time::Instant::now();
    NeverAnsweringBackend::wait("a hang-up", &mut backend.hang_ups).await;
    assert!(
        gone.elapsed() < Duration::from_secs(2),
        "the upstream outlived the client disconnect"
    );
    backend.handle.abort();
}

// Four real HTTP stubs hold streams until the test explicitly ends them.
async fn borrowing_gateway() -> (
    axum::Router,
    AppState,
    tokio::sync::watch::Sender<u8>,
    Vec<tokio::task::JoinHandle<()>>,
) {
    let (release, receiver) = tokio::sync::watch::channel(0u8);
    let mut urls = Vec::new();
    let mut tasks = Vec::new();
    for _ in 0..4 {
        let receiver = receiver.clone();
        let serve = move || {
            let mut receiver = receiver.clone();
            async move {
                let (tx, rx) =
                    tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(4);
                tokio::spawn(async move {
                    let first = "data: {\"id\":\"synthetic\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
                    if tx.send(Ok(first.into())).await.is_err() {
                        return;
                    }
                    tokio::select! {
                        _ = tx.closed() => return,
                        _ = receiver.changed() => {},
                    }
                    let tail = if *receiver.borrow() == 2 {
                        "data: {\"error\":{\"message\":\"synthetic failure\",\"type\":\"api_error\",\"code\":500}}\n\ndata: [DONE]\n\n"
                    } else {
                        "data: {\"id\":\"synthetic\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\ndata: [DONE]\n\n"
                    };
                    let _ = tx.send(Ok(tail.into())).await;
                });
                axum::response::Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(
                        tokio_stream::wrappers::ReceiverStream::new(rx),
                    ))
                    .unwrap()
            }
        };
        let stub = axum::Router::new()
            .route("/v1/chat/completions", axum::routing::post(serve.clone()))
            .route("/v1/completions", axum::routing::post(serve));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        urls.push(format!("http://{}", listener.local_addr().unwrap()));
        tasks.push(tokio::spawn(async move {
            axum::serve(listener, stub).await.unwrap();
        }));
    }
    let (app, state) = build_gateway_with_state(
        &urls[0],
        GatewayOptions {
            backend_urls: urls[..3].to_vec(),
            backend_long_context_urls: vec![urls[3].clone()],
            long_context_above_tokens: ABOVE_TOKENS,
            admission_max_inflight: 48,
            admission_tier_borrowing: true,
            admission_long_max_inflight_per_host: 12,
            backend_connect_failover: true,
            ..Default::default()
        },
    );
    (app, state, release, tasks)
}

async fn borrowing_request(
    app: axum::Router,
    long: bool,
    completions: bool,
) -> axum::response::Response {
    let body = if completions {
        serde_json::json!({"model":"test-model","prompt":"x".repeat(if long {4000} else {4}),"stream":true})
    } else {
        let mut body = sized_body(if long { 4000 } else { 4 });
        body["stream"] = true.into();
        body
    };
    let mut request = chat_request(body);
    if completions {
        *request.uri_mut() = "/v1/completions".parse().unwrap();
    }
    app.oneshot(request).await.unwrap()
}

async fn assert_borrowing_released(state: &AppState) {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if state.admission.inflight() == 0
                && state
                    .backend_pool
                    .backends()
                    .iter()
                    .all(|b| b.lane_conns.load(std::sync::atomic::Ordering::Acquire) == 0)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("all global and backend slots released");
}

#[tokio::test]
async fn borrowing_serves_48_base_streams_and_releases_after_done() {
    let (app, state, release, tasks) = borrowing_gateway().await;
    let responses = futures_util::future::join_all(
        (0..48).map(|_| borrowing_request(app.clone(), false, false)),
    )
    .await;
    assert!(responses.iter().all(|r| r.status() == StatusCode::OK));
    assert_eq!(state.admission.inflight(), 48);
    for backend in &state.backend_pool.backends()[..3] {
        assert_eq!(
            backend
                .lane_conns
                .load(std::sync::atomic::Ordering::Acquire),
            16
        );
    }
    assert_overloaded(borrowing_request(app.clone(), false, false).await).await;
    assert_overloaded(borrowing_request(app.clone(), true, false).await).await;
    release.send(1).unwrap();
    for response in responses {
        let body = stream_frames(response).await.concat();
        assert!(
            body.contains("[DONE]") && body.contains("finish_reason") && body.contains("usage"),
            "{body}"
        );
    }
    assert_borrowing_released(&state).await;
    for task in tasks {
        task.abort();
    }
}

#[tokio::test]
async fn borrowing_caps_long_completions_and_releases_on_cancel() {
    let (app, state, _release, tasks) = borrowing_gateway().await;
    let mut held = Vec::new();
    for _ in 0..12 {
        let response = borrowing_request(app.clone(), true, true).await;
        assert_eq!(response.status(), StatusCode::OK);
        held.push(response);
    }
    assert_overloaded(borrowing_request(app.clone(), true, true).await).await;
    for _ in 0..36 {
        let response = borrowing_request(app.clone(), false, true).await;
        assert_eq!(response.status(), StatusCode::OK);
        held.push(response);
    }
    assert_eq!(state.admission.inflight(), 48);
    assert_overloaded(borrowing_request(app.clone(), false, true).await).await;
    drop(held);
    assert_borrowing_released(&state).await;
    for task in tasks {
        task.abort();
    }
}

#[tokio::test]
async fn borrowing_fallback_uses_destination_limit_and_late_error_releases() {
    let (app, state, release, tasks) = borrowing_gateway().await;
    // Simulate the entire base tier disappearing: base requests land on long,
    // whose bound remains 12 rather than inheriting the base allowance of 16.
    for backend in &state.backend_pool.backends()[..3] {
        backend
            .healthy
            .store(false, std::sync::atomic::Ordering::Release);
    }
    let mut responses = Vec::new();
    for _ in 0..12 {
        let response = borrowing_request(app.clone(), false, false).await;
        assert_eq!(response.status(), StatusCode::OK);
        responses.push(response);
    }
    assert_overloaded(borrowing_request(app.clone(), false, false).await).await;
    release.send(2).unwrap();
    for response in responses {
        assert!(stream_frames(response).await.concat().contains("error"));
    }
    assert_borrowing_released(&state).await;
    for task in tasks {
        task.abort();
    }
}

#[tokio::test]
async fn borrowing_connect_failover_keeps_destination_cap_and_one_permit() {
    let (app, state, release, mut tasks) = borrowing_gateway().await;
    let failed = tasks.remove(0);
    failed.abort();
    let _ = failed.await;
    for backend in &state.backend_pool.backends()[1..3] {
        backend
            .healthy
            .store(false, std::sync::atomic::Ordering::Release);
    }
    // The last nominally healthy base host refuses the TCP connection.
    let first = borrowing_request(app.clone(), false, false).await;
    assert_eq!(first.status(), StatusCode::OK);
    assert!(!state.backend_pool.backends()[0]
        .healthy
        .load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(state.admission.inflight(), 1);
    assert_eq!(
        state.backend_pool.backends()[0]
            .lane_conns
            .load(std::sync::atomic::Ordering::Acquire),
        0
    );
    let mut held = vec![first];
    for _ in 0..11 {
        let r = borrowing_request(app.clone(), false, false).await;
        assert_eq!(r.status(), StatusCode::OK);
        held.push(r);
    }
    assert_overloaded(borrowing_request(app.clone(), false, false).await).await;
    assert_eq!(
        state.backend_pool.backends()[3]
            .lane_conns
            .load(std::sync::atomic::Ordering::Acquire),
        12
    );
    release.send(1).unwrap();
    for r in held {
        assert!(stream_frames(r).await.concat().contains("[DONE]"));
    }
    assert_borrowing_released(&state).await;
    for task in tasks {
        task.abort();
    }
}
