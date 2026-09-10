//! Gateway-mode behaviors: the proxy fronting a fleet of inference-proxies
//! (dynamic membership from the model-proxy registry, a backend-only bearer,
//! modality policy, a strictly stateless route surface, keep-alives).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware;
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use vllm_proxy_rs::backend_discovery::{poll_once, BackendDiscoveryConfig};
use vllm_proxy_rs::backend_pool::BackendPool;
use vllm_proxy_rs::*;

#[derive(Default)]
struct GatewayOptions {
    /// Explicit pool members; `None` means "just the mock URL", `Some(vec![])`
    /// an empty pool.
    backend_urls: Option<Vec<String>>,
    backend_token: Option<String>,
    backend_health_path: Option<String>,
    non_tee_deployment: bool,
    map_queue_full_to_429: bool,
    stream_error_peek_ms: u64,
    rejected_content_part_types: Vec<String>,
    catch_all_disabled: bool,
    sse_keepalive_secs: u64,
}

fn build_gateway(mock_url: &str, options: GatewayOptions) -> (axum::Router, Arc<BackendPool>) {
    let base = mock_url.trim_end_matches('/');
    let backend_urls = options
        .backend_urls
        .unwrap_or_else(|| vec![mock_url.to_string()]);
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
        cloud_api_url: None,
        cloud_api_auth_max_attempts: 1,
        cloud_api_auth_initial_backoff_ms: 0,
        cloud_api_auth_timeout_secs: 5,
        cloud_api_usage_token: None,
        compose_manager_url: None,
        tls_cert_path: None,
        timeout_secs: 30,
        stream_idle_timeout_secs: 0,
        timeout_tokenize_secs: 5,
        openai_chat_compatibility_check_enabled: false,
        startup_check_retries: 1,
        startup_check_retry_delay_secs: 0,
        startup_check_timeout_secs: 5,
        backend_urls: backend_urls.clone(),
        vllm_data_parallel_size: None,
        backend_conversation_affinity: false,
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
        backend_health_path: options
            .backend_health_path
            .unwrap_or_else(|| "/health".to_string()),
        non_tee_deployment: options.non_tee_deployment,
        map_queue_full_to_429: options.map_queue_full_to_429,
        stream_error_peek_ms: options.stream_error_peek_ms,
        backend_discovery: None,
        rejected_content_part_types: options.rejected_content_part_types,
        catch_all_disabled: options.catch_all_disabled,
        sse_keepalive_secs: options.sse_keepalive_secs,
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

    // Mirror main.rs: the backend bearer is a default header on a dedicated
    // client, never on the general-purpose one.
    let http_client = reqwest::Client::new();
    let backend_client = match &options.backend_token {
        Some(token) => {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
            );
            reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .unwrap()
        }
        None => http_client.clone(),
    };

    let backend_pool = Arc::new(BackendPool::new(backend_urls));
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
        backend_pool: backend_pool.clone(),
        ohttp_gateway: None,
        ohttp_attestation_ed25519: None,
        fusion_caches: Arc::new(fusion::FusionCaches::default()),
        vllm_dp_affinity: Arc::new(vllm_dp_affinity::VllmDpAffinity::new(None, 1_200)),
        backend_affinity: Arc::new(backend_affinity::BackendConversationAffinity::new(
            false, 8, 1_200,
        )),
    };
    let rate_limit_state = rate_limit::RateLimitState {
        limiter: rate_limit::build_rate_limiter(100, 200),
        trust_proxy_headers: true,
    };
    let app = routes::build_router()
        .layer(middleware::from_fn(rate_limit::rate_limit_middleware))
        .layer(axum::Extension(rate_limit_state))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state);
    (app, backend_pool)
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

    let (app, _) = build_gateway(
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

    let (app, _) = build_gateway(&mock.uri(), GatewayOptions::default());
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

    let (app, _) = build_gateway(
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

    // The same route alias (trailing slash → catch-all) is gated too.
    let alias = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions/")
        .header("authorization", "Bearer test-token")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&video).unwrap()))
        .unwrap();
    let response = app.clone().oneshot(alias).await.unwrap();
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
// VLLM_PROXY_CATCH_ALL_DISABLED
// ---------------------------------------------------------------------------

#[tokio::test]
async fn catch_all_disabled_refuses_undeclared_paths() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
        .mount(&mock)
        .await;
    // Nothing else may reach the backend.
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/conversations"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&mock)
        .await;

    let (app, _) = build_gateway(
        &mock.uri(),
        GatewayOptions {
            catch_all_disabled: true,
            ..Default::default()
        },
    );

    for (m, uri) in [
        ("POST", "/v1/responses"),
        ("GET", "/v1/conversations"),
        ("POST", "/v1/anything/else"),
        ("GET", "/health"),
    ] {
        let request = Request::builder()
            .method(m)
            .uri(uri)
            .header("authorization", "Bearer test-token")
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{m} {uri}");
        let body = json_body(response).await;
        assert_eq!(body["error"]["type"], "not_found", "{m} {uri}");
    }

    // Declared routes are unaffected.
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
// /healthz in a non-TEE deployment and with an empty pool
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

    let (app, _) = build_gateway(
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
    let (app, _) = build_gateway(&mock.uri(), GatewayOptions::default());
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

#[tokio::test]
async fn empty_pool_is_503_on_inference_and_unhealthy_on_healthz() {
    let mock = MockServer::start().await;
    let (app, pool) = build_gateway(
        &mock.uri(),
        GatewayOptions {
            backend_urls: Some(Vec::new()),
            non_tee_deployment: true,
            ..Default::default()
        },
    );
    assert!(pool.is_empty());

    let response = app
        .clone()
        .oneshot(chat_request(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = json_body(response).await;
    assert_eq!(body["error"]["type"], "service_unavailable");

    let response = app
        .clone()
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
    assert_eq!(body["checks"]["backend"], "no_backends");

    // Discovery later fills the pool; the same app serves without restart.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_completion_json()))
        .expect(1)
        .mount(&mock)
        .await;
    pool.set_backends(vec![(mock.uri(), true)]);
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
// Registry discovery
// ---------------------------------------------------------------------------

fn backend_url(handle: &str) -> String {
    format!("http://glm-b{handle}.backend.test")
}

fn discovery_config(registry: &MockServer) -> BackendDiscoveryConfig {
    BackendDiscoveryConfig {
        url: format!("{}/backends/list?domain=glm.test", registry.uri()),
        token: Some("registry-token".to_string()),
        url_template: "http://glm-b{handle}.backend.test".to_string(),
        interval_secs: 5,
        timeout_secs: 3,
    }
}

#[tokio::test]
async fn discovery_reconciles_pool_from_registry_listing() {
    let registry = MockServer::start().await;
    let pool = BackendPool::new(Vec::new());
    let client = reqwest::Client::new();
    let cfg = discovery_config(&registry);

    // First listing: two backends, one currently unhealthy per the registry.
    Mock::given(method("GET"))
        .and(path("/backends/list"))
        .and(query_param("domain", "glm.test"))
        .and(header("authorization", "Bearer registry-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "domain": "glm.test",
            "backends": [
                {"handle": "bbbbbbbbbbbb", "healthy": false},
                {"handle": "aaaaaaaaaaaa", "healthy": true}
            ]
        })))
        .up_to_n_times(1)
        .mount(&registry)
        .await;
    let change = poll_once(&client, &cfg, &pool).await.unwrap();
    assert_eq!(
        change.added,
        vec![backend_url("aaaaaaaaaaaa"), backend_url("bbbbbbbbbbbb")]
    );
    assert!(change.removed.is_empty());
    let backends = pool.backends();
    assert_eq!(backends.len(), 2);
    assert!(backends[0].healthy.load(Ordering::Relaxed));
    assert!(!backends[1].healthy.load(Ordering::Relaxed));
    let a = backends[0].clone();
    a.active_conns.store(4, Ordering::Relaxed);

    // Second listing: `b` withdrawn, `c` added. `a` keeps its instance/state.
    Mock::given(method("GET"))
        .and(path("/backends/list"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "backends": [{"handle": "aaaaaaaaaaaa"}, {"handle": "cccccccccccc"}]
        })))
        .up_to_n_times(1)
        .mount(&registry)
        .await;
    let change = poll_once(&client, &cfg, &pool).await.unwrap();
    assert_eq!(change.added, vec![backend_url("cccccccccccc")]);
    assert_eq!(change.removed, vec![backend_url("bbbbbbbbbbbb")]);
    let backends = pool.backends();
    assert_eq!(backends.len(), 2);
    assert!(Arc::ptr_eq(&backends[0], &a));
    assert_eq!(backends[0].active_conns.load(Ordering::Relaxed), 4);

    // Failure modes keep the last known membership untouched.
    for template in [
        ResponseTemplate::new(500),
        ResponseTemplate::new(401),
        ResponseTemplate::new(200).set_body_string("not json"),
        ResponseTemplate::new(200).set_body_json(serde_json::json!({"backends": []})),
        ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"backends": [{"handle": "../x"}]})),
    ] {
        Mock::given(method("GET"))
            .and(path("/backends/list"))
            .respond_with(template)
            .up_to_n_times(1)
            .mount(&registry)
            .await;
        let before = pool.backends();
        assert!(poll_once(&client, &cfg, &pool).await.is_err());
        assert!(Arc::ptr_eq(&before, &pool.backends()));
    }
}

#[tokio::test]
async fn discovery_without_token_sends_no_authorization() {
    let registry = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/backends/list"))
        .and(header("authorization", "Bearer registry-token"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&registry)
        .await;
    Mock::given(method("GET"))
        .and(path("/backends/list"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "backends": [{"handle": "aaaaaaaaaaaa"}]
        })))
        .expect(1)
        .mount(&registry)
        .await;
    let pool = BackendPool::new(Vec::new());
    let mut cfg = discovery_config(&registry);
    cfg.token = None;
    poll_once(&reqwest::Client::new(), &cfg, &pool)
        .await
        .unwrap();
    assert_eq!(pool.len(), 1);
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

    let (app, _) = build_gateway(
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
    let (app, _) = build_gateway(&backend, GatewayOptions::default());
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
    let (app, _) = build_gateway(
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
    let (app, _) = build_gateway(&mock.uri(), GatewayOptions::default());
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
        let (app, _) = build_gateway(
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
        let body = json_body(response).await;
        assert_eq!(body["error"]["message"], "The request queue is full.");
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
    let (app, _) = build_gateway(
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
    let (app, _) = build_gateway(
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
    let body = json_body(response).await;
    assert_eq!(body["error"]["message"], "The request queue is full.");
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
    let (app, _) = build_gateway(
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
    let (app, _) = build_gateway(
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
    let (app, _) = build_gateway(
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
    let (app, _) = build_gateway(
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
