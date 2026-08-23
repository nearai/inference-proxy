use std::sync::Arc;

use axum::middleware;

use vllm_proxy_rs::*;

#[derive(Default)]
pub(crate) struct TestAppOptions {
    pub(crate) cloud_api_url: Option<String>,
    pub(crate) dstack_socket_path: Option<String>,
}

pub(crate) fn build_test_app(mock_url: &str, options: TestAppOptions) -> axum::Router {
    let base = mock_url.trim_end_matches('/');
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
        cloud_api_url: options.cloud_api_url,
        cloud_api_auth_max_attempts: 1,
        cloud_api_auth_initial_backoff_ms: 0,
        cloud_api_auth_timeout_secs: 5,
        cloud_api_usage_token: Some("test-usage-token".to_string()),
        compose_manager_url: None,
        tls_cert_path: None,
        timeout_secs: 30,
        stream_idle_timeout_secs: 0,
        timeout_tokenize_secs: 5,
        openai_chat_compatibility_check_enabled: false,
        startup_check_retries: 1,
        startup_check_retry_delay_secs: 0,
        startup_check_timeout_secs: 5,
        backend_urls: vec![mock_url.to_string()],
        vllm_data_parallel_size: None,
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
        dstack_socket_path: options
            .dstack_socket_path
            .unwrap_or_else(|| "/var/run/dstack.sock".to_string()),
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
    let chat_cache = cache::ChatCache::new("test-model", 1200);
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let backend_pool = Arc::new(vllm_proxy_rs::backend_pool::BackendPool::new(vec![
        mock_url.to_string(),
    ]));

    let state = AppState {
        config: Arc::new(config),
        signing: Arc::new(signing_pair),
        cache: Arc::new(chat_cache),
        attestation_cache: Arc::new(vllm_proxy_rs::attestation::AttestationCache::new(300)),
        http_client: reqwest::Client::new(),
        metrics_handle,
        tls_cert_fingerprint: Arc::new(
            vllm_proxy_rs::attestation::TlsCertTracker::new(None).expect("tracker for None path"),
        ),
        backend_pool,
        ohttp_gateway: None,
        ohttp_attestation_ed25519: None,
        fusion_caches: Arc::new(fusion::FusionCaches::default()),
        vllm_dp_affinity: Arc::new(vllm_dp_affinity::VllmDpAffinity::new(None, 1_200)),
    };
    let rate_limiter = rate_limit::build_rate_limiter(100, 200);
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
