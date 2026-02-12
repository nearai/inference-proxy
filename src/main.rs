use std::sync::Arc;

use axum::middleware;
use tokio::net::TcpListener;
use tracing::info;

use vllm_proxy_rs::{
    cache, config, metrics_middleware, rate_limit, request_id_middleware, routes, signing, AppState,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Load config
    let config = config::Config::from_env()?;

    let listen_port: u16 = std::env::var("LISTEN_PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()
        .map_err(|_| anyhow::anyhow!("LISTEN_PORT must be a valid port number"))?;

    // Warn if backend URL points to the proxy's own listen address
    let backend_base = config.vllm_base_url.trim_end_matches('/');
    let self_local = format!("://localhost:{listen_port}");
    let self_ip = format!("://127.0.0.1:{listen_port}");
    if backend_base.contains(&self_local) || backend_base.contains(&self_ip) {
        tracing::warn!(
            backend = %config.vllm_base_url,
            listen_port,
            "VLLM_BASE_URL points to the proxy's own listen port. \
             Set VLLM_BASE_URL to the actual backend address or change LISTEN_PORT."
        );
    }

    info!(
        model = %config.model_name,
        backend = %config.vllm_base_url,
        listen_port,
        dev_mode = config.dev_mode,
        "Starting vllm-proxy-rs"
    );

    // Initialize signing keys
    let signing = signing::SigningPair::init(&config.model_name, config.dev_mode).await?;
    info!(
        ecdsa_address = %signing.ecdsa.signing_address,
        ed25519_address = %signing.ed25519.signing_address,
        "Signing keys ready"
    );

    // Initialize cache
    let chat_cache = cache::ChatCache::new(&config.model_name, config.chat_cache_expiration_secs);

    // Initialize HTTP client with connection pooling
    let http_client = reqwest::Client::builder()
        .pool_max_idle_per_host(config.max_keepalive)
        .timeout(std::time::Duration::from_secs(config.timeout_secs))
        .build()?;

    // Initialize metrics
    let metrics_handle = metrics_middleware::setup_metrics_recorder();

    // Build app state
    let state = AppState {
        config: Arc::new(config),
        signing: Arc::new(signing),
        cache: Arc::new(chat_cache),
        http_client,
        metrics_handle,
    };

    // Build rate limiter
    let rate_limiter = rate_limit::build_rate_limiter(
        state.config.rate_limit_per_second,
        state.config.rate_limit_burst_size,
    );
    let rate_limit_state = rate_limit::RateLimitState {
        limiter: rate_limiter,
        trust_proxy_headers: state.config.rate_limit_trust_proxy_headers,
    };
    info!(
        per_second = state.config.rate_limit_per_second,
        burst = state.config.rate_limit_burst_size,
        trust_proxy_headers = state.config.rate_limit_trust_proxy_headers,
        "Rate limiter configured"
    );

    // Build router
    let app = routes::build_router()
        .layer(middleware::from_fn(rate_limit::rate_limit_middleware))
        .layer(axum::Extension(rate_limit_state))
        .layer(middleware::from_fn(request_id_middleware))
        .layer(middleware::from_fn(metrics_middleware::metrics_middleware))
        .with_state(state);

    // Bind and serve
    let addr = format!("0.0.0.0:{listen_port}");
    let listener = TcpListener::bind(&addr).await?;
    info!("Listening on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shut down");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("Shutdown signal received");
}
