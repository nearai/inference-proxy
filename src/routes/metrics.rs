use axum::extract::State;
use axum::response::Response;

use crate::error::AppError;
use crate::proxy;
use crate::AppState;

/// GET /v1/metrics — plain text passthrough (no auth).
pub async fn metrics(State(state): State<AppState>) -> Result<Response, AppError> {
    let (url, _guard) = state.backend_pool.select_url("/metrics");
    proxy::proxy_simple(
        &state.http_client,
        &url,
        reqwest::Method::GET,
        None,
        "text/plain; charset=utf-8",
        None,
    )
    .await
}

/// GET /v1/models — JSON passthrough (no auth).
pub async fn models(State(state): State<AppState>) -> Result<Response, AppError> {
    let (url, _guard) = state.backend_pool.select_url("/v1/models");
    proxy::proxy_simple(
        &state.http_client,
        &url,
        reqwest::Method::GET,
        None,
        "application/json",
        None,
    )
    .await
}
