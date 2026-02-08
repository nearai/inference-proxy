use axum::extract::State;
use axum::response::Response;

use crate::error::AppError;
use crate::proxy;
use crate::AppState;

/// GET /v1/metrics — plain text passthrough (no auth).
pub async fn metrics(State(state): State<AppState>) -> Result<Response, AppError> {
    proxy::proxy_simple(
        &state.http_client,
        &state.config.metrics_url,
        reqwest::Method::GET,
        None,
        "text/plain; charset=utf-8",
        None,
    )
    .await
}

/// GET /v1/models — JSON passthrough (no auth).
pub async fn models(State(state): State<AppState>) -> Result<Response, AppError> {
    proxy::proxy_simple(
        &state.http_client,
        &state.config.models_url,
        reqwest::Method::GET,
        None,
        "application/json",
        None,
    )
    .await
}
