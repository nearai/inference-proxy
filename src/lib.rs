use std::sync::Arc;

use axum::http::{HeaderValue, Request};
use axum::middleware::Next;
use axum::response::Response;
use tracing::Instrument;

pub mod attestation;
pub mod auth;
pub mod cache;
pub mod config;
pub mod error;
pub mod metrics_middleware;
pub mod proxy;
pub mod rate_limit;
pub mod routes;
pub mod signing;
pub mod types;

/// Shared application state available to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<config::Config>,
    pub signing: Arc<signing::SigningPair>,
    pub cache: Arc<cache::ChatCache>,
    pub http_client: reqwest::Client,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
}

/// Request ID middleware: generates or passes through X-Request-ID header.
pub async fn request_id_middleware(mut request: Request<axum::body::Body>, next: Next) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Store in request extensions for handlers
    request.extensions_mut().insert(request_id.clone());

    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    let span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %method,
        path = %path,
    );

    let mut response = next.run(request).instrument(span).await;
    if let Ok(val) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", val);
    }
    response
}
