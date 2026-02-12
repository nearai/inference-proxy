use std::sync::Arc;

use axum::http::{HeaderValue, Request};
use axum::middleware::Next;
use axum::response::Response;

pub mod attestation;
pub mod auth;
pub mod cache;
pub mod config;
pub mod error;
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

    let mut response = next.run(request).await;
    if let Ok(val) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", val);
    }
    response
}
