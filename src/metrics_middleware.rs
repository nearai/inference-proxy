use std::time::Instant;

use axum::extract::{MatchedPath, State};
use axum::http::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use crate::AppState;

/// Create a Prometheus recorder and return its handle for rendering metrics.
pub fn setup_metrics_recorder() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}

/// Axum middleware that tracks HTTP request metrics.
pub async fn metrics_middleware(request: Request<axum::body::Body>, next: Next) -> Response {
    let method = request.method().to_string();
    let endpoint = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "CATCH_ALL".to_string());

    gauge!("active_connections").increment(1);
    let start = Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    counter!("http_requests_total", "method" => method.clone(), "endpoint" => endpoint.clone(), "status" => status).increment(1);
    histogram!("http_request_duration_seconds", "method" => method, "endpoint" => endpoint)
        .record(duration);
    gauge!("active_connections").decrement(1);

    response
}

/// Handler for `GET /metrics` — renders Prometheus text format.
pub async fn prometheus_metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics_handle.render()
}
