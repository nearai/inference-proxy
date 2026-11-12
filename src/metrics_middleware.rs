use std::time::Instant;

use axum::extract::{MatchedPath, State};
use axum::http::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use metrics::{counter, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use crate::AppState;

/// Create a Prometheus recorder and return its handle for rendering metrics.
pub fn setup_metrics_recorder() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .unwrap_or_else(|err| {
            panic!(
                "failed to install Prometheus recorder. \
             This may occur if a recorder is already installed: {err}"
            )
        })
}

/// Axum middleware that tracks HTTP request metrics. `active_connections` is
/// kept by the request guard (`request_log`), which also sees a dropped
/// request and a response body still being sent.
pub async fn metrics_middleware(request: Request<axum::body::Body>, next: Next) -> Response {
    let method = request.method().to_string();
    let endpoint = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "CATCH_ALL".to_string());

    let start = Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    counter!("http_requests_total", "method" => method.clone(), "endpoint" => endpoint.clone(), "status" => status).increment(1);
    histogram!("http_request_duration_seconds", "method" => method, "endpoint" => endpoint)
        .record(duration);

    response
}

/// Handler for `GET /metrics` — renders Prometheus text format.
pub async fn prometheus_metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    match state.models.as_deref() {
        Some(models) => {
            for model in models.iter() {
                model.admission.record_backend_metrics(&model.backend_pool);
            }
        }
        None => state.admission.record_backend_metrics(&state.backend_pool),
    }
    state.metrics_handle.render()
}
