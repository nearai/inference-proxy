use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Extension;
use serde_json::Value;

use crate::error::AppError;
use crate::proxy;
use crate::{AppState, TracingIds};

/// GET /v1/metrics — plain text passthrough (no auth).
pub async fn metrics(
    State(state): State<AppState>,
    Extension(tracing_ids): Extension<TracingIds>,
) -> Result<Response, AppError> {
    let (url, _guard) = state.backend_pool.select_url("/metrics");
    proxy::proxy_simple(
        &state.http_client,
        &url,
        reqwest::Method::GET,
        None,
        "text/plain; charset=utf-8",
        None,
        Some(&tracing_ids),
    )
    .await
}

/// GET /v1/models — the models document (no auth).
///
/// With `VLLM_PROXY_MODELS_DOCUMENT_URL` set (gateway mode) the document is
/// read from that URL (cloud-api's `/v1/models`: pricing, modalities,
/// `is_ready`, `openrouter.slug`), reduced to this deployment's `MODEL_NAME`
/// and completed with the lane's declared `capacity`, so an aggregator can
/// read one URL for both inference and the listing. Without it, or when the
/// source cannot be read, the engine's own list is passed through as before;
/// with `VLLM_PROXY_DISCOUNT_TO_USER` set (which requires the document URL),
/// that list carries the discount on every entry.
pub async fn models(
    State(state): State<AppState>,
    Extension(tracing_ids): Extension<TracingIds>,
) -> Result<Response, AppError> {
    if let Some(source) = state.config.models_document_url.as_deref() {
        match fetch_models_document(&state, source).await {
            Ok(document) => {
                return Ok((StatusCode::OK, axum::Json(document)).into_response());
            }
            Err(error) => {
                metrics::counter!("models_document_source_failures_total").increment(1);
                tracing::warn!(
                    error = %error,
                    "Models document source unavailable, passing the engine list through"
                );
            }
        }
    }
    let (url, _guard) = state.backend_pool.select_url("/v1/models");
    let engine_list = proxy::proxy_simple(
        &state.http_client,
        &url,
        reqwest::Method::GET,
        None,
        "application/json",
        None,
        Some(&tracing_ids),
    )
    .await?;
    match state.config.discount_to_user {
        // Usage reports carry the discount whether or not the source answered,
        // so the list served in its place has to carry it too.
        Some(discount) => engine_list_with_discount(engine_list, discount).await,
        None => Ok(engine_list),
    }
}

/// The engine's model list with the lane's discount on every entry, for when
/// the models document source is unavailable. An engine answer that is not a
/// model list cannot carry the discount, so it is refused with a 502 instead of
/// being published without it.
async fn engine_list_with_discount(
    engine_list: Response,
    discount: f64,
) -> Result<Response, AppError> {
    let body = axum::body::to_bytes(engine_list.into_body(), usize::MAX)
        .await
        .map_err(|error| AppError::Internal(anyhow::anyhow!("engine model list: {error}")))?;
    let mut list = serde_json::from_slice::<Value>(&body).unwrap_or(Value::Null);
    let Some(entries) = list.get_mut("data").and_then(Value::as_array_mut) else {
        tracing::warn!(
            "Engine answer is not a model list, refusing to serve it without the discount"
        );
        return Err(AppError::UpstreamParsed {
            status: StatusCode::BAD_GATEWAY,
            message: "The model list is temporarily unavailable".to_string(),
            error_type: "upstream_invalid_response".to_string(),
        });
    };
    for entry in entries.iter_mut().filter_map(Value::as_object_mut) {
        set_discount(entry, Some(discount));
    }
    Ok((StatusCode::OK, axum::Json(list)).into_response())
}

/// Read the source document, keep only this deployment's model and attach
/// the declared capacity and, when configured, the lane's discount. Any
/// failure falls back to the engine list.
async fn fetch_models_document(state: &AppState, source: &str) -> anyhow::Result<Value> {
    let response = state
        .http_client
        .get(source)
        .timeout(Duration::from_secs(MODELS_DOCUMENT_TIMEOUT_SECS))
        .send()
        .await?;
    if !response.status().is_success() {
        anyhow::bail!("source answered {}", response.status());
    }
    let mut document: Value = response.json().await?;
    let model_name = state.config.model_name.as_str();
    let entries = document
        .get_mut("data")
        .and_then(|d| d.as_array_mut())
        .ok_or_else(|| anyhow::anyhow!("source document has no `data` array"))?;
    entries.retain(|entry| entry.get("id").and_then(|id| id.as_str()) == Some(model_name));
    if entries.is_empty() {
        anyhow::bail!("model {model_name} is not in the source document");
    }
    let capacity = capacity_entries(&state.config);
    for entry in entries.iter_mut() {
        let Some(entry) = entry.as_object_mut() else {
            continue;
        };
        if !capacity.is_empty() {
            entry.insert("capacity".to_string(), Value::Array(capacity.clone()));
        }
        set_discount(entry, state.config.discount_to_user);
    }
    Ok(document)
}

/// Put the lane's discount on a model entry, or take one off when none is
/// configured. The value is the one usage reports carry (`UsageReporter`), so
/// the published and the billed price come from the same setting; a value
/// already in the source is not this lane's and is dropped.
fn set_discount(entry: &mut serde_json::Map<String, Value>, discount: Option<f64>) {
    match discount {
        Some(discount) => {
            entry.insert("discount_to_user".to_string(), Value::from(discount));
        }
        None => {
            entry.remove("discount_to_user");
        }
    }
}

const MODELS_DOCUMENT_TIMEOUT_SECS: u64 = 5;

/// The lane's declared capacity, in the aggregator's schema: concurrency from
/// the admission budget's ceiling, a per-minute request rate when configured.
pub fn capacity_entries(config: &crate::config::Config) -> Vec<Value> {
    let mut entries = Vec::new();
    if config.admission_max_inflight > 0 {
        entries.push(serde_json::json!({
            "type": "concurrency",
            "unit": "request",
            "value": config.admission_max_inflight,
        }));
    }
    if config.capacity_requests_per_minute > 0 {
        entries.push(serde_json::json!({
            "type": "request",
            "unit": "request",
            "per": "minute",
            "value": config.capacity_requests_per_minute,
        }));
    }
    entries
}
