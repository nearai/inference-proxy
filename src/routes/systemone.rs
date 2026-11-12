//! `POST /v1/systemone`: typed decisions (TypeSafe System One protocol) from
//! a decision model behind this proxy, such as Levanto Sage.
//!
//! The route exists only when `SYSTEMONE_UPSTREAM_MODEL` is set (see
//! `build_router_for`); everywhere else it is the same 404 as any undeclared
//! path. Its only caller is cloud-api's NEAR adapter, with the config `TOKEN`:
//! it sends the catalog id as `model` and the client's body hash as
//! `X-Request-Hash`, and fetches the receipt from `/v1/signature/{id}` with
//! the `id` of the answer.
//!
//! Unlike the privacy route, a successful answer is never returned unsigned:
//! anything but a JSON object from the backend is a 502.

use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use sha2::{Digest, Sha256};

use crate::auth::RequireTrustedAuth;
use crate::error::AppError;
use crate::proxy;
use crate::routes::chat::{read_body_with_limit, resolve_request_hash_for_signing};
use crate::{AppState, TracingIds};

/// Prefix of the answer ids this route generates.
const ID_PREFIX: &str = "systemone";

/// How long `/v1/models` waits for the backend's health route.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn systemone(
    State(state): State<AppState>,
    _auth: RequireTrustedAuth,
    Extension(tracing_ids): Extension<TracingIds>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let Some(upstream_model) = state.config.systemone_upstream_model.as_deref() else {
        return Err(AppError::NotFound("Endpoint not found".to_string()));
    };
    if headers.contains_key("x-signing-algo") || headers.contains_key("x-client-pub-key") {
        return Err(AppError::BadRequest(
            "Encryption is not supported on /v1/systemone".to_string(),
        ));
    }

    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;
    // The caller holds the config `TOKEN`, so its `X-Request-Hash` (the hash of
    // the client's own bytes, before cloud-api re-serialized them) is signed.
    let request_sha256 = resolve_request_hash_for_signing(&headers, &request_body, true);

    // cloud-api names the model by its catalog id (`MODEL_NAME`); the backend
    // knows it by its own name.
    let mut payload: serde_json::Value = serde_json::from_slice(&request_body)
        .map_err(|_| AppError::BadRequest("Request body must be a JSON object".to_string()))?;
    let Some(fields) = payload.as_object_mut() else {
        return Err(AppError::BadRequest(
            "Request body must be a JSON object".to_string(),
        ));
    };
    fields.insert(
        "model".to_string(),
        serde_json::Value::String(upstream_model.to_string()),
    );
    let upstream_body = serde_json::to_vec(&payload).map_err(|e| AppError::Internal(e.into()))?;

    let (backend_url, _backend_guard) = state.backend_pool.select_url(super::ROUTE_SYSTEMONE);
    // No client header is forwarded: `backend_client` carries the backend
    // bearer (`VLLM_BACKEND_TOKEN`), the rest is the request id.
    let request = state
        .backend_client
        .post(&backend_url)
        .header("content-type", "application/json");
    let request = proxy::apply_tracing_headers(request, Some(&tracing_ids)).body(upstream_body);

    let upstream_start = std::time::Instant::now();
    let response = request
        .send()
        .await
        .map_err(|error| AppError::Internal(error.into()))?;
    metrics::histogram!(
        "upstream_request_duration_seconds",
        "endpoint" => "systemone"
    )
    .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_default();
        let info = proxy::log_upstream_error(status, &backend_url, &body, Some(&tracing_ids));
        return Err(AppError::UpstreamParsed {
            status: proxy::effective_error_status(status.as_u16(), info.as_ref(), false),
            message: info
                .as_ref()
                .map(|error| error.message.clone())
                .unwrap_or_else(|| format!("Upstream request failed with status {status}")),
            error_type: info
                .as_ref()
                .map(|error| error.error_type.clone())
                .unwrap_or_else(|| "upstream_error".to_string()),
        });
    }

    let is_json = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("application/json"));
    let response_bytes = response
        .bytes()
        .await
        .map_err(|error| AppError::Internal(error.into()))?;
    if response_bytes.len() > state.config.max_request_size {
        return Err(AppError::PayloadTooLarge {
            max_size: state.config.max_request_size,
        });
    }
    let mut answer = match serde_json::from_slice::<serde_json::Value>(&response_bytes) {
        Ok(value) if is_json && value.is_object() => value,
        _ => {
            tracing::warn!(
                status = status.as_u16(),
                is_json,
                body_len = response_bytes.len(),
                "System One backend answered without a JSON object"
            );
            return Err(AppError::UpstreamParsed {
                status: StatusCode::BAD_GATEWAY,
                message: "Invalid response from the decision model".to_string(),
                error_type: "upstream_error".to_string(),
            });
        }
    };

    // The receipt is cached under the answer's id, so it is always ours: an id
    // from the backend could repeat and overwrite another answer's receipt.
    let id = format!(
        "{ID_PREFIX}-{}",
        &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
    );
    if let Some(fields) = answer.as_object_mut() {
        fields.insert("id".to_string(), serde_json::Value::String(id.clone()));
    }

    // The signature covers the exact bytes the caller receives.
    let final_body = serde_json::to_string(&answer).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(final_body.as_bytes()));
    let text = format!(
        "{}:{request_sha256}:{response_sha256}",
        state.config.model_name
    );
    let signed = state.signing.sign_chat(&text).map_err(|error| {
        tracing::error!(error = %error, "Signing failed");
        AppError::Internal(error)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    state.cache.set_chat(&id, &signed_json);
    tracing::info!(
        id = %id,
        duration_ms = upstream_start.elapsed().as_millis() as u64,
        "System One decision signed"
    );

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(final_body))
        .unwrap())
}

/// `GET /v1/models` on a decision-model proxy: this deployment's one model,
/// named as cloud-api and model-proxy know it (`MODEL_NAME`), served only
/// while the backend's health route (`VLLM_BACKEND_HEALTH_PATH`) answers 2xx.
/// model-proxy learns from this list which domain the port serves (OpenAI
/// shape, `data[].id`); the backend's own list has another shape and name.
pub async fn models(state: &AppState) -> Result<Response, AppError> {
    let (url, _guard) = state
        .backend_pool
        .select_url(&state.config.backend_health_path);
    let healthy = state
        .backend_client
        .get(&url)
        .timeout(HEALTH_TIMEOUT)
        .send()
        .await
        .is_ok_and(|response| response.status().is_success());
    if !healthy {
        return Err(AppError::UpstreamParsed {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "The decision model is not ready".to_string(),
            error_type: "upstream_unavailable".to_string(),
        });
    }
    let list = serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.config.model_name,
            "object": "model",
            "owned_by": "nearai",
        }],
    });
    Ok((StatusCode::OK, axum::Json(list)).into_response())
}
