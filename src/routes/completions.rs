use axum::body::Body;
use axum::extract::State;
use axum::response::Response;

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::proxy::{self, ProxyOpts};
use crate::routes::chat::read_body_with_limit;
use crate::AppState;

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    _auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;

    let request_json: serde_json::Value = serde_json::from_slice(&request_body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "cmpl".to_string(),
    };

    if is_stream {
        proxy::proxy_streaming_request(
            &state.http_client,
            &state.config.completions_url,
            &request_body,
            &opts,
        )
        .await
    } else {
        proxy::proxy_json_request(
            &state.http_client,
            &state.config.completions_url,
            &request_body,
            &opts,
        )
        .await
    }
}
