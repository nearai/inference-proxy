use axum::body::Body;
use axum::extract::State;
use axum::response::Response;

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::proxy::{self, ProxyOpts};
use crate::AppState;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    _auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_audio_request_size).await?;

    let mut request_json: serde_json::Value = serde_json::from_slice(&request_body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    // Strip empty tool_calls (vLLM bug workaround)
    strip_empty_tool_calls(&mut request_json);

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let modified_body =
        serde_json::to_vec(&request_json).map_err(|e| AppError::Internal(e.into()))?;

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "chatcmpl".to_string(),
    };

    if is_stream {
        proxy::proxy_streaming_request(
            &state.http_client,
            &state.config.chat_completions_url,
            &modified_body,
            &opts,
        )
        .await
    } else {
        proxy::proxy_json_request(
            &state.http_client,
            &state.config.chat_completions_url,
            &modified_body,
            &opts,
        )
        .await
    }
}

/// Strip empty tool_calls arrays from messages (vLLM bug workaround).
fn strip_empty_tool_calls(payload: &mut serde_json::Value) {
    if let Some(messages) = payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for message in messages.iter_mut() {
            if let Some(obj) = message.as_object_mut() {
                if let Some(tool_calls) = obj.get("tool_calls") {
                    if tool_calls.as_array().map(|a| a.is_empty()).unwrap_or(false) {
                        obj.remove("tool_calls");
                    }
                }
            }
        }
    }
}

/// Read body with size limit.
pub async fn read_body_with_limit(body: Body, max_size: usize) -> Result<Vec<u8>, AppError> {
    use http_body_util::BodyExt;

    let mut chunks: Vec<bytes::Bytes> = Vec::new();
    let mut total_size = 0usize;
    let mut body = body;

    loop {
        match body.frame().await {
            Some(Ok(frame)) => {
                if let Ok(data) = frame.into_data() {
                    total_size += data.len();
                    if total_size > max_size {
                        return Err(AppError::PayloadTooLarge { max_size });
                    }
                    chunks.push(data);
                }
            }
            Some(Err(e)) => {
                return Err(AppError::Internal(anyhow::anyhow!(
                    "Error reading body: {e}"
                )));
            }
            None => break,
        }
    }

    let mut result = Vec::with_capacity(total_size);
    for chunk in chunks {
        result.extend_from_slice(&chunk);
    }
    Ok(result)
}
