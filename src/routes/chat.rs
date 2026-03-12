use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;

use sha2::Digest;

use crate::auth::RequireAuth;
use crate::encryption::{self, Endpoint};
use crate::error::AppError;
use crate::proxy::{self, make_usage_reporter, ProxyOpts, UsageType};
use crate::AppState;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    auth: RequireAuth,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;

    let mut request_json: serde_json::Value = serde_json::from_slice(&request_body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    // Strip empty tool_calls (vLLM bug workaround)
    strip_empty_tool_calls(&mut request_json);

    // Extract encryption context from headers
    let enc_ctx = encryption::extract_encryption_context(&headers)?;

    // Request hash for signing: SHA256(wire body) by default. If X-Request-Hash is a
    // valid 64-hex string different from the wire body hash, use the header (cloud-api
    // and others may re-serialize JSON; header carries the client's original hash).
    let original_request_hash = Some(resolve_request_hash_for_signing(&headers, &request_body));

    // Decrypt request fields if encryption is active
    if let Some(ref ctx) = enc_ctx {
        encryption::decrypt_request_fields(
            &mut request_json,
            Endpoint::ChatCompletions,
            ctx,
            &state.signing,
        )?;
    }

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // For cloud API key requests with streaming, force include_usage
    // so the backend always sends token counts for billing.
    if auth.cloud_api_key.is_some() && is_stream {
        let stream_opts = request_json
            .get("stream_options")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let mut stream_opts = stream_opts;
        stream_opts.insert("include_usage".into(), true.into());
        request_json["stream_options"] = serde_json::Value::Object(stream_opts);
    }

    let modified_body =
        serde_json::to_vec(&request_json).map_err(|e| AppError::Internal(e.into()))?;

    // Build encryption transforms if active
    let (response_transform, chunk_transform) = if let Some(ctx) = enc_ctx {
        let signing = state.signing.clone();
        (
            Some(encryption::make_response_transform(
                Endpoint::ChatCompletions,
                ctx.clone(),
                signing.clone(),
            )),
            Some(encryption::make_chunk_transform(
                Endpoint::ChatCompletions,
                ctx,
                signing,
            )),
        )
    } else {
        (None, None)
    };

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "chatcmpl".to_string(),
        model_name: state.config.model_name.clone(),
        usage_reporter: make_usage_reporter(auth.cloud_api_key.as_ref(), &state),
        usage_type: UsageType::ChatCompletion,
        request_hash: original_request_hash,
        response_transform,
        chunk_transform,
    };

    if is_stream {
        proxy::proxy_streaming_request(
            &state.http_client,
            &state.config.chat_completions_url,
            modified_body,
            opts,
        )
        .await
    } else {
        proxy::proxy_json_request(
            &state.http_client,
            &state.config.chat_completions_url,
            modified_body,
            opts,
        )
        .await
    }
}

/// Resolve SHA-256 hex digest to use as request_sha256 in signed text.
///
/// Default: hash of the wire body. If `X-Request-Hash` is present, is 64 hex chars,
/// and differs from the wire body hash, returns the header value so gateways that
/// re-serialize the body (plaintext or ciphertext envelope) can still bind signatures
/// to the client's original byte sequence. When header equals body hash or is invalid,
/// the wire body hash is used so direct clients sending a garbage header are unaffected.
pub fn resolve_request_hash_for_signing(headers: &HeaderMap, body_bytes: &[u8]) -> String {
    let body_hash = hex::encode(sha2::Sha256::digest(body_bytes));
    for name in ["x-request-hash", "X-Request-Hash"] {
        if let Some(hv) = headers.get(name) {
            if let Ok(s) = hv.to_str() {
                let s = s.trim().to_lowercase();
                if s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit()) && s != body_hash {
                    return s;
                }
            }
            break;
        }
    }
    body_hash
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
