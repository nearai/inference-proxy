use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::Extension;

use sha2::Digest;

use crate::auth::RequireAuth;
use crate::encryption::{self, Endpoint};
use crate::error::AppError;
use crate::proxy::{self, make_usage_reporter, ProxyOpts, ResponseShape, UsageType};
use crate::{agent_loop, fusion};
use crate::{AppState, TracingIds};

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    auth: RequireAuth,
    Extension(tracing_ids): Extension<TracingIds>,
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
    let tracing_ids = tracing_ids.with_authenticated_context(&headers, &auth);

    // Request hash for signing: SHA256(wire body) by default. X-Request-Hash is only
    // honored when authenticated with config.token (trusted gateway); sk- clients
    // always bind signatures to the wire body so they cannot forge a hash for a
    // different payload.
    let request_hash =
        resolve_request_hash_for_signing(&headers, &request_body, auth.cloud_api_key.is_none());

    // Decrypt request fields if encryption is active
    if let Some(ref ctx) = enc_ctx {
        encryption::decrypt_request_fields(
            &mut request_json,
            Endpoint::ChatCompletions,
            ctx,
            &state.signing,
        )?;
    }

    // Reject clearly-bad image inputs (unfetchable / non-image) before forwarding
    // to the engine, so a flood of dead URLs can't load the model. Runs only when
    // the request actually contains images; conservative/fail-open otherwise.
    // See nearai/infra#159, #172.
    crate::image_validation::reject_invalid_images(&request_json, &state.config.image_validation())
        .await?;

    let is_stream = request_json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if state.config.fusion_enabled && fusion::has_fusion_tool(&request_json) {
        let depth = fusion::trusted_request_depth(&headers, &state);
        if depth >= state.config.fusion_max_depth {
            return Err(AppError::BadRequest("fusion_depth_exceeded".to_string()));
        }

        let (response_transform, chunk_transform) = if let Some(ctx) = enc_ctx.clone() {
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

        return fusion::run_chat_completion(
            fusion::ChatCompletionContext {
                state,
                auth,
                tracing_ids,
                request_hash,
                is_stream,
                response_transform,
                chunk_transform,
            },
            request_json,
        )
        .await;
    }

    // Server-side agent loop opt-in: the request advertises exactly
    // `{"type":"web_context_search"}` and nothing else. Requires streaming
    // (we splice tool-result chunks between iterations) and Brave creds
    // configured on this CVM. Anything that doesn't match falls through to
    // the existing pass-through path below, byte-for-byte identical.
    if agent_loop::is_web_context_search_request(&request_json) {
        if !is_stream {
            return Err(AppError::BadRequest(
                "web_context_search requires stream:true".to_string(),
            ));
        }
        if state.config.web_context_search_url.is_none()
            || state.config.web_context_search_api_key.is_none()
        {
            return Err(AppError::BadRequest(
                "web_context_search is not configured on this deployment".to_string(),
            ));
        }

        // Build chunk transform if E2EE is active. The agent loop is the
        // privacy-critical path — both our synthetic `nearai_tool_result`
        // chunks AND the model's own `tool_calls[].function.{name,arguments}`
        // (which contain the search query the model just generated from the
        // user's E2EE-decrypted prompt) must travel encrypted. Force
        // `encrypt_all_fields: true` on the context used to build this
        // transform so clients don't need to remember to send
        // `X-Encrypt-All-Fields: true` to get the full privacy guarantee.
        // This only affects the agent-loop path; the regular chat path
        // below still honors the client's `X-Encrypt-All-Fields` choice.
        let chunk_transform = enc_ctx.map(|mut ctx| {
            ctx.encrypt_all_fields = true;
            encryption::make_chunk_transform(Endpoint::ChatCompletions, ctx, state.signing.clone())
        });
        return agent_loop::run_chat_completion(
            state,
            auth,
            tracing_ids,
            request_hash,
            request_json,
            chunk_transform,
        )
        .await;
    }

    // For cloud API key requests with streaming, force include_usage AND
    // continuous_usage_stats so the backend sends running cumulative token
    // counts on every chunk — not just the final one. This is what makes an
    // interrupted stream billable (nearai/infra#98): on a client disconnect or
    // upstream error before [DONE] the last chunk we saw still carries usage, so
    // `proxy_streaming_request` can report partial usage. include_usage alone only
    // emits usage in the final chunk, which an interrupted stream never reaches.
    // (Non-streaming requests also stream internally via proxy_json_request,
    // which injects stream_options itself, so this only matters for true streaming.)
    if auth.cloud_api_key.is_some() && is_stream {
        let stream_opts = request_json
            .get("stream_options")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let mut stream_opts = stream_opts;
        stream_opts.insert("include_usage".into(), true.into());
        stream_opts.insert("continuous_usage_stats".into(), true.into());
        request_json["stream_options"] = serde_json::Value::Object(stream_opts);
    }

    let upstream_data_parallel_rank = state
        .vllm_dp_affinity
        .rank_for_chat_request(&request_json, &state.config.model_name);

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

    let (url, guard) = state.backend_pool.select_url("/v1/chat/completions");

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "chatcmpl".to_string(),
        model_name: state.config.model_name.clone(),
        usage_reporter: make_usage_reporter(&auth, &state),
        usage_type: UsageType::ChatCompletion,
        request_hash: Some(request_hash),
        response_transform,
        chunk_transform,
        backend_guard: Some(guard),
        stream_idle_timeout_secs: state.config.stream_idle_timeout_secs,
        response_shape: ResponseShape::ChatCompletion,
        tracing_ids: Some(tracing_ids),
        upstream_data_parallel_rank,
    };

    if is_stream {
        proxy::proxy_streaming_request(&state.http_client, &url, modified_body, opts).await
    } else {
        proxy::proxy_json_request(&state.http_client, &url, modified_body, opts).await
    }
}

/// Resolve SHA-256 hex digest to use as request_sha256 in signed text.
///
/// Default: hash of the wire body. When `allow_x_request_hash_override` is true
/// (caller authenticated with `config.token`, not `sk-`), if `X-Request-Hash` is
/// present, decodes to 32 bytes, and differs from the wire body hash, returns the
/// header value so trusted gateways that re-serialize JSON can bind signatures to
/// the original client body hash. When false, the header is ignored so end-user
/// API keys cannot bind signatures to an arbitrary hash.
pub fn resolve_request_hash_for_signing(
    headers: &HeaderMap,
    body_bytes: &[u8],
    allow_x_request_hash_override: bool,
) -> String {
    let body_hash = hex::encode(sha2::Sha256::digest(body_bytes));
    if !allow_x_request_hash_override {
        return body_hash;
    }
    // HeaderMap::get is case-insensitive; no need to try multiple spellings.
    if let Some(hv) = headers.get("x-request-hash") {
        if let Ok(s) = hv.to_str() {
            let s = s.trim();
            // hex::decode validates digits and rejects odd length; 32 bytes == SHA-256.
            if let Ok(bytes) = hex::decode(s) {
                if bytes.len() == 32 {
                    let header_hash = hex::encode(&bytes);
                    if header_hash != body_hash {
                        return header_hash;
                    }
                }
            }
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
