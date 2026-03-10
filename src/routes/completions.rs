use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;

use sha2::Digest;

use crate::auth::RequireAuth;
use crate::encryption::{self, Endpoint};
use crate::error::AppError;
use crate::proxy::{self, make_usage_reporter, ProxyOpts, UsageType};
use crate::routes::chat::read_body_with_limit;
use crate::AppState;

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    auth: RequireAuth,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;

    let mut request_json: serde_json::Value = serde_json::from_slice(&request_body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    // Extract encryption context from headers
    let enc_ctx = encryption::extract_encryption_context(&headers)?;

    // Always hash the original client-sent body for signatures.
    // The body gets re-serialized after parsing (which reorders keys), so we must
    // hash the original bytes to let clients verify the exact request they sent.
    let original_request_hash = Some(hex::encode(sha2::Sha256::digest(&request_body)));

    // Decrypt request fields if encryption is active
    if let Some(ref ctx) = enc_ctx {
        encryption::decrypt_request_fields(
            &mut request_json,
            Endpoint::Completions,
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
                Endpoint::Completions,
                ctx.clone(),
                signing.clone(),
            )),
            Some(encryption::make_chunk_transform(
                Endpoint::Completions,
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
        id_prefix: "cmpl".to_string(),
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
            &state.config.completions_url,
            modified_body,
            opts,
        )
        .await
    } else {
        proxy::proxy_json_request(
            &state.http_client,
            &state.config.completions_url,
            modified_body,
            opts,
        )
        .await
    }
}
