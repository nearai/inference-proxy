use axum::body::Body;
use axum::extract::multipart::Field;
use axum::extract::{Multipart, State};
use axum::http::HeaderMap;
use axum::response::Response;
use sha2::Digest;

use crate::auth::RequireAuth;
use crate::encryption::{self, Endpoint};
use crate::error::AppError;
use crate::proxy::{self, make_usage_reporter, ProxyOpts, UsageReporter, UsageType};
use crate::routes::chat::read_body_with_limit;
use crate::AppState;

/// POST /v1/tokenize — simple proxy, no signing.
pub async fn tokenize(
    State(state): State<AppState>,
    _auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;

    let (url, _guard) = state.backend_pool.select_url("/tokenize");
    proxy::proxy_simple(
        &state.http_client,
        &url,
        reqwest::Method::POST,
        Some(&request_body),
        "application/json",
        Some(std::time::Duration::from_secs(
            state.config.timeout_tokenize_secs,
        )),
    )
    .await
}

/// POST /v1/embeddings — JSON proxy with signing.
pub async fn embeddings(
    State(state): State<AppState>,
    auth: RequireAuth,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let reporter = make_usage_reporter(auth.cloud_api_key.as_ref(), &state);
    let enc_ctx = encryption::extract_encryption_context(&headers)?;

    json_passthrough_encrypted(
        state,
        body,
        "/v1/embeddings",
        None,
        "emb",
        |c| c.max_request_size,
        reporter,
        UsageType::ChatCompletion,
        enc_ctx,
        Endpoint::Embeddings,
    )
    .await
}

/// POST /v1/rerank — JSON proxy with signing.
pub async fn rerank(
    State(state): State<AppState>,
    auth: RequireAuth,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let reporter = make_usage_reporter(auth.cloud_api_key.as_ref(), &state);
    let enc_ctx = encryption::extract_encryption_context(&headers)?;
    let url_override = state.config.rerank_url_override.clone();

    json_passthrough_encrypted(
        state,
        body,
        "/v1/rerank",
        url_override.as_deref(),
        "rerank",
        |c| c.max_request_size,
        reporter,
        UsageType::ChatCompletion,
        enc_ctx,
        Endpoint::Rerank,
    )
    .await
}

/// POST /v1/score — JSON proxy with signing.
pub async fn score(
    State(state): State<AppState>,
    auth: RequireAuth,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let reporter = make_usage_reporter(auth.cloud_api_key.as_ref(), &state);
    let enc_ctx = encryption::extract_encryption_context(&headers)?;
    let url_override = state.config.score_url_override.clone();

    json_passthrough_encrypted(
        state,
        body,
        "/v1/score",
        url_override.as_deref(),
        "score",
        |c| c.max_request_size,
        reporter,
        UsageType::ChatCompletion,
        enc_ctx,
        Endpoint::Score,
    )
    .await
}

/// POST /v1/images/generations — JSON proxy with signing.
pub async fn images_generations(
    State(state): State<AppState>,
    auth: RequireAuth,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let reporter = make_usage_reporter(auth.cloud_api_key.as_ref(), &state);
    let enc_ctx = encryption::extract_encryption_context(&headers)?;
    let url_override = state.config.images_url_override.clone();

    json_passthrough_encrypted(
        state,
        body,
        "/v1/images/generations",
        url_override.as_deref(),
        "img",
        |c| c.max_image_request_size,
        reporter,
        UsageType::ImageGeneration,
        enc_ctx,
        Endpoint::ImagesGenerations,
    )
    .await
}

/// POST /v1/images/edits — multipart proxy with signing.
pub async fn images_edits(
    State(state): State<AppState>,
    auth: RequireAuth,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
    let enc_ctx = encryption::extract_encryption_context(&headers)?;
    let max_size = state.config.max_image_request_size;
    let mut form = reqwest::multipart::Form::new();
    let mut total_size: usize = 0;
    let mut hasher = sha2::Sha256::new();

    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Invalid multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        let file_name = field.file_name().map(|s| s.to_string());
        let content_type = field.content_type().map(|s| s.to_string());

        if let (true, Some(ctx)) = (name == "prompt", enc_ctx.as_ref()) {
            // Read field, hash the raw (encrypted) bytes, then decrypt for forwarding
            let raw_data = read_field_data(&mut field, &mut total_size, max_size).await?;
            hasher.update(&raw_data);
            let text = String::from_utf8(raw_data)
                .map_err(|_| AppError::BadRequest("prompt field is not UTF-8".to_string()))?;
            let data = if !text.is_empty() {
                encryption::decrypt_string(&text, ctx, &state.signing)?.into_bytes()
            } else {
                text.into_bytes()
            };
            let part = build_multipart_part(data, file_name, content_type)?;
            form = form.part(name, part);
        } else {
            let data =
                read_field_chunks(&mut field, &mut total_size, max_size, &mut hasher).await?;
            let part = build_multipart_part(data, file_name, content_type)?;
            form = form.part(name, part);
        }
    }

    let request_sha256 = hex::encode(hasher.finalize());

    let response_transform = enc_ctx.as_ref().map(|ctx| {
        encryption::make_response_transform(
            Endpoint::ImagesEdits,
            ctx.clone(),
            state.signing.clone(),
        )
    });

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "img".to_string(),
        model_name: state.config.model_name.clone(),
        usage_reporter: make_usage_reporter(auth.cloud_api_key.as_ref(), &state),
        usage_type: UsageType::ImageGeneration,
        request_hash: None,
        response_transform,
        chunk_transform: None,
    };

    let url = match &state.config.images_edits_url_override {
        Some(override_url) => override_url.clone(),
        None => state.backend_pool.select_url("/v1/images/edits").0,
    };

    proxy::proxy_multipart_request(&state.http_client, &url, form, &request_sha256, opts).await
}

/// POST /v1/audio/transcriptions — multipart proxy with signing.
pub async fn audio_transcriptions(
    State(state): State<AppState>,
    auth: RequireAuth,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
    let enc_ctx = encryption::extract_encryption_context(&headers)?;
    let max_size = state.config.max_audio_request_size;
    let mut form = reqwest::multipart::Form::new();
    let mut total_size: usize = 0;
    // Always hash the raw (original) bytes for signatures, matching the Python proxy.
    // When encryption is active, the prompt field is hashed in its encrypted form,
    // then decrypted for forwarding to the backend.
    let mut hasher = sha2::Sha256::new();

    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Invalid multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        let file_name = field.file_name().map(|s| s.to_string());
        let content_type = field.content_type().map(|s| s.to_string());

        if let (true, Some(ctx)) = (name == "prompt", enc_ctx.as_ref()) {
            // Read field, hash the raw (encrypted) bytes, then decrypt for forwarding
            let raw_data = read_field_data(&mut field, &mut total_size, max_size).await?;
            hasher.update(&raw_data);
            let text = String::from_utf8(raw_data)
                .map_err(|_| AppError::BadRequest("prompt field is not UTF-8".to_string()))?;
            let data = if !text.is_empty() {
                encryption::decrypt_string(&text, ctx, &state.signing)?.into_bytes()
            } else {
                text.into_bytes()
            };
            let part = build_multipart_part(data, file_name, content_type)?;
            form = form.part(name, part);
        } else {
            let data =
                read_field_chunks(&mut field, &mut total_size, max_size, &mut hasher).await?;
            let part = build_multipart_part(data, file_name, content_type)?;
            form = form.part(name, part);
        }
    }

    let request_sha256 = hex::encode(hasher.finalize());

    let response_transform = enc_ctx.as_ref().map(|ctx| {
        encryption::make_response_transform(
            Endpoint::AudioTranscriptions,
            ctx.clone(),
            state.signing.clone(),
        )
    });

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "trans".to_string(),
        model_name: state.config.model_name.clone(),
        usage_reporter: make_usage_reporter(auth.cloud_api_key.as_ref(), &state),
        usage_type: UsageType::ChatCompletion,
        request_hash: None,
        response_transform,
        chunk_transform: None,
    };

    let url = match &state.config.transcriptions_url_override {
        Some(override_url) => override_url.clone(),
        None => state.backend_pool.select_url("/v1/audio/transcriptions").0,
    };

    proxy::proxy_multipart_request(&state.http_client, &url, form, &request_sha256, opts).await
}

/// Generic JSON passthrough with signing and optional encryption support.
#[allow(clippy::too_many_arguments)]
async fn json_passthrough_encrypted(
    state: AppState,
    body: Body,
    pool_path: &str,
    url_override: Option<&str>,
    id_prefix: &str,
    size_fn: fn(&crate::config::Config) -> usize,
    usage_reporter: Option<UsageReporter>,
    usage_type: UsageType,
    enc_ctx: Option<encryption::EncryptionContext>,
    endpoint: Endpoint,
) -> Result<Response, AppError> {
    let max_size = size_fn(&state.config);
    let request_body = read_body_with_limit(body, max_size).await?;

    // When encryption is active, decrypt fields and re-serialize.
    // When inactive, validate JSON but forward the original bytes to
    // preserve the exact request body for hashing.
    let (forward_body, original_request_hash, response_transform) = if let Some(ctx) = enc_ctx {
        // Hash original client-sent body before decryption for signatures
        let original_hash = hex::encode(sha2::Sha256::digest(&request_body));
        let mut request_json: serde_json::Value = serde_json::from_slice(&request_body)
            .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;
        encryption::decrypt_request_fields(&mut request_json, endpoint, &ctx, &state.signing)?;
        let modified =
            serde_json::to_vec(&request_json).map_err(|e| AppError::Internal(e.into()))?;
        let transform = encryption::make_response_transform(endpoint, ctx, state.signing.clone());
        (modified, Some(original_hash), Some(transform))
    } else {
        // Validate JSON without re-serializing
        let _: serde_json::Value = serde_json::from_slice(&request_body)
            .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;
        (request_body, None, None)
    };

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: id_prefix.to_string(),
        model_name: state.config.model_name.clone(),
        usage_reporter,
        usage_type,
        request_hash: original_request_hash,
        response_transform,
        chunk_transform: None,
    };

    let url = match url_override {
        Some(u) => u.to_string(),
        None => state.backend_pool.select_url(pool_path).0,
    };
    proxy::proxy_json_request(&state.http_client, &url, forward_body, opts).await
}

/// Read a multipart field incrementally, checking cumulative size (without hashing).
/// Used when the raw bytes should not be hashed (e.g., encrypted fields that will
/// be decrypted and hashed separately).
async fn read_field_data(
    field: &mut Field<'_>,
    total_size: &mut usize,
    max_size: usize,
) -> Result<Vec<u8>, AppError> {
    let mut data = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|e| AppError::BadRequest(format!("Error reading field: {e}")))?
    {
        *total_size = total_size.saturating_add(chunk.len());
        if *total_size > max_size {
            return Err(AppError::PayloadTooLarge { max_size });
        }
        data.extend_from_slice(&chunk);
    }
    Ok(data)
}

/// Read a multipart field incrementally, checking cumulative size and hashing all bytes.
async fn read_field_chunks(
    field: &mut Field<'_>,
    total_size: &mut usize,
    max_size: usize,
    hasher: &mut sha2::Sha256,
) -> Result<Vec<u8>, AppError> {
    let mut data = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|e| AppError::BadRequest(format!("Error reading field: {e}")))?
    {
        *total_size = total_size.saturating_add(chunk.len());
        if *total_size > max_size {
            return Err(AppError::PayloadTooLarge { max_size });
        }
        hasher.update(&chunk);
        data.extend_from_slice(&chunk);
    }
    Ok(data)
}

/// Build a multipart part with optional filename and content type.
fn build_multipart_part(
    data: Vec<u8>,
    file_name: Option<String>,
    content_type: Option<String>,
) -> Result<reqwest::multipart::Part, AppError> {
    let mut part = reqwest::multipart::Part::bytes(data);
    if let Some(fname) = file_name {
        part = part.file_name(fname);
    }
    if let Some(ct) = content_type {
        part = part
            .mime_str(&ct)
            .map_err(|_| AppError::BadRequest(format!("Invalid MIME type: {ct}")))?;
    }
    Ok(part)
}
