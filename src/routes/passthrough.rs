use axum::body::Body;
use axum::extract::multipart::Field;
use axum::extract::{Multipart, State};
use axum::response::Response;
use sha2::Digest;

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::proxy::{self, ProxyOpts};
use crate::routes::chat::read_body_with_limit;
use crate::AppState;

/// POST /v1/tokenize — simple proxy, no signing.
pub async fn tokenize(
    State(state): State<AppState>,
    _auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    let request_body = read_body_with_limit(body, state.config.max_request_size).await?;

    proxy::proxy_simple(
        &state.http_client,
        &state.config.tokenize_url,
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
    _auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    json_passthrough(
        state,
        body,
        |c| &c.embeddings_url,
        "emb",
        |c| c.max_request_size,
    )
    .await
}

/// POST /v1/rerank — JSON proxy with signing.
pub async fn rerank(
    State(state): State<AppState>,
    _auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    json_passthrough(
        state,
        body,
        |c| &c.rerank_url,
        "rerank",
        |c| c.max_request_size,
    )
    .await
}

/// POST /v1/score — JSON proxy with signing.
pub async fn score(
    State(state): State<AppState>,
    _auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    json_passthrough(
        state,
        body,
        |c| &c.score_url,
        "score",
        |c| c.max_request_size,
    )
    .await
}

/// POST /v1/images/generations — JSON proxy with signing, 50MB limit.
pub async fn images_generations(
    State(state): State<AppState>,
    _auth: RequireAuth,
    body: Body,
) -> Result<Response, AppError> {
    json_passthrough(
        state,
        body,
        |c| &c.images_url,
        "img",
        |c| c.max_image_request_size,
    )
    .await
}

/// POST /v1/images/edits — multipart proxy with signing.
pub async fn images_edits(
    State(state): State<AppState>,
    _auth: RequireAuth,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
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

        let data = read_field_chunks(&mut field, &mut total_size, max_size, &mut hasher).await?;

        let part = build_multipart_part(data, file_name, content_type)?;
        form = form.part(name, part);
    }

    let request_sha256 = hex::encode(hasher.finalize());

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "img".to_string(),
    };

    proxy::proxy_multipart_request(
        &state.http_client,
        &state.config.images_edits_url,
        form,
        &request_sha256,
        &opts,
    )
    .await
}

/// POST /v1/audio/transcriptions — multipart proxy with signing.
pub async fn audio_transcriptions(
    State(state): State<AppState>,
    _auth: RequireAuth,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
    let max_size = state.config.max_audio_request_size;
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

        let data = read_field_chunks(&mut field, &mut total_size, max_size, &mut hasher).await?;

        let part = build_multipart_part(data, file_name, content_type)?;
        form = form.part(name, part);
    }

    let request_sha256 = hex::encode(hasher.finalize());

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),
        id_prefix: "trans".to_string(),
    };

    proxy::proxy_multipart_request(
        &state.http_client,
        &state.config.transcriptions_url,
        form,
        &request_sha256,
        &opts,
    )
    .await
}

/// Generic JSON passthrough with signing.
async fn json_passthrough(
    state: AppState,
    body: Body,
    url_fn: fn(&crate::config::Config) -> &str,
    id_prefix: &str,
    size_fn: fn(&crate::config::Config) -> usize,
) -> Result<Response, AppError> {
    let max_size = size_fn(&state.config);
    let request_body = read_body_with_limit(body, max_size).await?;

    // Validate JSON
    let _: serde_json::Value = serde_json::from_slice(&request_body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    let opts = ProxyOpts {
        signing: state.signing.clone(),
        cache: state.cache.clone(),

        id_prefix: id_prefix.to_string(),
    };

    let url = url_fn(&state.config);
    proxy::proxy_json_request(&state.http_client, url, request_body, &opts).await
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
