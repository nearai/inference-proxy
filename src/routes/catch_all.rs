use axum::body::Body;
use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, Method};
use axum::response::Response;
use sha2::{Digest, Sha256};
use tracing::debug;

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::proxy::{self, ProxyOpts};
use crate::routes::chat::read_body_with_limit;
use crate::AppState;

/// Headers to exclude when forwarding requests to the backend.
const EXCLUDED_REQUEST_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    // Hop-by-hop headers (RFC 7230)
    "connection",
    "keep-alive",
    "te",
    "trailer",
    "upgrade",
    // Proxy headers
    "proxy-authenticate",
    "proxy-authorization",
    // Don't leak the proxy's auth token to backend
    "authorization",
];

/// Headers to exclude when forwarding response headers from the backend.
const EXCLUDED_RESPONSE_HEADERS: &[&str] = &[
    "transfer-encoding",
    "connection",
    "keep-alive",
    "te",
    "trailer",
    "upgrade",
];

/// Catch-all handler for undefined API paths. Forwards any request to the
/// backend inference engine, signing and caching the response when it's
/// JSON or SSE streaming.
pub async fn catch_all(
    State(state): State<AppState>,
    _auth: RequireAuth,
    method: Method,
    uri: OriginalUri,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let path = uri.path();
    let query = uri.query();

    // Path traversal check: reject if any decoded segment is ".."
    let decoded_path = percent_encoding::percent_decode_str(path).decode_utf8_lossy();
    for segment in decoded_path.split('/') {
        if segment == ".." {
            return Err(AppError::BadRequest(
                "Path traversal not allowed".to_string(),
            ));
        }
    }

    // Read body (use max_audio_request_size = 100MB since content type is unknown)
    let body_bytes = read_body_with_limit(body, state.config.max_audio_request_size).await?;

    // Compute request hash
    let request_sha256 = if body_bytes.is_empty() {
        // For bodyless requests, hash the path+query
        let hash_input = match query {
            Some(q) => format!("{path}?{q}"),
            None => path.to_string(),
        };
        hex::encode(Sha256::digest(hash_input.as_bytes()))
    } else {
        hex::encode(Sha256::digest(&body_bytes))
    };

    // Build backend URL
    let base = state.config.vllm_base_url.trim_end_matches('/');
    let backend_url = match query {
        Some(q) => format!("{base}{path}?{q}"),
        None => format!("{base}{path}"),
    };

    debug!(method = %method, backend_url = %backend_url, "Catch-all passthrough");

    // Build backend request
    let mut builder = state.http_client.request(
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap(),
        &backend_url,
    );

    // Forward headers, excluding hop-by-hop and security-sensitive ones
    for (name, value) in headers.iter() {
        let name_lower = name.as_str().to_lowercase();
        if !EXCLUDED_REQUEST_HEADERS.contains(&name_lower.as_str()) {
            builder = builder.header(name.as_str(), value);
        }
    }

    // Attach body if non-empty
    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes);
    }

    let response = builder
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    let upstream_status = response.status();
    if !upstream_status.is_success() {
        let axum_status = axum::http::StatusCode::from_u16(upstream_status.as_u16())
            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
        let error_body = serde_json::json!({
            "error": {
                "message": format!("Upstream request failed with status {upstream_status}"),
                "type": "upstream_error",
                "param": null,
                "code": null,
            }
        });
        return Ok(Response::builder()
            .status(axum_status)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&error_body).unwrap()))
            .unwrap());
    }

    let axum_status = axum::http::StatusCode::from_u16(upstream_status.as_u16())
        .unwrap_or(axum::http::StatusCode::OK);

    // Route by response content-type
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if content_type.contains("text/event-stream") {
        let opts = ProxyOpts {
            signing: state.signing.clone(),
            cache: state.cache.clone(),
            id_prefix: "pt".to_string(),
        };
        proxy::proxy_streaming_response(response, &request_sha256, &opts, axum_status).await
    } else if content_type.contains("application/json") {
        // Buffer JSON response with size guard
        let response_bytes = response
            .bytes()
            .await
            .map_err(|e| AppError::Internal(e.into()))?;

        if response_bytes.len() > state.config.max_request_size {
            return Err(AppError::PayloadTooLarge {
                max_size: state.config.max_request_size,
            });
        }

        let opts = ProxyOpts {
            signing: state.signing.clone(),
            cache: state.cache.clone(),
            id_prefix: "pt".to_string(),
        };
        proxy::sign_and_cache_json_response(&response_bytes, &request_sha256, &opts, axum_status)
            .await
    } else {
        // Raw passthrough — stream without signing, preserve upstream headers
        let mut builder = Response::builder().status(axum_status);
        for (name, value) in response.headers().iter() {
            let name_lower = name.as_str().to_lowercase();
            if !EXCLUDED_RESPONSE_HEADERS.contains(&name_lower.as_str()) {
                builder = builder.header(name.as_str(), value);
            }
        }
        Ok(builder
            .body(Body::from_stream(response.bytes_stream()))
            .unwrap())
    }
}
