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

/// Validate that a request path is safe to forward to the backend.
///
/// Performs multi-pass percent-decoding to catch double/triple-encoded traversal
/// attacks (e.g. `%252e%252e` → `%2e%2e` → `..`), and rejects paths containing
/// encoded slashes, backslashes, or null bytes.
fn validate_path(path: &str) -> Result<(), AppError> {
    // Reject encoded slashes and backslashes in the raw path — these can bypass
    // segment splitting. Check case-insensitively.
    let path_lower = path.to_ascii_lowercase();
    if path_lower.contains("%2f") || path_lower.contains("%5c") {
        return Err(AppError::BadRequest(
            "Path traversal not allowed".to_string(),
        ));
    }

    // Reject null bytes
    if path_lower.contains("%00") || path.contains('\0') {
        return Err(AppError::BadRequest(
            "Path traversal not allowed".to_string(),
        ));
    }

    // Multi-pass decode: iteratively decode until the string stabilizes,
    // checking for traversal patterns after each pass.
    let mut current = path.to_string();
    for _ in 0..5 {
        let decoded = percent_encoding::percent_decode_str(&current).decode_utf8_lossy();

        // Check for ".." segments and backslashes in the decoded form
        for segment in decoded.split('/') {
            if segment == ".." {
                return Err(AppError::BadRequest(
                    "Path traversal not allowed".to_string(),
                ));
            }
        }
        if decoded.contains('\\') {
            return Err(AppError::BadRequest(
                "Path traversal not allowed".to_string(),
            ));
        }

        if decoded == current {
            break; // Stable — no more decoding possible
        }
        current = decoded.into_owned();
    }

    Ok(())
}

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

    validate_path(path)?;

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

    let upstream_start = std::time::Instant::now();
    let response = builder
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "catch_all")
        .record(upstream_start.elapsed().as_secs_f64());

    let upstream_status = response.status();
    if !upstream_status.is_success() {
        let error_body = response
            .bytes()
            .await
            .unwrap_or_else(|_| bytes::Bytes::from("{}"));
        let error_info = crate::proxy::log_upstream_error(upstream_status, &backend_url, &error_body);
        let axum_status = axum::http::StatusCode::from_u16(upstream_status.as_u16())
            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
        return Err(AppError::UpstreamParsed {
            status: axum_status,
            message: error_info
                .as_ref()
                .map(|e| e.message.clone())
                .unwrap_or_else(|| {
                    format!("Upstream request failed with status {upstream_status}")
                }),
            error_type: error_info
                .as_ref()
                .map(|e| e.error_type.clone())
                .unwrap_or_else(|| "upstream_error".to_string()),
        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_paths_accepted() {
        assert!(validate_path("/v1/chat/completions").is_ok());
        assert!(validate_path("/v1/models").is_ok());
        assert!(validate_path("/v1/models/gpt-3.5-turbo").is_ok());
        assert!(validate_path("/").is_ok());
        assert!(validate_path("/v1/custom/endpoint").is_ok());
    }

    #[test]
    fn test_literal_dotdot_rejected() {
        assert!(validate_path("/../etc/passwd").is_err());
        assert!(validate_path("/v1/../../secret").is_err());
        assert!(validate_path("/v1/foo/../bar").is_err());
    }

    #[test]
    fn test_single_encoded_dotdot_rejected() {
        // %2e = '.'
        assert!(validate_path("/v1/%2e%2e/secret").is_err());
        assert!(validate_path("/v1/%2E%2E/secret").is_err());
    }

    #[test]
    fn test_double_encoded_dotdot_rejected() {
        // %252e → %2e (after first decode) → '.' (after second decode)
        assert!(validate_path("/v1/%252e%252e/secret").is_err());
        assert!(validate_path("/v1/%252E%252E/secret").is_err());
    }

    #[test]
    fn test_triple_encoded_dotdot_rejected() {
        // %25252e → %252e → %2e → '.'
        assert!(validate_path("/v1/%25252e%25252e/secret").is_err());
    }

    #[test]
    fn test_encoded_slash_rejected() {
        // %2f = '/' — could bypass segment splitting
        assert!(validate_path("/v1/foo%2fbar").is_err());
        assert!(validate_path("/v1/foo%2Fbar").is_err());
    }

    #[test]
    fn test_backslash_rejected() {
        assert!(validate_path("/v1/foo\\bar").is_err());
        // %5c = '\'
        assert!(validate_path("/v1/foo%5cbar").is_err());
        assert!(validate_path("/v1/foo%5Cbar").is_err());
    }

    #[test]
    fn test_null_byte_rejected() {
        assert!(validate_path("/v1/foo%00bar").is_err());
        assert!(validate_path("/v1/foo\0bar").is_err());
    }

    #[test]
    fn test_mixed_encoding_attacks() {
        // Mixed case encoding
        assert!(validate_path("/v1/%2e%2E/secret").is_err());
        // Encoded slash with dotdot
        assert!(validate_path("/v1%2f..%2fsecret").is_err());
        // Double-encoded slash
        assert!(validate_path("/v1/%252f../secret").is_err());
    }
}
