use std::sync::Arc;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info};

use crate::cache::ChatCache;
use crate::error::AppError;
use crate::signing::SigningPair;

/// Options for proxy requests that need signing.
pub struct ProxyOpts {
    pub signing: Arc<SigningPair>,
    pub cache: Arc<ChatCache>,
    /// Prefix for auto-generated IDs (e.g., "chatcmpl", "img", "emb").
    pub id_prefix: String,
}

/// Proxy a JSON request to the backend, sign the response, cache signature.
pub async fn proxy_json_request(
    client: &reqwest::Client,
    url: &str,
    request_body: &[u8],
    opts: &ProxyOpts,
) -> Result<Response, AppError> {
    let request_sha256 = hex::encode(Sha256::digest(request_body));

    let response = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(request_body.to_vec())
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        return Err(AppError::Upstream {
            status: StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
        });
    }

    let response_bytes = response
        .bytes()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    // Parse response to extract/generate ID
    let mut response_data: serde_json::Value =
        serde_json::from_slice(&response_bytes).map_err(|e| AppError::Internal(e.into()))?;

    let chat_id = match response_data.get("id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            let id = format!(
                "{}-{}",
                opts.id_prefix,
                &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
            );
            if let Some(obj) = response_data.as_object_mut() {
                obj.insert("id".to_string(), serde_json::Value::String(id.clone()));
            }
            debug!(id = %id, "Generated response ID");
            id
        }
    };

    // Serialize with compact separators (matching Python's separators=(",",":"))
    let response_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(response_body.as_bytes()));

    // Sign and cache
    let text = format!("{request_sha256}:{response_sha256}");
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&chat_id, &signed_json);

    Ok((
        StatusCode::OK,
        [("content-type", "application/json")],
        response_body,
    )
        .into_response())
}

/// Proxy a streaming SSE request. Hashes all chunks, signs at end, caches signature.
pub async fn proxy_streaming_request(
    client: &reqwest::Client,
    url: &str,
    request_body: &[u8],
    opts: &ProxyOpts,
) -> Result<Response, AppError> {
    let request_sha256 = hex::encode(Sha256::digest(request_body));

    let response = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(request_body.to_vec())
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        return Err(AppError::Upstream {
            status: StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
        });
    }

    let signing = opts.signing.clone();
    let cache = opts.cache.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    // Spawn a task to consume upstream and forward chunks
    let byte_stream = response.bytes_stream();
    tokio::spawn(async move {
        use futures_util::StreamExt;

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut hasher = Sha256::new();
        let mut parser = SseParser::new();
        let mut upstream_error = false;
        let mut downstream_closed = false;

        while let Some(chunk_result) = byte_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    hasher.update(&chunk);
                    parser.process_chunk(&chunk);

                    let chunk_bytes: Bytes = chunk;
                    if tx.send(Ok(chunk_bytes)).await.is_err() {
                        downstream_closed = true;
                        break;
                    }
                }
                Err(e) => {
                    error!(error = %e, "Error reading upstream stream");
                    upstream_error = true;
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    break;
                }
            }
        }

        // Only sign and cache for a fully completed stream
        if !upstream_error && !downstream_closed && parser.seen_done {
            let response_sha256 = hex::encode(hasher.finalize());
            if let Some(id) = parser.chat_id {
                let text = format!("{request_sha256}:{response_sha256}");
                match signing.sign_chat(&text) {
                    Ok(signed) => {
                        if let Ok(signed_json) = serde_json::to_string(&signed) {
                            cache.set_chat(&id, &signed_json);
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Signing failed for streaming response");
                    }
                }
            } else {
                error!("Chat id could not be extracted from the completed streaming response");
            }
        } else {
            info!(
                upstream_error,
                downstream_closed,
                seen_done = parser.seen_done,
                "Skipping streaming signature cache: stream did not complete cleanly"
            );
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .unwrap())
}

/// Proxy a multipart request. Caller provides pre-computed request hash covering all field bytes.
pub async fn proxy_multipart_request(
    client: &reqwest::Client,
    url: &str,
    form: reqwest::multipart::Form,
    request_sha256: &str,
    opts: &ProxyOpts,
) -> Result<Response, AppError> {
    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        return Err(AppError::Upstream {
            status: StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
        });
    }

    let response_bytes = response
        .bytes()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    let mut response_data: serde_json::Value =
        serde_json::from_slice(&response_bytes).map_err(|e| AppError::Internal(e.into()))?;

    let response_id = match response_data.get("id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            let id = format!(
                "{}-{}",
                opts.id_prefix,
                &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
            );
            if let Some(obj) = response_data.as_object_mut() {
                obj.insert("id".to_string(), serde_json::Value::String(id.clone()));
            }
            id
        }
    };

    let response_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(response_body.as_bytes()));

    let text = format!("{request_sha256}:{response_sha256}");
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&response_id, &signed_json);

    Ok((
        StatusCode::OK,
        [("content-type", "application/json")],
        response_body,
    )
        .into_response())
}

/// Simple proxy without signing (for tokenize, metrics, models).
pub async fn proxy_simple(
    client: &reqwest::Client,
    url: &str,
    method: reqwest::Method,
    body: Option<&[u8]>,
    content_type: &str,
    timeout: Option<std::time::Duration>,
) -> Result<Response, AppError> {
    let mut builder = client.request(method, url);

    if let Some(body) = body {
        builder = builder
            .header("content-type", "application/json")
            .body(body.to_vec());
    }

    if let Some(timeout) = timeout {
        builder = builder.timeout(timeout);
    }

    let response = builder
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        return Err(AppError::Upstream {
            status: StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
        });
    }

    let response_bytes = response
        .bytes()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", content_type)
        .body(Body::from(response_bytes))
        .unwrap())
}

/// Sign already-fetched JSON response bytes, cache the signature, and return a JSON response.
/// Used by catch-all when content-type is already known to be JSON.
pub async fn sign_and_cache_json_response(
    response_bytes: &[u8],
    request_sha256: &str,
    opts: &ProxyOpts,
    status: StatusCode,
) -> Result<Response, AppError> {
    // Parse JSON; if the backend sent content-type: application/json but the body
    // is empty or not valid JSON, wrap it in an empty object so we can still
    // generate an ID, sign, and cache.
    let mut response_data: serde_json::Value = match serde_json::from_slice(response_bytes) {
        Ok(data) => data,
        Err(e) => {
            debug!(error = %e, "Response body not valid JSON, wrapping in empty object");
            serde_json::json!({})
        }
    };

    let chat_id = match response_data.get("id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            let id = format!(
                "{}-{}",
                opts.id_prefix,
                &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
            );
            if let Some(obj) = response_data.as_object_mut() {
                obj.insert("id".to_string(), serde_json::Value::String(id.clone()));
            }
            debug!(id = %id, "Generated response ID");
            id
        }
    };

    let response_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(response_body.as_bytes()));

    let text = format!("{request_sha256}:{response_sha256}");
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&chat_id, &signed_json);

    Ok(Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(response_body))
        .unwrap())
}

/// Proxy an already-received streaming SSE response. Hashes all chunks, signs at end, caches.
/// Used by catch-all when content-type is already known to be SSE.
pub async fn proxy_streaming_response(
    response: reqwest::Response,
    request_sha256: &str,
    opts: &ProxyOpts,
    status: StatusCode,
) -> Result<Response, AppError> {
    let signing = opts.signing.clone();
    let cache = opts.cache.clone();
    let request_sha256 = request_sha256.to_string();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    let byte_stream = response.bytes_stream();
    tokio::spawn(async move {
        use futures_util::StreamExt;

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut hasher = Sha256::new();
        let mut parser = SseParser::new();
        let mut upstream_error = false;
        let mut downstream_closed = false;

        while let Some(chunk_result) = byte_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    hasher.update(&chunk);
                    parser.process_chunk(&chunk);

                    let chunk_bytes: Bytes = chunk;
                    if tx.send(Ok(chunk_bytes)).await.is_err() {
                        downstream_closed = true;
                        break;
                    }
                }
                Err(e) => {
                    error!(error = %e, "Error reading upstream stream");
                    upstream_error = true;
                    let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                    break;
                }
            }
        }

        if !upstream_error && !downstream_closed && parser.seen_done {
            let response_sha256 = hex::encode(hasher.finalize());
            if let Some(id) = parser.chat_id {
                let text = format!("{request_sha256}:{response_sha256}");
                match signing.sign_chat(&text) {
                    Ok(signed) => {
                        if let Ok(signed_json) = serde_json::to_string(&signed) {
                            cache.set_chat(&id, &signed_json);
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Signing failed for streaming response");
                    }
                }
            } else {
                error!("Chat id could not be extracted from the completed streaming response");
            }
        } else {
            info!(
                upstream_error,
                downstream_closed,
                seen_done = parser.seen_done,
                "Skipping streaming signature cache: stream did not complete cleanly"
            );
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(status)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(body)
        .unwrap())
}

/// Line-buffered SSE parser that handles data split across chunk boundaries.
/// Extracts `chat_id` from the first JSON chunk and detects the `[DONE]` marker.
pub(crate) struct SseParser {
    line_buffer: String,
    pub chat_id: Option<String>,
    pub seen_done: bool,
}

impl SseParser {
    fn new() -> Self {
        Self {
            line_buffer: String::new(),
            chat_id: None,
            seen_done: false,
        }
    }

    fn process_chunk(&mut self, chunk: &[u8]) {
        let chunk_str = String::from_utf8_lossy(chunk);
        self.line_buffer.push_str(&chunk_str);

        // Process all complete lines in the buffer
        while let Some(newline_pos) = self.line_buffer.find('\n') {
            let line = self.line_buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            self.line_buffer = self.line_buffer[newline_pos + 1..].to_string();

            let data = line
                .strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
                .unwrap_or(&line)
                .trim();

            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                self.seen_done = true;
                continue;
            }
            if self.chat_id.is_none() {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(id) = parsed.get("id").and_then(|v| v.as_str()) {
                        self.chat_id = Some(id.to_string());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::ChatCache;
    use crate::signing::{EcdsaContext, Ed25519Context, SigningPair};

    /// Build a ProxyOpts with fixed signing keys for deterministic tests.
    fn test_proxy_opts() -> ProxyOpts {
        let ecdsa_key: [u8; 32] = [
            0xac, 0x09, 0x74, 0xbe, 0xc3, 0x9a, 0x17, 0xe3, 0x6b, 0xa4, 0xa6, 0xb4, 0xd2, 0x38,
            0xff, 0x94, 0x4b, 0xac, 0xb3, 0x5e, 0x5d, 0xc4, 0xaf, 0x0f, 0x33, 0x47, 0xe5, 0x87,
            0x31, 0x79, 0x67, 0x0f,
        ];
        let ed25519_key: [u8; 32] = [
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec,
            0x2c, 0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03,
            0x1c, 0xae, 0x7f, 0x60,
        ];
        let ecdsa = EcdsaContext::from_key_bytes(&ecdsa_key).unwrap();
        let ed25519 = Ed25519Context::from_key_bytes(&ed25519_key).unwrap();
        let signing = Arc::new(SigningPair { ecdsa, ed25519 });
        let cache = Arc::new(ChatCache::new("test-model", 1200));
        ProxyOpts {
            signing,
            cache,
            id_prefix: "test".to_string(),
        }
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_empty_body() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));

        let result =
            sign_and_cache_json_response(b"", &request_sha256, &opts, StatusCode::OK).await;

        let resp = result.expect("empty body should not return error");
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Should have a generated ID starting with the prefix
        let id = parsed["id"].as_str().unwrap();
        assert!(id.starts_with("test-"), "id should start with prefix: {id}");
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_invalid_json_body() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));

        let result = sign_and_cache_json_response(
            b"this is not json",
            &request_sha256,
            &opts,
            StatusCode::OK,
        )
        .await;

        let resp = result.expect("invalid JSON should not return error");
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let id = parsed["id"].as_str().unwrap();
        assert!(id.starts_with("test-"), "id should start with prefix: {id}");
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_valid_body_with_id() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));
        let body = br#"{"id":"existing-id","text":"hello"}"#;

        let result =
            sign_and_cache_json_response(body, &request_sha256, &opts, StatusCode::OK).await;

        let resp = result.expect("valid JSON should succeed");
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(parsed["id"], "existing-id");

        // Signature should be cached under "existing-id"
        assert!(opts.cache.get_chat("existing-id").is_some());
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_valid_body_without_id() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));
        let body = br#"{"text":"hello"}"#;

        let result =
            sign_and_cache_json_response(body, &request_sha256, &opts, StatusCode::OK).await;

        let resp = result.expect("valid JSON without id should succeed");
        let resp_body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        let id = parsed["id"].as_str().unwrap();
        assert!(id.starts_with("test-"), "should generate id with prefix");

        // Signature should be cached under the generated id
        assert!(opts.cache.get_chat(id).is_some());
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_preserves_status_code() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test"));
        let body = br#"{"id":"s1"}"#;

        let resp = sign_and_cache_json_response(body, &request_sha256, &opts, StatusCode::CREATED)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[test]
    fn test_sse_parser_normal_sse() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-1\",\"content\":\"hi\"}\n\ndata: [DONE]\n\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-1"));
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_done_split_across_chunks() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-2\"}\n\ndata: [DO");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-2"));
        assert!(!parser.seen_done);

        parser.process_chunk(b"NE]\n\n");
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_id_split_across_chunks() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"cha");
        assert!(parser.chat_id.is_none());

        parser.process_chunk(b"t-3\",\"choices\":[]}\n\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-3"));
    }

    #[test]
    fn test_sse_parser_no_space_after_data_colon() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data:{\"id\":\"chat-4\"}\n\ndata:[DONE]\n\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-4"));
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_crlf_line_endings() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-5\"}\r\n\r\ndata: [DONE]\r\n\r\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-5"));
        assert!(parser.seen_done);
    }

    #[test]
    fn test_sse_parser_no_done_marker() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-6\"}\n\n");
        assert_eq!(parser.chat_id.as_deref(), Some("chat-6"));
        assert!(!parser.seen_done);
    }

    #[test]
    fn test_sse_parser_multiple_json_chunks() {
        let mut parser = SseParser::new();
        parser.process_chunk(b"data: {\"id\":\"chat-7\",\"delta\":\"a\"}\n\n");
        parser.process_chunk(b"data: {\"id\":\"chat-7\",\"delta\":\"b\"}\n\n");
        parser.process_chunk(b"data: [DONE]\n\n");
        // Should use the first id
        assert_eq!(parser.chat_id.as_deref(), Some("chat-7"));
        assert!(parser.seen_done);
    }
}
