use std::sync::Arc;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use crate::cache::ChatCache;
use crate::error::AppError;
use crate::signing::SigningPair;
use crate::AppState;

/// Parsed upstream error info for logging and re-wrapping.
pub struct UpstreamErrorInfo {
    pub message: String,
    pub error_type: String,
}

/// Parse an upstream error body to extract message and type.
/// Handles both vLLM flat format (`{"object":"error","message":"...","type":"..."}`)
/// and nested format (`{"error":{"message":"...","type":"..."}}`).
/// Returns None if the body is not parseable JSON with the expected fields.
///
/// The extracted message is sanitized to strip user data from validation error
/// details (the `'input'` and `'ctx'` fields in Python-formatted validation dicts).
pub fn parse_upstream_error(body: &[u8]) -> Option<UpstreamErrorInfo> {
    let json: serde_json::Value = serde_json::from_slice(body).ok()?;

    // Try nested format first: {"error": {"message": "...", "type": "..."}}
    if let Some(error_obj) = json.get("error").filter(|v| v.is_object()) {
        let message = error_obj.get("message")?.as_str()?.to_string();
        let error_type = error_obj
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        return Some(UpstreamErrorInfo {
            message: sanitize_validation_errors(&message),
            error_type,
        });
    }

    // Try vLLM flat format: {"message": "...", "type": "...", "object": "error"}
    let message = json.get("message")?.as_str()?.to_string();
    let error_type = json
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    Some(UpstreamErrorInfo {
        message: sanitize_validation_errors(&message),
        error_type,
    })
}

/// Regex to extract 'type' values from Python-formatted validation error dicts.
/// Handles both single-quoted and double-quoted strings.
static VALIDATION_TYPE_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r#"'type':\s*(?:'([^']+)'|"([^"]+)")"#).unwrap());

/// Regex to extract 'msg' values from Python-formatted validation error dicts.
/// Handles both single-quoted and double-quoted strings.
static VALIDATION_MSG_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r#"'msg':\s*(?:'([^']*)'|"([^"]*)")"#).unwrap());

/// Regex to extract error type from pydantic v2 bracket sections: [type=missing, ...]
static PYDANTIC_V2_TYPE_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"type=(\w+)").unwrap());

/// Sanitize validation error messages to prevent leaking user conversation content.
///
/// Backend validation errors (from SGLang/vLLM) include `'input'` and `'ctx'` fields
/// containing the original request data, which may include user messages, AI responses,
/// and other sensitive conversation content. This function strips those fields while
/// preserving useful error type and message information.
pub fn sanitize_validation_errors(message: &str) -> String {
    // Check for sensitive fields in both Python dict format ('input':, 'ctx':)
    // and pydantic v2 format (input_value=, input_type=)
    let has_python_dict_fields = message.contains("'input':") || message.contains("'ctx':");
    let has_pydantic_v2_fields = message.contains("input_value=") || message.contains("input_type=");

    if !has_python_dict_fields && !has_pydantic_v2_fields {
        return message.to_string();
    }

    message
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();

            // Skip stack traces, HTTP method lines, and pydantic v2 "For further information" URLs
            if trimmed.starts_with("File \"")
                || trimmed.starts_with("POST ")
                || trimmed.starts_with("GET ")
                || trimmed.starts_with("For further information visit")
            {
                return None;
            }

            // Python dict format: lines with 'input' or 'ctx' fields
            if trimmed.contains("'input':") || trimmed.contains("'ctx':") {
                let error_type = VALIDATION_TYPE_RE
                    .captures(trimmed)
                    .and_then(|c| c.get(1).or_else(|| c.get(2)))
                    .map(|m| m.as_str());
                let error_msg = VALIDATION_MSG_RE
                    .captures(trimmed)
                    .and_then(|c| c.get(1).or_else(|| c.get(2)))
                    .map(|m| m.as_str());

                let sanitized = match (error_type, error_msg) {
                    (Some(t), Some(m)) => format!("  {}: {}", t, m),
                    (Some(t), None) => format!("  {}", t),
                    _ => "  (validation error)".to_string(),
                };
                Some(sanitized)
            } else if trimmed.contains("input_value=") || trimmed.contains("input_type=") {
                // Pydantic v2 format: "Field required [type=missing, input_value={...}, input_type=dict]"
                // Extract only the description and error type, discard input_value which contains user data
                let desc = trimmed.split('[').next().unwrap_or("").trim();
                let error_type = PYDANTIC_V2_TYPE_RE
                    .captures(trimmed)
                    .and_then(|c| c.get(1))
                    .map(|m| m.as_str());
                match error_type {
                    Some(t) => Some(format!("  {} [type={}]", desc, t)),
                    None => Some(format!("  {}", desc)),
                }
            } else {
                Some(line.to_string())
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse an upstream error body, log it, and return the parsed info.
pub(crate) fn log_upstream_error(
    status: reqwest::StatusCode,
    url: &str,
    body: &[u8],
) -> Option<UpstreamErrorInfo> {
    let info = parse_upstream_error(body);
    warn!(
        upstream_status = %status,
        upstream_url = %url,
        error_message = info.as_ref().map(|e| e.message.as_str()).unwrap_or("unparseable"),
        error_type = info.as_ref().map(|e| e.error_type.as_str()).unwrap_or("unknown"),
        "Backend returned non-success status"
    );
    info
}

/// Reports usage to the cloud API for billing.
#[derive(Clone)]
pub struct UsageReporter {
    pub http_client: reqwest::Client,
    pub cloud_api_url: String,
    pub api_key: String,
    pub model_name: String,
}

/// What kind of usage to extract from the response.
#[derive(Clone, Default)]
pub enum UsageType {
    /// Extract prompt_tokens / completion_tokens from the `usage` object.
    #[default]
    ChatCompletion,
    /// Count items in the `data` array (for image generation).
    ImageGeneration,
}

/// Build a `UsageReporter` if the request was authenticated with a cloud API key.
pub fn make_usage_reporter(
    cloud_api_key: Option<&String>,
    state: &AppState,
) -> Option<UsageReporter> {
    let key = cloud_api_key?;
    let url = state.config.cloud_api_url.as_ref()?;
    Some(UsageReporter {
        http_client: state.http_client.clone(),
        cloud_api_url: url.clone(),
        api_key: key.clone(),
        model_name: state.config.model_name.clone(),
    })
}

/// Extract usage from a parsed JSON response and fire-and-forget a report to the cloud API.
fn try_report_usage(response_data: &serde_json::Value, id: &str, opts: &ProxyOpts) {
    let reporter = match &opts.usage_reporter {
        Some(r) => r,
        None => return,
    };
    let body = match &opts.usage_type {
        UsageType::ChatCompletion => {
            let usage = match response_data.get("usage") {
                Some(u) => u,
                None => return,
            };
            let input = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let output = usage
                .get("completion_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            if input == 0 && output == 0 {
                return;
            }
            serde_json::json!({
                "type": "chat_completion",
                "model": reporter.model_name,
                "input_tokens": input,
                "output_tokens": output,
                "id": id,
            })
        }
        UsageType::ImageGeneration => {
            let count = response_data
                .get("data")
                .and_then(|d| d.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            if count == 0 {
                return;
            }
            serde_json::json!({
                "type": "image_generation",
                "model": reporter.model_name,
                "image_count": count,
                "id": id,
            })
        }
    };
    spawn_usage_report(reporter, body);
}

/// Fire-and-forget POST to the cloud API usage endpoint.
fn spawn_usage_report(reporter: &UsageReporter, body: serde_json::Value) {
    let client = reporter.http_client.clone();
    let url = format!("{}/v1/usage", reporter.cloud_api_url);
    let auth = format!("Bearer {}", reporter.api_key);
    tokio::spawn(async move {
        match client
            .post(&url)
            .header("authorization", &auth)
            .json(&body)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if !resp.status().is_success() => {
                warn!(status = %resp.status(), "Usage reporting returned non-success");
            }
            Err(e) => warn!(error = %e, "Usage reporting failed"),
            _ => {}
        }
    });
}

/// Options for proxy requests that need signing.
pub struct ProxyOpts {
    pub signing: Arc<SigningPair>,
    pub cache: Arc<ChatCache>,
    /// Prefix for auto-generated IDs (e.g., "chatcmpl", "img", "emb").
    pub id_prefix: String,
    /// Model name included in the signed text.
    pub model_name: String,
    /// If set, report usage to the cloud API after a successful response.
    pub usage_reporter: Option<UsageReporter>,
    /// What kind of usage to extract from the response.
    pub usage_type: UsageType,
    /// Pre-computed SHA-256 hex hash of the original request body.
    /// When set, this hash is used in the signature instead of hashing the
    /// (possibly decrypted/modified) body that is forwarded to the backend.
    /// This matches the Python proxy behavior where signatures cover the
    /// original client-sent body, not the decrypted version.
    pub request_hash: Option<String>,
    /// Applied to response JSON after signing, before sending to client.
    pub response_transform: Option<crate::encryption::ResponseTransform>,
    /// Applied to each SSE chunk JSON before forwarding to client.
    pub chunk_transform: Option<crate::encryption::ChunkTransform>,
}

/// Proxy a JSON request to the backend, sign the response, cache signature.
pub async fn proxy_json_request(
    client: &reqwest::Client,
    url: &str,
    request_body: Vec<u8>,
    mut opts: ProxyOpts,
) -> Result<Response, AppError> {
    let request_sha256 = opts
        .request_hash
        .take()
        .unwrap_or_else(|| hex::encode(Sha256::digest(&request_body)));

    let upstream_start = std::time::Instant::now();
    let response = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(request_body)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "json")
        .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        log_upstream_error(status, url, &body);
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

    // Report usage for cloud API key requests (before encryption, needs plaintext fields)
    try_report_usage(&response_data, &chat_id, &opts);

    // Apply response transform (e.g., encryption) before hashing/signing.
    // The signature covers the response bytes the client actually receives.
    if let Some(transform) = opts.response_transform.take() {
        transform(&mut response_data)?;
    }

    // Serialize with compact separators (matching Python's separators=(",",":"))
    let final_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(final_body.as_bytes()));

    // Sign and cache
    let text = format!("{}:{request_sha256}:{response_sha256}", opts.model_name);
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&chat_id, &signed_json);

    Ok((
        StatusCode::OK,
        [("content-type", "application/json")],
        final_body,
    )
        .into_response())
}

/// Proxy a streaming SSE request. Hashes all chunks, signs at end, caches signature.
pub async fn proxy_streaming_request(
    client: &reqwest::Client,
    url: &str,
    request_body: Vec<u8>,
    mut opts: ProxyOpts,
) -> Result<Response, AppError> {
    let request_sha256 = opts
        .request_hash
        .take()
        .unwrap_or_else(|| hex::encode(Sha256::digest(&request_body)));

    let upstream_start = std::time::Instant::now();
    let response = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .body(request_body)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "streaming")
        .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        log_upstream_error(status, url, &body);
        return Err(AppError::Upstream {
            status: StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            body,
        });
    }

    let signing = opts.signing.clone();
    let cache = opts.cache.clone();
    let usage_reporter = opts.usage_reporter.clone();
    let model_name = opts.model_name.clone();
    let chunk_transform = opts.chunk_transform;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    // Spawn a task to consume upstream and forward chunks.
    // Uses select! on tx.closed() to detect client disconnect while waiting
    // for upstream data, preventing resource leaks from abandoned connections.
    let byte_stream = response.bytes_stream();
    tokio::spawn(async move {
        use futures_util::StreamExt;

        let _guard = StreamingGuard::new();

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut hasher = Sha256::new();
        let mut parser = SseParser::new();
        let mut upstream_error = false;
        let mut downstream_closed = false;
        let mut transformer = chunk_transform.map(SseTransformer::new);

        loop {
            tokio::select! {
                chunk = byte_stream.next() => {
                    match chunk {
                        Some(Ok(chunk)) => {
                            parser.process_chunk(&chunk);

                            // Transform (encrypt) the chunk if needed, then hash
                            // what the client actually receives for signatures.
                            let to_send = if let Some(ref mut xform) = transformer {
                                match xform.process_chunk(&chunk) {
                                    Ok(transformed) => transformed,
                                    Err(e) => {
                                        error!(error = %e, "Stream encryption failed");
                                        let _ = tx.send(Err(std::io::Error::other(
                                            "Stream encryption failed",
                                        ))).await;
                                        upstream_error = true;
                                        break;
                                    }
                                }
                            } else {
                                chunk
                            };
                            hasher.update(&to_send);

                            if tx.send(Ok(to_send)).await.is_err() {
                                downstream_closed = true;
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "Error reading upstream stream");
                            upstream_error = true;
                            let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                            break;
                        }
                        None => break, // stream ended
                    }
                }
                _ = tx.closed() => {
                    info!("Client disconnected, aborting upstream stream processing");
                    downstream_closed = true;
                    break;
                }
            }
        }

        // Flush any remaining buffered content in the transformer
        if !upstream_error && !downstream_closed {
            if let Some(ref mut xform) = transformer {
                match xform.flush() {
                    Ok(flushed) if !flushed.is_empty() => {
                        hasher.update(&flushed);
                        if tx.send(Ok(flushed)).await.is_err() {
                            downstream_closed = true;
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Stream encryption flush failed");
                        let _ = tx
                            .send(Err(std::io::Error::other("Stream encryption failed")))
                            .await;
                        upstream_error = true;
                    }
                    _ => {}
                }
            }
        }

        // Only sign and cache for a fully completed stream
        if !upstream_error && !downstream_closed && parser.seen_done {
            let response_sha256 = hex::encode(hasher.finalize());
            if let Some(ref id) = parser.chat_id {
                let text = format!("{model_name}:{request_sha256}:{response_sha256}");
                match signing.sign_chat(&text) {
                    Ok(signed) => {
                        if let Ok(signed_json) = serde_json::to_string(&signed) {
                            cache.set_chat(id, &signed_json);
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Signing failed for streaming response");
                    }
                }

                // Report usage for cloud API key requests
                if let (Some(reporter), Some((input, output))) = (&usage_reporter, parser.usage) {
                    let body = serde_json::json!({
                        "type": "chat_completion",
                        "model": reporter.model_name,
                        "input_tokens": input,
                        "output_tokens": output,
                        "id": id,
                    });
                    spawn_usage_report(reporter, body);
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

/// Line-buffered SSE transformer that handles data split across chunk boundaries.
/// Fail-closed: if a data line contains JSON that cannot be transformed, the stream errors.
struct SseTransformer {
    line_buffer: String,
    transform: crate::encryption::ChunkTransform,
}

impl SseTransformer {
    fn new(transform: crate::encryption::ChunkTransform) -> Self {
        Self {
            line_buffer: String::new(),
            transform,
        }
    }

    /// Feed raw bytes into the buffer and return all complete transformed lines.
    /// Incomplete lines are buffered for the next call.
    fn process_chunk(&mut self, chunk: &[u8]) -> Result<Bytes, AppError> {
        let s = std::str::from_utf8(chunk).map_err(|e| {
            AppError::Internal(anyhow::anyhow!("Received invalid UTF-8 in SSE stream: {e}"))
        })?;
        self.line_buffer.push_str(s);

        let mut output = String::new();

        loop {
            let Some(newline_pos) = self.line_buffer.find('\n') else {
                break;
            };

            // Extract the complete line including the newline
            let full_line = self.line_buffer[..=newline_pos].to_string();
            self.line_buffer.drain(..=newline_pos);

            let trimmed = full_line.trim_end_matches(['\n', '\r']);
            let data = trimmed
                .strip_prefix("data: ")
                .or_else(|| trimmed.strip_prefix("data:"));

            if let Some(data) = data {
                let data = data.trim();
                if !data.is_empty() && data != "[DONE]" {
                    // This is a JSON data line — must transform or fail
                    let mut parsed: serde_json::Value =
                        serde_json::from_str(data).map_err(|e| {
                            AppError::Internal(anyhow::anyhow!(
                                "Failed to parse SSE data for encryption: {e}"
                            ))
                        })?;
                    (self.transform)(&mut parsed)?;
                    let re_serialized =
                        serde_json::to_string(&parsed).map_err(|e| AppError::Internal(e.into()))?;
                    output.push_str("data: ");
                    output.push_str(&re_serialized);
                    // Preserve the original line ending
                    let ending = &full_line[trimmed.len()..];
                    output.push_str(ending);
                    continue;
                }
            }
            // Pass through non-data lines, empty lines, and [DONE]
            output.push_str(&full_line);
        }

        Ok(Bytes::from(output))
    }

    /// Flush any remaining buffered content at stream end.
    /// A well-formed SSE stream always ends lines with `\n`, but if the backend
    /// sends a final line without one, this ensures it is still transformed and
    /// forwarded so the signature hash (which covers all raw bytes) matches
    /// what the client receives.
    fn flush(&mut self) -> Result<Bytes, AppError> {
        if self.line_buffer.is_empty() {
            return Ok(Bytes::new());
        }
        let remaining = std::mem::take(&mut self.line_buffer);
        let trimmed = remaining.trim_end_matches(['\n', '\r']);
        let data = trimmed
            .strip_prefix("data: ")
            .or_else(|| trimmed.strip_prefix("data:"));

        if let Some(data) = data {
            let data = data.trim();
            if !data.is_empty() && data != "[DONE]" {
                let mut parsed: serde_json::Value = serde_json::from_str(data).map_err(|e| {
                    AppError::Internal(anyhow::anyhow!(
                        "Failed to parse SSE data for encryption: {e}"
                    ))
                })?;
                (self.transform)(&mut parsed)?;
                let re_serialized =
                    serde_json::to_string(&parsed).map_err(|e| AppError::Internal(e.into()))?;
                let mut output = String::from("data: ");
                output.push_str(&re_serialized);
                output.push_str(&remaining[trimmed.len()..]);
                return Ok(Bytes::from(output));
            }
        }
        Ok(Bytes::from(remaining))
    }
}

/// Proxy a multipart request. Caller provides pre-computed request hash covering all field bytes.
pub async fn proxy_multipart_request(
    client: &reqwest::Client,
    url: &str,
    form: reqwest::multipart::Form,
    request_sha256: &str,
    mut opts: ProxyOpts,
) -> Result<Response, AppError> {
    let upstream_start = std::time::Instant::now();
    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "multipart")
        .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        log_upstream_error(status, url, &body);
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

    // Report usage for cloud API key requests (before encryption, needs plaintext fields)
    try_report_usage(&response_data, &response_id, &opts);

    // Apply response transform (e.g., encryption) before hashing/signing.
    // The signature covers the response bytes the client actually receives.
    if let Some(transform) = opts.response_transform.take() {
        transform(&mut response_data)?;
    }

    // Serialize with compact separators (matching Python's separators=(",",":"))
    let final_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(final_body.as_bytes()));

    // Sign and cache
    let text = format!("{}:{request_sha256}:{response_sha256}", opts.model_name);
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&response_id, &signed_json);

    Ok((
        StatusCode::OK,
        [("content-type", "application/json")],
        final_body,
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

    let upstream_start = std::time::Instant::now();
    let response = builder
        .send()
        .await
        .map_err(|e| AppError::Internal(e.into()))?;
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => "simple")
        .record(upstream_start.elapsed().as_secs_f64());

    let status = response.status();
    if !status.is_success() {
        let body = response.bytes().await.unwrap_or_else(|_| Bytes::from("{}"));
        log_upstream_error(status, url, &body);
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
    mut opts: ProxyOpts,
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

    // Report usage for cloud API key requests (before encryption, needs plaintext fields)
    try_report_usage(&response_data, &chat_id, &opts);

    // Apply response transform (e.g., encryption) before hashing/signing.
    // The signature covers the response bytes the client actually receives.
    if let Some(transform) = opts.response_transform.take() {
        transform(&mut response_data)?;
    }

    // Serialize with compact separators (matching Python's separators=(",",":"))
    let final_body =
        serde_json::to_string(&response_data).map_err(|e| AppError::Internal(e.into()))?;
    let response_sha256 = hex::encode(Sha256::digest(final_body.as_bytes()));

    // Sign and cache
    let text = format!("{}:{request_sha256}:{response_sha256}", opts.model_name);
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&chat_id, &signed_json);

    Ok(Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(final_body))
        .unwrap())
}

/// Proxy an already-received streaming SSE response. Hashes all chunks, signs at end, caches.
/// Used by catch-all when content-type is already known to be SSE.
pub async fn proxy_streaming_response(
    response: reqwest::Response,
    request_sha256: &str,
    opts: ProxyOpts,
    status: StatusCode,
) -> Result<Response, AppError> {
    let signing = opts.signing.clone();
    let cache = opts.cache.clone();
    let usage_reporter = opts.usage_reporter.clone();
    let model_name = opts.model_name.clone();
    let chunk_transform = opts.chunk_transform;
    let request_sha256 = request_sha256.to_string();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

    let byte_stream = response.bytes_stream();
    tokio::spawn(async move {
        use futures_util::StreamExt;

        let _guard = StreamingGuard::new();

        let mut byte_stream = std::pin::pin!(byte_stream);
        let mut hasher = Sha256::new();
        let mut parser = SseParser::new();
        let mut upstream_error = false;
        let mut downstream_closed = false;
        let mut transformer = chunk_transform.map(SseTransformer::new);

        loop {
            tokio::select! {
                chunk = byte_stream.next() => {
                    match chunk {
                        Some(Ok(chunk)) => {
                            parser.process_chunk(&chunk);

                            // Transform (encrypt) the chunk if needed, then hash
                            // what the client actually receives for signatures.
                            let to_send = if let Some(ref mut xform) = transformer {
                                match xform.process_chunk(&chunk) {
                                    Ok(transformed) => transformed,
                                    Err(e) => {
                                        error!(error = %e, "Stream encryption failed");
                                        let _ = tx.send(Err(std::io::Error::other(
                                            "Stream encryption failed",
                                        ))).await;
                                        upstream_error = true;
                                        break;
                                    }
                                }
                            } else {
                                chunk
                            };

                            hasher.update(&to_send);

                            if tx.send(Ok(to_send)).await.is_err() {
                                downstream_closed = true;
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!(error = %e, "Error reading upstream stream");
                            upstream_error = true;
                            let _ = tx.send(Err(std::io::Error::other(e.to_string()))).await;
                            break;
                        }
                        None => break, // stream ended
                    }
                }
                _ = tx.closed() => {
                    info!("Client disconnected, aborting upstream stream processing");
                    downstream_closed = true;
                    break;
                }
            }
        }

        // Flush any remaining buffered content in the transformer
        if !upstream_error && !downstream_closed {
            if let Some(ref mut xform) = transformer {
                match xform.flush() {
                    Ok(flushed) if !flushed.is_empty() => {
                        hasher.update(&flushed);
                        if tx.send(Ok(flushed)).await.is_err() {
                            downstream_closed = true;
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Stream encryption flush failed");
                        let _ = tx
                            .send(Err(std::io::Error::other("Stream encryption failed")))
                            .await;
                        upstream_error = true;
                    }
                    _ => {}
                }
            }
        }

        if !upstream_error && !downstream_closed && parser.seen_done {
            let response_sha256 = hex::encode(hasher.finalize());
            if let Some(ref id) = parser.chat_id {
                let text = format!("{model_name}:{request_sha256}:{response_sha256}");
                match signing.sign_chat(&text) {
                    Ok(signed) => {
                        if let Ok(signed_json) = serde_json::to_string(&signed) {
                            cache.set_chat(id, &signed_json);
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Signing failed for streaming response");
                    }
                }

                // Report usage for cloud API key requests
                if let (Some(reporter), Some((input, output))) = (&usage_reporter, parser.usage) {
                    let body = serde_json::json!({
                        "type": "chat_completion",
                        "model": reporter.model_name,
                        "input_tokens": input,
                        "output_tokens": output,
                        "id": id,
                    });
                    spawn_usage_report(reporter, body);
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

/// Drop guard that tracks the streaming_connections gauge.
/// Increments on creation, decrements on drop — guarantees they stay paired.
struct StreamingGuard;

impl StreamingGuard {
    fn new() -> Self {
        metrics::gauge!("streaming_connections").increment(1);
        Self
    }
}

impl Drop for StreamingGuard {
    fn drop(&mut self) {
        metrics::gauge!("streaming_connections").decrement(1);
    }
}

/// Line-buffered SSE parser that handles data split across chunk boundaries.
/// Extracts `chat_id` from the first JSON chunk and detects the `[DONE]` marker.
pub struct SseParser {
    line_buffer: String,
    pub chat_id: Option<String>,
    pub seen_done: bool,
    /// Token usage extracted from the final SSE chunk (prompt_tokens, completion_tokens).
    pub usage: Option<(i64, i64)>,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            line_buffer: String::new(),
            chat_id: None,
            seen_done: false,
            usage: None,
        }
    }

    pub fn process_chunk(&mut self, chunk: &[u8]) {
        match std::str::from_utf8(chunk) {
            Ok(s) => self.line_buffer.push_str(s),
            Err(_) => self.line_buffer.push_str(&String::from_utf8_lossy(chunk)),
        }

        // Process all complete lines in the buffer.
        // We extract state changes from borrowed data first, then mutate,
        // to avoid allocating a String copy of each line.
        loop {
            let Some(newline_pos) = self.line_buffer.find('\n') else {
                break;
            };

            let line_end =
                if newline_pos > 0 && self.line_buffer.as_bytes()[newline_pos - 1] == b'\r' {
                    newline_pos - 1
                } else {
                    newline_pos
                };

            // Borrow the line from the buffer, extract what we need, then release the borrow
            let (is_done, extracted_id, extracted_usage) = {
                let line = &self.line_buffer[..line_end];
                let data = line
                    .strip_prefix("data: ")
                    .or_else(|| line.strip_prefix("data:"))
                    .unwrap_or(line)
                    .trim();

                if data.is_empty() {
                    (false, None, None)
                } else if data == "[DONE]" {
                    (true, None, None)
                } else if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                    let id = if self.chat_id.is_none() {
                        parsed
                            .get("id")
                            .and_then(|id| id.as_str().map(String::from))
                    } else {
                        None
                    };
                    // Capture usage from any chunk that has it (typically the final one)
                    let usage = parsed
                        .get("usage")
                        .filter(|u| u.is_object())
                        .and_then(|usage| {
                            let input = usage
                                .get("prompt_tokens")
                                .and_then(|v| v.as_i64())
                                .unwrap_or(0);
                            let output = usage
                                .get("completion_tokens")
                                .and_then(|v| v.as_i64())
                                .unwrap_or(0);
                            if input > 0 || output > 0 {
                                Some((input, output))
                            } else {
                                None
                            }
                        });
                    (false, id, usage)
                } else {
                    (false, None, None)
                }
            };

            if is_done {
                self.seen_done = true;
            }
            if let Some(id) = extracted_id {
                self.chat_id = Some(id);
            }
            if let Some(usage) = extracted_usage {
                self.usage = Some(usage);
            }

            // Remove the processed line in-place (no allocation, just memmove)
            self.line_buffer.drain(..newline_pos + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::ChatCache;
    use crate::encryption::ChunkTransform;
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
            model_name: "test-model".to_string(),
            usage_reporter: None,
            usage_type: UsageType::default(),
            request_hash: None,
            response_transform: None,
            chunk_transform: None,
        }
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_empty_body() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));

        let result = sign_and_cache_json_response(b"", &request_sha256, opts, StatusCode::OK).await;

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
            opts,
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
        let cache = opts.cache.clone();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));
        let body = br#"{"id":"existing-id","text":"hello"}"#;

        let result =
            sign_and_cache_json_response(body, &request_sha256, opts, StatusCode::OK).await;

        let resp = result.expect("valid JSON should succeed");
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(parsed["id"], "existing-id");

        // Signature should be cached under "existing-id"
        assert!(cache.get_chat("existing-id").is_some());
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_valid_body_without_id() {
        let opts = test_proxy_opts();
        let cache = opts.cache.clone();
        let request_sha256 = hex::encode(Sha256::digest(b"test-request"));
        let body = br#"{"text":"hello"}"#;

        let result =
            sign_and_cache_json_response(body, &request_sha256, opts, StatusCode::OK).await;

        let resp = result.expect("valid JSON without id should succeed");
        let resp_body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        let id = parsed["id"].as_str().unwrap();
        assert!(id.starts_with("test-"), "should generate id with prefix");

        // Signature should be cached under the generated id
        assert!(cache.get_chat(id).is_some());
    }

    #[tokio::test]
    async fn test_sign_and_cache_json_preserves_status_code() {
        let opts = test_proxy_opts();
        let request_sha256 = hex::encode(Sha256::digest(b"test"));
        let body = br#"{"id":"s1"}"#;

        let resp = sign_and_cache_json_response(body, &request_sha256, opts, StatusCode::CREATED)
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    /// Verifies that the streaming task exits promptly when the client disconnects
    /// (i.e. the response body receiver is dropped) even if the upstream is still
    /// producing data. Without the `tx.closed()` branch in `select!`, the task
    /// would block on `byte_stream.next().await` indefinitely.
    #[tokio::test]
    async fn test_streaming_task_cancels_on_client_disconnect() {
        use std::time::Duration;

        // Simulate a slow upstream that hasn't sent anything yet.
        // We keep _upstream_tx alive so the upstream channel stays open — this means
        // byte_stream.next() will block forever waiting for data, which is exactly
        // the scenario that tx.closed() needs to rescue us from.
        let (_upstream_tx, upstream_rx) =
            tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(1);

        // This is the downstream channel (proxy -> client)
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);

        let handle = tokio::spawn(async move {
            use futures_util::StreamExt;
            let byte_stream = tokio_stream::wrappers::ReceiverStream::new(upstream_rx);
            let mut byte_stream = std::pin::pin!(byte_stream);

            loop {
                tokio::select! {
                    chunk = byte_stream.next() => {
                        match chunk {
                            Some(Ok(data)) => {
                                if tx.send(Ok(data)).await.is_err() {
                                    break;
                                }
                            }
                            _ => break,
                        }
                    }
                    _ = tx.closed() => {
                        // Client disconnected — exit immediately
                        break;
                    }
                }
            }
        });

        // Drop the receiver to simulate client disconnect
        drop(rx);

        // The task should exit promptly thanks to tx.closed().
        // Without the tx.closed() branch in select!, the task would block forever
        // on byte_stream.next() since _upstream_tx is alive and never sends data.
        let result = tokio::time::timeout(Duration::from_millis(100), handle).await;
        assert!(
            result.is_ok(),
            "Streaming task should exit promptly on client disconnect"
        );
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

    #[test]
    fn test_parse_upstream_error_vllm_flat_format() {
        let body = br#"{"object":"error","message":"This model's maximum context length is 2048 tokens","type":"BadRequestError","param":null,"code":400}"#;
        let info = parse_upstream_error(body).unwrap();
        assert_eq!(
            info.message,
            "This model's maximum context length is 2048 tokens"
        );
        assert_eq!(info.error_type, "BadRequestError");
    }

    #[test]
    fn test_parse_upstream_error_nested_format() {
        let body = br#"{"error":{"message":"model not found","type":"not_found"}}"#;
        let info = parse_upstream_error(body).unwrap();
        assert_eq!(info.message, "model not found");
        assert_eq!(info.error_type, "not_found");
    }

    #[test]
    fn test_parse_upstream_error_nested_missing_type() {
        let body = br#"{"error":{"message":"something went wrong"}}"#;
        let info = parse_upstream_error(body).unwrap();
        assert_eq!(info.message, "something went wrong");
        assert_eq!(info.error_type, "unknown");
    }

    #[test]
    fn test_parse_upstream_error_non_json() {
        assert!(parse_upstream_error(b"internal secret error details").is_none());
    }

    #[test]
    fn test_parse_upstream_error_missing_message() {
        let body = br#"{"type":"BadRequestError","code":400}"#;
        assert!(parse_upstream_error(body).is_none());
    }

    #[test]
    fn test_parse_upstream_error_empty_body() {
        assert!(parse_upstream_error(b"").is_none());
    }

    #[test]
    fn test_parse_upstream_error_empty_json() {
        assert!(parse_upstream_error(b"{}").is_none());
    }

    // ── Fix 1: SseTransformer line buffering and fail-closed tests ──

    #[test]
    fn test_sse_transformer_data_split_across_chunks() {
        // Simulate a "data: {...}\n" line split across two TCP chunks
        let transform: ChunkTransform = Arc::new(|v| {
            if let Some(s) = v
                .get_mut("text")
                .and_then(|t| t.as_str().map(|s| s.to_string()))
            {
                v["text"] = serde_json::Value::String(format!("ENC:{s}"));
            }
            Ok(())
        });

        let mut transformer = SseTransformer::new(transform);

        // First chunk: incomplete line
        let out1 = transformer.process_chunk(b"data: {\"text\":\"hel").unwrap();
        assert_eq!(out1.as_ref(), b""); // No complete line yet

        // Second chunk: completes the line
        let out2 = transformer.process_chunk(b"lo\"}\n").unwrap();
        let out_str = std::str::from_utf8(&out2).unwrap();
        assert!(out_str.contains("\"ENC:hello\""), "Got: {out_str}");
    }

    #[test]
    fn test_sse_transformer_multiple_lines_in_one_chunk() {
        let transform: ChunkTransform = Arc::new(|v| {
            if let Some(s) = v
                .get_mut("x")
                .and_then(|t| t.as_str().map(|s| s.to_string()))
            {
                v["x"] = serde_json::Value::String(format!("T:{s}"));
            }
            Ok(())
        });

        let mut transformer = SseTransformer::new(transform);
        let chunk = b"data: {\"x\":\"a\"}\ndata: {\"x\":\"b\"}\n\n";
        let out = transformer.process_chunk(chunk).unwrap();
        let out_str = std::str::from_utf8(&out).unwrap();
        assert!(out_str.contains("\"T:a\""), "Got: {out_str}");
        assert!(out_str.contains("\"T:b\""), "Got: {out_str}");
    }

    #[test]
    fn test_sse_transformer_fail_closed_on_bad_json() {
        let transform: ChunkTransform = Arc::new(|_| Ok(()));

        let mut transformer = SseTransformer::new(transform);
        // Invalid JSON in data line
        let result = transformer.process_chunk(b"data: {not json}\n");
        assert!(result.is_err(), "Should fail-closed on bad JSON");
    }

    #[test]
    fn test_sse_transformer_fail_closed_on_transform_error() {
        let transform: ChunkTransform =
            Arc::new(|_| Err(AppError::Internal(anyhow::anyhow!("transform failed"))));

        let mut transformer = SseTransformer::new(transform);
        let result = transformer.process_chunk(b"data: {\"x\":1}\n");
        assert!(result.is_err(), "Should fail-closed on transform error");
    }

    #[test]
    fn test_sse_transformer_passes_through_done_and_empty_lines() {
        let transform: ChunkTransform = Arc::new(|_| Ok(()));

        let mut transformer = SseTransformer::new(transform);
        let chunk = b"data: [DONE]\n\n";
        let out = transformer.process_chunk(chunk).unwrap();
        let out_str = std::str::from_utf8(&out).unwrap();
        assert!(out_str.contains("[DONE]"));
    }

    #[test]
    fn test_sse_transformer_flush_incomplete_line() {
        // Simulate a data line that arrives without a trailing newline (stream ends mid-line).
        let transform: ChunkTransform = Arc::new(|val| {
            // Uppercase the "text" field to prove the transform ran
            if let Some(t) = val
                .get_mut("text")
                .and_then(|v| v.as_str().map(|s| s.to_uppercase()))
            {
                val["text"] = serde_json::Value::String(t);
            }
            Ok(())
        });

        let mut transformer = SseTransformer::new(transform);

        // Send a partial chunk with no trailing newline
        let chunk = b"data: {\"text\":\"hello\"}";
        let out = transformer.process_chunk(chunk).unwrap();
        // Should be buffered, nothing emitted yet
        assert!(
            out.is_empty(),
            "Expected buffered, got: {:?}",
            std::str::from_utf8(&out)
        );

        // Flush should emit the transformed line
        let flushed = transformer.flush().unwrap();
        let flushed_str = std::str::from_utf8(&flushed).unwrap();
        assert!(
            flushed_str.contains("HELLO"),
            "Expected transformed text, got: {flushed_str}"
        );

        // Second flush should be empty
        let flushed2 = transformer.flush().unwrap();
        assert!(flushed2.is_empty());
    }

    #[test]
    fn test_sse_transformer_flush_empty_buffer() {
        let transform: ChunkTransform = Arc::new(|_| Ok(()));

        let mut transformer = SseTransformer::new(transform);
        let flushed = transformer.flush().unwrap();
        assert!(flushed.is_empty());
    }

    #[test]
    fn test_sse_transformer_flush_done_marker() {
        // A [DONE] marker buffered without trailing newline should pass through unchanged.
        let transform: ChunkTransform = Arc::new(|_| Ok(()));

        let mut transformer = SseTransformer::new(transform);
        let chunk = b"data: [DONE]";
        let out = transformer.process_chunk(chunk).unwrap();
        assert!(out.is_empty());

        let flushed = transformer.flush().unwrap();
        let flushed_str = std::str::from_utf8(&flushed).unwrap();
        assert!(flushed_str.contains("[DONE]"));
    }

    // --- sanitize_validation_errors tests ---

    #[test]
    fn test_sanitize_strips_input_from_validation_errors() {
        let message = concat!(
            "2 validation errors:\n",
            "  {'type': 'value_error', 'loc': ('body', 'messages', 1), 'msg': \"Value error, invalid role\", 'input': 'user', 'ctx': {'error': ValueError(\"bad\")}}\n",
            "  {'type': 'string_type', 'loc': ('body', 'messages', 1, 'content'), 'msg': 'Input should be a valid string', 'input': [{'text': 'secret user conversation content', 'type': 'custom'}]}"
        );
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("secret user conversation"));
        assert!(!result.contains("'input':"));
        assert!(result.contains("2 validation errors:"));
        assert!(result.contains("value_error: Value error, invalid role"));
        assert!(result.contains("string_type: Input should be a valid string"));
    }

    #[test]
    fn test_sanitize_strips_ctx_only_lines() {
        let message = concat!(
            "1 validation errors:\n",
            "  {'type': 'value_error', 'msg': 'bad', 'ctx': {'error': ValueError('secret data')}}"
        );
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("secret data"));
        assert!(result.contains("value_error: bad"));
    }

    #[test]
    fn test_sanitize_strips_stack_traces() {
        let message = concat!(
            "1 validation errors:\n",
            "  {'type': 'value_error', 'msg': 'bad', 'input': 'x'}\n",
            "  File \"/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py\", line 1324\n",
            "    POST /v1/chat/completions some data"
        );
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("sgl-workspace"));
        assert!(!result.contains("POST /v1/chat"));
    }

    #[test]
    fn test_sanitize_preserves_non_validation_errors() {
        let message = "Context length exceeded: 32768 tokens requested, 16384 max";
        assert_eq!(sanitize_validation_errors(message), message);
    }

    #[test]
    fn test_sanitize_handles_non_dict_lines_with_input() {
        let message = concat!(
            "1 validation errors:\n",
            "  - {'type': 'value_error', 'msg': 'bad', 'input': 'secret user message'}"
        );
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("secret user message"));
        assert!(result.contains("value_error: bad"));
    }

    #[test]
    fn test_parse_upstream_error_sanitizes_sglang_validation() {
        let body = serde_json::json!({
            "object": "error",
            "message": "1 validation errors:\n  {'type': 'value_error', 'msg': 'bad request', 'input': 'sensitive user data', 'ctx': {'error': ValueError('details')}}"
        });
        let body_bytes = serde_json::to_vec(&body).unwrap();
        let info = parse_upstream_error(&body_bytes).unwrap();
        assert!(!info.message.contains("sensitive user data"));
        assert!(info.message.contains("value_error: bad request"));
    }

    #[test]
    fn test_parse_upstream_error_sanitizes_nested_format() {
        let body = serde_json::json!({
            "error": {
                "message": "1 validation errors:\n  {'type': 'string_type', 'msg': 'bad input', 'input': [{'text': 'secret conversation'}]}",
                "type": "invalid_request_error"
            }
        });
        let body_bytes = serde_json::to_vec(&body).unwrap();
        let info = parse_upstream_error(&body_bytes).unwrap();
        assert!(!info.message.contains("secret conversation"));
        assert!(info.message.contains("string_type: bad input"));
        assert_eq!(info.error_type, "invalid_request_error");
    }

    // --- pydantic v2 format tests ---

    #[test]
    fn test_sanitize_strips_pydantic_v2_input_value() {
        // Real vLLM pydantic v2 error format
        let message = concat!(
            "7 validation errors for ValidatorIterator\n",
            "0.ChatCompletionContentPartTextParam.text\n",
            "  Field required [type=missing, input_value={'content': 'secret user message', 'type': 'custom'}, input_type=dict]\n",
            "    For further information visit https://errors.pydantic.dev/2.10/v/missing\n",
            "0.ChatCompletionContentPartTextParam.type\n",
            "  Input should be 'text' [type=literal_error, input_value='custom', input_type=str]\n",
            "    For further information visit https://errors.pydantic.dev/2.10/v/literal_error"
        );
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("secret user message"), "leaked user content: {result}");
        assert!(!result.contains("input_value="), "leaked input_value: {result}");
        assert!(!result.contains("input_type="), "leaked input_type: {result}");
        assert!(!result.contains("pydantic.dev"), "leaked URL: {result}");
        assert!(result.contains("Field required [type=missing]"), "missing error desc: {result}");
        assert!(result.contains("Input should be 'text' [type=literal_error]"), "missing error desc: {result}");
        assert!(result.contains("0.ChatCompletionContentPartTextParam.text"), "missing field path: {result}");
    }

    #[test]
    fn test_sanitize_strips_pydantic_v2_nested_dict() {
        // input_value with deeply nested user content
        let message = "  Field required [type=missing, input_value={'messages': [{'role': 'user', 'content': 'tell me your secrets'}]}, input_type=dict]";
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("tell me your secrets"), "leaked user content: {result}");
        assert!(result.contains("Field required [type=missing]"));
    }

    #[test]
    fn test_parse_upstream_error_sanitizes_pydantic_v2() {
        // Full vLLM error response with pydantic v2 format
        let body = serde_json::json!({
            "message": "7 validation errors for ValidatorIterator\n0.ChatCompletionContentPartTextParam.text\n  Field required [type=missing, input_value={'file_id': 'file-abc', 'type': 'file'}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.10/v/missing",
            "type": "invalid_request_error"
        });
        let body_bytes = serde_json::to_vec(&body).unwrap();
        let info = parse_upstream_error(&body_bytes).unwrap();
        assert!(!info.message.contains("input_value"), "leaked input_value: {}", info.message);
        assert!(!info.message.contains("file-abc"), "leaked file id: {}", info.message);
        assert!(!info.message.contains("pydantic.dev"), "leaked URL: {}", info.message);
        assert!(info.message.contains("Field required [type=missing]"));
    }

    #[test]
    fn test_sanitize_mixed_python_dict_and_pydantic_v2() {
        // Mixed format (both SGLang and vLLM style in one message)
        let message = concat!(
            "2 errors:\n",
            "  {'type': 'value_error', 'msg': 'bad role', 'input': 'secret data'}\n",
            "  Input should be 'text' [type=literal_error, input_value='secret', input_type=str]"
        );
        let result = sanitize_validation_errors(message);
        assert!(!result.contains("secret"), "leaked data: {result}");
        assert!(result.contains("value_error: bad role"));
        assert!(result.contains("Input should be 'text' [type=literal_error]"));
    }
}
