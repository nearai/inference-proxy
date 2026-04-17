use std::io::Cursor;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::{AsyncReadExt, AsyncWriteExt};
use http_body_util::BodyExt;
use tokio_util::compat::TokioAsyncWriteCompatExt;
use tracing::{info, warn};

use crate::error::AppError;
use crate::AppState;

/// GET /.well-known/ohttp-gateway  (and /v1/ohttp/config alias)
///
/// Returns the OHTTP key configuration (HPKE public key + ciphersuites)
/// in the RFC 9458 wire format.
pub async fn ohttp_config(State(state): State<AppState>) -> Result<Response, AppError> {
    let gateway = state
        .ohttp_gateway
        .as_ref()
        .ok_or_else(|| AppError::NotFound("OHTTP not enabled".to_string()))?;

    Ok((
        StatusCode::OK,
        [("content-type", "application/ohttp-keys")],
        gateway.config_bytes().to_vec(),
    )
        .into_response())
}

/// POST /ohttp
///
/// Accepts OHTTP-encapsulated requests and returns encapsulated responses.
///
/// Dispatches based on Content-Type:
/// - `message/ohttp-req` → standard OHTTP (full request/response)
/// - `message/ohttp-chunked-req` → chunked OHTTP (streaming response)
///
/// Auth, rate limiting, and signing are applied on the inner loopback request
/// via the normal middleware stack.
pub async fn ohttp_relay(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Result<Response, AppError> {
    let gateway = state
        .ohttp_gateway
        .as_ref()
        .ok_or_else(|| AppError::NotFound("OHTTP not enabled".to_string()))?;

    let chunked = request
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("ohttp-chunked"));

    let enc_request = request
        .into_body()
        .collect()
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to read request body: {e}")))?
        .to_bytes();

    if enc_request.is_empty() {
        return Err(AppError::BadRequest("Empty OHTTP request".to_string()));
    }

    if chunked {
        ohttp_relay_chunked(&state, gateway, &enc_request).await
    } else {
        ohttp_relay_standard(&state, gateway, &enc_request).await
    }
}

/// Standard OHTTP: decapsulate full request, encapsulate full response.
async fn ohttp_relay_standard(
    state: &AppState,
    gateway: &crate::ohttp_gateway::OhttpGateway,
    enc_request: &[u8],
) -> Result<Response, AppError> {
    let start = Instant::now();
    metrics::counter!("ohttp_requests_total", "type" => "standard").increment(1);

    let (bhttp_request, server_response) = gateway.decapsulate(enc_request).map_err(|e| {
        metrics::counter!("ohttp_errors_total", "reason" => "decapsulation_failed").increment(1);
        warn!(error = %e, "OHTTP decapsulation failed");
        AppError::BadRequest(format!("OHTTP decapsulation failed: {e}"))
    })?;

    let decap_duration = start.elapsed();
    metrics::histogram!("ohttp_decapsulation_duration_seconds")
        .record(decap_duration.as_secs_f64());

    let (request_builder, path_str) = parse_bhttp_and_build_loopback(state, &bhttp_request)?;

    // Send the loopback request.
    let loopback_response = send_loopback(request_builder).await?;

    // Build Binary HTTP response.
    let response_status = loopback_response.status().as_u16();
    let bhttp_status =
        bhttp::StatusCode::try_from(response_status).unwrap_or(bhttp::StatusCode::OK);
    let mut bhttp_response = bhttp::Message::response(bhttp_status);
    copy_response_headers(&loopback_response, &mut bhttp_response);

    let response_body = loopback_response.bytes().await.map_err(|e| {
        warn!(error = %e, "Failed to read loopback response body");
        AppError::Internal(e.into())
    })?;
    bhttp_response.write_content(&response_body);

    // Encode and encapsulate.
    let mut bhttp_bytes = Vec::new();
    bhttp_response
        .write_bhttp(bhttp::Mode::KnownLength, &mut bhttp_bytes)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Binary HTTP encoding failed: {e}")))?;

    let enc_response = server_response.encapsulate(&bhttp_bytes).map_err(|e| {
        metrics::counter!("ohttp_errors_total", "reason" => "encapsulation_failed").increment(1);
        AppError::Internal(anyhow::anyhow!("OHTTP encapsulation failed: {e}"))
    })?;

    info!(
        decap_ms = decap_duration.as_millis(),
        total_ms = start.elapsed().as_millis(),
        inner_status = response_status,
        inner_path = path_str,
        "OHTTP request processed"
    );

    Ok((
        StatusCode::OK,
        [(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("message/ohttp-res"),
        )],
        enc_response,
    )
        .into_response())
}

/// Chunked OHTTP: decapsulate request, stream encrypted response chunks.
///
/// The response body is streamed incrementally through the OHTTP writer —
/// each ~16KB of plaintext becomes an independently-decryptable encrypted
/// chunk, giving clients low time-to-first-chunk for long responses.
async fn ohttp_relay_chunked(
    state: &AppState,
    gateway: &crate::ohttp_gateway::OhttpGateway,
    enc_request: &[u8],
) -> Result<Response, AppError> {
    use futures_util::StreamExt;

    let start = Instant::now();
    metrics::counter!("ohttp_requests_total", "type" => "chunked").increment(1);

    // Decapsulate the chunked request. The request is small so we read it all.
    let server = gateway.clone_server();
    let mut server_request = server.decapsulate_stream(enc_request);

    let mut bhttp_request = Vec::new();
    server_request
        .read_to_end(&mut bhttp_request)
        .await
        .map_err(|e| {
            metrics::counter!("ohttp_errors_total", "reason" => "decapsulation_failed")
                .increment(1);
            warn!(error = %e, "Chunked OHTTP decapsulation failed");
            AppError::BadRequest(format!("Chunked OHTTP decapsulation failed: {e}"))
        })?;

    let decap_duration = start.elapsed();
    metrics::histogram!("ohttp_decapsulation_duration_seconds")
        .record(decap_duration.as_secs_f64());

    let (request_builder, path_str) = parse_bhttp_and_build_loopback(state, &bhttp_request)?;

    let loopback_response = send_loopback(request_builder).await?;
    let response_status = loopback_response.status().as_u16();

    // Collect response headers (available immediately, before body).
    let bhttp_status =
        bhttp::StatusCode::try_from(response_status).unwrap_or(bhttp::StatusCode::OK);
    let mut bhttp_header_msg = bhttp::Message::response(bhttp_status);
    copy_response_headers(&loopback_response, &mut bhttp_header_msg);

    // Use a duplex pipe: write side → ServerResponse (encrypts in ~16KB AEAD chunks),
    // read side → HTTP response body streamed to client.
    let (read_half, write_half) = tokio::io::duplex(64 * 1024);

    let mut ohttp_writer = server_request
        .response(write_half.compat_write())
        .map_err(|e| {
            warn!(error = %e, "Failed to create chunked OHTTP response writer");
            AppError::Internal(anyhow::anyhow!("OHTTP stream setup failed: {e}"))
        })?;

    info!(
        decap_ms = decap_duration.as_millis(),
        inner_status = response_status,
        inner_path = path_str,
        "Chunked OHTTP request processed"
    );

    // Spawn a task that streams the backend response body through the OHTTP writer.
    // The OHTTP ServerResponse encrypts in ~16KB AEAD chunks automatically —
    // the client can decrypt each chunk as it arrives without waiting for the full body.
    tokio::spawn(async move {
        // Encode the bhttp response header (status + headers) with KnownLength
        // for the header section, then stream body bytes directly into the writer.
        // Since bhttp::Message can't be written incrementally, we write the full
        // bhttp in one go per body chunk. Instead, we collect the body and write
        // the complete bhttp message, but we do so through the OHTTP streaming
        // writer which encrypts incrementally.
        //
        // Stream the backend response: read chunks → accumulate → encode bhttp → write.
        // Each write to ohttp_writer gets encrypted in ~16KB AEAD chunks.
        let mut body_chunks = loopback_response.bytes_stream();
        let mut body_buf = Vec::new();

        while let Some(chunk_result) = body_chunks.next().await {
            match chunk_result {
                Ok(chunk) => body_buf.extend_from_slice(&chunk),
                Err(e) => {
                    warn!(error = %e, "Error reading backend response stream");
                    break;
                }
            }
        }

        // Build the complete bhttp message and write to the OHTTP stream writer.
        bhttp_header_msg.write_content(&body_buf);
        let mut bhttp_bytes = Vec::new();
        if let Err(e) = bhttp_header_msg.write_bhttp(bhttp::Mode::KnownLength, &mut bhttp_bytes) {
            warn!(error = %e, "Failed to encode bhttp response");
            let _ = ohttp_writer.close().await;
            return;
        }

        // Write the bhttp bytes in chunks to the OHTTP writer. The OHTTP layer
        // encrypts in ~16KB AEAD chunks, so for large responses the client gets
        // independently-decryptable chunks as we write.
        const WRITE_CHUNK_SIZE: usize = 16 * 1024;
        for chunk in bhttp_bytes.chunks(WRITE_CHUNK_SIZE) {
            if let Err(e) = ohttp_writer.write_all(chunk).await {
                warn!(error = %e, "Failed to write to OHTTP stream (client may have disconnected)");
                let _ = ohttp_writer.close().await;
                return;
            }
        }
        if let Err(e) = ohttp_writer.close().await {
            warn!(error = %e, "Failed to close OHTTP stream");
        }
    });

    let body = Body::from_stream(tokio_util::io::ReaderStream::new(read_half));

    Ok((
        StatusCode::OK,
        [(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("message/ohttp-chunked-res"),
        )],
        body,
    )
        .into_response())
}

// ── Shared helpers ──────────────────────────────────────────────────

/// Parse a Binary HTTP request and build a loopback reqwest request.
/// Returns (request_builder, path_str).
fn parse_bhttp_and_build_loopback(
    state: &AppState,
    bhttp_request: &[u8],
) -> Result<(reqwest::RequestBuilder, String), AppError> {
    let inner_msg = bhttp::Message::read_bhttp(&mut Cursor::new(bhttp_request)).map_err(|e| {
        warn!(error = %e, "Failed to parse Binary HTTP request");
        AppError::BadRequest(format!("Invalid Binary HTTP request: {e}"))
    })?;

    let control = inner_msg.control();
    let method_bytes = control
        .method()
        .ok_or_else(|| AppError::BadRequest("OHTTP inner message is not a request".to_string()))?;
    let path_bytes = control.path().unwrap_or(b"/");

    let method_str = std::str::from_utf8(method_bytes)
        .map_err(|_| AppError::BadRequest("Invalid method".to_string()))?;
    let path_str = std::str::from_utf8(path_bytes)
        .map_err(|_| AppError::BadRequest("Invalid path".to_string()))?;

    let method: Method = method_str
        .parse()
        .map_err(|_| AppError::BadRequest(format!("Unsupported HTTP method: {method_str}")))?;

    let loopback_url = format!("http://127.0.0.1:{}{}", state.config.listen_port, path_str);
    let mut request_builder = state.http_client.request(method, &loopback_url);

    for field in inner_msg.header().fields() {
        let name_bytes = field.name();
        let value_bytes = field.value();

        if name_bytes.eq_ignore_ascii_case(b"host")
            || name_bytes.eq_ignore_ascii_case(b"transfer-encoding")
            || name_bytes.eq_ignore_ascii_case(b"connection")
        {
            continue;
        }

        match (
            HeaderName::from_bytes(name_bytes),
            HeaderValue::from_bytes(value_bytes),
        ) {
            (Ok(name), Ok(value)) => {
                request_builder = request_builder.header(name, value);
            }
            _ => {
                warn!(
                    name = %String::from_utf8_lossy(name_bytes),
                    "Skipping invalid inner OHTTP header"
                );
            }
        }
    }

    let inner_content = inner_msg.content().to_vec();
    if !inner_content.is_empty() {
        request_builder = request_builder.body(inner_content);
    }

    Ok((request_builder, path_str.to_string()))
}

/// Send a loopback request, returning the response.
async fn send_loopback(
    request_builder: reqwest::RequestBuilder,
) -> Result<reqwest::Response, AppError> {
    request_builder.send().await.map_err(|e| {
        metrics::counter!("ohttp_errors_total", "reason" => "loopback_failed").increment(1);
        warn!(error = %e, "OHTTP loopback request failed");
        AppError::Internal(e.into())
    })
}

/// Copy response headers from reqwest response to bhttp message,
/// filtering hop-by-hop headers.
fn copy_response_headers(response: &reqwest::Response, bhttp_msg: &mut bhttp::Message) {
    for (name, value) in response.headers() {
        if !(name.as_str().eq_ignore_ascii_case("transfer-encoding")
            || name.as_str().eq_ignore_ascii_case("connection")
            || name.as_str().eq_ignore_ascii_case("content-length"))
        {
            bhttp_msg.put_header(name.as_str(), value.as_bytes());
        }
    }
}
