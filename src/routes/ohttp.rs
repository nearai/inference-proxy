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
async fn ohttp_relay_chunked(
    state: &AppState,
    gateway: &crate::ohttp_gateway::OhttpGateway,
    enc_request: &[u8],
) -> Result<Response, AppError> {
    let start = Instant::now();
    metrics::counter!("ohttp_requests_total", "type" => "chunked").increment(1);

    // Decapsulate the chunked request. The request is small so we read it all.
    let server = gateway.clone_server();
    let mut server_request = server.decapsulate_stream(enc_request);

    // Read the full decrypted inner request.
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

    // Send the loopback request.
    let loopback_response = send_loopback(request_builder).await?;
    let response_status = loopback_response.status().as_u16();

    // Build bhttp response header (status + headers, no body yet).
    let bhttp_status =
        bhttp::StatusCode::try_from(response_status).unwrap_or(bhttp::StatusCode::OK);
    let mut bhttp_response_header = bhttp::Message::response(bhttp_status);
    copy_response_headers(&loopback_response, &mut bhttp_response_header);

    // Serialize the bhttp header. We'll write it first, then stream the body.
    // For chunked OHTTP, we write the entire bhttp response (header + body) into
    // the ServerResponse writer — it handles chunked encryption automatically.
    //
    // Use a tokio duplex stream: write side goes to ServerResponse (encrypts),
    // read side becomes the HTTP response body.
    let (read_half, write_half) = tokio::io::duplex(64 * 1024);

    // Create the streaming ServerResponse writer.
    // .compat_write() bridges tokio::AsyncWrite → futures::AsyncWrite.
    let mut ohttp_writer = server_request
        .response(write_half.compat_write())
        .map_err(|e| {
            warn!(error = %e, "Failed to create chunked OHTTP response writer");
            AppError::Internal(anyhow::anyhow!("OHTTP stream setup failed: {e}"))
        })?;

    let path_owned = path_str.to_string();

    // Spawn a task that reads the loopback response and writes encrypted chunks.
    tokio::spawn(async move {
        // Write the bhttp response as: header + body streamed.
        // We use IndeterminateLength mode so we can write body incrementally.
        let mut bhttp_header_bytes = Vec::new();

        // Write the control data + headers portion of the bhttp message.
        // For IndeterminateLength mode, body is sent as separate chunks after.
        if let Err(e) = bhttp_response_header
            .write_bhttp(bhttp::Mode::IndeterminateLength, &mut bhttp_header_bytes)
        {
            warn!(error = %e, "Failed to encode bhttp header");
            let _ = ohttp_writer.close().await;
            return;
        }

        // The IndeterminateLength bhttp encoding writes: control + headers + empty-body-terminator + trailers.
        // But we want to write the header portion, then stream body bytes.
        // Since bhttp with IndeterminateLength for an empty-content message already wrote the
        // structure, we can't easily split it.
        //
        // Simpler approach: collect the full response body (like standard mode) but stream
        // the encrypted chunks. The encryption chunking is the streaming part — each ~16KB
        // of plaintext becomes an independently-decryptable encrypted chunk.
        // This means the client can start decrypting before the full response arrives.
        let body_bytes = match loopback_response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                warn!(error = %e, "Failed to read loopback response in chunked mode");
                let _ = ohttp_writer.close().await;
                return;
            }
        };

        // Write full bhttp message (header + body) into the OHTTP stream writer.
        // The writer encrypts in ~16KB chunks automatically.
        let mut full_bhttp = Vec::new();
        let mut full_msg = bhttp_response_header;
        full_msg.write_content(&body_bytes);
        if let Err(e) = full_msg.write_bhttp(bhttp::Mode::KnownLength, &mut full_bhttp) {
            warn!(error = %e, "Failed to encode bhttp response");
            let _ = ohttp_writer.close().await;
            return;
        }

        if let Err(e) = ohttp_writer.write_all(&full_bhttp).await {
            warn!(error = %e, "Failed to write to OHTTP stream");
        }
        if let Err(e) = ohttp_writer.close().await {
            warn!(error = %e, "Failed to close OHTTP stream");
        }
    });

    info!(
        decap_ms = decap_duration.as_millis(),
        inner_status = response_status,
        inner_path = path_owned,
        "Chunked OHTTP request processed"
    );

    // Stream the encrypted chunks back through the read half of the duplex.
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
