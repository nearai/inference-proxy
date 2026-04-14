use std::io::Cursor;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use http_body_util::BodyExt;
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
/// Accepts an OHTTP-encapsulated request (`Content-Type: message/ohttp-req`),
/// decapsulates it, forwards the inner HTTP request to localhost, encapsulates
/// the response, and returns it.
///
/// Auth, rate limiting, and signing are applied on the inner loopback request
/// via the normal middleware stack.
pub async fn ohttp_relay(State(state): State<AppState>, body: Body) -> Result<Response, AppError> {
    let start = Instant::now();
    metrics::counter!("ohttp_requests_total", "type" => "standard").increment(1);

    let gateway = state
        .ohttp_gateway
        .as_ref()
        .ok_or_else(|| AppError::NotFound("OHTTP not enabled".to_string()))?;

    // Read the full encapsulated request body.
    let enc_request = body
        .collect()
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to read request body: {e}")))?
        .to_bytes();

    if enc_request.is_empty() {
        return Err(AppError::BadRequest("Empty OHTTP request".to_string()));
    }

    // Decapsulate (HPKE decrypt).
    let (bhttp_request, server_response) = gateway.decapsulate(&enc_request).map_err(|e| {
        metrics::counter!("ohttp_errors_total", "reason" => "decapsulation_failed").increment(1);
        warn!(error = %e, "OHTTP decapsulation failed");
        AppError::BadRequest(format!("OHTTP decapsulation failed: {e}"))
    })?;

    let decap_duration = start.elapsed();
    metrics::histogram!("ohttp_decapsulation_duration_seconds")
        .record(decap_duration.as_secs_f64());

    // Parse inner Binary HTTP request (RFC 9292).
    let inner_msg =
        bhttp::Message::read_bhttp(&mut Cursor::new(&bhttp_request[..])).map_err(|e| {
            warn!(error = %e, "Failed to parse Binary HTTP request");
            AppError::BadRequest(format!("Invalid Binary HTTP request: {e}"))
        })?;

    // Extract method and path from the inner request's control data.
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

    // Build the loopback request to ourselves.
    let loopback_url = format!("http://127.0.0.1:{}{}", state.config.listen_port, path_str);

    let mut request_builder = state.http_client.request(method, &loopback_url);

    // Copy headers from the inner Binary HTTP message.
    for field in inner_msg.header().fields() {
        let name = std::str::from_utf8(field.name()).unwrap_or_default();
        let value = std::str::from_utf8(field.value()).unwrap_or_default();
        // Skip hop-by-hop headers and host (we set our own).
        if !matches!(
            name.to_lowercase().as_str(),
            "host" | "transfer-encoding" | "connection"
        ) {
            request_builder = request_builder.header(name, value);
        }
    }

    // Set the body from the inner message content.
    let inner_content = inner_msg.content().to_vec();
    if !inner_content.is_empty() {
        request_builder = request_builder.body(inner_content);
    }

    // Send the loopback request.
    let loopback_response = request_builder.send().await.map_err(|e| {
        metrics::counter!("ohttp_errors_total", "reason" => "loopback_failed").increment(1);
        warn!(error = %e, "OHTTP loopback request failed");
        AppError::Internal(e.into())
    })?;

    // Build Binary HTTP response from the loopback response.
    let response_status = loopback_response.status().as_u16();
    let bhttp_status =
        bhttp::StatusCode::try_from(response_status).unwrap_or(bhttp::StatusCode::OK);
    let mut bhttp_response = bhttp::Message::response(bhttp_status);

    // Copy response headers.
    for (name, value) in loopback_response.headers() {
        // Skip hop-by-hop headers.
        if !matches!(
            name.as_str(),
            "transfer-encoding" | "connection" | "content-length"
        ) {
            bhttp_response.put_header(name.as_str(), value.as_bytes());
        }
    }

    // Read the full response body.
    let response_body = loopback_response.bytes().await.map_err(|e| {
        warn!(error = %e, "Failed to read loopback response body");
        AppError::Internal(e.into())
    })?;
    bhttp_response.write_content(&response_body);

    // Encode as Binary HTTP.
    let mut bhttp_bytes = Vec::new();
    bhttp_response
        .write_bhttp(bhttp::Mode::KnownLength, &mut bhttp_bytes)
        .map_err(|e| {
            warn!(error = %e, "Failed to encode Binary HTTP response");
            AppError::Internal(anyhow::anyhow!("Binary HTTP encoding failed: {e}"))
        })?;

    // Encapsulate (HPKE encrypt) the response.
    let enc_response = server_response.encapsulate(&bhttp_bytes).map_err(|e| {
        metrics::counter!("ohttp_errors_total", "reason" => "encapsulation_failed").increment(1);
        warn!(error = %e, "OHTTP response encapsulation failed");
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
