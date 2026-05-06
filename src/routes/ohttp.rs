use std::io::Cursor;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderName, HeaderValue, Method, StatusCode};
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
/// **Authorization:** An OHTTP relay can send `Authorization: Bearer …` on this
/// HTTP request (outside the encrypted BHTTP payload). When present, it is applied
/// to the inner loopback request and overrides any `Authorization` field inside
/// the decrypted Binary HTTP message—so the relay can hold the API secret while
/// clients only encrypt the request line, headers, and body they need.
///
/// If the outer request has no `Authorization` header, auth is taken from the
/// inner Binary HTTP message only (backward-compatible with clients that embed
/// the token in the encrypted envelope).
///
/// **Trusted gateway semantics:** Outer `Bearer` auth is validated the same way
/// as a direct client call. In particular, features that apply only when the caller
/// uses the deployment `config.token` (and not cloud `sk-` keys)—for example
/// honoring inner `X-Request-Hash` for signing in chat routes—will apply once
/// the relay injects that token. Encrypted inner headers still come from the end
/// client. If the relay sits between an untrusted client and this server, **the relay
/// must strip or override sensitive inner headers** (for example `X-Request-Hash`)
/// when building BHTTP so clients cannot forge trusted-gateway-only behavior while
/// the relay supplies the shared secret. Hash binding rules are documented on
/// [`resolve_request_hash_for_signing`].
///
/// [`resolve_request_hash_for_signing`]: crate::routes::chat::resolve_request_hash_for_signing
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

    let outer_authorization = request.headers().get(header::AUTHORIZATION).cloned();

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
        ohttp_relay_chunked(&state, gateway, &enc_request, outer_authorization).await
    } else {
        ohttp_relay_standard(&state, gateway, &enc_request, outer_authorization).await
    }
}

/// Standard OHTTP: decapsulate full request, encapsulate full response.
async fn ohttp_relay_standard(
    state: &AppState,
    gateway: &crate::ohttp_gateway::OhttpGateway,
    enc_request: &[u8],
    outer_authorization: Option<HeaderValue>,
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

    let (request_builder, path_str) =
        parse_bhttp_and_build_loopback(state, &bhttp_request, outer_authorization.as_ref())?;

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
/// Backend body chunks are translated 1:1 into BHTTP indeterminate-length
/// content chunks (RFC 9292 §6) and written immediately to the OHTTP writer.
/// The OHTTP layer encrypts in ~16KB AEAD chunks, so once the BHTTP framing
/// has emitted enough data the client can decrypt and surface partial output
/// while the upstream is still producing — the previous implementation
/// buffered the whole upstream body before emitting any framing, which
/// collapsed back to the same TTFT as standard OHTTP.
async fn ohttp_relay_chunked(
    state: &AppState,
    gateway: &crate::ohttp_gateway::OhttpGateway,
    enc_request: &[u8],
    outer_authorization: Option<HeaderValue>,
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

    let (request_builder, path_str) =
        parse_bhttp_and_build_loopback(state, &bhttp_request, outer_authorization.as_ref())?;

    let loopback_response = send_loopback(request_builder).await?;
    let response_status = loopback_response.status().as_u16();

    // Collect response headers (available immediately, before body) as raw
    // (name, value) pairs — we write the BHTTP framing manually so we don't
    // build a `bhttp::Message` here.
    let response_headers = collect_response_headers(&loopback_response);

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
    // BHTTP indeterminate-length framing lets us emit each upstream chunk as a
    // length-prefixed content chunk without knowing the total body size up front.
    tokio::spawn(async move {
        // Header section: framing indicator + status + headers + terminator.
        if let Err(e) = write_indeterminate_response_header(
            &mut ohttp_writer,
            response_status,
            &response_headers,
        )
        .await
        {
            warn!(error = %e, "Failed to write BHTTP response header");
            let _ = ohttp_writer.close().await;
            return;
        }

        // Body section: each upstream chunk → one BHTTP content chunk.
        let mut body_chunks = loopback_response.bytes_stream();
        while let Some(chunk_result) = body_chunks.next().await {
            match chunk_result {
                Ok(chunk) if chunk.is_empty() => continue,
                Ok(chunk) => {
                    if let Err(e) = bhttp_write_vec(&mut ohttp_writer, &chunk).await {
                        warn!(
                            error = %e,
                            "Failed to write BHTTP body chunk (client may have disconnected)"
                        );
                        let _ = ohttp_writer.close().await;
                        return;
                    }
                    // Push the encrypted bytes downstream as soon as the OHTTP
                    // layer has accumulated a full AEAD chunk — without flush
                    // it only writes when its 16KB seal buffer fills up.
                    if let Err(e) = ohttp_writer.flush().await {
                        warn!(error = %e, "Failed to flush OHTTP stream");
                        let _ = ohttp_writer.close().await;
                        return;
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Error reading backend response stream");
                    break;
                }
            }
        }

        // Body terminator (also serves as "empty body" if no chunks were written)
        // followed by an empty trailer field section.
        if let Err(e) = bhttp_write_varint(&mut ohttp_writer, 0).await {
            warn!(error = %e, "Failed to write BHTTP body terminator");
            let _ = ohttp_writer.close().await;
            return;
        }
        if let Err(e) = bhttp_write_varint(&mut ohttp_writer, 0).await {
            warn!(error = %e, "Failed to write BHTTP trailer terminator");
            let _ = ohttp_writer.close().await;
            return;
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

// ── BHTTP indeterminate-length streaming framing ─────────────────────
//
// We can't use `bhttp::Message::write_bhttp` on the streaming path because it
// requires the whole body up front (it length-prefixes everything in a single
// `write_vec`). For RFC 9292 §6 indeterminate-length framing the wire format is
// just a sequence of QUIC-style varint-prefixed byte vectors, which is small
// enough to write directly.

/// Write an RFC 9000 §16 variable-length integer.
async fn bhttp_write_varint<W>(w: &mut W, v: u64) -> std::io::Result<()>
where
    W: futures_util::AsyncWrite + Unpin,
{
    if v < (1 << 6) {
        w.write_all(&[v as u8]).await
    } else if v < (1 << 14) {
        let bytes = ((v as u16) | (1 << 14)).to_be_bytes();
        w.write_all(&bytes).await
    } else if v < (1 << 30) {
        let bytes = ((v as u32) | (2 << 30)).to_be_bytes();
        w.write_all(&bytes).await
    } else if v < (1u64 << 62) {
        let bytes = (v | (3u64 << 62)).to_be_bytes();
        w.write_all(&bytes).await
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "BHTTP varint value too large",
        ))
    }
}

/// Write a varint length-prefix followed by the bytes (BHTTP "vec" encoding).
async fn bhttp_write_vec<W>(w: &mut W, bytes: &[u8]) -> std::io::Result<()>
where
    W: futures_util::AsyncWrite + Unpin,
{
    bhttp_write_varint(w, bytes.len() as u64).await?;
    if !bytes.is_empty() {
        w.write_all(bytes).await?;
    }
    Ok(())
}

/// Write the framing indicator + control data + header field section + terminator
/// for an indeterminate-length BHTTP response.
async fn write_indeterminate_response_header<W>(
    w: &mut W,
    status: u16,
    headers: &[(Vec<u8>, Vec<u8>)],
) -> std::io::Result<()>
where
    W: futures_util::AsyncWrite + Unpin,
{
    // Framing indicator: 3 = response, indeterminate-length.
    bhttp_write_varint(w, 3).await?;
    // Control data for response: just the status code as a varint.
    bhttp_write_varint(w, u64::from(status)).await?;
    // Header field section: each (name, value) as a pair of vecs, terminated
    // by an empty-name vec (a single varint(0)).
    for (name, value) in headers {
        bhttp_write_vec(w, name).await?;
        bhttp_write_vec(w, value).await?;
    }
    bhttp_write_varint(w, 0).await?;
    Ok(())
}

/// Collect response headers as raw byte pairs, filtering hop-by-hop headers.
/// `reqwest::HeaderName::as_str` is already lowercase, which matches the
/// BHTTP requirement that field names be lowercase (RFC 9292 §3.5.2).
fn collect_response_headers(response: &reqwest::Response) -> Vec<(Vec<u8>, Vec<u8>)> {
    response
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            let n = name.as_str();
            if n.eq_ignore_ascii_case("transfer-encoding")
                || n.eq_ignore_ascii_case("connection")
                || n.eq_ignore_ascii_case("content-length")
            {
                None
            } else {
                Some((n.as_bytes().to_vec(), value.as_bytes().to_vec()))
            }
        })
        .collect()
}

// ── Shared helpers ──────────────────────────────────────────────────

/// Parse a Binary HTTP request and build a loopback reqwest request.
/// Returns (request_builder, path_str).
///
/// `outer_authorization`: if it is a usable `Bearer` value (e.g. relay-injected
/// `Authorization` on `POST /ohttp`), it is attached to the loopback request,
/// the inner `Authorization` field is skipped, and trusted-only inner headers are
/// scrubbed because the outer bearer establishes trusted-gateway semantics.
fn parse_bhttp_and_build_loopback(
    state: &AppState,
    bhttp_request: &[u8],
    outer_authorization: Option<&HeaderValue>,
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

    let relay_outer_bearer = outer_authorization.filter(|value| {
        value
            .to_str()
            .is_ok_and(|header| header.starts_with("Bearer "))
    });

    for field in inner_msg.header().fields() {
        let name_bytes = field.name();
        let value_bytes = field.value();

        let skip = name_bytes.eq_ignore_ascii_case(b"host")
            || name_bytes.eq_ignore_ascii_case(b"transfer-encoding")
            || name_bytes.eq_ignore_ascii_case(b"connection")
            || (relay_outer_bearer.is_some() && name_bytes.eq_ignore_ascii_case(b"authorization"))
            || (relay_outer_bearer.is_some() && name_bytes.eq_ignore_ascii_case(b"x-request-hash"));
        if skip {
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

    if let Some(auth) = relay_outer_bearer {
        request_builder = request_builder.header(header::AUTHORIZATION, auth.clone());
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use futures_util::io::Cursor as FuturesCursor;

    async fn varint_bytes(v: u64) -> Vec<u8> {
        let mut buf = FuturesCursor::new(Vec::new());
        bhttp_write_varint(&mut buf, v).await.unwrap();
        buf.into_inner()
    }

    #[tokio::test]
    async fn varint_encoding_matches_quic_lengths() {
        // 1-byte form: 0..=63
        assert_eq!(varint_bytes(0).await, vec![0x00]);
        assert_eq!(varint_bytes(63).await, vec![0x3f]);
        // 2-byte form: 64..=16383
        assert_eq!(varint_bytes(64).await, vec![0x40, 0x40]);
        assert_eq!(varint_bytes(16383).await, vec![0x7f, 0xff]);
        // 4-byte form: 16384..=1073741823
        assert_eq!(varint_bytes(16384).await, vec![0x80, 0x00, 0x40, 0x00]);
        // 8-byte form: large value
        let big = varint_bytes(0x3fff_ffff_ffff_ffff).await;
        assert_eq!(big.len(), 8);
        assert_eq!(big[0], 0xff);
    }

    #[tokio::test]
    async fn varint_roundtrips_through_bhttp_reader() {
        // The reader is internal to bhttp, but Message::read_bhttp parses the
        // framing indicator as a varint, so we can exercise our writer end-to-end
        // by feeding a complete indeterminate-length message into it.
        let mut buf = FuturesCursor::new(Vec::new());
        write_indeterminate_response_header(
            &mut buf,
            200,
            &[
                (b"content-type".to_vec(), b"application/json".to_vec()),
                (b"x-test".to_vec(), b"hello".to_vec()),
            ],
        )
        .await
        .unwrap();
        // Two body chunks then body terminator + empty trailer.
        bhttp_write_vec(&mut buf, b"{\"a\":").await.unwrap();
        bhttp_write_vec(&mut buf, b"1}").await.unwrap();
        bhttp_write_varint(&mut buf, 0).await.unwrap(); // body terminator
        bhttp_write_varint(&mut buf, 0).await.unwrap(); // trailer terminator

        let bytes = buf.into_inner();
        // First byte should be the framing indicator (3 = response indeterminate).
        assert_eq!(bytes[0], 3);

        let msg = bhttp::Message::read_bhttp(&mut Cursor::new(&bytes[..])).unwrap();
        assert_eq!(msg.control().status().unwrap().code(), 200);
        assert_eq!(msg.content(), b"{\"a\":1}");
        assert_eq!(
            msg.header().get(b"content-type"),
            Some(b"application/json".as_ref())
        );
        assert_eq!(msg.header().get(b"x-test"), Some(b"hello".as_ref()));
    }

    #[tokio::test]
    async fn empty_body_roundtrips() {
        let mut buf = FuturesCursor::new(Vec::new());
        write_indeterminate_response_header(&mut buf, 204, &[])
            .await
            .unwrap();
        // No body chunks. Single body terminator + trailer terminator.
        bhttp_write_varint(&mut buf, 0).await.unwrap();
        bhttp_write_varint(&mut buf, 0).await.unwrap();

        let bytes = buf.into_inner();
        let msg = bhttp::Message::read_bhttp(&mut Cursor::new(&bytes[..])).unwrap();
        assert_eq!(msg.control().status().unwrap().code(), 204);
        assert!(msg.content().is_empty());
    }
}
