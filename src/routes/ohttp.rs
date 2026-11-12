use std::io::Cursor;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

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

// OHTTP must never follow an inner request away from the local proxy, including
// redirects returned by a local route. Keep this client separate from the
// shared HTTP client used for external services.
static OHTTP_LOOPBACK_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("build OHTTP loopback client")
});

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
/// the inner `Authorization` field is skipped, and trusted-only inner headers
/// (`X-Request-Hash`, `X-NearAI-Priority`) are scrubbed because the outer bearer
/// establishes trusted-gateway semantics.
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

    let method_str = std::str::from_utf8(method_bytes).map_err(|_| {
        log_rejected_inner_request("invalid_method", "-", path_bytes);
        AppError::BadRequest("Invalid method".to_string())
    })?;
    let path_str = std::str::from_utf8(path_bytes).map_err(|_| {
        log_rejected_inner_request("invalid_path", method_str, path_bytes);
        AppError::BadRequest("Invalid path".to_string())
    })?;

    let method: Method = method_str.parse().map_err(|_| {
        log_rejected_inner_request("invalid_method", "-", path_bytes);
        AppError::BadRequest("Unsupported HTTP method".to_string())
    })?;

    let loopback_url = validated_loopback_url(state.config.listen_port, &method, path_str)
        .inspect_err(|_| log_rejected_inner_request("not_allowed", method.as_str(), path_bytes))?;
    let mut request_builder = OHTTP_LOOPBACK_CLIENT
        .request(method, loopback_url)
        .timeout(Duration::from_secs(state.config.timeout_secs));

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
            || (relay_outer_bearer.is_some() && name_bytes.eq_ignore_ascii_case(b"x-request-hash"))
            // The loopback authenticates as a trusted caller; the inner headers
            // come from the end customer, who must not pick the engine priority.
            || (relay_outer_bearer.is_some()
                && name_bytes.eq_ignore_ascii_case(crate::priority::PRIORITY_HEADER.as_bytes()));
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

/// Log and count an inner OHTTP request that was refused before any loopback call.
///
/// Inner paths are attacker-controlled and have carried scripts and tokens (incident
/// 2026-11-07), so the raw path is never logged. The log carries only what is needed to
/// spot and correlate an attack: the method (standard methods only), the path's shape, any
/// `@host:port` / `//host` / `scheme://host` forward target (host and port only, no
/// userinfo), the path with long segments and the query removed, the path length, and a
/// SHA-256 prefix of the exact bytes so repeated attempts can be matched.
fn log_rejected_inner_request(reason: &'static str, method: &str, path: &[u8]) {
    metrics::counter!("ohttp_rejected_inner_requests_total", "reason" => reason).increment(1);
    let summary = summarize_inner_path(path);
    let method = match method {
        "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD" | "OPTIONS" | "CONNECT" | "TRACE" => {
            method
        }
        "-" => "-",
        _ => "other",
    };
    warn!(
        reason,
        inner_method = method,
        inner_path_shape = summary.shape,
        inner_target = summary.target.as_deref().unwrap_or("-"),
        inner_path_redacted = %summary.redacted,
        inner_path_len = path.len(),
        inner_query_len = summary.query_len,
        inner_path_sha256 = %summary.sha256_prefix,
        "OHTTP inner request rejected"
    );
}

struct InnerPathSummary {
    /// `origin` (/...), `userinfo` (@...), `scheme_relative` (//...), `absolute` (x://...),
    /// `empty`, or `other`.
    shape: &'static str,
    /// Forward target host[:port] for `userinfo`, `scheme_relative` and `absolute` shapes.
    target: Option<String>,
    /// Path without query or fragment. A segment is kept if it is at most 24 bytes of
    /// `[A-Za-z0-9._-]`, or at most 48 bytes of `[a-z_-]` (route names); anything else
    /// becomes `<N>` (its length). At most 8 segments are kept.
    redacted: String,
    query_len: usize,
    sha256_prefix: String,
}

fn summarize_inner_path(path: &[u8]) -> InnerPathSummary {
    use sha2::{Digest, Sha256};

    const MAX_SEGMENT: usize = 24;
    const MAX_ROUTE_SEGMENT: usize = 48;
    const MAX_SEGMENTS: usize = 8;

    let sha256_prefix = hex::encode(&Sha256::digest(path)[..8]);
    let text = String::from_utf8_lossy(path);
    let (before_fragment, _) = text.split_once('#').unwrap_or((&text, ""));
    let (path_part, query) = match before_fragment.split_once('?') {
        Some((p, q)) => (p, q.len()),
        None => (before_fragment, 0),
    };

    let (shape, rest) = if path_part.is_empty() {
        ("empty", "")
    } else if let Some(r) = path_part.strip_prefix('@') {
        ("userinfo", r)
    } else if let Some(r) = path_part.strip_prefix("//") {
        ("scheme_relative", r)
    } else if let Some((scheme, r)) = path_part.split_once("://") {
        if !scheme.is_empty()
            && scheme
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
        {
            ("absolute", r)
        } else {
            ("other", path_part)
        }
    } else if path_part.starts_with('/') {
        ("origin", path_part)
    } else {
        ("other", path_part)
    };

    let (target, path_rest) = match shape {
        "userinfo" | "scheme_relative" | "absolute" => {
            let (authority, tail) = match rest.find('/') {
                Some(i) => (&rest[..i], &rest[i..]),
                None => (rest, ""),
            };
            // Drop any userinfo (user:password@) in front of the host.
            let host_port = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
            let safe = !host_port.is_empty()
                && host_port.len() <= 64
                && host_port
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".-:[]_".contains(&b));
            let target = if safe {
                host_port.to_ascii_lowercase()
            } else {
                format!("<{}>", host_port.len())
            };
            (Some(target), tail)
        }
        _ => (None, rest),
    };

    let segments: Vec<&str> = path_rest.split('/').skip(1).collect();
    let mut redacted = String::new();
    for segment in segments.iter().take(MAX_SEGMENTS) {
        redacted.push('/');
        // Short plain segments, or longer route-like names (lowercase letters and
        // `_`/`-` only, e.g. update_weights_from_tensor). Tokens mix digits or case.
        let plain = (segment.len() <= MAX_SEGMENT
            && segment
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b)))
            || (segment.len() <= MAX_ROUTE_SEGMENT
                && segment
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b == b'_' || b == b'-'));
        if plain {
            redacted.push_str(segment);
        } else {
            redacted.push_str(&format!("<{}>", segment.len()));
        }
    }
    if segments.len() > MAX_SEGMENTS {
        redacted.push_str(&format!("/...+{}", segments.len() - MAX_SEGMENTS));
    }
    if redacted.is_empty() && shape == "other" {
        redacted = format!("<{}>", path_part.len());
    }

    InnerPathSummary {
        shape,
        target,
        redacted,
        query_len: query,
        sha256_prefix,
    }
}

/// Accept only GET /v1/models and POST to chat, text, or audio completions.
/// Verify that URL parsing preserves the fixed loopback authority: a path
/// starting with `@` would turn `127.0.0.1:PORT` into URL userinfo.
fn validated_loopback_url(
    port: u16,
    method: &Method,
    path: &str,
) -> Result<reqwest::Url, AppError> {
    let invalid_target = || AppError::BadRequest("Unsupported OHTTP inner request".to_string());
    let allowed = (method == Method::GET && path == crate::routes::ROUTE_V1_MODELS)
        || (method == Method::POST
            && [
                crate::routes::ROUTE_CHAT_COMPLETIONS,
                crate::routes::ROUTE_COMPLETIONS,
                crate::routes::ROUTE_AUDIO_TRANSCRIPTIONS,
            ]
            .contains(&path));
    if !allowed {
        return Err(invalid_target());
    }

    let url = reqwest::Url::parse(&format!("http://127.0.0.1:{port}{path}"))
        .map_err(|_| invalid_target())?;
    if url.scheme() != "http"
        || url.host_str() != Some("127.0.0.1")
        || url.port_or_known_default() != Some(port)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid_target());
    }

    Ok(url)
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
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn ohttp_only_accepts_inference_routes_on_loopback() {
        let url =
            validated_loopback_url(8000, &Method::POST, crate::routes::ROUTE_CHAT_COMPLETIONS)
                .unwrap();
        assert_eq!(url.host_str(), Some("127.0.0.1"));
        assert_eq!(url.port(), Some(8000));
        assert_eq!(url.path(), crate::routes::ROUTE_CHAT_COMPLETIONS);
        assert_eq!(url.query(), None);

        let models =
            validated_loopback_url(8000, &Method::GET, crate::routes::ROUTE_V1_MODELS).unwrap();
        assert_eq!(models.path(), crate::routes::ROUTE_V1_MODELS);
        assert_eq!(models.query(), None);

        for path in [
            crate::routes::ROUTE_COMPLETIONS,
            crate::routes::ROUTE_AUDIO_TRANSCRIPTIONS,
        ] {
            assert_eq!(
                validated_loopback_url(8000, &Method::POST, path)
                    .unwrap()
                    .path(),
                path
            );
        }

        // URL parsing removes the explicit default HTTP port.
        assert_eq!(
            validated_loopback_url(80, &Method::POST, crate::routes::ROUTE_CHAT_COMPLETIONS)
                .unwrap()
                .port_or_known_default(),
            Some(80)
        );

        for (method, path) in [
            (Method::GET, "/v1/chat/completions"),
            (Method::PUT, "/v1/chat/completions"),
            (Method::POST, "/v1/models"),
            (Method::GET, "/v1/completions"),
            (Method::GET, "/v1/audio/transcriptions"),
            (Method::POST, "/update_weights_from_tensor"),
            (Method::POST, "/v1/chat/completions?debug=1"),
            (Method::GET, "/v1/models?debug=1"),
            (Method::POST, "/v1/chat/completions/"),
            (Method::POST, "/v1/chat/%63ompletions"),
            (Method::POST, "@example.com/x"),
            (Method::POST, "//example.com/x"),
            (Method::POST, "http://example.com/x"),
            (Method::POST, ""),
        ] {
            assert!(
                matches!(
                    validated_loopback_url(8000, &method, path),
                    Err(AppError::BadRequest(_))
                ),
                "unexpectedly accepted {method} {path:?}"
            );
        }
    }

    #[test]
    fn rejected_inner_path_summary_keeps_the_target_and_drops_secrets() {
        // The 2026-11 attack shape: userinfo forward to a webshell with a command in the query.
        let s =
            summarize_inner_path(b"@172.18.0.5:9998/run?cmd=curl%20-H%20Bearer%20sk-SECRETSECRET");
        assert_eq!(s.shape, "userinfo");
        assert_eq!(s.target.as_deref(), Some("172.18.0.5:9998"));
        assert_eq!(s.redacted, "/run");
        assert_eq!(
            s.query_len,
            "cmd=curl%20-H%20Bearer%20sk-SECRETSECRET".len()
        );
        assert_eq!(s.sha256_prefix.len(), 16);

        // Userinfo inside the authority is dropped; long or odd segments become lengths.
        let s = summarize_inner_path(
            b"//admin:hunter2@10.0.2.2:8443/prpc/CreateVm/eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig",
        );
        assert_eq!(s.shape, "scheme_relative");
        assert_eq!(s.target.as_deref(), Some("10.0.2.2:8443"));
        assert_eq!(s.redacted, "/prpc/CreateVm/<40>");

        let s = summarize_inner_path(b"http://Example.COM/x");
        assert_eq!(
            (s.shape, s.target.as_deref()),
            ("absolute", Some("example.com"))
        );

        let s = summarize_inner_path(b"/update_weights_from_tensor");
        assert_eq!((s.shape, s.target.as_deref()), ("origin", None));
        assert_eq!(s.redacted, "/update_weights_from_tensor");

        let s = summarize_inner_path(b"/v1/chat/%63ompletions#frag");
        assert_eq!(s.redacted, "/v1/chat/<13>");

        let s = summarize_inner_path(b"");
        assert_eq!((s.shape, s.redacted.as_str()), ("empty", ""));

        let s = summarize_inner_path(b"@bad host with spaces/x");
        assert_eq!(s.target.as_deref(), Some("<20>"));

        let deep = "/a".repeat(12);
        let s = summarize_inner_path(deep.as_bytes());
        assert_eq!(s.redacted, format!("{}/...+4", "/a".repeat(8)));

        // No byte of a secret ever reaches the redacted fields.
        for secret_path in [
            "@h:1/x?token=SECRETSECRET".as_bytes(),
            b"/SECRETSECRETSECRETSECRETSECRET",
            b"/x/sk-SECRET0a1b2c3d4e5f6a7b8c9d0e",
            b"//u:SECRETSECRET@h/x",
        ] {
            let s = summarize_inner_path(secret_path);
            let all = format!("{} {:?} {}", s.redacted, s.target, s.shape);
            assert!(!all.contains("SECRET"), "{all}");
        }
    }

    #[tokio::test]
    async fn ohttp_loopback_client_does_not_follow_redirects() {
        let destination = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/secret"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&destination)
            .await;

        let source = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/redirect"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("location", format!("{}/secret", destination.uri())),
            )
            .mount(&source)
            .await;

        let response = OHTTP_LOOPBACK_CLIENT
            .get(format!("{}/redirect", source.uri()))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FOUND);
        assert!(destination.received_requests().await.unwrap().is_empty());
    }

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
