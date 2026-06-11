//! Pre-dispatch validation of client-supplied images.
//!
//! Inference engines (SGLang) fetch and decode the image URLs in a chat
//! request themselves. When a client sends an unreachable URL or non-image
//! content (e.g. an HTML error page from an expired Facebook image link), the
//! engine still performs the fetch + decode + multimodal-preprocessing work
//! before failing — so a flood of such requests loads the GPU and can degrade
//! serving for everyone (nearai/infra#159, #172). cloud-api's fix made these
//! non-retryable + 400 to the client, but that is *reactive*: the request
//! still reaches the engine.
//!
//! This module rejects clearly-bad images at the proxy, BEFORE the request is
//! forwarded to the engine.
//!
//! ## Conservative by design (fail-open)
//! It only rejects inputs it can *positively* identify as bad: a URL that can't
//! be fetched because of DNS/connect failures or is definitively gone
//! (404/410), content that sniffs as HTML/text/JSON, a `data:` payload that isn't decodable base64 under any
//! common alphabet, or (when enabled for Gemma-4) a raster image header that is
//! decisively non-RGB. Anything ambiguous (unknown-but-not-textual bytes,
//! unusual schemes, unknown content types) is **passed through** so the engine
//! stays the source of truth and valid multimodal traffic is never broken.
//!
//! ## SSRF
//! This introduces an outbound fetch of client-controlled URLs, so the
//! validation client is hardened against reaching internal addresses:
//!   * a dedicated [`reqwest::Client`] whose DNS resolver ([`SsrfResolver`])
//!     refuses any **domain** that resolves to a private/loopback/link-local/
//!     metadata range — this covers the initial request *and every redirect
//!     hop* (redirects are still followed, so legitimately-redirecting image
//!     URLs keep working);
//!   * a redirect policy that additionally refuses hops to a disallowed **IP
//!     literal** (the resolver isn't consulted for literals);
//!   * an explicit check on the initial URL's host when it is an IP literal.
//!
//! SGLang already fetches these URLs today, so this is a net security
//! improvement, not a new exposure.

use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use base64::Engine as _;
use futures_util::stream::{self, StreamExt};
use reqwest::header::CONTENT_TYPE;
use serde_json::Value;
use url::Host;

use crate::error::AppError;

/// Cap on remote (http/https) images fetched per request.
const MAX_IMAGES_PER_REQUEST: usize = 64;
/// Redirect hops followed by the validation client.
const MAX_REDIRECTS: usize = 5;
/// How many decoded bytes of a `data:` base64 payload to materialize for
/// type-sniffing (bounds allocation regardless of the payload's true size).
const SNIFF_DECODE_BYTES: usize = 4096;
const IMAGE_VALIDATION_USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

/// Generic, content-free rejection message. Never echoes the URL or any user
/// data (see proxy.rs sanitization rules / privacy requirements).
const REJECT_MSG: &str =
    "One or more image inputs could not be fetched or are not a supported image for this model. \
     Ensure each image URL is reachable and resolves to a compatible image.";

#[derive(Debug, Clone)]
pub struct ImageValidationConfig {
    pub enabled: bool,
    pub timeout: Duration,
    /// Max bytes downloaded from a fetched image before classifying it. Only
    /// the head is needed to sniff the type; the whole image is never buffered.
    pub max_bytes: usize,
    pub max_concurrency: usize,
    /// Permit private/loopback image hosts — both initial IP literals and
    /// domains that resolve to private ranges. Set in tests (loopback wiremock)
    /// and for trusted internal deployments; off in production. Redirect hops
    /// to private IP *literals* are always refused regardless, since redirect
    /// targets are server-controlled and must never be trusted to point inward.
    pub allow_private_hosts: bool,
    /// Reject decisively non-RGB raster images. This is strict opt-in because
    /// RGBA, CMYK, and palette handling still need real-engine verification.
    pub reject_non_rgb_images: bool,
    /// Reject observed one-channel PNG/JPEG headers for Gemma-4 deployments.
    /// This is narrower than `reject_non_rgb_images` and is auto-enabled for
    /// Gemma-4 because the production crash shape matched 1-channel patches.
    pub reject_single_channel_images: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Plausibly valid (or unknown) — let it through.
    Pass,
    /// Positively bad — reject the request before it reaches the engine.
    Reject,
}

/// Reject the request if any image input is positively identified as bad.
///
/// No-op (`Ok`) when disabled, when there are no image inputs, or when every
/// image is plausibly valid. `request_json` must already be decrypted.
pub async fn reject_invalid_images(
    request_json: &Value,
    cfg: &ImageValidationConfig,
) -> Result<(), AppError> {
    if !cfg.enabled {
        return Ok(());
    }
    let urls = match collect_image_urls(request_json) {
        Ok(u) => u,
        Err(()) => {
            tracing::debug!(
                reason = "malformed_image_part",
                "image validation rejected request"
            );
            return Err(AppError::BadRequest(
                "Malformed image_url in request.".to_string(),
            ));
        }
    };
    if urls.is_empty() {
        return Ok(());
    }
    // Only remote (http/https) URLs are fetched; count those against the cap so
    // long multi-turn vision histories full of cheap, locally-validated `data:`
    // URLs aren't rejected. Reject (don't silently skip) when the cap is hit so
    // a bad image past it can't slip through unvalidated.
    let remote = urls.iter().filter(|u| is_remote_scheme(u)).count();
    if remote > MAX_IMAGES_PER_REQUEST {
        return Err(AppError::BadRequest(format!(
            "Too many remote image inputs in a single request (max {MAX_IMAGES_PER_REQUEST})."
        )));
    }

    // Validate ALL images with bounded concurrency; short-circuit on the first
    // bad one. Each URL is moved into its own future (owning the String) so the
    // buffered stream stays `Send` for the axum handler.
    let mut results = stream::iter(urls)
        .map(|url| async move { validate_one(&url, cfg).await })
        .buffer_unordered(cfg.max_concurrency.max(1));

    while let Some(verdict) = results.next().await {
        if verdict == Verdict::Reject {
            return Err(AppError::BadRequest(REJECT_MSG.to_string()));
        }
    }
    Ok(())
}

/// Collect every image URL from `messages[*].content[*]` typed parts. Content
/// may be a string (no images) or an array of typed parts; only the array form
/// carries images.
///
/// Returns `Err(())` if an `image_url` part is structurally malformed — the
/// `image_url` value is neither an object with a string `url` nor a bare string
/// URL. Such a request is rejected rather than silently dropping the part and
/// forwarding an unvalidated value to a lenient engine.
fn collect_image_urls(request_json: &Value) -> Result<Vec<String>, ()> {
    let mut urls = Vec::new();
    let Some(messages) = request_json.get("messages").and_then(|m| m.as_array()) else {
        return Ok(urls);
    };
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(|t| t.as_str()) != Some("image_url") {
                continue;
            }
            match part.get("image_url") {
                // Standard shape: {"url": "..."} (extra fields like `detail` ok).
                Some(Value::Object(obj)) => match obj.get("url").and_then(|u| u.as_str()) {
                    Some(u) => urls.push(u.to_string()),
                    None => return Err(()),
                },
                // Lenient string shape `{"image_url":"http://…"}` — validate it.
                Some(Value::String(s)) => urls.push(s.clone()),
                // type=image_url but image_url missing or not object/string → malformed.
                _ => return Err(()),
            }
        }
    }
    Ok(urls)
}

/// Whether `url`'s scheme is http/https (case-insensitive). These are the only
/// schemes that trigger an outbound fetch.
fn is_remote_scheme(url: &str) -> bool {
    let url = normalize_image_url_input(url);
    matches!(
        url.split_once(':')
            .map(|(s, _)| s.to_ascii_lowercase())
            .as_deref(),
        Some("http") | Some("https")
    )
}

async fn validate_one(url: &str, cfg: &ImageValidationConfig) -> Verdict {
    // URL schemes are case-insensitive (RFC 3986), and common URL parsers
    // ignore leading ASCII whitespace/control characters. Normalize before
    // matching so `HTTP://`, ` DATA:`, and ` FILE:///` can't bypass validation
    // or the dangerous-scheme refusal.
    let url = normalize_image_url_input(url);
    let (scheme, rest) = match url.split_once(':') {
        Some((s, r)) => (s.to_ascii_lowercase(), r),
        None => return Verdict::Pass, // no scheme — let the engine decide
    };
    match scheme.as_str() {
        "data" => validate_data_url_with_policy(
            rest,
            cfg.reject_non_rgb_images,
            cfg.reject_single_channel_images,
        ),
        "http" | "https" => validate_http_url(url, cfg).await,
        "file" | "ftp" | "gopher" => {
            // Dangerous / never a valid web image — refuse outright.
            tracing::debug!(reason = "dangerous_scheme", "image validation rejected");
            Verdict::Reject
        }
        _ => Verdict::Pass, // unknown scheme: don't assume (fail-open)
    }
}

fn normalize_image_url_input(url: &str) -> &str {
    url.trim_start_matches(|c: char| c.is_ascii_whitespace() || c.is_ascii_control())
}

/// `rest` is the part after `data:` — e.g. `image/png;base64,iVBOR...`.
#[cfg(test)]
fn validate_data_url(rest: &str) -> Verdict {
    validate_data_url_with_policy(rest, false, false)
}

fn validate_data_url_with_policy(
    rest: &str,
    reject_non_rgb_images: bool,
    reject_single_channel_images: bool,
) -> Verdict {
    let Some((meta, data)) = rest.split_once(',') else {
        return Verdict::Reject; // malformed data URL
    };
    if meta.to_ascii_lowercase().contains("base64") {
        match decode_b64_prefix(data) {
            // The declared media type is client-controlled and is exactly what
            // lies in the mislabeled cases — classify the bytes, not the label.
            Some(bytes) => classify_bytes_with_policy(
                &bytes,
                None,
                reject_non_rgb_images,
                reject_single_channel_images,
            ),
            None => {
                tracing::debug!(reason = "data_url_bad_base64", "image validation rejected");
                Verdict::Reject // not decodable base64 under any alphabet
            }
        }
    } else {
        // Percent-encoded / plain data URLs are rare; don't risk a false reject.
        Verdict::Pass
    }
}

/// Decode the head of a base64 payload for type-sniffing. Tries standard and
/// URL-safe alphabets, padded and unpadded; embedded whitespace is stripped.
/// The input is truncated to roughly `SNIFF_DECODE_BYTES` worth of base64
/// first, so a large `data:` URL isn't fully allocated (four times) just to
/// read its header. Returns `None` only when the head decodes under no alphabet.
fn decode_b64_prefix(data: &str) -> Option<Vec<u8>> {
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};
    let cap_chars = (SNIFF_DECODE_BYTES / 3 + 1) * 4; // multiple of 4
    let mut cleaned_chars = data.chars().filter(|c| !c.is_whitespace());
    let cleaned: String = cleaned_chars.by_ref().take(cap_chars).collect();
    if cleaned_chars.next().is_none() {
        for engine in [STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD] {
            if let Ok(bytes) = engine.decode(cleaned.as_bytes()) {
                return Some(bytes);
            }
        }
        None
    } else {
        // Decode just the 4-char-aligned prefix (no trailing padding → no-pad).
        for engine in [STANDARD_NO_PAD, URL_SAFE_NO_PAD] {
            if let Ok(bytes) = engine.decode(cleaned.as_bytes()) {
                return Some(bytes);
            }
        }
        None
    }
}

async fn validate_http_url(url: &str, cfg: &ImageValidationConfig) -> Verdict {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return Verdict::Reject;
    };

    // SSRF guard for the *initial* URL when it's an IP literal: the client's
    // resolver isn't consulted for literals. Domains (initial + redirect hops)
    // are guarded by SsrfResolver; literal redirect hops by the redirect policy.
    match parsed.host() {
        None => return Verdict::Reject,
        Some(Host::Ipv4(ip)) if !cfg.allow_private_hosts && is_disallowed_ip(IpAddr::V4(ip)) => {
            tracing::debug!(reason = "ssrf_ip_literal", "image validation rejected");
            return Verdict::Reject;
        }
        Some(Host::Ipv6(ip)) if !cfg.allow_private_hosts && is_disallowed_ip(IpAddr::V6(ip)) => {
            tracing::debug!(reason = "ssrf_ip_literal", "image validation rejected");
            return Verdict::Reject;
        }
        _ => {}
    }

    // Global cap on concurrent outbound fetches across ALL requests, so a flood
    // can't turn the proxy into an unbounded fetcher (FD/socket pressure). The
    // per-request `buffer_unordered` bound is layered on top.
    //
    // The acquire itself is bounded by `timeout`: under saturation we must not
    // queue (and thereby block) legitimate requests unboundedly — the DoS would
    // just move from the GPU to the proxy. On expiry we fail-open (Pass) and let
    // the engine handle it, rather than holding the request.
    let _permit = match tokio::time::timeout(
        cfg.timeout,
        global_fetch_semaphore(cfg.max_concurrency).acquire(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        _ => {
            tracing::debug!(
                reason = "validator_saturated",
                "image validation passthrough"
            );
            return Verdict::Pass;
        }
    };

    // Present like a normal client: many image CDNs gate on User-Agent/Accept
    // (see catch_all.rs / cloud-api#606). A UA-less request would 403 here while
    // the engine's fetcher succeeds, producing false rejects.
    let resp = match validation_client(cfg.allow_private_hosts)
        .get(url)
        .timeout(cfg.timeout)
        .header(reqwest::header::USER_AGENT, IMAGE_VALIDATION_USER_AGENT)
        .header(reqwest::header::ACCEPT, "image/*,*/*;q=0.8")
        .send()
        .await
    {
        Ok(r) => r,
        // DNS/connect failure or blocked by SsrfResolver → a definitive failure
        // the engine would also hit. Timeouts are ambiguous and fail open.
        Err(err) if err.is_timeout() => {
            tracing::debug!(reason = "fetch_timeout", "image validation passthrough");
            return Verdict::Pass;
        }
        Err(_) => {
            tracing::debug!(reason = "fetch_error", "image validation rejected");
            return Verdict::Reject;
        }
    };

    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::GONE {
        // Definitively gone.
        tracing::debug!(status = %status.as_u16(), reason = "http_gone", "image validation rejected");
        return Verdict::Reject;
    }
    if !status.is_success() {
        // Ambiguous (401/403/429, 5xx, stopped redirect, other 4xx): could be
        // UA-gating, rate-limiting, or transient. Per fail-open, let the engine
        // be the judge rather than emit a false 400.
        tracing::debug!(status = %status.as_u16(), reason = "http_ambiguous_pass", "image validation passthrough");
        return Verdict::Pass;
    }

    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_ascii_lowercase());

    // Download only the head (bounded by max_bytes) — enough to sniff the type;
    // never buffer the whole image.
    let cap = cfg.max_bytes.max(64);
    let mut head = Vec::new();
    let mut body = resp.bytes_stream();
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(b) => {
                head.extend_from_slice(&b);
                if head.len() >= cap {
                    head.truncate(cap);
                    break;
                }
            }
            Err(err) if err.is_timeout() => {
                tracing::debug!(reason = "body_timeout", "image validation passthrough");
                return Verdict::Pass;
            }
            Err(_) if head.is_empty() => {
                tracing::debug!(
                    reason = "body_error_before_bytes",
                    "image validation passthrough"
                );
                return Verdict::Pass;
            }
            Err(_) => break, // truncated read; classify what we have
        }
    }

    let verdict = classify_bytes_with_policy(
        &head,
        content_type.as_deref(),
        cfg.reject_non_rgb_images,
        cfg.reject_single_channel_images,
    );
    if verdict == Verdict::Reject {
        tracing::debug!(reason = "non_image_body", "image validation rejected");
    }
    verdict
}

/// Global cap on concurrent outbound image-validation fetches, shared across
/// all requests. Sized once from `max_concurrency` on first use.
fn global_fetch_semaphore(permits: usize) -> &'static tokio::sync::Semaphore {
    static SEM: OnceLock<tokio::sync::Semaphore> = OnceLock::new();
    SEM.get_or_init(|| tokio::sync::Semaphore::new(permits.max(1)))
}

/// The dedicated client used for image validation fetches. Built once.
///
/// Differs from the shared client in two security-relevant ways: it resolves
/// through [`SsrfResolver`] (rejects domains pointing at non-public ranges) and
/// uses a redirect policy that refuses hops to disallowed IP literals while
/// still following ordinary redirects (so valid redirecting image URLs work).
fn validation_client(allow_private_hosts: bool) -> &'static reqwest::Client {
    static PUBLIC_ONLY_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    static PRIVATE_ALLOWED_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    let client = if allow_private_hosts {
        &PRIVATE_ALLOWED_CLIENT
    } else {
        &PUBLIC_ONLY_CLIENT
    };
    client.get_or_init(|| {
        let redirect = reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= MAX_REDIRECTS {
                return attempt.stop();
            }
            // Refuse redirects to a disallowed IP literal (regardless of
            // allow_private_hosts — redirect targets are server-controlled and
            // must never be trusted to point inward). Domain hops are guarded
            // by SsrfResolver during connection.
            let blocked = match attempt.url().host() {
                Some(Host::Ipv4(ip)) => is_disallowed_ip(IpAddr::V4(ip)),
                Some(Host::Ipv6(ip)) => is_disallowed_ip(IpAddr::V6(ip)),
                _ => false,
            };
            if blocked {
                // IMPORTANT: `error()`, not `stop()`. `stop()` returns the 3xx
                // response, which downstream treats as an ambiguous non-2xx and
                // passes through (fail-open) — defeating the SSRF guard. An
                // error surfaces as a transport failure → Reject.
                attempt.error("redirect to a non-public address refused")
            } else {
                attempt.follow()
            }
        });
        reqwest::Client::builder()
            .no_proxy()
            .redirect(redirect)
            .dns_resolver(Arc::new(SsrfResolver {
                allow_private_hosts,
            }))
            .build()
            .expect("build image-validation reqwest client")
    })
}

/// A reqwest DNS resolver that refuses any name that resolves to a non-public
/// address — closing SSRF via the initial domain and via redirect hops alike,
/// with the checked IP equal to the connected IP (no resolve-then-connect
/// TOCTOU window).
struct SsrfResolver {
    allow_private_hosts: bool,
}

impl reqwest::dns::Resolve for SsrfResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let allow_private_hosts = self.allow_private_hosts;
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host(format!("{}:0", name.as_str()))
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?
                .collect();
            // Honor allow_private_hosts for resolved domains (not just literals).
            if !allow_private_hosts && addrs.iter().any(|a| is_disallowed_ip(a.ip())) {
                return Err(Box::<dyn std::error::Error + Send + Sync>::from(
                    "image host resolves to a non-public address",
                ));
            }
            Ok(Box::new(addrs.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Decide whether bytes (and optional content-type) are a plausible image.
/// Returns `Reject` only for *positively* non-image content.
#[cfg(test)]
fn classify_bytes(bytes: &[u8], content_type: Option<&str>) -> Verdict {
    classify_bytes_with_policy(bytes, content_type, false, false)
}

fn classify_bytes_with_policy(
    bytes: &[u8],
    content_type: Option<&str>,
    reject_non_rgb_images: bool,
    reject_single_channel_images: bool,
) -> Verdict {
    // Empty payload is positively not a decodable image — reject regardless of
    // any (client-controlled) content type. Covers empty `data:` base64 bodies
    // and empty/204 HTTP responses, which would otherwise still hit the engine.
    if bytes.is_empty() {
        return Verdict::Reject;
    }
    // The actual bytes win over the (attacker/server-controlled) content type:
    // a dead URL serving an HTML/JSON error page with `Content-Type: image/png`
    // must still be rejected. Real images never match these markers, so this
    // can't false-reject a genuine image regardless of its declared type.
    if looks_textual(bytes) {
        return Verdict::Reject;
    }
    if is_rejected_raster(bytes, reject_non_rgb_images, reject_single_channel_images) {
        return Verdict::Reject;
    }
    if is_image_magic(bytes) {
        return Verdict::Pass;
    }
    // No image magic and the head is essentially printable text (e.g. a plain
    // "Not Found" body or a base64-encoded text blob): not a decodable raster
    // image. Real images are binary, so this won't false-reject one.
    if is_mostly_text(bytes) {
        return Verdict::Reject;
    }
    // Bytes are inconclusive — fall back to the content type as a tiebreaker.
    if let Some(ct) = content_type {
        let ct = ct.split(';').next().unwrap_or("").trim();
        if is_textual_content_type(ct) {
            return Verdict::Reject;
        }
        if ct.starts_with("image/") || ct.starts_with("video/") {
            return Verdict::Pass;
        }
    }
    // Unknown but not obviously textual — let the engine be the judge.
    Verdict::Pass
}

#[derive(Debug, PartialEq, Eq)]
enum RgbCompatibility {
    RgbCompatible,
    SingleChannel,
    OtherNonRgb,
    Unknown,
}

/// Header-only detection for image formats where channel count is available in
/// the head. Used only as a Gemma-4 crash guard: current SGLang Gemma-4 expects
/// 3-channel RGB patches (`16 * 16 * 3 = 768`) and has crashed on one-channel
/// inputs (`16 * 16 * 1 = 256`). Inconclusive formats fail open.
fn is_rejected_raster(
    bytes: &[u8],
    reject_non_rgb_images: bool,
    reject_single_channel_images: bool,
) -> bool {
    match rgb_compatibility(bytes) {
        RgbCompatibility::SingleChannel => reject_single_channel_images || reject_non_rgb_images,
        RgbCompatibility::OtherNonRgb => reject_non_rgb_images,
        RgbCompatibility::RgbCompatible | RgbCompatibility::Unknown => false,
    }
}

fn rgb_compatibility(bytes: &[u8]) -> RgbCompatibility {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return png_rgb_compatibility(bytes);
    }
    if bytes.starts_with(&[0xFF, 0xD8]) {
        return jpeg_rgb_compatibility(bytes);
    }
    RgbCompatibility::Unknown
}

fn png_rgb_compatibility(bytes: &[u8]) -> RgbCompatibility {
    // PNG: signature(8), IHDR length(4), "IHDR"(4), width(4), height(4),
    // bit depth(1), color type(1). Color type is byte 25.
    if bytes.len() <= 25 || &bytes[12..16] != b"IHDR" {
        return RgbCompatibility::Unknown;
    }
    match bytes[25] {
        0 => RgbCompatibility::SingleChannel,       // grayscale
        3 | 4 | 6 => RgbCompatibility::OtherNonRgb, // indexed palette/P-mode, grayscale+alpha, RGBA
        2 => RgbCompatibility::RgbCompatible,       // RGB
        _ => RgbCompatibility::Unknown,
    }
}

fn jpeg_rgb_compatibility(bytes: &[u8]) -> RgbCompatibility {
    let mut i = 2; // after SOI
    while i + 4 <= bytes.len() {
        while i < bytes.len() && bytes[i] != 0xFF {
            i += 1;
        }
        while i < bytes.len() && bytes[i] == 0xFF {
            i += 1;
        }
        if i >= bytes.len() {
            return RgbCompatibility::Unknown;
        }
        let marker = bytes[i];
        i += 1;

        // Standalone markers with no segment length.
        if marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            continue;
        }
        // Start of scan: channel count was not found in the header we have.
        if marker == 0xDA {
            return RgbCompatibility::Unknown;
        }
        if i + 2 > bytes.len() {
            return RgbCompatibility::Unknown;
        }
        let segment_len = u16::from_be_bytes([bytes[i], bytes[i + 1]]) as usize;
        if segment_len < 2 || i + segment_len > bytes.len() {
            return RgbCompatibility::Unknown;
        }
        let payload = &bytes[i + 2..i + segment_len];
        if is_jpeg_sof_marker(marker) {
            if payload.len() < 6 {
                return RgbCompatibility::Unknown;
            }
            return match payload[5] {
                1 => RgbCompatibility::SingleChannel,
                3 => RgbCompatibility::RgbCompatible,
                _ => RgbCompatibility::OtherNonRgb,
            };
        }
        i += segment_len;
    }
    RgbCompatibility::Unknown
}

fn is_jpeg_sof_marker(marker: u8) -> bool {
    matches!(
        marker,
        0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE | 0xCF
    )
}

/// Heuristic: the head is (almost) entirely printable ASCII / common
/// whitespace, with no image magic — i.e. plain text, not a raster image.
/// Requires a little signal (>= 8 bytes) to avoid misjudging tiny heads, and a
/// high printable ratio so genuinely-binary image data is never flagged.
fn is_mostly_text(b: &[u8]) -> bool {
    if b.len() < 8 {
        return false; // too little signal — fail-open
    }
    let printable = b
        .iter()
        .filter(|&&c| c == b'\t' || c == b'\n' || c == b'\r' || (0x20..=0x7E).contains(&c))
        .count();
    printable * 100 >= b.len() * 95
}

fn is_textual_content_type(ct: &str) -> bool {
    ct.starts_with("text/")
        || matches!(
            ct,
            "application/json"
                | "application/xml"
                | "application/xhtml+xml"
                | "application/javascript"
        )
}

/// Magic-byte detection for the common raster formats SGLang/PIL decode.
fn is_image_magic(b: &[u8]) -> bool {
    if b.len() < 4 {
        return false;
    }
    // PNG
    if b.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        return true;
    }
    // JPEG
    if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return true;
    }
    // GIF
    if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        return true;
    }
    // BMP
    if b.starts_with(b"BM") {
        return true;
    }
    // TIFF (LE / BE)
    if b.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || b.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
        return true;
    }
    // RIFF container (WebP): "RIFF"...."WEBP"
    if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        return true;
    }
    // ISO-BMFF (HEIC/HEIF/AVIF): "....ftyp" then a known image brand
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        let brand = &b[8..12];
        if matches!(
            brand,
            b"heic" | b"heif" | b"heix" | b"hevc" | b"mif1" | b"msf1" | b"avif" | b"avis"
        ) {
            return true;
        }
    }
    false
}

/// Whether the head looks like HTML / XML / JSON / plain text rather than an
/// image — i.e. positively not a decodable raster image.
fn looks_textual(b: &[u8]) -> bool {
    // Skip a UTF-8 BOM and leading ASCII whitespace.
    let mut s = b;
    if s.starts_with(&[0xEF, 0xBB, 0xBF]) {
        s = &s[3..];
    }
    let s = match s.iter().position(|&c| !c.is_ascii_whitespace()) {
        Some(i) => &s[i..],
        None => return false,
    };
    let lower_head: Vec<u8> = s.iter().take(16).map(|c| c.to_ascii_lowercase()).collect();
    lower_head.starts_with(b"<!doctype")
        || lower_head.starts_with(b"<html")
        || lower_head.starts_with(b"<head")
        || lower_head.starts_with(b"<?xml")
        || lower_head.starts_with(b"<svg")
        || s.starts_with(b"{")
        || s.starts_with(b"[")
}

fn is_disallowed_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local() // 169.254/16 incl. 169.254.169.254 metadata
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || o[0] == 0                                   // 0.0.0.0/8
                || (o[0] == 100 && (o[1] & 0xC0) == 0x40)      // 100.64.0.0/10 CGNAT
                || (224..=239).contains(&o[0])                 // 224.0.0.0/4 multicast
                || o[0] >= 240                                 // 240.0.0.0/4 reserved
                || (o[0] == 198 && (o[1] == 18 || o[1] == 19)) // 198.18.0.0/15 benchmark
                || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0.0/24
        }
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_disallowed_ip(IpAddr::V4(mapped));
            }
            let seg = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                || (seg[0] & 0xfe00) == 0xfc00 // fc00::/7 unique-local
                || (seg[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
                || (
                    seg[0] == 0x0064
                        && seg[1] == 0xff9b
                        && seg[2] == 0
                        && seg[3] == 0
                        && seg[4] == 0
                        && seg[5] == 0
                ) // 64:ff9b::/96 NAT64 well-known prefix
                || (seg[0] == 0x0064 && seg[1] == 0xff9b && seg[2] == 0x0001) // 64:ff9b:1::/48
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ImageValidationConfig {
        ImageValidationConfig {
            enabled: true,
            timeout: Duration::from_secs(2),
            max_bytes: 2048,
            max_concurrency: 4,
            allow_private_hosts: true,
            reject_non_rgb_images: false,
            reject_single_channel_images: false,
        }
    }

    fn minimal_jpeg_with_components(components: u8) -> Vec<u8> {
        let mut bytes = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
        ];
        let segment_len: u16 = 8 + (components as u16 * 3);
        bytes.extend_from_slice(&segment_len.to_be_bytes());
        bytes.extend_from_slice(&[
            0x08, // precision
            0x00, 0x01, // height
            0x00, 0x01, // width
            components,
        ]);
        for id in 1..=components {
            bytes.extend_from_slice(&[id, 0x11, 0x00]);
        }
        bytes
    }

    fn minimal_png_with_color_type(color_type: u8) -> Vec<u8> {
        let mut bytes = vec![
            0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, // signature
            0x00, 0x00, 0x00, 0x0D, // IHDR length
            b'I', b'H', b'D', b'R', // IHDR
            0x00, 0x00, 0x00, 0x01, // width
            0x00, 0x00, 0x00, 0x01, // height
            0x08, // bit depth
            color_type,
        ];
        bytes.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // compression/filter/interlace + partial CRC
        bytes
    }

    #[test]
    fn collects_only_image_url_parts() {
        let v = serde_json::json!({
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]}
            ]
        });
        assert_eq!(
            collect_image_urls(&v),
            Ok(vec![
                "https://x/a.png".to_string(),
                "data:image/png;base64,AAAA".to_string()
            ])
        );
    }

    #[test]
    fn text_only_request_has_no_images() {
        let v = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(collect_image_urls(&v).unwrap().is_empty());
    }

    #[test]
    fn string_form_image_url_is_collected() {
        let v = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":"https://x/a.png"}
        ]}]});
        assert_eq!(
            collect_image_urls(&v),
            Ok(vec!["https://x/a.png".to_string()])
        );
    }

    #[test]
    fn malformed_image_url_part_is_error() {
        // type=image_url but image_url is a number / missing url → malformed.
        for bad in [
            serde_json::json!({"type":"image_url","image_url": 42}),
            serde_json::json!({"type":"image_url","image_url": {"detail":"high"}}),
            serde_json::json!({"type":"image_url"}),
        ] {
            let v = serde_json::json!({"messages":[{"role":"user","content":[bad]}]});
            assert_eq!(collect_image_urls(&v), Err(()));
        }
    }

    #[test]
    fn image_magic_detected() {
        assert!(is_image_magic(&[
            0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A
        ]));
        assert!(is_image_magic(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0]));
        assert!(is_image_magic(b"GIF89a...."));
        assert!(is_image_magic(b"RIFF\0\0\0\0WEBPVP8 "));
        assert!(!is_image_magic(b"<!doctype html>"));
        assert!(!is_image_magic(b"not an image"));
    }

    #[test]
    fn textual_payloads_flagged() {
        assert!(looks_textual(b"<!DOCTYPE html><html>..."));
        assert!(looks_textual(b"   \n<html>"));
        assert!(looks_textual(b"<?xml version=\"1.0\"?>"));
        assert!(looks_textual(b"{\"error\":\"not found\"}"));
        assert!(looks_textual(b"  [1,2,3]"));
        assert!(!looks_textual(&[0x89, b'P', b'N', b'G']));
        assert!(!looks_textual(&[0xFF, 0xD8, 0xFF]));
    }

    #[test]
    fn classify_uses_content_type_then_magic() {
        assert_eq!(
            classify_bytes(
                b"<!doctype html><html>login</html>",
                Some("text/html; charset=utf-8")
            ),
            Verdict::Reject
        );
        assert_eq!(
            classify_bytes(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A], None),
            Verdict::Pass
        );
        assert_eq!(
            classify_bytes(&[0x00, 0x01], Some("image/png")),
            Verdict::Pass
        );
        assert_eq!(
            classify_bytes(&[0x00, 0x01, 0x02, 0x03], None),
            Verdict::Pass
        );
    }

    #[test]
    fn data_url_validation() {
        // Valid base64 PNG header → pass.
        let png = base64::engine::general_purpose::STANDARD
            .encode([0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        assert_eq!(
            validate_data_url(&format!("image/png;base64,{png}")),
            Verdict::Pass
        );
        // base64 of an HTML page → reject (bytes sniff as HTML despite image/* label).
        let html = base64::engine::general_purpose::STANDARD.encode("<!doctype html><html></html>");
        assert_eq!(
            validate_data_url(&format!("image/png;base64,{html}")),
            Verdict::Reject
        );
        // Truly non-base64 → reject.
        assert_eq!(
            validate_data_url("image/png;base64,!!! not base64 @@@"),
            Verdict::Reject
        );
        // Malformed (no comma) → reject.
        assert_eq!(validate_data_url("image/png;base64"), Verdict::Reject);
    }

    #[test]
    fn single_channel_guard_rejects_observed_crash_headers_when_enabled() {
        let gray_jpeg = minimal_jpeg_with_components(1);
        let rgb_jpeg = minimal_jpeg_with_components(3);
        let gray_png = minimal_png_with_color_type(0);
        let rgb_png = minimal_png_with_color_type(2);

        for single_channel in [&gray_jpeg, &gray_png] {
            assert_eq!(
                classify_bytes_with_policy(single_channel, Some("image/png"), false, true),
                Verdict::Reject
            );
            assert_eq!(
                classify_bytes_with_policy(single_channel, Some("image/png"), false, false),
                Verdict::Pass
            );
        }

        for rgb in [&rgb_jpeg, &rgb_png] {
            assert_eq!(
                classify_bytes_with_policy(rgb, Some("image/png"), false, true),
                Verdict::Pass
            );
        }
    }

    #[test]
    fn strict_non_rgb_guard_rejects_alpha_palette_and_cmyk_headers_when_enabled() {
        let cmyk_jpeg = minimal_jpeg_with_components(4);
        let palette_png = minimal_png_with_color_type(3);
        let gray_alpha_png = minimal_png_with_color_type(4);
        let rgba_png = minimal_png_with_color_type(6);

        for strict_only in [&cmyk_jpeg, &palette_png, &gray_alpha_png, &rgba_png] {
            assert_eq!(
                classify_bytes_with_policy(strict_only, Some("image/png"), true, false),
                Verdict::Reject
            );
            assert_eq!(
                classify_bytes_with_policy(strict_only, Some("image/png"), false, true),
                Verdict::Pass
            );
        }
    }

    #[test]
    fn data_url_single_channel_guard_rejects_grayscale_jpeg_when_enabled() {
        let gray_jpeg =
            base64::engine::general_purpose::STANDARD.encode(minimal_jpeg_with_components(1));
        assert_eq!(
            validate_data_url_with_policy(&format!("image/jpeg;base64,{gray_jpeg}"), false, true),
            Verdict::Reject
        );
        assert_eq!(
            validate_data_url_with_policy(&format!("image/jpeg;base64,{gray_jpeg}"), false, false),
            Verdict::Pass
        );
    }

    #[test]
    fn html_body_with_image_content_type_is_rejected() {
        // A dead URL / hostile server returning an HTML (or JSON) error page
        // with an image/* content type must NOT pass — the bytes win.
        assert_eq!(
            classify_bytes(b"<!doctype html><html>nope</html>", Some("image/png")),
            Verdict::Reject
        );
        assert_eq!(
            classify_bytes(b"{\"error\":\"gone\"}", Some("image/jpeg")),
            Verdict::Reject
        );
        // A genuine image is still accepted whatever the declared type.
        assert_eq!(
            classify_bytes(&[0xFF, 0xD8, 0xFF, 0xE0], Some("application/octet-stream")),
            Verdict::Pass
        );
    }

    #[tokio::test]
    async fn rejects_request_over_remote_image_cap() {
        // More than the cap of *remote* images → reject outright (before any
        // fetch), so a bad image past the cap can't slip through unvalidated.
        let parts: Vec<_> = (0..=MAX_IMAGES_PER_REQUEST)
            .map(|i| serde_json::json!({"type":"image_url","image_url":{"url": format!("https://example.invalid/{i}.png")}}))
            .collect();
        let v = serde_json::json!({"messages":[{"role":"user","content": parts}]});
        assert!(matches!(
            reject_invalid_images(&v, &cfg()).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[tokio::test]
    async fn data_urls_do_not_count_against_remote_cap() {
        // Long multi-turn histories full of cheap, locally-validated `data:`
        // URLs (well over the remote cap) must NOT be rejected.
        let png = base64::engine::general_purpose::STANDARD
            .encode([0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let parts: Vec<_> = (0..MAX_IMAGES_PER_REQUEST + 10)
            .map(|_| serde_json::json!({"type":"image_url","image_url":{"url": format!("data:image/png;base64,{png}")}}))
            .collect();
        let v = serde_json::json!({"messages":[{"role":"user","content": parts}]});
        assert!(reject_invalid_images(&v, &cfg()).await.is_ok());
    }

    #[tokio::test]
    async fn uppercase_and_leading_space_schemes_are_not_bypassed() {
        // Case-insensitive and parser-like leading-space handling: dangerous,
        // data, and remote schemes must still hit their validation paths.
        let html = base64::engine::general_purpose::STANDARD.encode("<!doctype html><html></html>");
        for url in [
            "FILE:///etc/passwd".to_string(),
            "File:///etc/passwd".to_string(),
            " FILE:///etc/passwd".to_string(),
            format!("DATA:image/png;base64,{html}"),
            format!("\tdata:image/png;base64,{html}"),
            " http://127.0.0.1:1/x.png".to_string(),
        ] {
            let v = serde_json::json!({"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url": url}}
            ]}]});
            assert!(
                matches!(
                    reject_invalid_images(&v, &cfg()).await,
                    Err(AppError::BadRequest(_))
                ),
                "{url} should be rejected"
            );
        }
    }

    #[test]
    fn plain_text_body_is_rejected() {
        // No image magic + (almost) all printable text → not a raster image.
        assert_eq!(
            classify_bytes(b"Not Found: the requested object does not exist.", None),
            Verdict::Reject
        );
        assert!(is_mostly_text(b"this is just plain text, not an image"));
        assert!(!is_mostly_text(&[
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46
        ]));
        // base64 of plain text in a data URL → reject.
        let txt =
            base64::engine::general_purpose::STANDARD.encode("totally not an image, just words");
        assert_eq!(
            validate_data_url(&format!("image/png;base64,{txt}")),
            Verdict::Reject
        );
    }

    #[tokio::test]
    async fn validates_all_images_not_just_first() {
        // A valid image first, a bad one second → the request is still rejected
        // (every image is validated, not just the first).
        let png = base64::engine::general_purpose::STANDARD
            .encode([0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let v = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url": format!("data:image/png;base64,{png}")}},
            {"type":"image_url","image_url":{"url":"file:///etc/passwd"}}
        ]}]});
        assert!(matches!(
            reject_invalid_images(&v, &cfg()).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[test]
    fn empty_payload_is_rejected() {
        // Empty bytes are positively not a decodable image — reject even when a
        // (client-controlled) image content-type claims otherwise.
        assert_eq!(classify_bytes(&[], None), Verdict::Reject);
        assert_eq!(classify_bytes(&[], Some("image/png")), Verdict::Reject);
        // Empty base64 data URL (`data:image/png;base64,`) decodes to no bytes.
        assert_eq!(validate_data_url("image/png;base64,"), Verdict::Reject);
    }

    #[test]
    fn data_url_url_safe_and_unpadded_decode() {
        // URL-safe alphabet + no padding must NOT be false-rejected: a PNG
        // header has bytes that encode with `-`/`_` and would fail STANDARD.
        let raw = [0xFB, 0xFF, 0xBFu8]; // -> "+/+/" std, "-_-_" url-safe
        let url_safe = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw);
        // Decodes fine; bytes are unknown/non-textual → fail-open Pass.
        assert_eq!(
            validate_data_url(&format!("image/png;base64,{url_safe}")),
            Verdict::Pass
        );
    }

    #[test]
    fn ssrf_ip_classification() {
        for ip in [
            "127.0.0.1",
            "10.0.0.5",
            "192.168.1.1",
            "169.254.169.254",
            "172.16.0.1",
            "100.64.0.1",
            "198.18.0.1",
            "224.0.0.1",
            "240.0.0.1",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fc00::1",
            "64:ff9b::0808:0808",
            "64:ff9b:1::1",
        ] {
            assert!(
                is_disallowed_ip(ip.parse().unwrap()),
                "{ip} should be disallowed"
            );
        }
        for ip in [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "2606:4700:4700::1111",
        ] {
            assert!(
                !is_disallowed_ip(ip.parse().unwrap()),
                "{ip} should be allowed"
            );
        }
    }

    #[tokio::test]
    async fn disabled_is_noop_even_with_images() {
        let v = serde_json::json!({
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"file:///etc/passwd"}}
            ]}]
        });
        let mut c = cfg();
        c.enabled = false;
        assert!(reject_invalid_images(&v, &c).await.is_ok());
    }

    #[tokio::test]
    async fn dangerous_scheme_rejected() {
        let v = serde_json::json!({
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"file:///etc/passwd"}}
            ]}]
        });
        assert!(matches!(
            reject_invalid_images(&v, &cfg()).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[tokio::test]
    async fn ipv6_literal_internal_host_rejected_without_fetch() {
        // Regression: bracketed IPv6 literals must be classified via host(),
        // not host_str(), so an internal IPv6 literal is refused.
        let mut c = cfg();
        c.allow_private_hosts = false;
        for url in ["http://[::1]:8000/x.png", "http://[fc00::1]/x.png"] {
            let v = serde_json::json!({"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url": url}}
            ]}]});
            assert!(
                matches!(
                    reject_invalid_images(&v, &c).await,
                    Err(AppError::BadRequest(_))
                ),
                "{url} should be rejected"
            );
        }
    }

    #[tokio::test]
    async fn ssrf_blocks_loopback_literal_when_not_allowed() {
        let mut c = cfg();
        c.allow_private_hosts = false;
        let v = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"http://127.0.0.1:1/x.png"}}
        ]}]});
        assert!(matches!(
            reject_invalid_images(&v, &c).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[tokio::test]
    async fn http_path_fetches_and_classifies() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/html.jpg"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string("<!doctype html><html><body>login</body></html>"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/real.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "image/png")
                    .set_body_bytes(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0]),
            )
            .mount(&server)
            .await;
        // 200 with an empty body but an image content-type.
        Mock::given(method("GET"))
            .and(path("/empty.png"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "image/png"))
            .mount(&server)
            .await;
        // 204 No Content (success status, no body).
        Mock::given(method("GET"))
            .and(path("/nocontent.png"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        // wiremock binds to a 127.0.0.1 *literal*, so DNS (and SsrfResolver) is
        // never consulted; allow_private_hosts lets the literal pre-check pass.
        let c = cfg();
        let req = |p: &str| {
            serde_json::json!({"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url": format!("{}{}", server.uri(), p)}}
            ]}]})
        };

        assert!(matches!(
            reject_invalid_images(&req("/html.jpg"), &c).await,
            Err(AppError::BadRequest(_))
        ));
        assert!(reject_invalid_images(&req("/real.png"), &c).await.is_ok());
        assert!(matches!(
            reject_invalid_images(&req("/missing.png"), &c).await,
            Err(AppError::BadRequest(_))
        ));
        // Empty body (even with image content-type) and 204 → reject.
        assert!(matches!(
            reject_invalid_images(&req("/empty.png"), &c).await,
            Err(AppError::BadRequest(_))
        ));
        assert!(matches!(
            reject_invalid_images(&req("/nocontent.png"), &c).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[tokio::test]
    async fn redirect_to_private_ip_literal_is_rejected() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/redirect.png"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("location", "http://169.254.169.254/latest/meta-data/"),
            )
            .mount(&server)
            .await;

        let c = cfg();
        let req = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url": format!("{}/redirect.png", server.uri())}}
        ]}]});
        assert!(matches!(
            reject_invalid_images(&req, &c).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[tokio::test]
    async fn redirect_chain_exhaustion_fails_open() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let server_uri = server.uri().replace("127.0.0.1", "localhost");
        for i in 0..=MAX_REDIRECTS {
            let current = format!("/r{i}.png");
            let next = format!("/r{}.png", i + 1);
            Mock::given(method("GET"))
                .and(path(current))
                .respond_with(
                    ResponseTemplate::new(302)
                        .insert_header("location", format!("{server_uri}{next}")),
                )
                .mount(&server)
                .await;
        }

        let c = cfg();
        let req = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url": format!("{server_uri}/r0.png")}}
        ]}]});
        assert!(reject_invalid_images(&req, &c).await.is_ok());
    }

    #[tokio::test]
    async fn named_private_hosts_honor_allow_private_hosts() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/real.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "image/png")
                    .set_body_bytes(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
            )
            .mount(&server)
            .await;

        let loopback_url = server.uri().replace("127.0.0.1", "localhost");
        let req = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url": format!("{loopback_url}/real.png")}}
        ]}]});

        let mut blocked = cfg();
        blocked.allow_private_hosts = false;
        assert!(matches!(
            reject_invalid_images(&req, &blocked).await,
            Err(AppError::BadRequest(_))
        ));

        let mut allowed = cfg();
        allowed.allow_private_hosts = true;
        assert!(reject_invalid_images(&req, &allowed).await.is_ok());
    }
}
