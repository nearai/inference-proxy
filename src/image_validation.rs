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
//! be fetched (DNS/connect/timeout/non-2xx), content that sniffs as
//! HTML/text/JSON, or a `data:` payload that isn't valid base64. Anything
//! ambiguous (unknown-but-not-textual bytes, unusual schemes, unknown content
//! types) is **passed through** so the engine stays the source of truth and
//! valid multimodal traffic is never broken by an over-eager check.
//!
//! ## SSRF
//! Because this introduces an outbound fetch of client-controlled URLs, hosts
//! that resolve to private/loopback/link-local/metadata ranges are refused
//! (SGLang already fetches these today, so this is a net security improvement,
//! not a new exposure). Pre-resolution is best-effort (TOCTOU is accepted; the
//! threat model here is dead public URLs, not active SSRF).

use std::net::IpAddr;
use std::time::Duration;

use base64::Engine as _;
use futures_util::stream::{self, StreamExt};
use reqwest::header::CONTENT_TYPE;
use serde_json::Value;

use crate::error::AppError;

/// Bytes sniffed from the head of a response/data payload to classify it.
const SNIFF_LEN: usize = 512;
/// Defensive cap on images validated per request (avoids unbounded fan-out).
const MAX_IMAGES_PER_REQUEST: usize = 64;

/// Generic, content-free rejection message. Never echoes the URL or any user
/// data (see proxy.rs sanitization rules / privacy requirements).
const REJECT_MSG: &str = "One or more image inputs could not be fetched or are not a valid image. \
     Ensure each image URL is reachable and resolves to a real image.";

#[derive(Debug, Clone)]
pub struct ImageValidationConfig {
    pub enabled: bool,
    pub timeout: Duration,
    /// Max bytes read from a fetched image before giving up (only the head is
    /// needed to sniff; we never buffer whole images).
    pub max_bytes: usize,
    pub max_concurrency: usize,
    /// Allow private/loopback hosts (set in tests so wiremock on 127.0.0.1
    /// works; never enabled in production).
    pub allow_private_hosts: bool,
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
    http_client: &reqwest::Client,
    cfg: &ImageValidationConfig,
) -> Result<(), AppError> {
    if !cfg.enabled {
        return Ok(());
    }
    let urls = collect_image_urls(request_json);
    if urls.is_empty() {
        return Ok(());
    }

    // Validate with bounded concurrency; short-circuit on the first bad image.
    // Each URL is moved into its own future (owning the String) so the buffered
    // stream stays `Send` for the axum handler.
    let mut results = stream::iter(urls.into_iter().take(MAX_IMAGES_PER_REQUEST))
        .map(|url| async move { validate_one(&url, http_client, cfg).await })
        .buffer_unordered(cfg.max_concurrency.max(1));

    while let Some(verdict) = results.next().await {
        if verdict == Verdict::Reject {
            return Err(AppError::BadRequest(REJECT_MSG.to_string()));
        }
    }
    Ok(())
}

/// Collect every `image_url.url` string from `messages[*].content[*]`.
/// Content may be a string (no images) or an array of typed parts; only the
/// array form carries images.
fn collect_image_urls(request_json: &Value) -> Vec<String> {
    let mut urls = Vec::new();
    let Some(messages) = request_json.get("messages").and_then(|m| m.as_array()) else {
        return urls;
    };
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            // OpenAI shape: {"type":"image_url","image_url":{"url":"..."}}
            if part.get("type").and_then(|t| t.as_str()) != Some("image_url") {
                continue;
            }
            if let Some(url) = part
                .get("image_url")
                .and_then(|iu| iu.get("url"))
                .and_then(|u| u.as_str())
            {
                urls.push(url.to_string());
            }
        }
    }
    urls
}

async fn validate_one(url: &str, client: &reqwest::Client, cfg: &ImageValidationConfig) -> Verdict {
    if let Some(rest) = url.strip_prefix("data:") {
        validate_data_url(rest)
    } else if url.starts_with("http://") || url.starts_with("https://") {
        validate_http_url(url, client, cfg).await
    } else if url.starts_with("file:") || url.starts_with("ftp:") || url.starts_with("gopher:") {
        // Dangerous / never a valid web image — refuse outright.
        Verdict::Reject
    } else {
        // Unknown scheme: don't assume; let the engine handle it (fail-open).
        Verdict::Pass
    }
}

/// `rest` is the part after `data:` — e.g. `image/png;base64,iVBOR...`.
fn validate_data_url(rest: &str) -> Verdict {
    let Some((meta, data)) = rest.split_once(',') else {
        return Verdict::Reject; // malformed data URL
    };
    if meta.contains("base64") {
        // Trim whitespace/newlines that some clients embed in base64.
        let cleaned: String = data.chars().filter(|c| !c.is_whitespace()).collect();
        match base64::engine::general_purpose::STANDARD.decode(cleaned.as_bytes()) {
            Ok(bytes) => classify_bytes(&bytes, None),
            Err(_) => Verdict::Reject, // not valid base64 → can't be a real image
        }
    } else {
        // Percent-encoded / plain data URLs are rare; don't risk a false reject.
        Verdict::Pass
    }
}

async fn validate_http_url(
    url: &str,
    client: &reqwest::Client,
    cfg: &ImageValidationConfig,
) -> Verdict {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return Verdict::Reject;
    };
    let Some(host) = parsed.host_str() else {
        return Verdict::Reject;
    };
    let port = parsed
        .port_or_known_default()
        .unwrap_or(if parsed.scheme() == "https" { 443 } else { 80 });

    // SSRF guard: refuse hosts that resolve to non-public ranges.
    if !cfg.allow_private_hosts && resolves_to_private(host, port).await {
        return Verdict::Reject;
    }

    let resp = match client.get(url).timeout(cfg.timeout).send().await {
        Ok(r) if r.status().is_success() => r,
        // Unfetchable (DNS/connect/timeout) or non-2xx → the engine would fail
        // on the same payload. Reject.
        _ => return Verdict::Reject,
    };

    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_ascii_lowercase());

    // Read only the head — enough to sniff magic bytes; never buffer the image.
    let cap = cfg.max_bytes.max(SNIFF_LEN);
    let mut head = Vec::with_capacity(SNIFF_LEN.min(cap));
    let mut body = resp.bytes_stream();
    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(b) => {
                head.extend_from_slice(&b);
                if head.len() >= SNIFF_LEN || head.len() >= cap {
                    break;
                }
            }
            Err(_) => break, // truncated read; classify what we have
        }
    }

    classify_bytes(&head, content_type.as_deref())
}

/// Decide whether bytes (and optional content-type) are a plausible image.
/// Returns `Reject` only for *positively* non-image content.
fn classify_bytes(bytes: &[u8], content_type: Option<&str>) -> Verdict {
    if let Some(ct) = content_type {
        let ct = ct.split(';').next().unwrap_or("").trim();
        if ct.starts_with("image/") || ct.starts_with("video/") {
            return Verdict::Pass;
        }
        if is_textual_content_type(ct) {
            return Verdict::Reject;
        }
    }
    if is_image_magic(bytes) {
        return Verdict::Pass;
    }
    if looks_textual(bytes) {
        return Verdict::Reject;
    }
    // Unknown but not obviously textual — let the engine be the judge.
    Verdict::Pass
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

/// Best-effort resolve `host` and report whether it maps to a non-public range
/// (loopback/private/link-local/CGNAT/metadata/ULA). Resolution failure is
/// treated as "not private" so the subsequent fetch produces the real
/// unfetchable verdict (a dead public host shouldn't be mistaken for SSRF).
async fn resolves_to_private(host: &str, port: u16) -> bool {
    // Literal IP?
    if let Ok(ip) = host.parse::<IpAddr>() {
        return is_disallowed_ip(ip);
    }
    match tokio::net::lookup_host((host, port)).await {
        Ok(addrs) => {
            let mut any = false;
            for a in addrs {
                any = true;
                if is_disallowed_ip(a.ip()) {
                    return true; // reject if ANY resolved IP is internal
                }
            }
            // If it didn't resolve to anything, let the fetch decide.
            let _ = any;
            false
        }
        Err(_) => false,
    }
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
            max_bytes: 65536,
            max_concurrency: 4,
            allow_private_hosts: true,
        }
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
            vec![
                "https://x/a.png".to_string(),
                "data:image/png;base64,AAAA".to_string()
            ]
        );
    }

    #[test]
    fn text_only_request_has_no_images() {
        let v = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(collect_image_urls(&v).is_empty());
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
        // HTML error page masquerading as an image URL → reject.
        assert_eq!(
            classify_bytes(
                b"<!doctype html><html>login</html>",
                Some("text/html; charset=utf-8")
            ),
            Verdict::Reject
        );
        // Real PNG → pass even with a missing/odd content type.
        assert_eq!(
            classify_bytes(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A], None),
            Verdict::Pass
        );
        // image/* content type → pass regardless of (truncated) bytes.
        assert_eq!(
            classify_bytes(&[0x00, 0x01], Some("image/png")),
            Verdict::Pass
        );
        // Unknown binary, no content type → fail-open (pass).
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
        // base64 of an HTML page → reject.
        let html = base64::engine::general_purpose::STANDARD.encode("<!doctype html><html></html>");
        assert_eq!(
            validate_data_url(&format!("image/png;base64,{html}")),
            Verdict::Reject
        );
        // Invalid base64 → reject.
        assert_eq!(
            validate_data_url("image/png;base64,!!!notb64!!!"),
            Verdict::Reject
        );
        // Malformed (no comma) → reject.
        assert_eq!(validate_data_url("image/png;base64"), Verdict::Reject);
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
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fc00::1",
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
        assert!(reject_invalid_images(&v, &reqwest::Client::new(), &c)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn dangerous_scheme_rejected() {
        let v = serde_json::json!({
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"file:///etc/passwd"}}
            ]}]
        });
        let err = reject_invalid_images(&v, &reqwest::Client::new(), &cfg()).await;
        assert!(matches!(err, Err(AppError::BadRequest(_))));
    }

    #[tokio::test]
    async fn http_path_fetches_and_classifies() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // A dead Facebook-style link that 200s with an HTML login page.
        Mock::given(method("GET"))
            .and(path("/html.jpg"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/html; charset=utf-8")
                    .set_body_string("<!doctype html><html><body>login</body></html>"),
            )
            .mount(&server)
            .await;
        // A real PNG.
        Mock::given(method("GET"))
            .and(path("/real.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "image/png")
                    .set_body_bytes(vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0]),
            )
            .mount(&server)
            .await;
        // Everything else 404s.

        let client = reqwest::Client::new();
        let c = cfg();

        let req = |p: &str| {
            serde_json::json!({"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url": format!("{}{}", server.uri(), p)}}
            ]}]})
        };

        // HTML masquerading as an image → reject.
        assert!(matches!(
            reject_invalid_images(&req("/html.jpg"), &client, &c).await,
            Err(AppError::BadRequest(_))
        ));
        // Real PNG → pass.
        assert!(reject_invalid_images(&req("/real.png"), &client, &c)
            .await
            .is_ok());
        // Unreachable (404) → reject.
        assert!(matches!(
            reject_invalid_images(&req("/missing.png"), &client, &c).await,
            Err(AppError::BadRequest(_))
        ));
    }

    #[tokio::test]
    async fn ssrf_blocks_loopback_when_not_allowed() {
        // With private hosts disallowed (production default), a loopback image
        // URL is refused without any fetch.
        let mut c = cfg();
        c.allow_private_hosts = false;
        let v = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"http://127.0.0.1:1/x.png"}}
        ]}]});
        assert!(matches!(
            reject_invalid_images(&v, &reqwest::Client::new(), &c).await,
            Err(AppError::BadRequest(_))
        ));
    }
}
