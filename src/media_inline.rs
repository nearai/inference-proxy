//! Fetch remote media in the proxy and hand the engine bytes only.
//!
//! With `VLLM_PROXY_INLINE_MEDIA=1` every http(s) `image_url` and `audio_url`
//! in a chat request is downloaded here and rewritten to a `data:` URL before
//! the request is forwarded, so the engine never has to reach the network. That
//! is what lets the engine run on a Docker network with no route out.
//!
//! Fail closed: a remote item that cannot be fetched, is too large, or is not a
//! supported image/audio format is a 400. Nothing remote ever reaches the
//! engine while this is on. `data:` items and `input_audio` (already base64)
//! are only checked against the size caps.
//!
//! The proxy moves bytes and reads headers; it never decodes media. Decoding
//! stays in the engine, which has no network and no secrets.
//!
//! The fetch reuses the image-validation client: no automatic redirects, every
//! hop re-checked against the SSRF rules and the media-domain allowlist, and a
//! DNS resolver that refuses names pointing at private ranges.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::OnceLock;
use std::time::Duration;

use base64::Engine as _;
use futures_util::stream::{self, StreamExt};
use serde_json::Value;
use url::Host;

use crate::error::AppError;
use crate::image_validation::{
    is_disallowed_ip, is_image_magic, normalize_image_url_input,
    remote_host_matches_allowed_domains, validation_client,
};

const USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

/// Generic, content-free rejection. Never echoes a URL or any user data.
const REJECT_MSG: &str = "One or more media inputs could not be fetched or are not a supported \
     image or audio file. Ensure each URL is reachable, or send the file inline as base64.";

#[derive(Debug, Clone)]
pub struct MediaInlineConfig {
    pub enabled: bool,
    /// Per-item fetch timeout (connect + full body).
    pub timeout: Duration,
    /// Deadline for all fetches of one request.
    pub request_deadline: Duration,
    /// Max remote items (images + audio) fetched per request.
    pub max_items: usize,
    pub max_image_bytes: usize,
    pub max_audio_bytes: usize,
    /// Max raw bytes of all media in one request, inline and fetched.
    pub max_total_bytes: usize,
    /// Max decoded pixels per image, read from the header (0 = no cap).
    pub max_image_pixels: u64,
    pub max_redirects: usize,
    /// Global cap on concurrent fetches across all requests.
    pub max_concurrency: usize,
    /// Shared with image validation: permit private hosts (tests only).
    pub allow_private_hosts: bool,
    /// Shared with image validation: exact host allowlist (empty = any).
    pub allowed_domains: Vec<String>,
}

impl Default for MediaInlineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout: Duration::from_secs(10),
            request_deadline: Duration::from_secs(30),
            max_items: 32,
            max_image_bytes: 10 * 1024 * 1024,
            max_audio_bytes: 25 * 1024 * 1024,
            max_total_bytes: 50 * 1024 * 1024,
            max_image_pixels: 25_000_000,
            max_redirects: 3,
            max_concurrency: 32,
            allow_private_hosts: false,
            allowed_domains: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Image,
    Audio,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Image => "image",
            Kind::Audio => "audio",
        }
    }
    fn max_bytes(self, cfg: &MediaInlineConfig) -> usize {
        match self {
            Kind::Image => cfg.max_image_bytes,
            Kind::Audio => cfg.max_audio_bytes,
        }
    }
}

/// Where a media URL lives in the request, so it can be rewritten in place.
struct Slot {
    msg: usize,
    part: usize,
    kind: Kind,
    /// `{"image_url": "..."}` (bare string) instead of `{"image_url": {"url": "..."}}`.
    bare: bool,
}

fn reject(reason: &'static str, kind: Option<Kind>) -> AppError {
    metrics::counter!(
        "inference_proxy_media_inline_total",
        "kind" => kind.map(Kind::label).unwrap_or("any"),
        "result" => reason
    )
    .increment(1);
    tracing::info!(
        reason,
        kind = kind.map(Kind::label),
        "media inline rejected request"
    );
    AppError::BadRequest(REJECT_MSG.to_string())
}

/// Rewrite every remote `image_url` / `audio_url` in `request_json` to a
/// `data:` URL. No-op when disabled. `request_json` must already be decrypted.
pub async fn inline_remote_media(
    request_json: &mut Value,
    cfg: &MediaInlineConfig,
) -> Result<(), AppError> {
    if !cfg.enabled {
        return Ok(());
    }
    let Some(messages) = request_json.get("messages").and_then(|m| m.as_array()) else {
        return Ok(());
    };

    let mut slots: Vec<(Slot, String)> = Vec::new();
    let mut inline_bytes: usize = 0;
    for (mi, msg) in messages.iter().enumerate() {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for (pi, part) in parts.iter().enumerate() {
            let ty = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let (kind, field) = match ty {
                "image_url" => (Kind::Image, "image_url"),
                "audio_url" => (Kind::Audio, "audio_url"),
                "input_audio" => {
                    // Already base64 (OpenAI shape). Size cap only.
                    let data = part
                        .get("input_audio")
                        .and_then(|a| a.get("data"))
                        .and_then(|d| d.as_str())
                        .ok_or_else(|| reject("malformed_part", Some(Kind::Audio)))?;
                    let n = b64_decoded_len(data);
                    if n > cfg.max_audio_bytes {
                        return Err(reject("too_large", Some(Kind::Audio)));
                    }
                    inline_bytes = inline_bytes.saturating_add(n);
                    continue;
                }
                _ => continue,
            };
            let (url, bare) = match part.get(field) {
                Some(Value::Object(o)) => match o.get("url").and_then(|u| u.as_str()) {
                    Some(u) => (u.to_string(), false),
                    None => return Err(reject("malformed_part", Some(kind))),
                },
                Some(Value::String(s)) => (s.clone(), true),
                _ => return Err(reject("malformed_part", Some(kind))),
            };
            let normalized = normalize_image_url_input(&url).into_owned();
            let scheme = normalized
                .split_once(':')
                .map(|(s, _)| s.to_ascii_lowercase())
                .unwrap_or_default();
            match scheme.as_str() {
                "data" => {
                    let n = data_url_decoded_len(&normalized)
                        .ok_or_else(|| reject("bad_data_url", Some(kind)))?;
                    if n > kind.max_bytes(cfg) {
                        return Err(reject("too_large", Some(kind)));
                    }
                    inline_bytes = inline_bytes.saturating_add(n);
                }
                "http" | "https" => slots.push((
                    Slot {
                        msg: mi,
                        part: pi,
                        kind,
                        bare,
                    },
                    normalized,
                )),
                // Anything else would make the engine reach out (or read a
                // local file): refuse.
                _ => return Err(reject("unsupported_scheme", Some(kind))),
            }
        }
    }
    if inline_bytes > cfg.max_total_bytes {
        return Err(reject("request_too_large", None));
    }
    if slots.is_empty() {
        return Ok(());
    }

    // Fetch each distinct URL once, but remember how many parts use it: every
    // occurrence gets its own copy of the bytes in the forwarded body, so every
    // occurrence is charged against the request budget.
    let mut unique: Vec<(String, Kind)> = Vec::new();
    let mut uses: HashMap<(String, Kind), usize> = HashMap::new();
    for (slot, url) in &slots {
        let n = uses.entry((url.clone(), slot.kind)).or_insert(0);
        if *n == 0 {
            unique.push((url.clone(), slot.kind));
        }
        *n += 1;
    }
    let uses = &uses;
    if unique.len() > cfg.max_items {
        return Err(reject("too_many_items", None));
    }

    let per_request = cfg.max_concurrency.clamp(1, 8);
    let budget = cfg.max_total_bytes.saturating_sub(inline_bytes);
    let fetch_all = async {
        let mut out: HashMap<(String, Kind), String> = HashMap::new();
        let mut total = 0usize;
        let mut results = stream::iter(unique)
            .map(|(url, kind)| async move {
                let r = fetch_one(&url, kind, cfg).await;
                (url, kind, r)
            })
            .buffer_unordered(per_request);
        while let Some((url, kind, r)) = results.next().await {
            let (mime, bytes) = r.map_err(|reason| reject(reason, Some(kind)))?;
            let copies = uses.get(&(url.clone(), kind)).copied().unwrap_or(1);
            total = total.saturating_add(bytes.len().saturating_mul(copies));
            if total > budget {
                return Err(reject("request_too_large", None));
            }
            let data_url = format!(
                "data:{mime};base64,{}",
                base64::engine::general_purpose::STANDARD.encode(&bytes)
            );
            out.insert((url, kind), data_url);
        }
        Ok(out)
    };
    let fetched = match tokio::time::timeout(cfg.request_deadline, fetch_all).await {
        Ok(r) => r?,
        Err(_) => return Err(reject("deadline", None)),
    };

    let messages = request_json
        .get_mut("messages")
        .and_then(|m| m.as_array_mut())
        .expect("messages checked above");
    for (slot, url) in slots {
        let data_url = fetched
            .get(&(url, slot.kind))
            .cloned()
            .expect("every slot was fetched");
        let field = match slot.kind {
            Kind::Image => "image_url",
            Kind::Audio => "audio_url",
        };
        let part = &mut messages[slot.msg]["content"][slot.part];
        if slot.bare {
            // Engines (SGLang) only take the object shape; normalize the
            // lenient bare-string form while we are rewriting it anyway.
            part[field] = serde_json::json!({ "url": data_url });
        } else {
            part[field]["url"] = Value::String(data_url);
        }
        metrics::counter!(
            "inference_proxy_media_inline_total",
            "kind" => slot.kind.label(),
            "result" => "inlined"
        )
        .increment(1);
    }
    Ok(())
}

fn global_semaphore(permits: usize) -> &'static tokio::sync::Semaphore {
    static SEM: OnceLock<tokio::sync::Semaphore> = OnceLock::new();
    SEM.get_or_init(|| tokio::sync::Semaphore::new(permits.max(1)))
}

fn preflight(
    url: &reqwest::Url,
    cfg: &MediaInlineConfig,
    redirect_hop: bool,
) -> Result<(), &'static str> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err("unsupported_scheme");
    }
    if !remote_host_matches_allowed_domains(url, &cfg.allowed_domains) {
        return Err("disallowed_domain");
    }
    let block_private_literal = redirect_hop || !cfg.allow_private_hosts;
    match url.host() {
        None => Err("no_host"),
        Some(Host::Ipv4(ip)) if block_private_literal && is_disallowed_ip(IpAddr::V4(ip)) => {
            Err("ssrf_ip_literal")
        }
        Some(Host::Ipv6(ip)) if block_private_literal && is_disallowed_ip(IpAddr::V6(ip)) => {
            Err("ssrf_ip_literal")
        }
        _ => Ok(()),
    }
}

/// Download one item: returns (mime, bytes) or a rejection reason.
async fn fetch_one(
    url: &str,
    kind: Kind,
    cfg: &MediaInlineConfig,
) -> Result<(&'static str, Vec<u8>), &'static str> {
    let mut current = reqwest::Url::parse(url).map_err(|_| "bad_url")?;
    let _permit =
        tokio::time::timeout(cfg.timeout, global_semaphore(cfg.max_concurrency).acquire())
            .await
            .map_err(|_| "saturated")?
            .map_err(|_| "saturated")?;
    // One deadline for the whole item: every redirect hop and the body.
    tokio::time::timeout(
        cfg.timeout,
        fetch_following_redirects(&mut current, kind, cfg),
    )
    .await
    .map_err(|_| "timeout")?
}

async fn fetch_following_redirects(
    current: &mut reqwest::Url,
    kind: Kind,
    cfg: &MediaInlineConfig,
) -> Result<(&'static str, Vec<u8>), &'static str> {
    let cap = kind.max_bytes(cfg);
    let accept = match kind {
        Kind::Image => "image/*,*/*;q=0.8",
        Kind::Audio => "audio/*,*/*;q=0.8",
    };
    let mut hops = 0;
    loop {
        preflight(current, cfg, hops > 0)?;
        let resp = validation_client(cfg.allow_private_hosts)
            .get(current.clone())
            .timeout(cfg.timeout)
            .header(reqwest::header::USER_AGENT, USER_AGENT)
            .header(reqwest::header::ACCEPT, accept)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    "timeout"
                } else {
                    "fetch_error"
                }
            })?;
        if resp.status().is_redirection() {
            if hops >= cfg.max_redirects {
                return Err("too_many_redirects");
            }
            let loc = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or("bad_redirect")?;
            *current = current.join(loc).map_err(|_| "bad_redirect")?;
            hops += 1;
            continue;
        }
        if !resp.status().is_success() {
            return Err("http_status");
        }
        if resp.content_length().is_some_and(|n| n as usize > cap) {
            return Err("too_large");
        }
        let mut body = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                if e.is_timeout() {
                    "timeout"
                } else {
                    "body_error"
                }
            })?;
            if body.len() + chunk.len() > cap {
                return Err("too_large");
            }
            body.extend_from_slice(&chunk);
        }
        let mime = match kind {
            Kind::Image => image_mime(&body).ok_or("unsupported_type")?,
            Kind::Audio => audio_mime(&body).ok_or("unsupported_type")?,
        };
        if kind == Kind::Image && cfg.max_image_pixels > 0 {
            // Fail closed: a header we cannot read cannot be checked against
            // the cap, so it is refused rather than passed to the engine.
            let (w, h) = image_dimensions(&body).ok_or("unknown_dimensions")?;
            if (w as u64).saturating_mul(h as u64) > cfg.max_image_pixels {
                return Err("too_many_pixels");
            }
        }
        return Ok((mime, body));
    }
}

/// Accepted image formats, by magic bytes (the content-type header is not trusted).
fn image_mime(b: &[u8]) -> Option<&'static str> {
    if !is_image_magic(b) {
        return None;
    }
    if b.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if b.starts_with(b"GIF8") {
        Some("image/gif")
    } else if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        // BMP, TIFF, HEIC/AVIF: not in the accepted set.
        None
    }
}

/// Accepted audio formats, by magic bytes.
fn audio_mime(b: &[u8]) -> Option<&'static str> {
    if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE" {
        Some("audio/wav")
    } else if b.starts_with(b"ID3") || (b.len() >= 2 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0) {
        Some("audio/mpeg")
    } else if b.starts_with(b"fLaC") {
        Some("audio/flac")
    } else if b.starts_with(b"OggS") {
        Some("audio/ogg")
    } else if b.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        Some("audio/webm")
    } else if b.len() >= 12 && &b[4..8] == b"ftyp" && matches!(&b[8..11], b"M4A" | b"mp4" | b"iso")
    {
        Some("audio/mp4")
    } else {
        None
    }
}

/// Width and height from the header of a PNG, GIF, JPEG or WebP. `None` when
/// the header can't be read.
fn image_dimensions(b: &[u8]) -> Option<(u32, u32)> {
    let be32 = |i: usize| {
        b.get(i..i + 4)
            .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    };
    let le16 = |i: usize| {
        b.get(i..i + 2)
            .map(|s| u16::from_le_bytes([s[0], s[1]]) as u32)
    };
    let be16 = |i: usize| {
        b.get(i..i + 2)
            .map(|s| u16::from_be_bytes([s[0], s[1]]) as u32)
    };
    if b.starts_with(&[0x89, b'P', b'N', b'G']) {
        return Some((be32(16)?, be32(20)?));
    }
    if b.starts_with(b"GIF8") {
        return Some((le16(6)?, le16(8)?));
    }
    if b.len() >= 30 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        return match &b[12..16] {
            b"VP8X" => {
                let w = 1 + (b[24] as u32 | (b[25] as u32) << 8 | (b[26] as u32) << 16);
                let h = 1 + (b[27] as u32 | (b[28] as u32) << 8 | (b[29] as u32) << 16);
                Some((w, h))
            }
            b"VP8 " => Some((le16(26)? & 0x3FFF, le16(28)? & 0x3FFF)),
            b"VP8L" => {
                let bits = u32::from_le_bytes([b[21], b[22], b[23], b[24]]);
                Some(((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1))
            }
            _ => None,
        };
    }
    if b.starts_with(&[0xFF, 0xD8]) {
        let mut i = 2;
        while i + 9 < b.len() {
            if b[i] != 0xFF {
                i += 1;
                continue;
            }
            let marker = b[i + 1];
            if marker == 0xFF {
                // Fill byte before a marker (allowed by the JPEG spec).
                i += 1;
                continue;
            }
            if matches!(marker, 0xD8 | 0x01) || (0xD0..=0xD7).contains(&marker) {
                i += 2;
                continue;
            }
            let len = be16(i + 2)? as usize;
            let is_sof = matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC);
            if is_sof {
                return Some((be16(i + 7)?, be16(i + 5)?));
            }
            i += 2 + len;
        }
    }
    None
}

/// Decoded length of a base64 string, without decoding it.
fn b64_decoded_len(s: &str) -> usize {
    let s = s.trim_end_matches('=');
    s.len() / 4 * 3 + (s.len() % 4).saturating_sub(1)
}

/// Decoded length of a `data:...;base64,...` URL (or raw length for a non-base64 one).
fn data_url_decoded_len(url: &str) -> Option<usize> {
    let rest = url.split_once(':')?.1;
    let (meta, data) = rest.split_once(',')?;
    if meta.to_ascii_lowercase().ends_with(";base64") {
        Some(b64_decoded_len(data))
    } else {
        Some(data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const PNG_1X1: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0x0D, b'I', b'H', b'D', b'R', 0,
        0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0,
    ];

    fn wav() -> Vec<u8> {
        let mut v = b"RIFF\x24\x00\x00\x00WAVEfmt ".to_vec();
        v.extend_from_slice(&[0u8; 32]);
        v
    }

    fn cfg() -> MediaInlineConfig {
        MediaInlineConfig {
            enabled: true,
            allow_private_hosts: true,
            timeout: Duration::from_secs(3),
            ..Default::default()
        }
    }

    fn chat(parts: Value) -> Value {
        json!({"model": "m", "messages": [{"role": "user", "content": parts}]})
    }

    #[tokio::test]
    async fn disabled_is_a_no_op() {
        let mut req =
            chat(json!([{"type":"image_url","image_url":{"url":"https://x.invalid/a.png"}}]));
        let before = req.clone();
        inline_remote_media(&mut req, &MediaInlineConfig::default())
            .await
            .unwrap();
        assert_eq!(req, before);
    }

    #[tokio::test]
    async fn remote_image_and_audio_become_data_urls() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/a.png"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(PNG_1X1))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/a.wav"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(wav()))
            .mount(&server)
            .await;
        let img = format!("{}/a.png", server.uri());
        let aud = format!("{}/a.wav", server.uri());
        let mut req = chat(json!([
            {"type":"text","text":"hi"},
            {"type":"image_url","image_url":{"url": img, "detail":"low"}},
            {"type":"image_url","image_url": img},
            {"type":"audio_url","audio_url":{"url": aud}}
        ]));
        inline_remote_media(&mut req, &cfg()).await.unwrap();
        let parts = &req["messages"][0]["content"];
        assert!(parts[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
        assert_eq!(parts[1]["image_url"]["detail"], "low");
        assert!(parts[2]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
        assert!(parts[3]["audio_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:audio/wav;base64,"));
    }

    #[tokio::test]
    async fn unfetchable_or_wrong_type_is_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/page"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>no</html>"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/gone"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        for p in ["/page", "/gone"] {
            let mut req = chat(
                json!([{"type":"image_url","image_url":{"url": format!("{}{p}", server.uri())}}]),
            );
            assert!(inline_remote_media(&mut req, &cfg()).await.is_err(), "{p}");
        }
        // An image is not audio.
        Mock::given(method("GET"))
            .and(path("/img.png"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(PNG_1X1))
            .mount(&server)
            .await;
        let mut req = chat(
            json!([{"type":"audio_url","audio_url":{"url": format!("{}/img.png", server.uri())}}]),
        );
        assert!(inline_remote_media(&mut req, &cfg()).await.is_err());
    }

    #[tokio::test]
    async fn caps_are_enforced() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/big.png"))
            .respond_with(
                ResponseTemplate::new(200).set_body_bytes([PNG_1X1, &[0u8; 2048]].concat()),
            )
            .mount(&server)
            .await;
        let c = MediaInlineConfig {
            max_image_bytes: 1024,
            ..cfg()
        };
        let mut req = chat(
            json!([{"type":"image_url","image_url":{"url": format!("{}/big.png", server.uri())}}]),
        );
        assert!(inline_remote_media(&mut req, &c).await.is_err());

        // Pixel cap from the PNG header (100000 x 100000).
        let mut huge = PNG_1X1.to_vec();
        huge[16..20].copy_from_slice(&100_000u32.to_be_bytes());
        huge[20..24].copy_from_slice(&100_000u32.to_be_bytes());
        Mock::given(method("GET"))
            .and(path("/huge.png"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(huge))
            .mount(&server)
            .await;
        let mut req = chat(
            json!([{"type":"image_url","image_url":{"url": format!("{}/huge.png", server.uri())}}]),
        );
        assert!(inline_remote_media(&mut req, &cfg()).await.is_err());

        // Too many distinct items.
        let c = MediaInlineConfig {
            max_items: 1,
            ..cfg()
        };
        let mut req = chat(json!([
            {"type":"image_url","image_url":{"url": format!("{}/a", server.uri())}},
            {"type":"image_url","image_url":{"url": format!("{}/b", server.uri())}}
        ]));
        assert!(inline_remote_media(&mut req, &c).await.is_err());

        // input_audio over the audio cap.
        let c = MediaInlineConfig {
            max_audio_bytes: 3,
            ..cfg()
        };
        let mut req =
            chat(json!([{"type":"input_audio","input_audio":{"data":"AAAAAAAA","format":"wav"}}]));
        assert!(inline_remote_media(&mut req, &c).await.is_err());
    }

    #[tokio::test]
    async fn data_urls_pass_and_other_schemes_are_refused() {
        let mut req = chat(json!([
            {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}},
            {"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}
        ]));
        let before = req.clone();
        inline_remote_media(&mut req, &cfg()).await.unwrap();
        assert_eq!(req, before);
        for u in ["file:///etc/passwd", "ftp://x/a.png", "gopher://x"] {
            let mut req = chat(json!([{"type":"image_url","image_url":{"url": u}}]));
            assert!(inline_remote_media(&mut req, &cfg()).await.is_err(), "{u}");
        }
    }

    #[tokio::test]
    async fn private_hosts_are_refused_in_production_mode() {
        let c = MediaInlineConfig {
            allow_private_hosts: false,
            ..cfg()
        };
        for u in [
            "http://127.0.0.1/a.png",
            "http://169.254.169.254/latest",
            "http://[::1]/a.png",
        ] {
            let mut req = chat(json!([{"type":"image_url","image_url":{"url": u}}]));
            assert!(inline_remote_media(&mut req, &c).await.is_err(), "{u}");
        }
    }

    #[tokio::test]
    async fn redirect_to_private_literal_is_refused() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/r"))
            .respond_with(
                ResponseTemplate::new(302).insert_header("location", "http://10.0.0.1/a.png"),
            )
            .mount(&server)
            .await;
        let mut req =
            chat(json!([{"type":"image_url","image_url":{"url": format!("{}/r", server.uri())}}]));
        assert!(inline_remote_media(&mut req, &cfg()).await.is_err());
    }

    fn jpeg(w: u16, h: u16, fill: usize) -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00];
        v.extend(std::iter::repeat_n(0xFF, fill));
        v.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
        v.extend_from_slice(&h.to_be_bytes());
        v.extend_from_slice(&w.to_be_bytes());
        v.extend_from_slice(&[0u8; 16]);
        v
    }

    #[test]
    fn jpeg_dimensions_survive_marker_fill_bytes() {
        assert_eq!(image_dimensions(&jpeg(6000, 5000, 0)), Some((6000, 5000)));
        assert_eq!(image_dimensions(&jpeg(6000, 5000, 1)), Some((6000, 5000)));
        assert_eq!(image_dimensions(&jpeg(6000, 5000, 5)), Some((6000, 5000)));
    }

    #[tokio::test]
    async fn padded_oversized_jpeg_and_unreadable_header_are_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/padded.jpg"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(jpeg(6000, 5000, 1)))
            .mount(&server)
            .await;
        // A JPEG with no SOF segment: dimensions unknown.
        Mock::given(method("GET"))
            .and(path("/nosof.jpg"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![
                0xFF, 0xD8, 0xFF, 0xE0, 0, 4, 0, 0, 0xFF, 0xD9, 0, 0, 0, 0,
            ]))
            .mount(&server)
            .await;
        for p in ["/padded.jpg", "/nosof.jpg"] {
            let mut req = chat(
                json!([{"type":"image_url","image_url":{"url": format!("{}{p}", server.uri())}}]),
            );
            assert!(inline_remote_media(&mut req, &cfg()).await.is_err(), "{p}");
        }
    }

    #[tokio::test]
    async fn repeated_url_is_fetched_once_but_charged_per_use() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/a.png"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(PNG_1X1))
            .expect(2)
            .mount(&server)
            .await;
        let url = format!("{}/a.png", server.uri());
        let parts = |n: usize| {
            Value::Array(
                (0..n)
                    .map(|_| json!({"type":"image_url","image_url":{"url": url}}))
                    .collect(),
            )
        };
        // 3 uses of a 29-byte image fit a 100-byte budget, 4 do not.
        let c = MediaInlineConfig {
            max_total_bytes: 100,
            ..cfg()
        };
        let mut ok = chat(parts(3));
        inline_remote_media(&mut ok, &c).await.unwrap();
        let mut too_many = chat(parts(4));
        assert!(inline_remote_media(&mut too_many, &c).await.is_err());
    }

    #[tokio::test]
    async fn item_timeout_covers_the_redirect_chain() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/r"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("location", "/a.png")
                    .set_delay(Duration::from_millis(700)),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/a.png"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(PNG_1X1)
                    .set_delay(Duration::from_millis(700)),
            )
            .mount(&server)
            .await;
        // Each hop is under the 1 s item timeout, the chain is not.
        let c = MediaInlineConfig {
            timeout: Duration::from_secs(1),
            ..cfg()
        };
        let mut req =
            chat(json!([{"type":"image_url","image_url":{"url": format!("{}/r", server.uri())}}]));
        assert!(inline_remote_media(&mut req, &c).await.is_err());
    }

    #[test]
    fn dimensions_and_types() {
        assert_eq!(image_dimensions(PNG_1X1), Some((1, 1)));
        assert_eq!(image_mime(PNG_1X1), Some("image/png"));
        assert_eq!(audio_mime(&wav()), Some("audio/wav"));
        assert_eq!(audio_mime(b"ID3\x04\x00"), Some("audio/mpeg"));
        assert_eq!(audio_mime(b"fLaC\x00"), Some("audio/flac"));
        assert_eq!(image_mime(b"BM\x00\x00\x00"), None);
        assert_eq!(b64_decoded_len("AAAA"), 3);
        assert_eq!(b64_decoded_len("AAAAAA=="), 4);
        assert_eq!(data_url_decoded_len("data:image/png;base64,AAAA"), Some(3));
    }
}
