//! Forward GPU evidence collection to a *delegate* inference-proxy on
//! the same host.
//!
//! ## Why
//!
//! The NVML/firmware race that caused production `NONCE_NOT_MATCHING`
//! rejections (see [`crate::attestation`] and PR #107) happens when
//! multiple processes on the same host call NVML concurrently. PR #118
//! added an in-process `Mutex` around the SDK call, which prevents
//! within-process contention but does *not* prevent cross-process
//! contention. On hosts where multiple `vllm-proxy-rs` containers share
//! the same physical GPUs (notably gpu04 small-models with 8 proxies),
//! NVML can still see concurrent attestation queries from up to N
//! processes simultaneously.
//!
//! With this feature, all proxies on a host except one are configured
//! with `GPU_EVIDENCE_DELEGATE_URL=http://<leader>:8000`. When a
//! delegating proxy needs GPU evidence, it forwards the request to the
//! leader's `POST /internal/gpu_evidence` endpoint. Only the leader's
//! process touches NVML — its in-process `Mutex` then suffices to
//! serialize host-wide.
//!
//! ## Authentication
//!
//! Reuses `TOKEN`. The delegate endpoint is a sibling of the existing
//! authenticated routes; we send the same Bearer the caller already
//! shares with the rest of the inference-proxy ecosystem (cloud-api
//! → vllm-proxy-rs). No new secret to rotate.
//!
//! ## What this does NOT change
//!
//! - The on-the-wire shape returned to cloud-api: still the same
//!   `evidence_list` JSON shape NVIDIA's NRAS expects.
//! - The self-check + retry from PR #107: still wraps every collection
//!   path. The delegate path applies the self-check to the bytes the
//!   delegate returns, exactly like the local paths.
//! - The Python / SDK / subprocess dispatch: those still apply on the
//!   *delegate* side. Delegating proxies just don't reach them.

use std::time::Duration;

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// Wire shape for `POST /internal/gpu_evidence`.
///
/// Keep this stable — it's the only contract between the delegating
/// proxy and the leader on a host. Older delegators against newer
/// leaders (or vice versa) is the rolling-deploy reality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegateRequest {
    /// Hex-encoded 32-byte nonce. The leader binds it into the
    /// firmware-signed evidence; the delegate caller verifies the
    /// returned binary at offset 4..36 matches.
    pub nonce: String,
    /// Pass through the gpu-no-hw-mode flag for dev/test environments
    /// without GPUs. The leader's local fallbacks (Python paths) honor
    /// this; the SDK path doesn't support it.
    #[serde(default)]
    pub no_gpu_mode: bool,
}

/// Wire shape for the response. Just the `evidence_list` array — the
/// caller embeds it in its own outer `nvidia_payload`. We deliberately
/// don't return the full attestation report: GPU evidence is the only
/// thing that needs serialization across processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegateResponse {
    pub evidence_list: serde_json::Value,
}

/// Returns true when the operator has configured a delegate URL via
/// `GPU_EVIDENCE_DELEGATE_URL`. Currently unused by the dispatch path
/// in `attestation.rs` (which checks the `Option<&DelegateContext>`
/// the caller passes in directly), but kept as a single source of
/// truth for "is delegation configured?" if other callers need it.
pub fn is_active(cfg: &crate::config::Config) -> bool {
    cfg.gpu_evidence_delegate_url.is_some()
}

/// Collect GPU evidence by forwarding the request to the configured
/// delegate.
///
/// Returns the same `evidence_list` shape any of the local paths
/// (SDK / Python worker / subprocess) returns, so the caller's
/// downstream logic — `build_nvidia_payload`, the self-check on
/// bytes 4..36 — works uniformly.
///
/// Caller-side timeout comes from
/// `cfg.gpu_evidence_delegate_timeout_secs`. Network errors and
/// non-2xx responses bubble up as `anyhow::Error`. Note that the
/// outer retry loop in `collect_gpu_evidence_with_nonce_check`
/// only retries on *nonce mismatches* (the firmware race we're
/// guarding against) — transport failures from the delegate
/// short-circuit the loop and are surfaced to cloud-api so it can
/// rotate to a different backend, rather than burning the retry
/// budget on a delegate that's flat-out down.
pub async fn collect_via_delegate(
    cfg: &crate::config::Config,
    http_client: &reqwest::Client,
    nonce_hex: &str,
    gpu_no_hw_mode: bool,
) -> anyhow::Result<serde_json::Value> {
    let base_url = cfg
        .gpu_evidence_delegate_url
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("collect_via_delegate called without configured URL"))?;
    let token = cfg.tokens.first().ok_or_else(|| {
        anyhow::anyhow!(
            "collect_via_delegate: no admin TOKEN configured to authenticate with delegate"
        )
    })?;
    collect_via_delegate_inner(
        http_client,
        base_url,
        token,
        Duration::from_secs(cfg.gpu_evidence_delegate_timeout_secs),
        nonce_hex,
        gpu_no_hw_mode,
    )
    .await
}

/// Lower-level helper that doesn't require a full `Config` — kept
/// `pub(crate)` so unit tests can drive it against a wiremock server
/// without standing up the whole config struct.
pub(crate) async fn collect_via_delegate_inner(
    http_client: &reqwest::Client,
    base_url: &str,
    token: &str,
    timeout: Duration,
    nonce_hex: &str,
    gpu_no_hw_mode: bool,
) -> anyhow::Result<serde_json::Value> {
    let url = format!("{base_url}/internal/gpu_evidence");
    let req = DelegateRequest {
        nonce: nonce_hex.to_string(),
        no_gpu_mode: gpu_no_hw_mode,
    };

    let response = http_client
        .post(&url)
        .timeout(timeout)
        .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
        .json(&req)
        .send()
        .await
        .with_context(|| format!("delegate request to {url} failed"))?;

    let status = response.status();
    if !status.is_success() {
        // Read at most ERROR_BODY_CAP bytes — some error responses can
        // carry full evidence payloads on certain misconfigs and we
        // don't want to allocate megabytes just to log a snippet.
        const ERROR_BODY_CAP: usize = 2048;
        let bytes = response.bytes().await.unwrap_or_default();
        let snippet = String::from_utf8_lossy(&bytes[..bytes.len().min(ERROR_BODY_CAP)]);
        anyhow::bail!("delegate {url} returned HTTP {status}: {}", snippet);
    }

    let parsed: DelegateResponse = response
        .json()
        .await
        .with_context(|| format!("parsing delegate response from {url}"))?;
    Ok(parsed.evidence_list)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delegate_request_roundtrip() {
        let req = DelegateRequest {
            nonce: "abc123".into(),
            no_gpu_mode: false,
        };
        let s = serde_json::to_string(&req).unwrap();
        // no_gpu_mode default tested via #[serde(default)]
        let parsed: DelegateRequest = serde_json::from_str(r#"{"nonce":"abc123"}"#).unwrap();
        assert_eq!(parsed.nonce, "abc123");
        assert!(!parsed.no_gpu_mode);
        assert!(s.contains("\"nonce\":\"abc123\""));
    }

    #[test]
    fn delegate_response_accepts_evidence_array() {
        // The wire shape must accept whatever NVIDIA's evidence_list
        // happens to look like — that's an array of objects today,
        // but we treat it opaquely with serde_json::Value.
        let body = r#"{"evidence_list":[{"arch":"HOPPER","certificate":"…","evidence":"…"}]}"#;
        let resp: DelegateResponse = serde_json::from_str(body).unwrap();
        assert!(resp.evidence_list.is_array());
        assert_eq!(resp.evidence_list.as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn collect_via_delegate_inner_forwards_request_and_unwraps_evidence_list() {
        use wiremock::matchers::{body_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let nonce = "a".repeat(64);

        Mock::given(method("POST"))
            .and(path("/internal/gpu_evidence"))
            .and(header("authorization", "Bearer test-token"))
            .and(body_json(serde_json::json!({
                "nonce": nonce,
                "no_gpu_mode": false,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "evidence_list": [{"arch": "HOPPER", "evidence": "…"}],
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let evidence_list = collect_via_delegate_inner(
            &client,
            &server.uri(),
            "test-token",
            Duration::from_secs(5),
            &nonce,
            false,
        )
        .await
        .expect("delegate call succeeded");

        // Caller receives the unwrapped `evidence_list` array, ready
        // to splice into the outer nvidia_payload.
        let arr = evidence_list.as_array().expect("evidence_list is an array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["arch"], "HOPPER");
    }

    #[tokio::test]
    async fn collect_via_delegate_inner_surfaces_non_2xx_with_capped_body() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // 8 KiB body — should be truncated to ~2 KiB in the error message.
        let big_body = "x".repeat(8192);

        Mock::given(method("POST"))
            .and(path("/internal/gpu_evidence"))
            .respond_with(ResponseTemplate::new(503).set_body_string(big_body))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let err = collect_via_delegate_inner(
            &client,
            &server.uri(),
            "test-token",
            Duration::from_secs(5),
            &"b".repeat(64),
            false,
        )
        .await
        .expect_err("delegate should surface non-2xx as Err");

        let msg = format!("{err}");
        assert!(msg.contains("503"), "error preserves status: {msg}");
        // Body cap is 2048 bytes; allow some slack for the surrounding
        // template ("delegate <url> returned HTTP 503: <snippet>").
        assert!(
            msg.len() < 4096,
            "error message should be capped, got {} bytes",
            msg.len()
        );
    }
}
