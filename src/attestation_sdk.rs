//! GPU evidence collection via NVIDIA's official attestation SDK Rust
//! bindings (`nv-attestation-sdk`), as an alternative to the Python
//! subprocess that shells out to `cc_admin.collect_gpu_evidence_remote`.
//!
//! This module is gated behind the `nv-attestation-sdk` Cargo feature
//! AND the runtime `USE_NV_ATTESTATION_SDK=true` env var. With the
//! feature off, the module compiles down to a stub that always reports
//! "unavailable" so the existing Python path keeps running unchanged.
//!
//! ## Why bother
//!
//! The Python path adds ~0.5–2s startup overhead per cold call (worked
//! around with a persistent worker), a fragile stdin/stdout JSON-line
//! protocol, and a stdout-pollution bug class (cc_admin printing C
//! library messages straight to fd 1). The Rust bindings are direct
//! FFI calls into NVIDIA's C++ SDK — no IPC, no subprocess.
//!
//! ## What this does NOT fix
//!
//! The production `NONCE_NOT_MATCHING` race (PR #107) lives below this
//! abstraction layer — at NVML/GPU firmware level. Switching backends
//! doesn't change that. The self-check + retry in
//! `collect_gpu_evidence_with_nonce_check` still applies regardless of
//! which backend produced the evidence.
//!
//! ## Threading
//!
//! Every type in `nv-attestation-sdk` except the zero-sized
//! `NvatSdk` lifecycle marker is `!Send + !Sync`. We therefore do all
//! evidence collection inside `tokio::task::spawn_blocking`, creating
//! the source/nonce/collection inside the closure and only crossing
//! threads with the resulting JSON `String`. Initialization happens
//! once via a `OnceCell` initialized at first use.

#[cfg(feature = "nv-attestation-sdk")]
mod inner {
    use anyhow::Context;
    use nv_attestation_sdk::{GpuEvidenceSource, Nonce, NvatSdk, SdkOptions};
    use std::time::Instant;
    use tokio::sync::{Mutex, OnceCell};

    /// Process-lifetime SDK handle.
    ///
    /// `NvatSdk` is a zero-sized lifecycle marker — keeping it alive
    /// keeps the underlying C SDK initialized; dropping it would call
    /// `nvat_sdk_shutdown`. We initialize once at first use and never
    /// drop (the `OnceCell` outlives anything that would call us).
    static SDK: OnceCell<NvatSdk> = OnceCell::const_new();

    /// Serializes GPU evidence collection across all in-process callers.
    ///
    /// **Why**: NVML and the GPU firmware do not handle concurrent
    /// attestation queries from a single process gracefully — under load,
    /// the C++ SDK's `GpuEvidenceSource::collect` returns errors that
    /// surface as TCP RSTs to clients. Reproduced 2026-05-08 on gpu07
    /// GLM-5 with 5 concurrent /v1/attestation/report calls: rounds 4–10
    /// returned 5/5 `Connection reset by peer` after rounds 1–3 worked
    /// fine. The Python path doesn't show this because `AttestationCache`
    /// has its own `gpu_worker: Mutex<...>` that already serializes
    /// `cc_admin.collect_gpu_evidence_remote` calls — we lost that
    /// serialization when bypassing the cache for the SDK path.
    ///
    /// Cost: serial NVML calls add wait time when concurrency is high.
    /// In practice cloud-api fires ~10 parallel discovery calls per
    /// model per refresh (every 5min), spread across backends; per-proxy
    /// concurrency is typically 2–5. With NVML evidence collection at
    /// ~1.5–2.5s per call on H100/H200, the worst-case wait under the
    /// Mutex for 5 queued requests is ~10s — within cloud-api's
    /// per-request timeout, and far better than the alternative (the
    /// proxy returning RSTs to half its callers).
    ///
    /// We deliberately use `tokio::sync::Mutex` (not std) so contended
    /// callers yield to the runtime instead of blocking the whole
    /// thread; the lock is held across `spawn_blocking`, so awaiting
    /// it is effectively awaiting "your turn at the SDK".
    static EVIDENCE_COLLECTION: Mutex<()> = Mutex::const_new(());

    async fn ensure_sdk_initialized() -> anyhow::Result<()> {
        SDK.get_or_try_init(|| async {
            // SDK init is fast (no NVML touch yet) but we run it on a
            // blocking thread anyway so we never hold up the runtime
            // on FFI work, no matter what the C side decides to do.
            tokio::task::spawn_blocking(|| {
                let opts = SdkOptions::new().context("SdkOptions::new")?;
                NvatSdk::init(opts).context("NvatSdk::init")
            })
            .await
            .context("spawn_blocking for NvatSdk::init")?
        })
        .await?;
        Ok(())
    }

    /// Collect GPU evidence via the SDK and return the JSON the C SDK
    /// emits — same wire shape we currently put under
    /// `nvidia_payload.evidence_list`.
    ///
    /// Runs on a blocking-pool thread so the (`!Send`) source/nonce/
    /// collection objects never cross await points, and so the FFI
    /// calls can't stall the async runtime.
    ///
    /// Serialized by `EVIDENCE_COLLECTION` so only one NVML call is in
    /// flight per process — see that static's docs for why.
    pub async fn collect_gpu_evidence_via_sdk(
        nonce_hex: &str,
    ) -> anyhow::Result<serde_json::Value> {
        ensure_sdk_initialized().await?;

        // Acquire the per-process serialization lock before we kick off
        // the FFI call. Wait time is observable via the metric below;
        // a sustained non-zero `gpu_evidence_sdk_lock_wait_total` is the
        // signal that NVML is the bottleneck under the current load.
        let wait_start = Instant::now();
        let _permit = EVIDENCE_COLLECTION.lock().await;
        let wait_ms = wait_start.elapsed().as_millis() as u64;
        if wait_ms > 0 {
            metrics::counter!("gpu_evidence_sdk_lock_wait_total").increment(wait_ms);
        }

        let nonce_hex_owned = nonce_hex.to_string();
        let json_str: String = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            // Per SDK guidance, create a fresh source per call rather
            // than sharing across threads — `GpuEvidenceSource` is
            // `!Send + !Sync`. NVML init lives inside `from_nvml`.
            let source = GpuEvidenceSource::from_nvml().context("GpuEvidenceSource::from_nvml")?;
            let nonce = Nonce::from_hex(&nonce_hex_owned).context("Nonce::from_hex")?;
            let evidence = source
                .collect(&nonce)
                .context("GpuEvidenceSource::collect")?;
            evidence.to_json().context("GpuEvidenceCollection::to_json")
        })
        .await
        .context("spawn_blocking for SDK evidence collection")??;

        // _permit dropped here, releasing the next waiter.
        drop(_permit);

        serde_json::from_str(&json_str).context("parsing SDK evidence JSON")
    }

    /// Whether the operator has opted into the SDK backend at runtime.
    /// Even with the Cargo feature compiled in, we keep the toggle so
    /// the Python path stays the default until staging proves the
    /// migration is safe.
    pub fn runtime_enabled() -> bool {
        std::env::var("USE_NV_ATTESTATION_SDK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }
}

#[cfg(not(feature = "nv-attestation-sdk"))]
mod inner {
    /// Stub used when the SDK feature is compiled out. Always reports
    /// "not available" so the cache keeps using the Python path.
    pub fn runtime_enabled() -> bool {
        false
    }

    pub async fn collect_gpu_evidence_via_sdk(
        _nonce_hex: &str,
    ) -> anyhow::Result<serde_json::Value> {
        anyhow::bail!("nv-attestation-sdk feature not compiled in")
    }
}

pub use inner::{collect_gpu_evidence_via_sdk, runtime_enabled};

/// Returns true when both the Cargo feature is compiled in AND the
/// operator has opted in via `USE_NV_ATTESTATION_SDK=true`. Callers
/// should branch on this to dispatch between the SDK and the existing
/// Python subprocess path.
pub fn is_active() -> bool {
    runtime_enabled()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_active_defaults_to_false_without_env_var() {
        // SAFETY: each test sets and clears its own env var; tests that
        // touch process-global state are scoped narrowly.
        // SAFETY: clearing env vars in tests requires `unsafe` on
        // recent Rust due to the Send/Sync contract on env mutation.
        unsafe {
            std::env::remove_var("USE_NV_ATTESTATION_SDK");
        }
        assert!(!is_active());
    }

    #[test]
    fn runtime_enabled_accepts_truthy_strings() {
        // Without the feature compiled in, runtime_enabled is the stub
        // that always returns false; with the feature, the env var
        // gates it. This test exercises both branches via the public
        // surface.
        for (val, expected_with_feature) in [
            ("true", true),
            ("True", true),
            ("TRUE", true),
            ("1", true),
            ("0", false),
            ("false", false),
            ("", false),
        ] {
            unsafe {
                std::env::set_var("USE_NV_ATTESTATION_SDK", val);
            }
            #[cfg(feature = "nv-attestation-sdk")]
            assert_eq!(
                runtime_enabled(),
                expected_with_feature,
                "USE_NV_ATTESTATION_SDK={val:?}"
            );
            #[cfg(not(feature = "nv-attestation-sdk"))]
            {
                let _ = expected_with_feature;
                // Stub always returns false regardless of env value.
                assert!(!runtime_enabled());
            }
        }
        unsafe {
            std::env::remove_var("USE_NV_ATTESTATION_SDK");
        }
    }
}
