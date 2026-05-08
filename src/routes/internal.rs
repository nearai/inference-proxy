//! Internal-only endpoints intended for sibling proxies on the same
//! host. NOT meant to be exposed to cloud-api or end users.
//!
//! Currently:
//! - `POST /internal/gpu_evidence` — collects GPU evidence locally and
//!   returns just the `evidence_list`. Used by sibling proxies that
//!   delegate evidence collection to centralize NVML access. See
//!   [`crate::gpu_evidence_delegate`] for the why.
//!
//! Auth: shares the standard `TOKEN`-based `RequireAuth` extractor.
//! On a multi-proxy host the leader and delegators all carry the same
//! `PROXY_TOKEN`, so no new secret is needed.

use axum::extract::State;
use axum::Json;

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::gpu_evidence_delegate::{DelegateRequest, DelegateResponse};
use crate::AppState;

/// `POST /internal/gpu_evidence`
///
/// Collects GPU evidence locally on this proxy (using the SDK or
/// Python path, whichever is configured) and returns the resulting
/// `evidence_list`. **Always uses local collection** — never recurses
/// into another delegate, even if `GPU_EVIDENCE_DELEGATE_URL` is set
/// on this proxy. Recursion would defeat the purpose and risk loops.
///
/// The body is a `DelegateRequest` (nonce + optional no_gpu_mode). The
/// response is a `DelegateResponse` carrying the JSON `evidence_list`
/// the caller will splice into its outer `nvidia_payload`.
///
/// The same self-check + retry from PR #107 applies here, so the
/// caller can trust that bytes 4..36 of each evidence binary match the
/// nonce they sent.
pub async fn gpu_evidence(
    State(state): State<AppState>,
    _auth: RequireAuth,
    Json(req): Json<DelegateRequest>,
) -> Result<Json<DelegateResponse>, AppError> {
    // Decode + length-check the nonce up front so a malformed request
    // surfaces a clean 400 instead of a generic 500 from the SDK /
    // Python paths deep in the call stack. We need the bytes anyway
    // for `collect_gpu_evidence_with_nonce_check` to verify the
    // self-check binding at offset 4..36.
    let nonce_bytes: [u8; 32] = hex::decode(&req.nonce)
        .ok()
        .and_then(|v| <[u8; 32]>::try_from(v).ok())
        .ok_or_else(|| AppError::BadRequest("nonce must be a 32-byte hex string".to_string()))?;

    // Critical: pass `delegate_ctx = None` so we use a local backend
    // path and never recurse into another delegate. If this proxy
    // happens to also have GPU_EVIDENCE_DELEGATE_URL set (rare, but
    // possible on a misconfiguration), forwarding from here would
    // create a loop.
    let evidence_list = crate::attestation::collect_gpu_evidence_with_nonce_check(
        &req.nonce,
        &nonce_bytes,
        req.no_gpu_mode,
        Some(&state.attestation_cache),
        None, // bypass_delegate
    )
    .await
    .map_err(AppError::Internal)?;

    Ok(Json(DelegateResponse { evidence_list }))
}
