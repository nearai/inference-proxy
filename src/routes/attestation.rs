use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::attestation::AttestationResult;
use crate::error::AppError;
use crate::types::AttestationResponse;
use crate::AppState;

#[derive(Deserialize)]
pub struct AttestationQuery {
    pub signing_algo: Option<String>,
    pub nonce: Option<String>,
    pub signing_address: Option<String>,
    /// Include TLS certificate fingerprint in the report data.
    /// Defaults to false; when true, report_data[..32] = SHA256(signing_address || cert_fingerprint).
    pub include_tls_fingerprint: Option<bool>,
}

/// GET /v1/attestation/report
pub async fn attestation_report(
    State(state): State<AppState>,
    Query(query): Query<AttestationQuery>,
) -> Result<impl IntoResponse, AppError> {
    let signing_algo = query.signing_algo.as_deref().unwrap_or("ecdsa");

    if signing_algo != "ecdsa" && signing_algo != "ed25519" {
        return Err(AppError::BadRequest(
            "Invalid signing algorithm. Must be 'ed25519' or 'ecdsa'".to_string(),
        ));
    }

    let (signing_address, signing_address_bytes, signing_public_key) = match signing_algo {
        "ecdsa" => (
            state.signing.ecdsa.signing_address.clone(),
            state.signing.ecdsa.signing_address_bytes.clone(),
            state.signing.ecdsa.signing_public_key.clone(),
        ),
        "ed25519" => (
            state.signing.ed25519.signing_address.clone(),
            state.signing.ed25519.signing_address_bytes.clone(),
            state.signing.ed25519.signing_public_key.clone(),
        ),
        _ => unreachable!(),
    };

    // If signing_address is specified and doesn't match, return 404
    if let Some(requested_addr) = &query.signing_address {
        if signing_address.to_lowercase() != requested_addr.to_lowercase() {
            return Err(AppError::NotFound(
                "Signing address not found on this server".to_string(),
            ));
        }
    }

    let include_tls = query.include_tls_fingerprint.unwrap_or(false);
    let delegate_ctx = state.config.gpu_evidence_delegate_url.as_ref().map(|_| {
        crate::attestation::DelegateContext {
            config: &state.config,
            http_client: &state.http_client,
        }
    });
    // Snapshot the current TLS fingerprint into a local so its `&str` is
    // alive for the `AttestationParams` borrow below. The tracker's value
    // may have just been refreshed by the background cache task.
    let tls_fp_snapshot = if include_tls {
        state.tls_cert_fingerprint.current()
    } else {
        None
    };
    // For nonce-provided requests we know the result will be a fresh report —
    // the nonce is cryptographically bound so the cache can't be used. Start
    // the compose-manager fetch immediately in a background task so it runs
    // concurrently with GPU-evidence collection + TDX-quote generation, saving
    // ~50 ms server-side when compose-manager is configured.
    // For nonce-less requests we skip the early fetch: they usually hit the
    // cache (background refresh keeps it warm) so the compose-manager
    // attestation is already included in the cached bytes.
    let early_cm_handle = if query.nonce.is_some() {
        if let Some(ref url) = state.config.compose_manager_url {
            let url = url.clone();
            let client = state.http_client.clone();
            let nonce = query.nonce.clone();
            Some(tokio::spawn(async move {
                crate::attestation::fetch_compose_manager_attestation(
                    &client,
                    &url,
                    nonce.as_deref(),
                )
                .await
            }))
        } else {
            None
        }
    } else {
        None
    };

    let result = crate::attestation::generate_attestation(
        crate::attestation::AttestationParams {
            model_name: &state.config.model_name,
            signing_address: &signing_address,
            signing_algo,
            signing_public_key: &signing_public_key,
            signing_address_bytes: &signing_address_bytes,
            nonce: query.nonce.as_deref(),
            gpu_no_hw_mode: state.config.gpu_no_hw_mode,
            tls_cert_fingerprint: tls_fp_snapshot.as_deref(),
        },
        Some(&state.attestation_cache),
        delegate_ctx.as_ref(),
    )
    .await
    .map_err(|e| match e {
        crate::attestation::AttestationError::InvalidNonce(msg) => AppError::BadRequest(msg),
        crate::attestation::AttestationError::Internal(e) => AppError::Internal(e),
    })?;

    match result {
        // Cache hit: return pre-serialized bytes directly (no clone, no serialization).
        // Compose-manager attestation is included in cached bytes by the background refresh task.
        AttestationResult::CachedBytes(bytes) => Ok((
            StatusCode::OK,
            [("content-type", "application/json")],
            bytes,
        )
            .into_response()),
        // Fresh report: obtain compose-manager attestation (already in flight if nonce
        // was provided) and return.
        AttestationResult::Fresh(report) => {
            let cm_attestation = if let Some(handle) = early_cm_handle {
                // Nonce-provided: compose-manager fetch was running concurrently.
                handle.await.unwrap_or(None)
            } else if let Some(ref url) = state.config.compose_manager_url {
                // Nonce-less cache miss (rare — background refresh keeps cache warm):
                // fall back to sequential fetch.
                crate::attestation::fetch_compose_manager_attestation(&state.http_client, url, None)
                    .await
            } else {
                None
            };

            // Cache nonce-less reports so subsequent requests get the full response
            // including compose-manager attestation.
            let ohttp_attestation = state.ohttp_attestation_ed25519.clone();

            // Cache nonce-less reports so subsequent requests get the full response
            // including compose-manager attestation.
            if query.nonce.is_none() {
                state
                    .attestation_cache
                    .set(
                        signing_algo,
                        include_tls,
                        report.as_ref().clone(),
                        cm_attestation.clone(),
                        ohttp_attestation.clone(),
                    )
                    .await;
            }

            let response = AttestationResponse::new(
                report.as_ref().clone(),
                vec![*report],
                cm_attestation,
                ohttp_attestation,
            );
            Ok(Json(response).into_response())
        }
    }
}
