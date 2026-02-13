use axum::extract::{Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::types::AttestationResponse;
use crate::AppState;

#[derive(Deserialize)]
pub struct AttestationQuery {
    pub signing_algo: Option<String>,
    pub nonce: Option<String>,
    pub signing_address: Option<String>,
}

/// GET /v1/attestation/report
pub async fn attestation_report(
    State(state): State<AppState>,
    _auth: RequireAuth,
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

    let report = crate::attestation::generate_attestation(
        &signing_address,
        signing_algo,
        &signing_public_key,
        &signing_address_bytes,
        query.nonce.as_deref(),
        state.config.gpu_no_hw_mode,
        state.tls_cert_fingerprint.as_deref(),
    )
    .await
    .map_err(|e| match e {
        crate::attestation::AttestationError::InvalidNonce(msg) => AppError::BadRequest(msg),
        crate::attestation::AttestationError::Internal(e) => AppError::Internal(e),
    })?;

    let response = AttestationResponse {
        report: report.clone(),
        all_attestations: vec![report],
    };

    Ok(Json(serde_json::to_value(response).unwrap()))
}
