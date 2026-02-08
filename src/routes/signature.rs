use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::auth::RequireAuth;
use crate::error::AppError;
use crate::types::{SignatureResponse, SignedChat};
use crate::AppState;

#[derive(Deserialize)]
pub struct SignatureQuery {
    pub signing_algo: Option<String>,
}

/// GET /v1/signature/{chat_id}?signing_algo=ecdsa
pub async fn signature(
    State(state): State<AppState>,
    _auth: RequireAuth,
    Path(chat_id): Path<String>,
    Query(query): Query<SignatureQuery>,
) -> Result<impl IntoResponse, AppError> {
    let cache_value = state
        .cache
        .get_chat(&chat_id)
        .ok_or_else(|| AppError::NotFound("Chat id not found or expired".to_string()))?;

    let signed: SignedChat = serde_json::from_str(&cache_value)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Failed to parse cache value: {e}")))?;

    let signing_algo = query.signing_algo.as_deref().unwrap_or("ecdsa");

    let (signature, signing_address) = match signing_algo {
        "ecdsa" => (signed.signature_ecdsa, signed.signing_address_ecdsa),
        "ed25519" => (signed.signature_ed25519, signed.signing_address_ed25519),
        _ => {
            return Err(AppError::BadRequest(
                "Invalid signing algorithm. Must be 'ed25519' or 'ecdsa'".to_string(),
            ));
        }
    };

    Ok(Json(
        serde_json::to_value(SignatureResponse {
            text: signed.text,
            signature,
            signing_address,
            signing_algo: signing_algo.to_string(),
        })
        .unwrap(),
    ))
}
