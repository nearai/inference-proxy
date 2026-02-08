use axum::extract::FromRequestParts;
use axum::http::request::Parts;

use crate::error::AppError;
use crate::AppState;

/// Extractor that validates Bearer token authentication.
/// Use as a handler parameter to require auth on a route.
pub struct RequireAuth;

impl FromRequestParts<AppState> for RequireAuth {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let auth_header = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok());

        match auth_header {
            Some(header) if header.starts_with("Bearer ") => {
                let token = &header[7..];
                if token == state.config.token {
                    Ok(RequireAuth)
                } else {
                    Err(AppError::Unauthorized)
                }
            }
            _ => Err(AppError::Unauthorized),
        }
    }
}
