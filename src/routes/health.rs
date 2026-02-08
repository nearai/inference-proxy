use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::AppState;

/// GET / → {}
pub async fn root() -> impl IntoResponse {
    Json(serde_json::json!({}))
}

/// GET /version → {"version": "...", "type": "proxy"}
pub async fn version(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "version": state.config.git_rev,
        "type": "proxy",
    }))
}
