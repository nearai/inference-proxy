pub mod attestation;
pub mod catch_all;
pub mod chat;
pub mod completions;
pub mod health;
pub mod metrics;
pub mod passthrough;
pub mod signature;

use axum::routing::{get, post};
use axum::Router;

use crate::AppState;

pub fn build_router() -> Router<AppState> {
    Router::new()
        // Unauthenticated health endpoints
        .route("/", get(health::root))
        .route("/version", get(health::version))
        // Unauthenticated metrics/models
        .route("/v1/metrics", get(metrics::metrics))
        .route("/v1/models", get(metrics::models))
        // Authenticated endpoints
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/tokenize", post(passthrough::tokenize))
        .route("/v1/embeddings", post(passthrough::embeddings))
        .route("/v1/rerank", post(passthrough::rerank))
        .route("/v1/score", post(passthrough::score))
        .route(
            "/v1/images/generations",
            post(passthrough::images_generations),
        )
        .route("/v1/images/edits", post(passthrough::images_edits))
        .route(
            "/v1/audio/transcriptions",
            post(passthrough::audio_transcriptions),
        )
        .route("/v1/signature/{chat_id}", get(signature::signature))
        .route(
            "/v1/attestation/report",
            get(attestation::attestation_report),
        )
        .fallback(catch_all::catch_all)
}
