pub mod attestation;
pub mod catch_all;
pub mod chat;
pub mod completions;
pub mod health;
pub mod internal;
pub mod metrics;
pub mod ohttp;
pub mod passthrough;
pub mod signature;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;

use crate::AppState;

pub fn build_router() -> Router<AppState> {
    Router::new()
        // Unauthenticated health endpoints
        .route("/", get(health::root))
        .route("/version", get(health::version))
        .route("/healthz", get(health::healthz))
        // Unauthenticated Prometheus metrics
        .route(
            "/metrics",
            get(crate::metrics_middleware::prometheus_metrics_handler),
        )
        // Unauthenticated backend metrics/models
        .route("/v1/metrics", get(metrics::metrics))
        .route("/v1/models", get(metrics::models))
        // Unauthenticated attestation report
        .route(
            "/v1/attestation/report",
            get(attestation::attestation_report),
        )
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
        // Multipart upload routes: axum's `Multipart` extractor enforces the
        // global 2 MiB `DefaultBodyLimit`, which rejects the body before the
        // handler's own size guard (`max_audio_request_size` / `max_image_request_size`)
        // can run. Disabling the default limit here makes those per-type limits —
        // enforced incrementally in `read_field_chunks` / `read_field_data` — the
        // single source of truth. Without this, audio/images > ~2 MiB fail upstream
        // with a 502 (connection reset before the body is consumed).
        .route(
            "/v1/images/edits",
            post(passthrough::images_edits).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/v1/audio/transcriptions",
            post(passthrough::audio_transcriptions).layer(DefaultBodyLimit::disable()),
        )
        .route("/v1/signature/{chat_id}", get(signature::signature))
        // Internal — sibling proxies on the same host call this when
        // configured with GPU_EVIDENCE_DELEGATE_URL pointed at us.
        .route("/internal/gpu_evidence", post(internal::gpu_evidence))
        // OHTTP Gateway (RFC 9458) — POST /ohttp is unauthenticated; auth may be
        // inside the encrypted Binary HTTP message or on the outer HTTP request
        // (relay-injected Authorization). See `ohttp_relay` docs.
        .route("/.well-known/ohttp-gateway", get(ohttp::ohttp_config))
        .route("/v1/ohttp/config", get(ohttp::ohttp_config))
        .route("/ohttp", post(ohttp::ohttp_relay))
        .fallback(catch_all::catch_all)
}
