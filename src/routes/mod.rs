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

pub const ROUTE_ROOT: &str = "/";
pub const ROUTE_VERSION: &str = "/version";
pub const ROUTE_HEALTHZ: &str = "/healthz";
pub const ROUTE_METRICS: &str = "/metrics";
pub const ROUTE_V1_METRICS: &str = "/v1/metrics";
pub const ROUTE_V1_MODELS: &str = "/v1/models";
pub const ROUTE_ATTESTATION_REPORT: &str = "/v1/attestation/report";
pub const ROUTE_CHAT_COMPLETIONS: &str = "/v1/chat/completions";
pub const ROUTE_COMPLETIONS: &str = "/v1/completions";
pub const ROUTE_TOKENIZE: &str = "/v1/tokenize";
pub const ROUTE_EMBEDDINGS: &str = "/v1/embeddings";
pub const ROUTE_RERANK: &str = "/v1/rerank";
pub const ROUTE_SCORE: &str = "/v1/score";
pub const ROUTE_IMAGES_GENERATIONS: &str = "/v1/images/generations";
pub const ROUTE_IMAGES_EDITS: &str = "/v1/images/edits";
pub const ROUTE_AUDIO_TRANSCRIPTIONS: &str = "/v1/audio/transcriptions";
pub const ROUTE_SIGNATURE: &str = "/v1/signature/{chat_id}";
pub const ROUTE_INTERNAL_GPU_EVIDENCE: &str = "/internal/gpu_evidence";
pub const ROUTE_OHTTP_WELL_KNOWN: &str = "/.well-known/ohttp-gateway";
pub const ROUTE_OHTTP_CONFIG: &str = "/v1/ohttp/config";
pub const ROUTE_OHTTP_RELAY: &str = "/ohttp";

pub fn build_router() -> Router<AppState> {
    Router::new()
        // Unauthenticated health endpoints
        .route(ROUTE_ROOT, get(health::root))
        .route(ROUTE_VERSION, get(health::version))
        .route(ROUTE_HEALTHZ, get(health::healthz))
        // Unauthenticated Prometheus metrics
        .route(
            ROUTE_METRICS,
            get(crate::metrics_middleware::prometheus_metrics_handler),
        )
        // Unauthenticated backend metrics/models
        .route(ROUTE_V1_METRICS, get(metrics::metrics))
        .route(ROUTE_V1_MODELS, get(metrics::models))
        // Unauthenticated attestation report
        .route(
            ROUTE_ATTESTATION_REPORT,
            get(attestation::attestation_report),
        )
        // Authenticated endpoints
        .route(ROUTE_CHAT_COMPLETIONS, post(chat::chat_completions))
        .route(ROUTE_COMPLETIONS, post(completions::completions))
        .route(ROUTE_TOKENIZE, post(passthrough::tokenize))
        .route(ROUTE_EMBEDDINGS, post(passthrough::embeddings))
        .route(ROUTE_RERANK, post(passthrough::rerank))
        .route(ROUTE_SCORE, post(passthrough::score))
        .route(
            ROUTE_IMAGES_GENERATIONS,
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
            ROUTE_IMAGES_EDITS,
            post(passthrough::images_edits).layer(DefaultBodyLimit::disable()),
        )
        .route(
            ROUTE_AUDIO_TRANSCRIPTIONS,
            post(passthrough::audio_transcriptions).layer(DefaultBodyLimit::disable()),
        )
        .route(ROUTE_SIGNATURE, get(signature::signature))
        // Internal — sibling proxies on the same host call this when
        // configured with GPU_EVIDENCE_DELEGATE_URL pointed at us.
        .route(ROUTE_INTERNAL_GPU_EVIDENCE, post(internal::gpu_evidence))
        // OHTTP Gateway (RFC 9458) — POST /ohttp is unauthenticated; auth may be
        // inside the encrypted Binary HTTP message or on the outer HTTP request
        // (relay-injected Authorization). See `ohttp_relay` docs.
        .route(ROUTE_OHTTP_WELL_KNOWN, get(ohttp::ohttp_config))
        .route(ROUTE_OHTTP_CONFIG, get(ohttp::ohttp_config))
        .route(ROUTE_OHTTP_RELAY, post(ohttp::ohttp_relay))
        .fallback(catch_all::catch_all)
}
