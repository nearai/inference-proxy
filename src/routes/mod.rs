pub mod attestation;
pub mod chat;
mod completion_placement;
pub mod completions;
pub mod health;
pub mod internal;
pub mod metrics;
pub mod ohttp;
pub mod passthrough;
pub mod privacy;
pub mod signature;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post, MethodRouter};
use axum::Router;

use crate::error::AppError;
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
pub const ROUTE_PRIVACY_CLASSIFY: &str = "/v1/privacy/classify";
pub const ROUTE_IMAGES_GENERATIONS: &str = "/v1/images/generations";
pub const ROUTE_IMAGES_EDITS: &str = "/v1/images/edits";
pub const ROUTE_AUDIO_TRANSCRIPTIONS: &str = "/v1/audio/transcriptions";
pub const ROUTE_SIGNATURE: &str = "/v1/signature/{chat_id}";
pub const ROUTE_INTERNAL_GPU_EVIDENCE: &str = "/internal/gpu_evidence";
pub const ROUTE_OHTTP_WELL_KNOWN: &str = "/.well-known/ohttp-gateway";
pub const ROUTE_OHTTP_CONFIG: &str = "/v1/ohttp/config";
pub const ROUTE_OHTTP_RELAY: &str = "/ohttp";

/// Fail closed for every route that is not explicitly registered below.
///
/// This handler intentionally has no request extractors: unknown requests are
/// rejected without authenticating, buffering their body, or contacting an
/// inference backend.
async fn unknown_route() -> AppError {
    AppError::NotFound("Endpoint not found".to_string())
}

/// What the process serves: the router for `state` behind the middleware
/// stack, outermost first: HTTP metrics, the request id, the per-IP rate
/// limit. `main` serves exactly this, and the list-mode tests drive it.
pub fn build_app(state: AppState) -> Router {
    let rate_limit_state = crate::rate_limit::RateLimitState {
        limiter: crate::rate_limit::build_rate_limiter(
            state.config.rate_limit_per_second,
            state.config.rate_limit_burst_size,
        ),
        trust_proxy_headers: state.config.rate_limit_trust_proxy_headers,
    };
    build_router_for(&state)
        .layer(axum::middleware::from_fn(
            crate::rate_limit::rate_limit_middleware,
        ))
        .layer(axum::Extension(rate_limit_state))
        .layer(axum::middleware::from_fn(crate::request_id_middleware))
        .layer(axum::middleware::from_fn(
            crate::metrics_middleware::metrics_middleware,
        ))
        .with_state(state)
}

/// The router for `state`: every route for a process that serves one model,
/// the model-list subset in gateway list mode.
pub fn build_router_for(state: &AppState) -> Router<AppState> {
    if state.models.is_some() {
        build_model_list_router()
    } else {
        build_router()
    }
}

/// One route: its path and its handlers.
pub type Route = (&'static str, MethodRouter<AppState>);

/// A router with exactly `routes`, failing closed on everything else.
fn router_of(routes: Vec<Route>) -> Router<AppState> {
    routes
        .into_iter()
        .fold(Router::new(), |router, (path, handlers)| {
            router.route(path, handlers)
        })
        .fallback(unknown_route)
}

pub fn build_model_list_router() -> Router<AppState> {
    router_of(model_list_routes())
}

pub fn build_router() -> Router<AppState> {
    router_of(single_model_routes())
}

/// Gateway list mode (`model_list.rs`): only what is defined for several
/// models — chat completions, completions, the models document, health,
/// metrics and the version. The other inference routes (tokenize, embeddings,
/// rerank, score, images, audio, privacy), the engine metrics passthrough,
/// attestation, signatures and OHTTP all assume the one model, backend pool
/// and signing identity of a single-model process, so here they are the same
/// 404 as any undeclared route.
///
/// The router is built from this list and from nothing else, so a test that
/// walks it sees every route (see `single_model_routes`).
pub fn model_list_routes() -> Vec<Route> {
    vec![
        (ROUTE_VERSION, get(health::version)),
        (ROUTE_HEALTHZ, get(health::healthz)),
        (
            ROUTE_METRICS,
            get(crate::metrics_middleware::prometheus_metrics_handler),
        ),
        (ROUTE_V1_MODELS, get(metrics::models)),
        (ROUTE_CHAT_COMPLETIONS, post(chat::chat_completions)),
        (ROUTE_COMPLETIONS, post(completions::completions)),
    ]
}

/// Every route of a process that serves one model. `build_router` registers
/// exactly these, so the list is the router: the test that every route which
/// takes a customer key applies per-organization model access
/// (`model_access.rs`) walks it, and a route added here is covered by that
/// test without anyone remembering to add it there.
pub fn single_model_routes() -> Vec<Route> {
    vec![
        // Unauthenticated health endpoints
        (ROUTE_ROOT, get(health::root)),
        (ROUTE_VERSION, get(health::version)),
        (ROUTE_HEALTHZ, get(health::healthz)),
        // Unauthenticated Prometheus metrics
        (
            ROUTE_METRICS,
            get(crate::metrics_middleware::prometheus_metrics_handler),
        ),
        // Unauthenticated backend metrics/models
        (ROUTE_V1_METRICS, get(metrics::metrics)),
        (ROUTE_V1_MODELS, get(metrics::models)),
        // Unauthenticated attestation report
        (
            ROUTE_ATTESTATION_REPORT,
            get(attestation::attestation_report),
        ),
        // Authenticated endpoints
        (ROUTE_CHAT_COMPLETIONS, post(chat::chat_completions)),
        (ROUTE_COMPLETIONS, post(completions::completions)),
        (ROUTE_TOKENIZE, post(passthrough::tokenize)),
        (ROUTE_EMBEDDINGS, post(passthrough::embeddings)),
        (ROUTE_RERANK, post(passthrough::rerank)),
        (ROUTE_SCORE, post(passthrough::score)),
        (ROUTE_PRIVACY_CLASSIFY, post(privacy::classify)),
        (
            ROUTE_IMAGES_GENERATIONS,
            post(passthrough::images_generations),
        ),
        // Multipart upload routes: axum's `Multipart` extractor enforces the
        // global 2 MiB `DefaultBodyLimit`, which rejects the body before the
        // handler's own size guard (`max_audio_request_size` / `max_image_request_size`)
        // can run. Disabling the default limit here makes those per-type limits —
        // enforced incrementally in `read_field_chunks` — the single source of
        // truth. Without this, audio/images > ~2 MiB fail upstream
        // with a 502 (connection reset before the body is consumed).
        (
            ROUTE_IMAGES_EDITS,
            post(passthrough::images_edits).layer(DefaultBodyLimit::disable()),
        ),
        (
            ROUTE_AUDIO_TRANSCRIPTIONS,
            post(passthrough::audio_transcriptions).layer(DefaultBodyLimit::disable()),
        ),
        (ROUTE_SIGNATURE, get(signature::signature)),
        // Internal — sibling proxies on the same host call this when
        // configured with GPU_EVIDENCE_DELEGATE_URL pointed at us.
        (ROUTE_INTERNAL_GPU_EVIDENCE, post(internal::gpu_evidence)),
        // OHTTP Gateway (RFC 9458) — POST /ohttp is unauthenticated; auth may be
        // inside the encrypted Binary HTTP message or on the outer HTTP request
        // (relay-injected Authorization). See `ohttp_relay` docs.
        (ROUTE_OHTTP_WELL_KNOWN, get(ohttp::ohttp_config)),
        (ROUTE_OHTTP_CONFIG, get(ohttp::ohttp_config)),
        (ROUTE_OHTTP_RELAY, post(ohttp::ohttp_relay)),
    ]
}
