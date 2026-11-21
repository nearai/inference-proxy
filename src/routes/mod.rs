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

/// What a POST route does with the client body. Every POST route is declared
/// in [`POST_ROUTES`] with one of these, so a new route has to say how its
/// body reaches an engine before it can be registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientBody {
    /// A JSON body forwarded to an engine or backend. It goes through
    /// `forward_body::prepare` (the chat route runs its two halves
    /// separately): the request field rules, and media inlining when
    /// `VLLM_PROXY_INLINE_MEDIA` is on, so no media URL reaches the engine.
    Forwarded,
    /// Not run through `forward_body::prepare`; the text says why.
    OptOut(&'static str),
}

/// One POST route: its path, what happens to its body, whether gateway list
/// mode serves it, and its handler.
pub struct PostRoute {
    pub path: &'static str,
    pub body: ClientBody,
    pub in_model_list: bool,
    handler: fn() -> MethodRouter<AppState>,
}

/// Every POST route of both routers. `build_router` and
/// `build_model_list_router` register POST routes from this table only.
pub const POST_ROUTES: &[PostRoute] = &[
    PostRoute {
        path: ROUTE_CHAT_COMPLETIONS,
        body: ClientBody::Forwarded,
        in_model_list: true,
        handler: || post(chat::chat_completions),
    },
    PostRoute {
        path: ROUTE_COMPLETIONS,
        body: ClientBody::Forwarded,
        in_model_list: true,
        handler: || post(completions::completions),
    },
    PostRoute {
        path: ROUTE_TOKENIZE,
        body: ClientBody::Forwarded,
        in_model_list: false,
        handler: || post(passthrough::tokenize),
    },
    PostRoute {
        path: ROUTE_EMBEDDINGS,
        body: ClientBody::Forwarded,
        in_model_list: false,
        handler: || post(passthrough::embeddings),
    },
    PostRoute {
        path: ROUTE_RERANK,
        body: ClientBody::Forwarded,
        in_model_list: false,
        handler: || post(passthrough::rerank),
    },
    PostRoute {
        path: ROUTE_SCORE,
        body: ClientBody::Forwarded,
        in_model_list: false,
        handler: || post(passthrough::score),
    },
    PostRoute {
        path: ROUTE_PRIVACY_CLASSIFY,
        body: ClientBody::Forwarded,
        in_model_list: false,
        handler: || post(privacy::classify),
    },
    PostRoute {
        path: ROUTE_IMAGES_GENERATIONS,
        body: ClientBody::Forwarded,
        in_model_list: false,
        handler: || post(passthrough::images_generations),
    },
    // Multipart upload routes: axum's `Multipart` extractor enforces the
    // global 2 MiB `DefaultBodyLimit`, which rejects the body before the
    // handler's own size guard (`max_audio_request_size` / `max_image_request_size`)
    // can run. Disabling the default limit here makes those per-type limits —
    // enforced incrementally in `read_field_chunks` — the single source of
    // truth. Without this, audio/images > ~2 MiB fail upstream
    // with a 502 (connection reset before the body is consumed).
    PostRoute {
        path: ROUTE_IMAGES_EDITS,
        body: ClientBody::OptOut(
            "multipart form: only the field names in IMAGE_EDIT_FORM_FIELDS are forwarded \
             (none takes a URL; media arrives as uploaded bytes), each field passes \
             `RequestGuardConfig::keep_form_field` and the image bounds",
        ),
        in_model_list: false,
        handler: || post(passthrough::images_edits).layer(DefaultBodyLimit::disable()),
    },
    PostRoute {
        path: ROUTE_AUDIO_TRANSCRIPTIONS,
        body: ClientBody::OptOut(
            "multipart form: only the field names in AUDIO_TRANSCRIPTION_FORM_FIELDS are \
             forwarded (none takes a URL; audio arrives as uploaded bytes), each field passes \
             `RequestGuardConfig::keep_form_field`",
        ),
        in_model_list: false,
        handler: || post(passthrough::audio_transcriptions).layer(DefaultBodyLimit::disable()),
    },
    // Internal — sibling proxies on the same host call this when
    // configured with GPU_EVIDENCE_DELEGATE_URL pointed at us.
    PostRoute {
        path: ROUTE_INTERNAL_GPU_EVIDENCE,
        body: ClientBody::OptOut(
            "not forwarded to an engine: the typed body (a nonce) only drives local GPU \
             evidence collection",
        ),
        in_model_list: false,
        handler: || post(internal::gpu_evidence),
    },
    // OHTTP Gateway (RFC 9458) — POST /ohttp is unauthenticated; auth may be
    // inside the encrypted Binary HTTP message or on the outer HTTP request
    // (relay-injected Authorization). See `ohttp_relay` docs.
    PostRoute {
        path: ROUTE_OHTTP_RELAY,
        body: ClientBody::OptOut(
            "the decapsulated inner request is sent back through this router \
             (`validated_loopback_url`: chat, completions, transcriptions), so it gets that \
             route's checks",
        ),
        in_model_list: false,
        handler: || post(ohttp::ohttp_relay),
    },
];

fn with_post_routes(mut router: Router<AppState>, model_list: bool) -> Router<AppState> {
    for route in POST_ROUTES {
        if !model_list || route.in_model_list {
            router = router.route(route.path, (route.handler)());
        }
    }
    router
}

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
/// With OHTTP enabled, the process-wide `POST /ohttp` in-flight cap
/// (`OHTTP_MAX_INFLIGHT`) is installed here for the relay handler to read.
pub fn build_app(state: AppState) -> Router {
    let rate_limit_state = crate::rate_limit::RateLimitState {
        limiter: crate::rate_limit::build_rate_limiter(
            state.config.rate_limit_per_second,
            state.config.rate_limit_burst_size,
        ),
        trusted_proxies: state.config.rate_limit_trusted_proxies.clone().into(),
    };
    let mut router = build_router_for(&state);
    if state.ohttp_gateway.is_some() {
        router = router.layer(axum::Extension(ohttp::BufferBudget::new(
            state.config.ohttp_buffer_budget,
        )));
        if let Some(limit) = ohttp::OhttpRelayLimit::new(state.config.ohttp_max_inflight) {
            router = router.layer(axum::Extension(limit));
        }
    }
    router
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

/// Gateway list mode (`model_list.rs`): only what is defined for several
/// models — chat completions, completions, the models document, health,
/// metrics and the version. The other inference routes (tokenize, embeddings,
/// rerank, score, images, audio, privacy), the engine metrics passthrough,
/// attestation, signatures and OHTTP all assume the one model, backend pool
/// and signing identity of a single-model process, so here they are the same
/// 404 as any undeclared route.
pub fn build_model_list_router() -> Router<AppState> {
    let router = Router::new()
        .route(ROUTE_VERSION, get(health::version))
        .route(ROUTE_HEALTHZ, get(health::healthz))
        .route(
            ROUTE_METRICS,
            get(crate::metrics_middleware::prometheus_metrics_handler),
        )
        .route(ROUTE_V1_MODELS, get(metrics::models));
    with_post_routes(router, true).fallback(unknown_route)
}

pub fn build_router() -> Router<AppState> {
    let router = Router::new()
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
        .route(ROUTE_SIGNATURE, get(signature::signature))
        .route(ROUTE_OHTTP_WELL_KNOWN, get(ohttp::ohttp_config))
        .route(ROUTE_OHTTP_CONFIG, get(ohttp::ohttp_config));
    // Authenticated endpoints, the internal route and the OHTTP relay.
    with_post_routes(router, false).fallback(unknown_route)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// POST routes are registered from `POST_ROUTES` only, so each one has
    /// declared what happens to its body. A `.route(..., post(...))` added
    /// anywhere else in this file fails here.
    #[test]
    fn post_routes_come_from_the_table_only() {
        let source = include_str!("mod.rs");
        let source = &source[..source.find("#[cfg(test)]").unwrap()];
        let start = source.find("pub const POST_ROUTES").unwrap();
        let end = start + source[start..].find("\n];\n").unwrap();
        let outside = format!("{}{}", &source[..start], &source[end..]);
        for other_way in [
            "post(",
            "any(",
            "routing::on",
            "MethodFilter",
            "post_service(",
            ".merge(",
            ".nest(",
        ] {
            assert!(
                !outside.contains(other_way),
                "register POST routes through POST_ROUTES, not `{other_way}`"
            );
        }
        // The only fallback answers 404.
        assert_eq!(
            outside.matches(".fallback(").count(),
            outside.matches(".fallback(unknown_route)").count()
        );
    }

    #[test]
    fn every_post_route_is_declared_once_with_a_reason_to_opt_out() {
        let mut paths: Vec<&str> = POST_ROUTES.iter().map(|r| r.path).collect();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), POST_ROUTES.len());
        for route in POST_ROUTES {
            if let ClientBody::OptOut(reason) = route.body {
                assert!(reason.len() > 20, "{}", route.path);
            }
        }
        // Gateway list mode serves chat and completions only.
        let list: Vec<&str> = POST_ROUTES
            .iter()
            .filter(|r| r.in_model_list)
            .map(|r| r.path)
            .collect();
        assert_eq!(list, [ROUTE_CHAT_COMPLETIONS, ROUTE_COMPLETIONS]);
    }
}
