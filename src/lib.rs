use std::sync::Arc;

pub mod agent_loop;
pub mod attestation;
pub mod attestation_sdk;
pub mod auth;
pub mod backend_pool;
pub mod cache;
pub mod config;
pub mod encryption;
pub mod error;
pub mod fusion;
pub mod gpu_evidence_delegate;
pub mod image_validation;
pub mod metrics_middleware;
pub mod ohttp_gateway;
pub mod proxy;
pub mod rate_limit;
pub mod request_tracing;
pub mod routes;
pub mod signing;
pub mod startup_checks;
pub mod types;
pub mod vllm_dp_affinity;

pub use request_tracing::{request_id_middleware, TracingIds};

/// Shared application state available to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<config::Config>,
    pub signing: Arc<signing::SigningPair>,
    pub cache: Arc<cache::ChatCache>,
    pub attestation_cache: Arc<attestation::AttestationCache>,
    pub http_client: reqwest::Client,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    /// Live SHA-256 hash of the TLS certificate's SPKI (Subject Public Key
    /// Info). The tracker re-stats the cert on every attestation-cache
    /// refresh tick and re-hashes when `mtime` advances, so a certbot-driven
    /// rotation in the co-located ingress sidecar is picked up automatically
    /// without restarting the proxy. Routes call `.current()` on the hot path.
    pub tls_cert_fingerprint: Arc<attestation::TlsCertTracker>,
    pub backend_pool: Arc<backend_pool::BackendPool>,
    pub ohttp_gateway: Option<Arc<ohttp_gateway::OhttpGateway>>,
    pub ohttp_attestation_ed25519: Option<types::OhttpAttestation>,
    pub fusion_caches: Arc<fusion::FusionCaches>,
    pub vllm_dp_affinity: Arc<vllm_dp_affinity::VllmDpAffinity>,
}
