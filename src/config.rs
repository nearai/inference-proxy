use std::env;
use tracing::warn;

const FUSION_INTERNAL_MAX_ATTEMPTS_LIMIT: usize = 5;
const DEFAULT_GEMMA4_ALLOWED_MEDIA_DOMAIN: &str = "prod-files-secure.s3.us-west-2.amazonaws.com";

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_int(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parse an optional numeric variable, failing loudly on garbage (unlike
/// `env_int`, which silently falls back to the default).
fn env_parse<T>(name: &str, default: T) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(raw) if !raw.trim().is_empty() => raw
            .trim()
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("{name}: {e}")),
        _ => Ok(default),
    }
}

/// Comma-separated backend/probe base URLs, trimmed of blanks and trailing
/// slashes. Missing or empty = no URLs.
fn url_list(name: &str) -> Vec<String> {
    env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(|u| u.trim().trim_end_matches('/').to_string())
        .filter(|u| !u.is_empty())
        .collect()
}

fn parse_bool(v: &str) -> bool {
    matches!(v.to_lowercase().as_str(), "1" | "true" | "yes")
}

fn env_bool(name: &str) -> bool {
    env::var(name).map(|v| parse_bool(&v)).unwrap_or(false)
}

fn env_bool_optional(name: &str) -> Option<bool> {
    env::var(name).ok().map(|v| parse_bool(&v))
}

fn is_gemma4_model_name(model_name: &str) -> bool {
    let name = model_name.to_ascii_lowercase();
    ["gemma-4", "gemma4"].iter().any(|needle| {
        name.match_indices(needle).any(|(idx, _)| {
            name[idx + needle.len()..]
                .chars()
                .next()
                .map(|c| !c.is_ascii_alphanumeric())
                .unwrap_or(true)
        })
    })
}

fn parse_allowed_media_domains(value: &str) -> Vec<String> {
    value
        .split(',')
        .filter_map(normalize_allowed_media_domain)
        .collect()
}

fn normalize_allowed_media_domain(raw: &str) -> Option<String> {
    let trimmed = raw.trim().trim_end_matches('.');
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(url) = reqwest::Url::parse(trimmed) {
        return url.host_str().map(normalize_host);
    }
    Some(normalize_host(trimmed))
}

fn normalize_host(host: &str) -> String {
    host.trim().trim_end_matches('.').to_ascii_lowercase()
}

#[derive(Debug, Clone)]
pub struct Config {
    pub model_name: String,
    /// Accepted admin tokens. Parsed from `TOKEN` (comma-separated) so multiple
    /// tokens can be active at once during a rotation.
    pub tokens: Vec<String>,

    // Backend URLs
    pub vllm_base_url: String,
    pub chat_completions_url: String,
    pub completions_url: String,
    pub tokenize_url: String,
    pub metrics_url: String,
    pub models_url: String,
    pub images_url: String,
    pub images_edits_url: String,
    pub transcriptions_url: String,
    pub embeddings_url: String,
    pub rerank_url: String,
    pub score_url: String,

    // Connection pool
    pub max_keepalive: usize,
    /// How long to keep an idle HTTP connection in the pool before closing it.
    /// Must be shorter than the upstream's keepalive_timeout to avoid
    /// reusing connections the server has already closed (which surfaces as
    /// `error sending request for url ...` transport errors). 0 disables
    /// pooling entirely.
    pub pool_idle_timeout_secs: u64,

    // Request size limits
    pub max_request_size: usize,
    pub max_image_request_size: usize,
    pub max_audio_request_size: usize,

    // Pre-dispatch image validation (reject unfetchable/non-image inputs before
    // they reach the engine — nearai/infra#159, #172). Env vars:
    //   VLLM_PROXY_IMAGE_VALIDATION_DISABLED=1          disable (default: on)
    //   VLLM_PROXY_IMAGE_VALIDATION_TIMEOUT_SECS=5      per-fetch timeout
    //   VLLM_PROXY_IMAGE_VALIDATION_MAX_BYTES=8192      head bytes read to sniff
    //   VLLM_PROXY_IMAGE_VALIDATION_MAX_CONCURRENCY=8   global concurrent fetches
    //   VLLM_PROXY_IMAGE_VALIDATION_ALLOW_PRIVATE_HOSTS=1  permit private/loopback
    //       image hosts (tests / trusted internal deployments; default: off)
    //   VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS=example.com,cdn.example.com
    //       exact remote image_url host allowlist. If unset, falls back to
    //       VLLM_ALLOWED_MEDIA_DOMAINS when present so deployments can mirror
    //       vLLM's --allowed-media-domains policy. Gemma-4 defaults to the
    //       current vLLM default domain because its backend already enforces it.
    //   VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB=0|1  force broad non-RGB
    //       rejection. By default Gemma-4 model names reject only observed
    //       one-channel crash inputs; broader RGBA/CMYK/palette rejection is
    //       opt-in until real-engine verification covers those classes.
    // NOTE: the validation fetcher bypasses system proxies (no_proxy) and uses
    // rustls/webpki-roots. Deployments whose outbound HTTP requires an egress
    // proxy (HTTPS_PROXY) or a custom CA that the engine trusts but rustls does
    // not should set _DISABLED=1 — otherwise remote-image requests 400 on a
    // connect/TLS error while the engine itself fetches fine.
    pub image_validation_enabled: bool,
    pub image_validation_timeout_secs: u64,
    pub image_validation_max_bytes: usize,
    pub image_validation_max_concurrency: usize,
    pub image_validation_allow_private_hosts: bool,
    pub image_validation_allowed_domains: Vec<String>,
    pub image_validation_reject_non_rgb_images: bool,
    pub image_validation_reject_single_channel_images: bool,

    // Cache
    pub chat_cache_expiration_secs: u64,
    /// TTL for cached nonce-less attestation reports (seconds).
    pub attestation_cache_ttl_secs: u64,

    // TLS certificate binding
    pub tls_cert_path: Option<String>,

    // Modes
    pub dev_mode: bool,
    pub gpu_no_hw_mode: bool,

    // Version
    pub git_rev: String,

    // Rate limiting
    pub rate_limit_per_second: u64,
    pub rate_limit_burst_size: u32,
    /// Trust X-Forwarded-For / X-Real-IP headers for rate-limit IP extraction.
    /// Set to false when the proxy is directly internet-facing (no trusted
    /// reverse proxy) to prevent IP spoofing that bypasses rate limits.
    pub rate_limit_trust_proxy_headers: bool,

    // Timeouts
    pub timeout_secs: u64,
    /// Maximum idle time between upstream SSE chunks after the first
    /// generation-progress event. Zero disables the watchdog. This deliberately
    /// does not bound queueing, prefill, or metadata-only role events;
    /// `timeout_secs` remains the total reqwest request bound.
    pub stream_idle_timeout_secs: u64,
    pub timeout_tokenize_secs: u64,

    // Cloud API for sk- key validation
    pub cloud_api_url: Option<String>,
    /// Maximum attempts (initial + retries) for `POST /v1/check_api_key`.
    /// 1 disables retry. Retries are issued only on transport errors and 5xx.
    pub cloud_api_auth_max_attempts: usize,
    /// Initial backoff between auth retries; doubles each attempt with full jitter.
    pub cloud_api_auth_initial_backoff_ms: u64,
    /// Per-attempt timeout for `POST /v1/check_api_key`.
    pub cloud_api_auth_timeout_secs: u64,
    /// Shared service-token presented to cloud-api on the `/v1/internal/usage`
    /// path. Required for usage reporting: when set AND the auth response carried
    /// `organization_id + workspace_id + api_key_id`, the reporter posts to
    /// `/v1/internal/usage` with this token as `Bearer` and the subject identity
    /// in the body. When unset (or the auth response is missing identity fields),
    /// usage reporting is skipped — cloud-api removed the legacy `Bearer sk-…`
    /// `/v1/usage` endpoint, so there is no fallback.
    pub cloud_api_usage_token: Option<String>,

    // Compose-manager attestation (deployment actions attestation)
    pub compose_manager_url: Option<String>,

    // GPU evidence delegation (host-level NVML serialization)
    /// HTTP base URL of another inference-proxy on the same host that
    /// owns NVML evidence collection (e.g. `http://vllm-proxy-leader:8000`).
    /// When set, this proxy forwards GPU evidence requests to the
    /// delegate's `POST /internal/gpu_evidence` endpoint instead of
    /// calling NVML locally. The intent is to serialize NVML access
    /// across the *host*, not just within one process — multiple
    /// inference-proxy instances sharing the same physical GPUs were
    /// observed to race at the firmware level (see #107). When unset,
    /// the proxy collects evidence locally via the SDK or Python path.
    pub gpu_evidence_delegate_url: Option<String>,
    /// Per-attempt timeout for the delegate HTTP call. Default 30s —
    /// the delegate's own evidence collection plus its NVML wait
    /// dominates this; we want enough headroom to not surface as
    /// timeouts under contended load.
    pub gpu_evidence_delegate_timeout_secs: u64,

    // OpenAI Chat Compatibility Checks
    // Validates that hosted models (qwen, glm, etc.) send OpenAI-compliant responses:
    // - /v1/models API format
    // - /v1/chat/completions with tool_calls (streaming & non-streaming)
    // Only enable for models serving OpenAI-compatible chat API. Disable for:
    // - Image generation models (FLUX, etc.)
    // - Embedding models
    // - Reranker models
    // - Cohere or other non-OpenAI-compliant APIs
    pub openai_chat_compatibility_check_enabled: bool,
    pub startup_check_retries: usize,
    pub startup_check_retry_delay_secs: u64,
    pub startup_check_timeout_secs: u64,

    // Multi-backend support
    /// All backend base URLs (derived from VLLM_BACKEND_URLS or VLLM_BASE_URL).
    pub backend_urls: Vec<String>,
    /// Number of independent vLLM data-parallel engines behind this proxy.
    /// When set, append-only chat conversations are pinned to one rank so
    /// their turns reuse that engine's local prefix cache.
    pub vllm_data_parallel_size: Option<usize>,
    /// Pin append-only chat conversations to one backend of `backend_urls`
    /// so their turns reuse that engine's prefix cache instead of being
    /// re-prefilled on whichever backend has the fewest connections. Bounded
    /// by `backend_affinity_max_imbalance`; no effect with a single backend.
    pub backend_conversation_affinity: bool,
    /// Max extra in-flight requests the pinned backend may carry over the
    /// least-loaded healthy backend before a turn is rebalanced (and the
    /// conversation re-pinned) to the least-loaded backend.
    pub backend_affinity_max_imbalance: u32,
    /// Health check interval in seconds (only used when multiple backends).
    pub health_check_interval_secs: u64,
    /// Consecutive failures before marking a backend unhealthy.
    pub health_check_max_failures: u32,
    /// Health check timeout in seconds.
    pub health_check_timeout_secs: u64,

    // OHTTP Gateway (RFC 9458)
    pub ohttp_enabled: bool,
    /// Listen port for the proxy (used by OHTTP handler for loopback requests).
    pub listen_port: u16,
    /// Interface to bind (`LISTEN_ADDR`, default `0.0.0.0`). Gateway
    /// deployments behind a local TLS terminator bind `127.0.0.1`.
    pub listen_addr: String,
    /// Bearer token attached to every request sent to the inference backends
    /// (`VLLM_BACKEND_TOKEN`). Used when the backends are themselves
    /// inference-proxies (gateway mode): they accept it as a trusted config
    /// token, so they neither re-validate the customer key nor double-report
    /// usage. Never sent to cloud-api or any other service.
    pub backend_token: Option<String>,
    /// Engine priority for this proxy's requests, sent as `X-NearAI-Priority`
    /// on every backend request (`VLLM_BACKEND_PRIORITY`, gateway mode). The
    /// CVM proxy honors it because the gateway authenticates with the trusted
    /// token; everything else gets 0. See `priority.rs`.
    pub backend_priority: Option<i64>,
    /// Path probed on each backend by the pool health checker and by
    /// `/healthz` (`VLLM_BACKEND_HEALTH_PATH`, default `/health`, the engine's
    /// lightweight route). Gateway mode points it at the inference-proxy's
    /// unauthenticated `/healthz`.
    pub backend_health_path: String,
    /// This proxy does not run inside a TEE (`NON_TEE_DEPLOYMENT=1`): no
    /// dstack guest agent, no hardware evidence, dev signing keys. Effects:
    /// `/healthz` skips the dstack probe, the attestation cache refresh is not
    /// started, and `/v1/attestation/report`, `/v1/signature/{id}` and
    /// `/internal/gpu_evidence` answer 404 so nothing unverifiable is
    /// advertised. Inference routes are unaffected.
    pub non_tee_deployment: bool,
    /// Rewrite the engine's queue-full rejection (HTTP 503 / SSE error event
    /// `"The request queue is full."`) to 429 (`VLLM_PROXY_MAP_QUEUE_FULL_TO_429`).
    /// Aggregators treat 429 as back-pressure and 5xx as an outage; off by
    /// default because cloud-api's peer fallback keys on the 503.
    pub map_queue_full_to_429: bool,
    /// For streaming requests, wait up to this many milliseconds for the
    /// first upstream SSE chunk before committing a 200 to the client
    /// (`VLLM_PROXY_STREAM_ERROR_PEEK_MS`, 0 = off). An engine that rejects
    /// at admission (queue full, aborted) emits `data: {"error": …}` as its
    /// first event on an HTTP 200 stream; peeking turns that into a real
    /// error status instead of a 200 that fails mid-stream.
    pub stream_error_peek_ms: u64,
    /// Commit `200 text/event-stream` to the client after this many
    /// milliseconds even when the upstream has not answered yet, so the
    /// keep-alive comments can start during a long prefill (an engine sends
    /// its response headers only with its first event). Zero disables it and
    /// the status always comes from the upstream. A failure that arrives after
    /// the commit is delivered as a terminal SSE `error` event instead of a
    /// status code, so set this above the slowest error a deployment produces.
    pub stream_commit_ms: u64,
    /// Chat content part `type`s refused with 400 before dispatch
    /// (`VLLM_PROXY_REJECTED_CONTENT_PART_TYPES`, e.g. `video_url,input_audio,file`).
    pub rejected_content_part_types: Vec<String>,
    /// Gateway mode: serve `/v1/models` from this URL (cloud-api's
    /// `/v1/models`) reduced to `MODEL_NAME` and completed with the lane's
    /// declared capacity, instead of passing the engine's list through
    /// (`VLLM_PROXY_MODELS_DOCUMENT_URL`). Unset = engine passthrough.
    pub models_document_url: Option<String>,
    /// Requests per minute declared in the models document's `capacity`
    /// (`VLLM_PROXY_CAPACITY_REQUESTS_PER_MINUTE`, 0 = not declared). The
    /// concurrency entry comes from `VLLM_PROXY_ADMISSION_MAX_INFLIGHT`.
    pub capacity_requests_per_minute: u64,
    /// Gateway mode: the `reasoning_effort` that stands for "as little
    /// reasoning as possible" on the served model
    /// (`VLLM_PROXY_REASONING_OFF_EFFORT`, default `none`). Applied to an
    /// aggregator's `reasoning.enabled: false`, to an effort of `none` or
    /// `minimal`, and to those values sent as `reasoning_effort` directly.
    /// GLM-5.3 Flash needs `low`: its template only knows `low` and `high`,
    /// and switched off outright it writes its reasoning as visible content.
    pub reasoning_off_effort: String,
    /// Organizations whose cloud-api keys may use this deployment
    /// (`VLLM_PROXY_ALLOWED_ORG_IDS`, comma-separated organization ids). Empty
    /// = every valid key. Config-token callers are not affected. Gateway mode
    /// uses it to keep a partner lane to that partner.
    pub allowed_org_ids: Vec<String>,
    /// Emit an SSE comment (`: keep-alive`) on client streams whenever the
    /// upstream has been silent for this many seconds
    /// (`VLLM_PROXY_SSE_KEEPALIVE_SECS`, 0 = off). Comments are not hashed into
    /// the response signature, so leave this off where clients verify
    /// signatures over the raw stream bytes.
    pub sse_keepalive_secs: u64,
    /// Lane admission (gateway mode, see `admission.rs`): hard ceiling on
    /// chat/completions requests in flight across the fleet
    /// (`VLLM_PROXY_ADMISSION_MAX_INFLIGHT`, 0 = off, the default).
    pub admission_max_inflight: u32,
    /// Budget at start-up (`VLLM_PROXY_ADMISSION_START_INFLIGHT`, default =
    /// the maximum, i.e. no ramp).
    pub admission_start_inflight: u32,
    /// Budget increase per clean ramp interval
    /// (`VLLM_PROXY_ADMISSION_RAMP_STEP`, default 8).
    pub admission_ramp_step: u32,
    /// Ramp interval (`VLLM_PROXY_ADMISSION_RAMP_INTERVAL_SECS`, default 1800).
    pub admission_ramp_interval_secs: u64,
    /// Refuse new work while the lane's time-to-first-chunk p95 over the last
    /// minute is above this (`VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS`, default
    /// 30000, 0 = no TTFT check).
    pub admission_ttft_p95_max_ms: u64,
    /// How long an engine admission rejection counts against its backend
    /// (`VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS`, default 10).
    pub admission_backpressure_secs: u64,
    /// `Retry-After` on refusals (`VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS`,
    /// default 2).
    pub admission_retry_after_secs: u64,
    /// Retry a chat/completions request once on another healthy backend when
    /// the connection to the chosen one fails before anything was sent
    /// (`VLLM_BACKEND_CONNECT_FAILOVER`). HTTP errors, queue-full included,
    /// are never retried.
    pub backend_connect_failover: bool,
    /// Gateway mode: one plain-HTTP probe base URL per backend (same order as
    /// `VLLM_BACKEND_URLS`) whose `/v1/metrics` is polled for the engine's
    /// running and queued request counts (`VLLM_BACKEND_PROBE_URLS`). Empty =
    /// no engine view; placement and admission use the gateway's own counts.
    pub backend_probe_urls: Vec<String>,
    /// Poll interval for the probes (`VLLM_BACKEND_PROBE_INTERVAL_SECS`,
    /// default 2).
    pub backend_probe_interval_secs: u64,
    /// Gateway mode: backend URLs of the long-context tier
    /// (`VLLM_BACKEND_LONG_CONTEXT_URLS`, the same hosts' handle URLs under
    /// the model's `-long` model-proxy domain). Appended to the pool after
    /// `backend_urls`, so the base backends keep their indexes.
    pub backend_long_context_urls: Vec<String>,
    /// One engine-load probe URL per long-context backend, same order
    /// (`VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS`). Required with
    /// `VLLM_BACKEND_PROBE_URLS`, empty without it.
    pub backend_long_context_probe_urls: Vec<String>,
    /// Estimated input tokens above which a request is placed on the
    /// long-context tier (`VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS`, 0 = off,
    /// the default). See `context_tier.rs` for the estimate.
    pub long_context_above_tokens: u64,

    // Endpoint URL overrides (Some = explicitly set, bypasses backend pool)
    pub images_url_override: Option<String>,
    pub images_edits_url_override: Option<String>,
    pub transcriptions_url_override: Option<String>,
    pub rerank_url_override: Option<String>,
    pub score_url_override: Option<String>,

    /// Path to the dstack guest agent unix socket. Probed by `/healthz` so
    /// upstream load balancers can detach this instance when the socket is
    /// unreachable (otherwise `/v1/attestation/report` silently 500s while
    /// `/v1/models` still passes). Default: `/var/run/dstack.sock`.
    pub dstack_socket_path: String,

    // Agent loop (server-side web_context_search tool)
    /// Brave LLM Context API endpoint. When unset, requests advertising the
    /// `{"type":"web_context_search"}` tool are rejected with 400. All
    /// tool execution happens inside the CVM; the query is the only thing
    /// that egresses, going directly to Brave under TLS.
    pub web_context_search_url: Option<String>,
    /// Brave subscription token for the LLM Context endpoint. Sent as
    /// `X-Subscription-Token`. Required when `web_context_search_url` is set.
    pub web_context_search_api_key: Option<String>,
    /// Hard cap on tool-call iterations within a single chat completion.
    /// Once hit, the loop emits a synthetic terminator chunk and stops.
    pub agent_loop_max_iterations: u32,
    /// Per-tool-call timeout for the Brave HTTP request.
    pub web_context_search_timeout_secs: u64,

    // Fusion (server-side multi-model deliberation)
    /// Feature flag for server-side Fusion orchestration. When false, Fusion
    /// tool entries pass through unchanged.
    pub fusion_enabled: bool,
    /// Endpoint discovery URL returning `{ "endpoints": [{ "domain": "...", "models": [...] }] }`.
    /// This URL is a Fusion trust anchor: returned domains receive the shared
    /// internal bearer token, so operators must keep it under trusted control.
    pub fusion_endpoints_url: String,
    /// TTL for the endpoint discovery cache.
    pub fusion_endpoints_ttl_secs: u64,
    /// Internal bearer token used for direct completions calls to every
    /// discovered panel and judge backend. V1 uses one shared secret; keep
    /// `FUSION_ENDPOINTS_URL` trusted and do not expose per-request overrides.
    pub fusion_internal_bearer_token: Option<String>,
    /// Default panel models when the tool configuration omits `analysis_models`.
    pub fusion_default_analysis_models: Vec<String>,
    /// Hard cap on panel size.
    pub fusion_max_panel_models: usize,
    /// Maximum accepted Fusion recursion depth.
    pub fusion_max_depth: u32,
    /// Per-request timeout for Fusion panel, judge, and synthesis chat calls.
    pub fusion_panel_timeout_secs: u64,
    /// Maximum bytes buffered from Fusion endpoint discovery and model responses.
    pub fusion_max_response_bytes: usize,
    /// Total attempts for transient Fusion direct model HTTP calls. 1 disables
    /// retry; retries are only for connect errors, timeouts, and 5xx.
    pub fusion_internal_max_attempts: usize,
    /// Initial backoff before retrying Fusion direct model calls. Backoff
    /// doubles per attempt.
    pub fusion_internal_retry_initial_backoff_ms: u64,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let model_name = env::var("MODEL_NAME")
            .map_err(|_| anyhow::anyhow!("MODEL_NAME environment variable is required"))?;
        let raw_tokens = env::var("TOKEN")
            .map_err(|_| anyhow::anyhow!("TOKEN environment variable is required"))?;
        let tokens: Vec<String> = raw_tokens
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        if tokens.is_empty() {
            anyhow::bail!("TOKEN must contain at least one non-empty token");
        }

        let vllm_base_url = env_or("VLLM_BASE_URL", "http://localhost:8000");
        let base = vllm_base_url.trim_end_matches('/');

        // Multi-backend: VLLM_BACKEND_URLS takes precedence over VLLM_BASE_URL
        let backend_urls: Vec<String> = env::var("VLLM_BACKEND_URLS")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.split(',')
                    .map(|u| u.trim().trim_end_matches('/').to_string())
                    .filter(|u| !u.is_empty())
                    .collect()
            })
            .unwrap_or_else(|| vec![vllm_base_url.clone()]);
        if backend_urls.is_empty() {
            anyhow::bail!("VLLM_BACKEND_URLS is set but contains no valid URLs");
        }

        let vllm_data_parallel_size = match env::var("VLLM_DATA_PARALLEL_SIZE") {
            Ok(raw) => {
                let value = raw.trim().parse::<usize>().map_err(|_| {
                    anyhow::anyhow!("VLLM_DATA_PARALLEL_SIZE must be a positive integer")
                })?;
                if value == 0 {
                    anyhow::bail!("VLLM_DATA_PARALLEL_SIZE must be a positive integer");
                }
                Some(value)
            }
            Err(env::VarError::NotPresent) => None,
            Err(env::VarError::NotUnicode(_)) => {
                anyhow::bail!("VLLM_DATA_PARALLEL_SIZE must be valid UTF-8")
            }
        };
        if vllm_data_parallel_size.is_some() && backend_urls.len() != 1 {
            anyhow::bail!(
                "VLLM_DATA_PARALLEL_SIZE requires exactly one vLLM backend; multiple VLLM_BACKEND_URLS have independent prefix caches"
            );
        }

        let backend_conversation_affinity = env_bool("VLLM_BACKEND_CONVERSATION_AFFINITY");
        if backend_conversation_affinity && vllm_data_parallel_size.is_some() {
            anyhow::bail!(
                "VLLM_BACKEND_CONVERSATION_AFFINITY and VLLM_DATA_PARALLEL_SIZE are mutually exclusive; data-parallel affinity already pins conversations inside the single backend"
            );
        }
        let backend_affinity_max_imbalance =
            u32::try_from(env_int("VLLM_BACKEND_AFFINITY_MAX_IMBALANCE", 8)).map_err(|_| {
                anyhow::anyhow!("VLLM_BACKEND_AFFINITY_MAX_IMBALANCE exceeds the u32 range")
            })?;

        // Track which endpoint URLs are explicitly overridden (should bypass pool)
        let images_url_override = env::var("VLLM_IMAGES_URL").ok().filter(|s| !s.is_empty());
        let images_edits_url_override = env::var("VLLM_IMAGES_EDITS_URL")
            .ok()
            .filter(|s| !s.is_empty());
        let transcriptions_url_override = env::var("VLLM_TRANSCRIPTIONS_URL")
            .ok()
            .filter(|s| !s.is_empty());
        let rerank_url_override = env::var("VLLM_RERANK_URL").ok().filter(|s| !s.is_empty());
        let score_url_override = env::var("VLLM_SCORE_URL").ok().filter(|s| !s.is_empty());

        let images_url = images_url_override
            .clone()
            .unwrap_or_else(|| format!("{base}/v1/images/generations"));
        let images_edits_url = images_edits_url_override
            .clone()
            .unwrap_or_else(|| format!("{base}/v1/images/edits"));
        let transcriptions_url = transcriptions_url_override
            .clone()
            .unwrap_or_else(|| format!("{base}/v1/audio/transcriptions"));
        let rerank_url = rerank_url_override
            .clone()
            .unwrap_or_else(|| format!("{base}/v1/rerank"));
        let score_url = score_url_override
            .clone()
            .unwrap_or_else(|| format!("{base}/v1/score"));

        let listen_port: u16 = env::var("LISTEN_PORT")
            .unwrap_or_else(|_| "8000".to_string())
            .parse()
            .map_err(|_| anyhow::anyhow!("LISTEN_PORT must be a valid port number"))?;
        let listen_addr = env_or("LISTEN_ADDR", "0.0.0.0");
        let listen_ip: std::net::IpAddr = listen_addr
            .parse()
            .map_err(|_| anyhow::anyhow!("LISTEN_ADDR must be an IP address"))?;
        // The OHTTP gateway re-dispatches decoded requests to 127.0.0.1 on
        // the listen port, so it needs a bind that loopback can reach.
        if env_bool("OHTTP_ENABLED") && !(listen_ip.is_unspecified() || listen_ip.is_loopback()) {
            anyhow::bail!(
                "OHTTP_ENABLED requires LISTEN_ADDR to be unspecified (0.0.0.0/::) or loopback"
            );
        }
        let backend_token = env::var("VLLM_BACKEND_TOKEN")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        // A backend bearer means the backends treat this proxy as trusted and
        // do not bill its requests, so this proxy must be able to: fail closed
        // at startup instead of serving unbilled inference.
        if backend_token.is_some() {
            let set = |name: &str| env::var(name).is_ok_and(|v| !v.trim().is_empty());
            if !set("CLOUD_API_URL") || !set("CLOUD_API_USAGE_TOKEN") {
                anyhow::bail!(
                    "VLLM_BACKEND_TOKEN requires CLOUD_API_URL and CLOUD_API_USAGE_TOKEN: backends do not bill trusted-token requests"
                );
            }
        }
        let backend_priority = match env::var("VLLM_BACKEND_PRIORITY")
            .ok()
            .filter(|s| !s.trim().is_empty())
        {
            Some(raw) => Some(
                crate::priority::validate_priority(&raw)
                    .map_err(|e| anyhow::anyhow!("VLLM_BACKEND_PRIORITY: {e}"))?,
            ),
            None => None,
        };
        let backend_health_path = env_or("VLLM_BACKEND_HEALTH_PATH", "/health");
        if !backend_health_path.starts_with('/') {
            anyhow::bail!("VLLM_BACKEND_HEALTH_PATH must start with '/'");
        }
        let allowed_org_ids: Vec<String> = env::var("VLLM_PROXY_ALLOWED_ORG_IDS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let models_document_url = env::var("VLLM_PROXY_MODELS_DOCUMENT_URL")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let capacity_requests_per_minute: u64 =
            env_parse("VLLM_PROXY_CAPACITY_REQUESTS_PER_MINUTE", 0)?;
        let reasoning_off_effort = env_or("VLLM_PROXY_REASONING_OFF_EFFORT", "none")
            .trim()
            .to_string();
        let rejected_content_part_types = crate::content_policy::parse_rejected_types(
            &env::var("VLLM_PROXY_REJECTED_CONTENT_PART_TYPES").unwrap_or_default(),
        );

        let git_rev = std::fs::read_to_string("/etc/.GIT_REV")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        // Handle TLS certificate path with logging
        let tls_cert_path = env::var("TLS_CERT_PATH")
            .ok()
            .filter(|s| !s.is_empty())
            .and_then(|path| {
                if std::path::Path::new(&path).exists() {
                    Some(path)
                } else {
                    warn!(tls_cert_path = %path, "TLS_CERT_PATH is set but file does not exist");
                    None
                }
            });

        let cloud_api_url = env::var("CLOUD_API_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_end_matches('/').to_string());

        let compose_manager_url = env::var("COMPOSE_MANAGER_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_end_matches('/').to_string());

        let image_validation_reject_non_rgb_override =
            env_bool_optional("VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB");
        let image_validation_reject_single_channel_images =
            image_validation_reject_non_rgb_override
                .unwrap_or_else(|| is_gemma4_model_name(&model_name));
        let image_validation_reject_non_rgb_images =
            image_validation_reject_non_rgb_override.unwrap_or(false);
        let image_validation_allowed_domains =
            env::var("VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS")
                .ok()
                .map(|s| parse_allowed_media_domains(&s))
                .or_else(|| {
                    env::var("VLLM_ALLOWED_MEDIA_DOMAINS")
                        .ok()
                        .map(|s| parse_allowed_media_domains(&s))
                })
                .unwrap_or_else(|| {
                    if is_gemma4_model_name(&model_name) {
                        vec![DEFAULT_GEMMA4_ALLOWED_MEDIA_DOMAIN.to_string()]
                    } else {
                        Vec::new()
                    }
                });

        let fusion_default_analysis_models = env::var("FUSION_DEFAULT_ANALYSIS_MODELS")
            .ok()
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().trim_start_matches('~').to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let admission_max_inflight: u32 = env_parse("VLLM_PROXY_ADMISSION_MAX_INFLIGHT", 0)?;
        let admission_start_inflight: u32 = env_parse(
            "VLLM_PROXY_ADMISSION_START_INFLIGHT",
            admission_max_inflight,
        )?;
        let admission_ramp_step: u32 = env_parse("VLLM_PROXY_ADMISSION_RAMP_STEP", 8)?;
        let admission_ramp_interval_secs: u64 =
            env_parse("VLLM_PROXY_ADMISSION_RAMP_INTERVAL_SECS", 1800)?;
        let admission_ttft_p95_max_ms: u64 =
            env_parse("VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS", 30_000)?;
        let admission_backpressure_secs: u64 =
            env_parse("VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS", 10)?;
        let admission_retry_after_secs: u64 =
            env_parse("VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS", 2)?;
        if admission_max_inflight > 0 {
            if admission_start_inflight == 0 || admission_start_inflight > admission_max_inflight {
                anyhow::bail!(
                    "VLLM_PROXY_ADMISSION_START_INFLIGHT must be between 1 and VLLM_PROXY_ADMISSION_MAX_INFLIGHT"
                );
            }
            if admission_start_inflight < admission_max_inflight && admission_ramp_step == 0 {
                anyhow::bail!(
                    "VLLM_PROXY_ADMISSION_RAMP_STEP must be at least 1 when the budget ramps"
                );
            }
            if admission_ramp_interval_secs == 0 {
                anyhow::bail!("VLLM_PROXY_ADMISSION_RAMP_INTERVAL_SECS must be at least 1");
            }
            if admission_backpressure_secs == 0 {
                anyhow::bail!("VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS must be at least 1");
            }
            if admission_retry_after_secs == 0 {
                anyhow::bail!("VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS must be at least 1");
            }
        }
        let backend_connect_failover = env_bool("VLLM_BACKEND_CONNECT_FAILOVER");
        let backend_probe_urls = url_list("VLLM_BACKEND_PROBE_URLS");
        if !backend_probe_urls.is_empty() && backend_probe_urls.len() != backend_urls.len() {
            anyhow::bail!(
                "VLLM_BACKEND_PROBE_URLS must list one probe URL per VLLM_BACKEND_URLS entry, in the same order"
            );
        }
        let backend_probe_interval_secs: u64 = env_parse("VLLM_BACKEND_PROBE_INTERVAL_SECS", 2)?;
        if backend_probe_interval_secs == 0 {
            anyhow::bail!("VLLM_BACKEND_PROBE_INTERVAL_SECS must be at least 1");
        }

        // Long-context tier: a second set of hosts, registered under the
        // model's `-long` model-proxy domain, for oversized prompts.
        let backend_long_context_urls = url_list("VLLM_BACKEND_LONG_CONTEXT_URLS");
        let backend_long_context_probe_urls = url_list("VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS");
        let long_context_above_tokens: u64 =
            env_parse("VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS", 0)?;
        if !backend_long_context_urls.is_empty() && long_context_above_tokens == 0 {
            anyhow::bail!(
                "VLLM_BACKEND_LONG_CONTEXT_URLS requires VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS: without a threshold nothing would ever be placed there"
            );
        }
        if long_context_above_tokens > 0 && backend_long_context_urls.is_empty() {
            anyhow::bail!(
                "VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS requires VLLM_BACKEND_LONG_CONTEXT_URLS"
            );
        }
        if let Some(both) = backend_long_context_urls
            .iter()
            .find(|url| backend_urls.contains(url))
        {
            anyhow::bail!(
                "{both} is listed in both VLLM_BACKEND_URLS and VLLM_BACKEND_LONG_CONTEXT_URLS; one pool entry serves one tier"
            );
        }
        let expected_long_probes = if backend_probe_urls.is_empty() {
            0
        } else {
            backend_long_context_urls.len()
        };
        if backend_long_context_probe_urls.len() != expected_long_probes {
            anyhow::bail!(
                "VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS must list one probe URL per VLLM_BACKEND_LONG_CONTEXT_URLS entry when VLLM_BACKEND_PROBE_URLS is set, and none when it is not"
            );
        }
        if !backend_long_context_urls.is_empty() && vllm_data_parallel_size.is_some() {
            anyhow::bail!(
                "VLLM_BACKEND_LONG_CONTEXT_URLS and VLLM_DATA_PARALLEL_SIZE are mutually exclusive; data-parallel affinity serves one backend"
            );
        }

        let config = Config {
            model_name,
            tokens,
            vllm_base_url: vllm_base_url.clone(),
            chat_completions_url: format!("{base}/v1/chat/completions"),
            completions_url: format!("{base}/v1/completions"),
            tokenize_url: format!("{base}/tokenize"),
            metrics_url: format!("{base}/metrics"),
            models_url: format!("{base}/v1/models"),
            images_url,
            images_edits_url,
            transcriptions_url,
            embeddings_url: format!("{base}/v1/embeddings"),
            rerank_url,
            score_url,
            cloud_api_url,
            cloud_api_auth_max_attempts: env_int("CLOUD_API_AUTH_MAX_ATTEMPTS", 3),
            cloud_api_auth_initial_backoff_ms: env_int("CLOUD_API_AUTH_INITIAL_BACKOFF_MS", 100)
                as u64,
            cloud_api_auth_timeout_secs: env_int("CLOUD_API_AUTH_TIMEOUT_SECS", 5) as u64,
            cloud_api_usage_token: env::var("CLOUD_API_USAGE_TOKEN")
                .ok()
                .filter(|s| !s.is_empty()),
            compose_manager_url,
            gpu_evidence_delegate_url: env::var("GPU_EVIDENCE_DELEGATE_URL")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| s.trim_end_matches('/').to_string()),
            gpu_evidence_delegate_timeout_secs: env_int("GPU_EVIDENCE_DELEGATE_TIMEOUT_SECS", 30)
                as u64,
            tls_cert_path,
            max_keepalive: env_int("VLLM_PROXY_MAX_KEEPALIVE", 100),
            pool_idle_timeout_secs: env_int("VLLM_PROXY_POOL_IDLE_TIMEOUT_SECS", 60) as u64,
            max_request_size: env_int("VLLM_PROXY_MAX_REQUEST_SIZE", 10 * 1024 * 1024),
            max_image_request_size: env_int("VLLM_PROXY_MAX_IMAGE_REQUEST_SIZE", 50 * 1024 * 1024),
            max_audio_request_size: env_int("VLLM_PROXY_MAX_AUDIO_REQUEST_SIZE", 100 * 1024 * 1024),
            image_validation_enabled: !env_bool("VLLM_PROXY_IMAGE_VALIDATION_DISABLED"),
            image_validation_timeout_secs: env_int("VLLM_PROXY_IMAGE_VALIDATION_TIMEOUT_SECS", 5)
                as u64,
            image_validation_max_bytes: env_int("VLLM_PROXY_IMAGE_VALIDATION_MAX_BYTES", 8192),
            image_validation_max_concurrency: env_int(
                "VLLM_PROXY_IMAGE_VALIDATION_MAX_CONCURRENCY",
                8,
            ),
            image_validation_allow_private_hosts: env_bool(
                "VLLM_PROXY_IMAGE_VALIDATION_ALLOW_PRIVATE_HOSTS",
            ),
            image_validation_allowed_domains,
            image_validation_reject_non_rgb_images,
            image_validation_reject_single_channel_images,
            chat_cache_expiration_secs: env_int("CHAT_CACHE_EXPIRATION", 1200) as u64,
            attestation_cache_ttl_secs: env_int("ATTESTATION_CACHE_TTL", 300) as u64,
            dev_mode: env_bool("DEV"),
            gpu_no_hw_mode: env_bool("GPU_NO_HW_MODE"),
            git_rev,
            rate_limit_per_second: env_int("RATE_LIMIT_PER_SECOND", 100) as u64,
            rate_limit_burst_size: env_int("RATE_LIMIT_BURST_SIZE", 200) as u32,
            rate_limit_trust_proxy_headers: !env_bool("RATE_LIMIT_NO_TRUST_PROXY"),
            timeout_secs: env_int("VLLM_PROXY_TIMEOUT_SECS", 3600) as u64,
            stream_idle_timeout_secs: env_int("VLLM_PROXY_STREAM_IDLE_TIMEOUT_SECS", 0) as u64,
            timeout_tokenize_secs: 10,
            openai_chat_compatibility_check_enabled: env_bool("OPENAI_CHAT_COMPATIBILITY_CHECK"),
            startup_check_retries: env_int("STARTUP_CHECK_RETRIES", 3),
            startup_check_retry_delay_secs: env_int("STARTUP_CHECK_RETRY_DELAY_SECS", 5) as u64,
            startup_check_timeout_secs: env_int("STARTUP_CHECK_TIMEOUT_SECS", 30) as u64,
            backend_urls,
            vllm_data_parallel_size,
            backend_conversation_affinity,
            backend_affinity_max_imbalance,
            health_check_interval_secs: env_int("HEALTH_CHECK_INTERVAL_SECS", 5) as u64,
            health_check_max_failures: env_int("HEALTH_CHECK_MAX_FAILURES", 3) as u32,
            health_check_timeout_secs: env_int("HEALTH_CHECK_TIMEOUT_SECS", 3) as u64,
            ohttp_enabled: env_bool("OHTTP_ENABLED"),
            listen_port,
            listen_addr,
            backend_token,
            backend_priority,
            backend_health_path,
            non_tee_deployment: env_bool("NON_TEE_DEPLOYMENT"),
            map_queue_full_to_429: env_bool("VLLM_PROXY_MAP_QUEUE_FULL_TO_429"),
            stream_error_peek_ms: env_int("VLLM_PROXY_STREAM_ERROR_PEEK_MS", 0) as u64,
            stream_commit_ms: env_int("VLLM_PROXY_STREAM_COMMIT_MS", 0) as u64,
            rejected_content_part_types,
            models_document_url,
            capacity_requests_per_minute,
            reasoning_off_effort,
            allowed_org_ids,
            sse_keepalive_secs: env_int("VLLM_PROXY_SSE_KEEPALIVE_SECS", 0) as u64,
            admission_max_inflight,
            admission_start_inflight,
            admission_ramp_step,
            admission_ramp_interval_secs,
            admission_ttft_p95_max_ms,
            admission_backpressure_secs,
            admission_retry_after_secs,
            backend_connect_failover,
            backend_probe_urls,
            backend_probe_interval_secs,
            backend_long_context_urls,
            backend_long_context_probe_urls,
            long_context_above_tokens,
            images_url_override,
            images_edits_url_override,
            transcriptions_url_override,
            rerank_url_override,
            score_url_override,
            dstack_socket_path: env_or("DSTACK_SOCKET_PATH", "/var/run/dstack.sock"),
            web_context_search_url: env::var("WEB_CONTEXT_SEARCH_URL")
                .ok()
                .filter(|s| !s.is_empty()),
            web_context_search_api_key: env::var("WEB_CONTEXT_SEARCH_API_KEY")
                .ok()
                .filter(|s| !s.is_empty())
                .or_else(|| {
                    env::var("BRAVE_LLM_CONTEXT_API_KEY")
                        .ok()
                        .filter(|s| !s.is_empty())
                }),
            // `env_int` returns `usize`; on 64-bit hosts a user-supplied value
            // > u32::MAX would silently wrap. `try_from` surfaces it as a
            // config error instead so a typo can't become a tiny iteration cap.
            agent_loop_max_iterations: u32::try_from(env_int("AGENT_LOOP_MAX_ITERATIONS", 5))
                .map_err(|_| anyhow::anyhow!("AGENT_LOOP_MAX_ITERATIONS exceeds the u32 range"))?,
            web_context_search_timeout_secs: env_int("WEB_CONTEXT_SEARCH_TIMEOUT_SECS", 30) as u64,
            fusion_enabled: env_bool("FUSION_ENABLED"),
            fusion_endpoints_url: env_or(
                "FUSION_ENDPOINTS_URL",
                "https://completions.near.ai/endpoints",
            ),
            fusion_endpoints_ttl_secs: env_int("FUSION_ENDPOINTS_TTL_SECS", 300) as u64,
            fusion_internal_bearer_token: env::var("FUSION_INTERNAL_BEARER_TOKEN")
                .ok()
                .filter(|s| !s.is_empty()),
            fusion_default_analysis_models,
            fusion_max_panel_models: env_int("FUSION_MAX_PANEL_MODELS", 8),
            fusion_max_depth: u32::try_from(env_int("FUSION_MAX_DEPTH", 1))
                .map_err(|_| anyhow::anyhow!("FUSION_MAX_DEPTH exceeds the u32 range"))?,
            fusion_panel_timeout_secs: env_int("FUSION_PANEL_TIMEOUT_SECS", 120) as u64,
            fusion_max_response_bytes: env_int("FUSION_MAX_RESPONSE_BYTES", 10 * 1024 * 1024),
            fusion_internal_max_attempts: env_int("FUSION_INTERNAL_MAX_ATTEMPTS", 2),
            fusion_internal_retry_initial_backoff_ms: env_int(
                "FUSION_INTERNAL_RETRY_INITIAL_BACKOFF_MS",
                250,
            ) as u64,
        };

        // Validate attestation cache TTL (TTL/2 is used as refresh interval, so TTL < 2 would cause a busy loop)
        if config.attestation_cache_ttl_secs < 2 {
            anyhow::bail!(
                "ATTESTATION_CACHE_TTL must be at least 2 (got {})",
                config.attestation_cache_ttl_secs
            );
        }

        // Validate startup check configuration
        if config.startup_check_retries == 0 {
            anyhow::bail!("STARTUP_CHECK_RETRIES must be at least 1");
        }
        if config.startup_check_timeout_secs == 0 {
            anyhow::bail!("STARTUP_CHECK_TIMEOUT_SECS must be greater than 0");
        }

        // Agent loop: URL and key must both be set or both unset; iteration cap must be positive.
        if config.web_context_search_url.is_some() != config.web_context_search_api_key.is_some() {
            anyhow::bail!(
                "WEB_CONTEXT_SEARCH_URL and WEB_CONTEXT_SEARCH_API_KEY or BRAVE_LLM_CONTEXT_API_KEY must both be set or both unset"
            );
        }
        if config.agent_loop_max_iterations == 0 {
            anyhow::bail!("AGENT_LOOP_MAX_ITERATIONS must be at least 1");
        }
        if config.web_context_search_timeout_secs == 0 {
            anyhow::bail!("WEB_CONTEXT_SEARCH_TIMEOUT_SECS must be greater than 0");
        }

        if config.fusion_enabled {
            if config.fusion_internal_bearer_token.is_none() {
                anyhow::bail!("FUSION_INTERNAL_BEARER_TOKEN must be set when FUSION_ENABLED=true");
            }
            if config.fusion_endpoints_url.is_empty() {
                anyhow::bail!("FUSION_ENDPOINTS_URL must not be empty");
            }
            if config.fusion_endpoints_ttl_secs == 0 {
                anyhow::bail!("FUSION_ENDPOINTS_TTL_SECS must be greater than 0");
            }
            if config.fusion_max_panel_models == 0 {
                anyhow::bail!("FUSION_MAX_PANEL_MODELS must be at least 1");
            }
            if config.fusion_max_depth == 0 {
                anyhow::bail!("FUSION_MAX_DEPTH must be at least 1");
            }
            if config.fusion_panel_timeout_secs == 0 {
                anyhow::bail!("FUSION_PANEL_TIMEOUT_SECS must be greater than 0");
            }
            if config.fusion_max_response_bytes == 0 {
                anyhow::bail!("FUSION_MAX_RESPONSE_BYTES must be greater than 0");
            }
            if config.fusion_internal_max_attempts == 0 {
                anyhow::bail!("FUSION_INTERNAL_MAX_ATTEMPTS must be at least 1");
            }
            if config.fusion_internal_max_attempts > FUSION_INTERNAL_MAX_ATTEMPTS_LIMIT {
                anyhow::bail!(
                    "FUSION_INTERNAL_MAX_ATTEMPTS must be at most {}",
                    FUSION_INTERNAL_MAX_ATTEMPTS_LIMIT
                );
            }
            if config.fusion_internal_max_attempts > 1
                && config.fusion_internal_retry_initial_backoff_ms == 0
            {
                anyhow::bail!(
                    "FUSION_INTERNAL_RETRY_INITIAL_BACKOFF_MS must be greater than 0 when FUSION_INTERNAL_MAX_ATTEMPTS > 1"
                );
            }
            if config.fusion_default_analysis_models.is_empty() {
                warn!(
                    "FUSION_ENABLED=true with no FUSION_DEFAULT_ANALYSIS_MODELS; clients must provide analysis_models per request"
                );
            }
        }

        if config.admission_max_inflight > 0
            && (config.fusion_enabled || config.web_context_search_url.is_some())
        {
            anyhow::bail!(
                "VLLM_PROXY_ADMISSION_MAX_INFLIGHT cannot be combined with FUSION_ENABLED or WEB_CONTEXT_SEARCH_URL: those execution modes run outside the lane budget"
            );
        }
        if !config.backend_long_context_urls.is_empty()
            && (config.fusion_enabled || config.web_context_search_url.is_some())
        {
            anyhow::bail!(
                "VLLM_BACKEND_LONG_CONTEXT_URLS cannot be combined with FUSION_ENABLED or WEB_CONTEXT_SEARCH_URL: those execution modes place their own backend requests, outside the tier"
            );
        }
        Ok(config)
    }

    /// Engine-load probe URLs in pool order: the base tier, then the
    /// long-context one, matching how `BackendPool` is built.
    pub fn pool_probe_urls(&self) -> Vec<String> {
        self.backend_probe_urls
            .iter()
            .chain(&self.backend_long_context_probe_urls)
            .cloned()
            .collect()
    }

    /// Lane admission settings, `None` unless `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` is set.
    pub fn admission(&self) -> Option<crate::admission::AdmissionConfig> {
        if self.admission_max_inflight == 0 {
            return None;
        }
        Some(crate::admission::AdmissionConfig {
            max_inflight: self.admission_max_inflight,
            start_inflight: self.admission_start_inflight,
            ramp_step: self.admission_ramp_step,
            ramp_interval: std::time::Duration::from_secs(self.admission_ramp_interval_secs),
            ttft_p95_max: (self.admission_ttft_p95_max_ms > 0)
                .then(|| std::time::Duration::from_millis(self.admission_ttft_p95_max_ms)),
            backpressure_ttl: std::time::Duration::from_secs(self.admission_backpressure_secs),
            retry_after: std::time::Duration::from_secs(self.admission_retry_after_secs),
        })
    }

    /// Build the runtime config for pre-dispatch image validation.
    pub fn image_validation(&self) -> crate::image_validation::ImageValidationConfig {
        crate::image_validation::ImageValidationConfig {
            enabled: self.image_validation_enabled,
            timeout: std::time::Duration::from_secs(self.image_validation_timeout_secs),
            max_bytes: self.image_validation_max_bytes,
            max_concurrency: self.image_validation_max_concurrency,
            allow_private_hosts: self.image_validation_allow_private_hosts,
            allowed_domains: self.image_validation_allowed_domains.clone(),
            reject_non_rgb_images: self.image_validation_reject_non_rgb_images,
            reject_single_channel_images: self.image_validation_reject_single_channel_images,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-modifying tests to avoid races
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env_vars<F, R>(vars: &[(&str, &str)], f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _guard = ENV_LOCK.lock().unwrap();
        // Capture old values
        let old_values: Vec<(&str, Option<String>)> =
            vars.iter().map(|(k, _)| (*k, env::var(k).ok())).collect();
        // Set new values
        for (k, v) in vars {
            env::set_var(k, v);
        }
        let result = f();
        // Restore old values
        for (k, old) in &old_values {
            match old {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }
        result
    }

    #[test]
    fn test_config_requires_model_name() {
        with_env_vars(&[("TOKEN", "test")], || {
            env::remove_var("MODEL_NAME");
            let result = Config::from_env();
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("MODEL_NAME"));
        });
    }

    #[test]
    fn test_config_requires_token() {
        with_env_vars(&[("MODEL_NAME", "test")], || {
            env::remove_var("TOKEN");
            let result = Config::from_env();
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("TOKEN"));
        });
    }

    #[test]
    fn test_config_rejects_empty_token() {
        with_env_vars(&[("MODEL_NAME", "test"), ("TOKEN", "")], || {
            let result = Config::from_env();
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("at least one non-empty token"));
        });
    }

    #[test]
    fn test_config_rejects_token_list_of_only_empties() {
        with_env_vars(&[("MODEL_NAME", "test"), ("TOKEN", " , , ")], || {
            let result = Config::from_env();
            assert!(result.is_err());
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("at least one non-empty token"));
        });
    }

    #[test]
    fn test_config_parses_multiple_tokens() {
        with_env_vars(
            &[("MODEL_NAME", "test"), ("TOKEN", "tok-a, tok-b ,tok-c")],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(config.tokens, vec!["tok-a", "tok-b", "tok-c"]);
            },
        );
    }

    fn gateway_env_cleanup() {
        for key in [
            "VLLM_BACKEND_URLS",
            "VLLM_DATA_PARALLEL_SIZE",
            "VLLM_BACKEND_TOKEN",
            "VLLM_BACKEND_PRIORITY",
            "VLLM_BACKEND_HEALTH_PATH",
            "NON_TEE_DEPLOYMENT",
            "VLLM_PROXY_MAP_QUEUE_FULL_TO_429",
            "VLLM_PROXY_STREAM_ERROR_PEEK_MS",
            "VLLM_PROXY_REJECTED_CONTENT_PART_TYPES",
            "VLLM_PROXY_ALLOWED_ORG_IDS",
            "VLLM_PROXY_SSE_KEEPALIVE_SECS",
            "VLLM_PROXY_ADMISSION_MAX_INFLIGHT",
            "VLLM_PROXY_ADMISSION_START_INFLIGHT",
            "VLLM_PROXY_ADMISSION_RAMP_STEP",
            "VLLM_PROXY_ADMISSION_RAMP_INTERVAL_SECS",
            "VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS",
            "VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS",
            "VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS",
            "VLLM_BACKEND_CONNECT_FAILOVER",
            "VLLM_BACKEND_PROBE_URLS",
            "VLLM_BACKEND_PROBE_INTERVAL_SECS",
            "VLLM_BACKEND_LONG_CONTEXT_URLS",
            "VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS",
            "VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS",
            "LISTEN_ADDR",
        ] {
            env::remove_var(key);
        }
    }

    #[test]
    fn test_gateway_defaults_leave_existing_deployments_unchanged() {
        with_env_vars(&[("MODEL_NAME", "m"), ("TOKEN", "t")], || {
            gateway_env_cleanup();
            let config = Config::from_env().unwrap();
            assert_eq!(config.listen_addr, "0.0.0.0");
            assert!(config.backend_token.is_none());
            assert!(config.backend_priority.is_none());
            assert_eq!(config.backend_health_path, "/health");
            assert!(!config.non_tee_deployment);
            assert!(!config.map_queue_full_to_429);
            assert_eq!(config.stream_error_peek_ms, 0);
            assert_eq!(config.stream_commit_ms, 0);
            assert!(config.rejected_content_part_types.is_empty());
            assert!(config.allowed_org_ids.is_empty());
            assert_eq!(config.sse_keepalive_secs, 0);
            assert_eq!(config.admission_max_inflight, 0);
            assert!(config.admission().is_none());
            assert!(!config.backend_connect_failover);
            assert!(config.backend_probe_urls.is_empty());
            assert_eq!(config.backend_probe_interval_secs, 2);
            assert!(config.backend_long_context_urls.is_empty());
            assert!(config.pool_probe_urls().is_empty());
            assert_eq!(config.long_context_above_tokens, 0);
            assert_eq!(config.backend_urls, vec!["http://localhost:8000"]);
        });
    }

    #[test]
    fn test_gateway_settings_parse() {
        with_env_vars(
            &[
                ("MODEL_NAME", "m"),
                ("TOKEN", "t"),
                ("VLLM_BACKEND_TOKEN", " backend-secret "),
                ("CLOUD_API_URL", "https://cloud-api.test"),
                ("CLOUD_API_USAGE_TOKEN", "usage-secret"),
                ("VLLM_BACKEND_HEALTH_PATH", "/healthz"),
                ("NON_TEE_DEPLOYMENT", "1"),
                ("VLLM_PROXY_MAP_QUEUE_FULL_TO_429", "1"),
                ("VLLM_PROXY_STREAM_ERROR_PEEK_MS", "750"),
                (
                    "VLLM_PROXY_REJECTED_CONTENT_PART_TYPES",
                    "video_url, input_audio",
                ),
                ("VLLM_PROXY_SSE_KEEPALIVE_SECS", "15"),
                ("LISTEN_ADDR", "127.0.0.1"),
                ("VLLM_PROXY_ALLOWED_ORG_IDS", " org-a, org-b ,,"),
            ],
            || {
                env::remove_var("VLLM_BACKEND_URLS");
                env::remove_var("VLLM_DATA_PARALLEL_SIZE");
                let config = Config::from_env().unwrap();
                assert_eq!(config.allowed_org_ids, vec!["org-a", "org-b"]);
                assert_eq!(config.backend_token.as_deref(), Some("backend-secret"));
                assert_eq!(config.backend_health_path, "/healthz");
                assert!(config.non_tee_deployment);
                assert!(config.map_queue_full_to_429);
                assert_eq!(config.stream_error_peek_ms, 750);
                assert_eq!(
                    config.rejected_content_part_types,
                    vec!["video_url", "input_audio"]
                );
                assert_eq!(config.sse_keepalive_secs, 15);
                assert_eq!(config.listen_addr, "127.0.0.1");
                gateway_env_cleanup();
            },
        );
    }

    #[test]
    fn test_backend_priority_parses_and_validates() {
        with_env_vars(
            &[
                ("MODEL_NAME", "m"),
                ("TOKEN", "t"),
                ("VLLM_BACKEND_PRIORITY", " -1 "),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(config.backend_priority, Some(-1));
                for bad in ["high", "1.5", "5000"] {
                    env::set_var("VLLM_BACKEND_PRIORITY", bad);
                    let err = Config::from_env().unwrap_err().to_string();
                    assert!(err.contains("VLLM_BACKEND_PRIORITY"), "{bad}: {err}");
                }
                gateway_env_cleanup();
            },
        );
    }

    #[test]
    fn test_backend_token_requires_cloud_api_billing() {
        with_env_vars(
            &[
                ("MODEL_NAME", "m"),
                ("TOKEN", "t"),
                ("VLLM_BACKEND_TOKEN", "backend-secret"),
                ("CLOUD_API_URL", "https://cloud-api.test"),
                ("CLOUD_API_USAGE_TOKEN", ""),
            ],
            || {
                gateway_env_cleanup();
                env::set_var("VLLM_BACKEND_TOKEN", "backend-secret");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("CLOUD_API_USAGE_TOKEN"), "{err}");
                env::set_var("CLOUD_API_USAGE_TOKEN", "usage-secret");
                assert!(Config::from_env().is_ok());
                gateway_env_cleanup();
            },
        );
    }

    #[test]
    fn test_gateway_listen_addr_and_health_path_are_validated() {
        with_env_vars(
            &[
                ("MODEL_NAME", "m"),
                ("TOKEN", "t"),
                ("LISTEN_ADDR", "not-an-ip"),
            ],
            || {
                gateway_env_cleanup();
                env::set_var("LISTEN_ADDR", "not-an-ip");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("LISTEN_ADDR"), "{err}");
                env::set_var("LISTEN_ADDR", "10.0.0.5");
                env::set_var("OHTTP_ENABLED", "1");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("OHTTP_ENABLED requires"), "{err}");
                env::set_var("LISTEN_ADDR", "::1");
                assert!(
                    Config::from_env().is_ok(),
                    "loopback v6 bind is fine with OHTTP"
                );
                env::remove_var("OHTTP_ENABLED");
                env::remove_var("LISTEN_ADDR");
                env::set_var("VLLM_BACKEND_HEALTH_PATH", "healthz");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("VLLM_BACKEND_HEALTH_PATH"), "{err}");
                gateway_env_cleanup();
            },
        );
    }

    #[test]
    fn test_long_context_tier_settings_are_validated() {
        with_env_vars(&[("MODEL_NAME", "m"), ("TOKEN", "t")], || {
            gateway_env_cleanup();
            let err = || Config::from_env().unwrap_err().to_string();
            env::set_var("VLLM_BACKEND_URLS", "https://m-b1.test,https://m-b2.test");
            // The tier and its threshold only make sense together.
            env::set_var("VLLM_BACKEND_LONG_CONTEXT_URLS", "https://m-long-b3.test");
            assert!(
                err().contains("VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS"),
                "{}",
                err()
            );
            env::remove_var("VLLM_BACKEND_LONG_CONTEXT_URLS");
            env::set_var("VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS", "100000");
            assert!(
                err().contains("VLLM_BACKEND_LONG_CONTEXT_URLS"),
                "{}",
                err()
            );
            // One pool entry serves one tier.
            env::set_var("VLLM_BACKEND_LONG_CONTEXT_URLS", "https://m-b2.test");
            assert!(err().contains("both"), "{}", err());

            env::set_var("VLLM_BACKEND_LONG_CONTEXT_URLS", "https://m-long-b3.test/");
            let config = Config::from_env().unwrap();
            assert_eq!(config.backend_long_context_urls, ["https://m-long-b3.test"]);
            assert_eq!(config.long_context_above_tokens, 100_000);
            assert!(config.pool_probe_urls().is_empty());

            // Probes: one per backend of each tier, in pool order.
            env::set_var("VLLM_BACKEND_PROBE_URLS", "http://p1:8000,http://p2:8000");
            assert!(
                err().contains("VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS"),
                "{}",
                err()
            );
            env::set_var("VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS", "http://p3:8000");
            assert_eq!(
                Config::from_env().unwrap().pool_probe_urls(),
                ["http://p1:8000", "http://p2:8000", "http://p3:8000"]
            );
            // Long probes alone would poll a tier nothing else is polled for.
            env::remove_var("VLLM_BACKEND_PROBE_URLS");
            assert!(
                err().contains("VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS"),
                "{}",
                err()
            );
            env::remove_var("VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS");

            // A data-parallel backend is a single engine: no second tier.
            env::set_var("VLLM_BACKEND_URLS", "https://m-b1.test");
            env::set_var("VLLM_DATA_PARALLEL_SIZE", "4");
            assert!(err().contains("VLLM_DATA_PARALLEL_SIZE"), "{}", err());
            env::remove_var("VLLM_DATA_PARALLEL_SIZE");

            // Fusion and the agent loop place their own backend requests,
            // which no tier restriction reaches.
            env::set_var("WEB_CONTEXT_SEARCH_URL", "https://brave.test");
            assert!(err().contains("WEB_CONTEXT_SEARCH_URL"), "{}", err());
            env::remove_var("WEB_CONTEXT_SEARCH_URL");
            env::set_var("FUSION_ENABLED", "1");
            env::set_var("FUSION_INTERNAL_BEARER_TOKEN", "fusion-secret");
            assert!(err().contains("FUSION_ENABLED"), "{}", err());
            env::remove_var("FUSION_ENABLED");
            env::remove_var("FUSION_INTERNAL_BEARER_TOKEN");
            gateway_env_cleanup();
        });
    }

    #[test]
    fn test_config_single_token_backward_compatible() {
        with_env_vars(&[("MODEL_NAME", "test"), ("TOKEN", "only-one")], || {
            let config = Config::from_env().unwrap();
            assert_eq!(config.tokens, vec!["only-one"]);
        });
    }

    #[test]
    fn test_config_default_values() {
        with_env_vars(&[("MODEL_NAME", "my-model"), ("TOKEN", "secret")], || {
            // Remove optional vars to test defaults
            env::remove_var("VLLM_BASE_URL");
            env::remove_var("VLLM_BACKEND_URLS");
            env::remove_var("VLLM_DATA_PARALLEL_SIZE");
            env::remove_var("VLLM_IMAGES_URL");
            env::remove_var("VLLM_IMAGES_EDITS_URL");
            env::remove_var("VLLM_TRANSCRIPTIONS_URL");
            env::remove_var("VLLM_RERANK_URL");
            env::remove_var("VLLM_SCORE_URL");
            env::remove_var("DEV");
            env::remove_var("GPU_NO_HW_MODE");
            env::remove_var("CHAT_CACHE_EXPIRATION");
            env::remove_var("VLLM_ALLOWED_MEDIA_DOMAINS");
            env::remove_var("VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS");
            env::remove_var("VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB");
            env::remove_var("VLLM_PROXY_STREAM_IDLE_TIMEOUT_SECS");
            env::remove_var("WEB_CONTEXT_SEARCH_URL");
            env::remove_var("WEB_CONTEXT_SEARCH_API_KEY");
            env::remove_var("FUSION_ENABLED");
            env::remove_var("FUSION_ENDPOINTS_URL");
            env::remove_var("FUSION_ENDPOINTS_TTL_SECS");
            env::remove_var("FUSION_INTERNAL_BEARER_TOKEN");
            env::remove_var("FUSION_DEFAULT_ANALYSIS_MODELS");
            env::remove_var("FUSION_MAX_PANEL_MODELS");
            env::remove_var("FUSION_MAX_DEPTH");
            env::remove_var("FUSION_PANEL_TIMEOUT_SECS");
            env::remove_var("FUSION_MAX_RESPONSE_BYTES");
            env::remove_var("FUSION_INTERNAL_MAX_ATTEMPTS");
            env::remove_var("FUSION_INTERNAL_RETRY_INITIAL_BACKOFF_MS");
            env::remove_var("BRAVE_LLM_CONTEXT_API_KEY");

            let config = Config::from_env().unwrap();

            assert_eq!(config.model_name, "my-model");
            assert_eq!(config.tokens, vec!["secret"]);
            assert_eq!(config.vllm_base_url, "http://localhost:8000");
            assert_eq!(
                config.chat_completions_url,
                "http://localhost:8000/v1/chat/completions"
            );
            assert_eq!(
                config.completions_url,
                "http://localhost:8000/v1/completions"
            );
            assert_eq!(config.tokenize_url, "http://localhost:8000/tokenize");
            assert_eq!(config.metrics_url, "http://localhost:8000/metrics");
            assert_eq!(config.models_url, "http://localhost:8000/v1/models");
            assert_eq!(config.max_request_size, 10 * 1024 * 1024);
            assert_eq!(config.max_image_request_size, 50 * 1024 * 1024);
            assert_eq!(config.max_audio_request_size, 100 * 1024 * 1024);
            assert_eq!(config.chat_cache_expiration_secs, 1200);
            assert!(!config.dev_mode);
            assert!(!config.gpu_no_hw_mode);
            assert_eq!(config.backend_urls, vec!["http://localhost:8000"]);
            assert_eq!(config.vllm_data_parallel_size, None);
            assert!(config.images_url_override.is_none());
            assert!(config.rerank_url_override.is_none());
            assert!(config.image_validation_allowed_domains.is_empty());
            assert!(!config.image_validation_reject_non_rgb_images);
            assert!(!config.image_validation_reject_single_channel_images);
            assert_eq!(config.stream_idle_timeout_secs, 0);
            assert!(!config.fusion_enabled);
            assert_eq!(
                config.fusion_endpoints_url,
                "https://completions.near.ai/endpoints"
            );
            assert_eq!(config.fusion_endpoints_ttl_secs, 300);
            assert!(config.fusion_internal_bearer_token.is_none());
            assert!(config.fusion_default_analysis_models.is_empty());
            assert_eq!(config.fusion_max_panel_models, 8);
            assert_eq!(config.fusion_max_depth, 1);
            assert_eq!(config.fusion_panel_timeout_secs, 120);
            assert_eq!(config.fusion_max_response_bytes, 10 * 1024 * 1024);
            assert_eq!(config.fusion_internal_max_attempts, 2);
            assert_eq!(config.fusion_internal_retry_initial_backoff_ms, 250);
        });
    }

    #[test]
    fn test_stream_idle_timeout_env_override() {
        with_env_vars(
            &[
                ("MODEL_NAME", "test-model"),
                ("TOKEN", "tok"),
                ("VLLM_PROXY_STREAM_IDLE_TIMEOUT_SECS", "20"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(config.stream_idle_timeout_secs, 20);
            },
        );
    }

    #[test]
    fn test_image_validation_env_vars_override_defaults() {
        with_env_vars(
            &[
                ("MODEL_NAME", "plain-model"),
                ("TOKEN", "tok"),
                ("VLLM_PROXY_IMAGE_VALIDATION_DISABLED", "1"),
                ("VLLM_PROXY_IMAGE_VALIDATION_TIMEOUT_SECS", "7"),
                ("VLLM_PROXY_IMAGE_VALIDATION_MAX_BYTES", "1234"),
                ("VLLM_PROXY_IMAGE_VALIDATION_MAX_CONCURRENCY", "3"),
                ("VLLM_PROXY_IMAGE_VALIDATION_ALLOW_PRIVATE_HOSTS", "true"),
                (
                    "VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS",
                    " https://CDN.Example.COM, images.example.com. ",
                ),
                ("VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB", "1"),
            ],
            || {
                let config = Config::from_env().unwrap();

                assert!(!config.image_validation_enabled);
                assert_eq!(config.image_validation_timeout_secs, 7);
                assert_eq!(config.image_validation_max_bytes, 1234);
                assert_eq!(config.image_validation_max_concurrency, 3);
                assert!(config.image_validation_allow_private_hosts);
                assert_eq!(
                    config.image_validation_allowed_domains,
                    vec!["cdn.example.com", "images.example.com"]
                );
                assert!(config.image_validation_reject_non_rgb_images);
                assert!(config.image_validation_reject_single_channel_images);

                let image_validation = config.image_validation();
                assert!(!image_validation.enabled);
                assert_eq!(image_validation.timeout, std::time::Duration::from_secs(7));
                assert_eq!(image_validation.max_bytes, 1234);
                assert_eq!(image_validation.max_concurrency, 3);
                assert!(image_validation.allow_private_hosts);
                assert_eq!(
                    image_validation.allowed_domains,
                    vec!["cdn.example.com", "images.example.com"]
                );
                assert!(image_validation.reject_non_rgb_images);
                assert!(image_validation.reject_single_channel_images);
            },
        );
    }

    #[test]
    fn test_image_validation_allowed_domains_fallback_and_gemma_default() {
        with_env_vars(
            &[
                ("MODEL_NAME", "plain-model"),
                ("TOKEN", "tok"),
                (
                    "VLLM_ALLOWED_MEDIA_DOMAINS",
                    "prod-files-secure.s3.us-west-2.amazonaws.com, CDN.GeneralContext.COM",
                ),
            ],
            || {
                env::remove_var("VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS");
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.image_validation_allowed_domains,
                    vec![
                        "prod-files-secure.s3.us-west-2.amazonaws.com",
                        "cdn.generalcontext.com"
                    ]
                );
            },
        );

        with_env_vars(
            &[("MODEL_NAME", "google/gemma-4-31B-it"), ("TOKEN", "tok")],
            || {
                env::remove_var("VLLM_ALLOWED_MEDIA_DOMAINS");
                env::remove_var("VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS");
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.image_validation_allowed_domains,
                    vec![DEFAULT_GEMMA4_ALLOWED_MEDIA_DOMAIN]
                );
            },
        );

        with_env_vars(
            &[
                ("MODEL_NAME", "google/gemma-4-31B-it"),
                ("TOKEN", "tok"),
                ("VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS", ""),
                (
                    "VLLM_ALLOWED_MEDIA_DOMAINS",
                    "prod-files-secure.s3.us-west-2.amazonaws.com",
                ),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert!(config.image_validation_allowed_domains.is_empty());
            },
        );
    }

    #[test]
    fn test_config_accepts_brave_llm_context_api_key_alias() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("WEB_CONTEXT_SEARCH_URL", "https://brave.test/context"),
                ("WEB_CONTEXT_SEARCH_API_KEY", ""),
                ("BRAVE_LLM_CONTEXT_API_KEY", "brave-alias-key"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.web_context_search_api_key.as_deref(),
                    Some("brave-alias-key")
                );
            },
        );
    }

    #[test]
    fn test_gemma4_enables_single_channel_guard_by_default_with_env_override() {
        with_env_vars(
            &[
                ("MODEL_NAME", "RedHatAI/gemma-4-31B-it-FP8-Dynamic"),
                ("TOKEN", "tok"),
                ("VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB", "0"),
            ],
            || {
                env::remove_var("VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB");
                let config = Config::from_env().unwrap();
                assert!(config.image_validation_reject_single_channel_images);
                assert!(!config.image_validation_reject_non_rgb_images);

                env::set_var("VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB", "1");
                let config = Config::from_env().unwrap();
                assert!(config.image_validation_reject_single_channel_images);
                assert!(config.image_validation_reject_non_rgb_images);

                env::set_var("VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB", "0");
                let config = Config::from_env().unwrap();
                assert!(!config.image_validation_reject_single_channel_images);
                assert!(!config.image_validation_reject_non_rgb_images);
            },
        );
    }

    #[test]
    fn test_config_requires_fusion_token_when_enabled() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("FUSION_ENABLED", "true"),
                ("FUSION_INTERNAL_BEARER_TOKEN", ""),
            ],
            || {
                let result = Config::from_env();
                assert!(result.is_err());
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("FUSION_INTERNAL_BEARER_TOKEN"));
            },
        );
    }

    #[test]
    fn test_config_rejects_zero_fusion_limits_when_enabled() {
        for (name, expected) in [
            (
                "FUSION_PANEL_TIMEOUT_SECS",
                "FUSION_PANEL_TIMEOUT_SECS must be greater than 0",
            ),
            (
                "FUSION_MAX_RESPONSE_BYTES",
                "FUSION_MAX_RESPONSE_BYTES must be greater than 0",
            ),
            (
                "FUSION_INTERNAL_MAX_ATTEMPTS",
                "FUSION_INTERNAL_MAX_ATTEMPTS must be at least 1",
            ),
            (
                "FUSION_INTERNAL_RETRY_INITIAL_BACKOFF_MS",
                "FUSION_INTERNAL_RETRY_INITIAL_BACKOFF_MS must be greater than 0",
            ),
        ] {
            with_env_vars(
                &[
                    ("MODEL_NAME", "model"),
                    ("TOKEN", "tok"),
                    ("FUSION_ENABLED", "true"),
                    ("FUSION_INTERNAL_BEARER_TOKEN", "internal"),
                    (name, "0"),
                ],
                || {
                    let result = Config::from_env();
                    assert!(result.is_err());
                    assert!(result.unwrap_err().to_string().contains(expected));
                },
            );
        }
    }

    #[test]
    fn test_config_rejects_excessive_fusion_internal_attempts_when_enabled() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("FUSION_ENABLED", "true"),
                ("FUSION_INTERNAL_BEARER_TOKEN", "internal"),
                ("FUSION_INTERNAL_MAX_ATTEMPTS", "6"),
            ],
            || {
                let result = Config::from_env();
                assert!(result.is_err());
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("FUSION_INTERNAL_MAX_ATTEMPTS must be at most 5"));
            },
        );
    }

    #[test]
    fn test_gemma4_guard_does_not_match_gemma_4b() {
        with_env_vars(
            &[("MODEL_NAME", "google/gemma-4b-it"), ("TOKEN", "tok")],
            || {
                env::remove_var("VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB");

                let config = Config::from_env().unwrap();

                assert!(!config.image_validation_reject_single_channel_images);
                assert!(!config.image_validation_reject_non_rgb_images);
            },
        );
    }

    #[test]
    fn test_config_parses_fusion_settings() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("FUSION_ENABLED", "true"),
                ("FUSION_INTERNAL_BEARER_TOKEN", "internal"),
                ("FUSION_ENDPOINTS_URL", "http://endpoints.test/list"),
                ("FUSION_ENDPOINTS_TTL_SECS", "42"),
                ("FUSION_DEFAULT_ANALYSIS_MODELS", "~model-a, model-b"),
                ("FUSION_MAX_PANEL_MODELS", "3"),
                ("FUSION_MAX_DEPTH", "2"),
                ("FUSION_PANEL_TIMEOUT_SECS", "9"),
                ("FUSION_MAX_RESPONSE_BYTES", "4096"),
                ("FUSION_INTERNAL_MAX_ATTEMPTS", "4"),
                ("FUSION_INTERNAL_RETRY_INITIAL_BACKOFF_MS", "17"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert!(config.fusion_enabled);
                assert_eq!(
                    config.fusion_internal_bearer_token.as_deref(),
                    Some("internal")
                );
                assert_eq!(config.fusion_endpoints_url, "http://endpoints.test/list");
                assert_eq!(config.fusion_endpoints_ttl_secs, 42);
                assert_eq!(
                    config.fusion_default_analysis_models,
                    vec!["model-a".to_string(), "model-b".to_string()]
                );
                assert_eq!(config.fusion_max_panel_models, 3);
                assert_eq!(config.fusion_max_depth, 2);
                assert_eq!(config.fusion_panel_timeout_secs, 9);
                assert_eq!(config.fusion_max_response_bytes, 4096);
                assert_eq!(config.fusion_internal_max_attempts, 4);
                assert_eq!(config.fusion_internal_retry_initial_backoff_ms, 17);
            },
        );
    }

    #[test]
    fn test_config_backend_urls() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                (
                    "VLLM_BACKEND_URLS",
                    "http://b1:8000, http://b2:8000 , http://b3:8000/",
                ),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.backend_urls,
                    vec!["http://b1:8000", "http://b2:8000", "http://b3:8000"]
                );
                // vllm_base_url should still be set for backward compat
                assert!(!config.vllm_base_url.is_empty());
            },
        );
    }

    #[test]
    fn test_config_backend_urls_fallback_to_base() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("VLLM_BASE_URL", "http://myhost:9000"),
            ],
            || {
                env::remove_var("VLLM_BACKEND_URLS");
                let config = Config::from_env().unwrap();
                assert_eq!(config.backend_urls, vec!["http://myhost:9000"]);
            },
        );
    }

    #[test]
    fn test_config_parses_vllm_data_parallel_size() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("VLLM_DATA_PARALLEL_SIZE", "4"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(config.vllm_data_parallel_size, Some(4));
            },
        );
    }

    #[test]
    fn test_config_rejects_invalid_vllm_data_parallel_size() {
        for invalid in ["0", "not-a-number", ""] {
            with_env_vars(
                &[
                    ("MODEL_NAME", "model"),
                    ("TOKEN", "tok"),
                    ("VLLM_DATA_PARALLEL_SIZE", invalid),
                ],
                || {
                    let error = Config::from_env().unwrap_err().to_string();
                    assert!(error.contains("VLLM_DATA_PARALLEL_SIZE"));
                },
            );
        }
    }

    #[test]
    fn test_config_rejects_dp_affinity_with_multiple_backends() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("VLLM_DATA_PARALLEL_SIZE", "4"),
                ("VLLM_BACKEND_URLS", "http://backend-a,http://backend-b"),
            ],
            || {
                let error = Config::from_env().unwrap_err().to_string();
                assert!(error.contains("requires exactly one vLLM backend"));
            },
        );
    }

    #[test]
    fn test_config_backend_conversation_affinity_defaults_off() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("VLLM_BACKEND_URLS", "http://backend-a,http://backend-b"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert!(!config.backend_conversation_affinity);
                assert_eq!(config.backend_affinity_max_imbalance, 8);
            },
        );
    }

    #[test]
    fn test_config_parses_backend_conversation_affinity() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("VLLM_BACKEND_URLS", "http://backend-a,http://backend-b"),
                ("VLLM_BACKEND_CONVERSATION_AFFINITY", "true"),
                ("VLLM_BACKEND_AFFINITY_MAX_IMBALANCE", "3"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert!(config.backend_conversation_affinity);
                assert_eq!(config.backend_affinity_max_imbalance, 3);
            },
        );
    }

    #[test]
    fn test_config_rejects_backend_affinity_combined_with_dp_affinity() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("VLLM_DATA_PARALLEL_SIZE", "4"),
                ("VLLM_BACKEND_CONVERSATION_AFFINITY", "1"),
            ],
            || {
                let error = Config::from_env().unwrap_err().to_string();
                assert!(error.contains("mutually exclusive"));
            },
        );
    }

    #[test]
    fn test_config_url_overrides_tracked() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                (
                    "VLLM_IMAGES_URL",
                    "http://image-service/v1/images/generations",
                ),
                ("VLLM_RERANK_URL", "http://rerank-service/v1/rerank"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.images_url_override.as_deref(),
                    Some("http://image-service/v1/images/generations")
                );
                assert_eq!(
                    config.rerank_url_override.as_deref(),
                    Some("http://rerank-service/v1/rerank")
                );
                assert!(config.score_url_override.is_none());
            },
        );
    }

    #[test]
    fn test_config_custom_base_url() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("VLLM_BASE_URL", "http://gpu-server:9000"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.chat_completions_url,
                    "http://gpu-server:9000/v1/chat/completions"
                );
                assert_eq!(config.metrics_url, "http://gpu-server:9000/metrics");
            },
        );
    }

    #[test]
    fn test_config_url_overrides() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                (
                    "VLLM_IMAGES_URL",
                    "http://image-service/v1/images/generations",
                ),
                ("VLLM_RERANK_URL", "http://rerank-service/v1/rerank"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.images_url,
                    "http://image-service/v1/images/generations"
                );
                assert_eq!(config.rerank_url, "http://rerank-service/v1/rerank");
            },
        );
    }

    #[test]
    fn test_config_dev_mode_flags() {
        with_env_vars(
            &[
                ("MODEL_NAME", "model"),
                ("TOKEN", "tok"),
                ("DEV", "1"),
                ("GPU_NO_HW_MODE", "true"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert!(config.dev_mode);
                assert!(config.gpu_no_hw_mode);
            },
        );
    }

    #[test]
    fn test_env_bool_variants() {
        for val in &["1", "true", "yes", "True", "YES"] {
            with_env_vars(&[("_TEST_BOOL", val)], || {
                assert!(env_bool("_TEST_BOOL"), "Expected true for '{val}'");
            });
        }
        for val in &["0", "false", "no", "anything"] {
            with_env_vars(&[("_TEST_BOOL", val)], || {
                assert!(!env_bool("_TEST_BOOL"), "Expected false for '{val}'");
            });
        }
    }

    #[test]
    fn test_env_int_fallback() {
        env::remove_var("_TEST_INT_NONEXISTENT");
        assert_eq!(env_int("_TEST_INT_NONEXISTENT", 42), 42);

        with_env_vars(&[("_TEST_INT_INVALID", "not_a_number")], || {
            assert_eq!(env_int("_TEST_INT_INVALID", 42), 42);
        });

        with_env_vars(&[("_TEST_INT_VALID", "99")], || {
            assert_eq!(env_int("_TEST_INT_VALID", 42), 99);
        });
    }

    #[test]
    fn test_admission_config_parses_and_validates() {
        with_env_vars(
            &[
                ("MODEL_NAME", "m"),
                ("TOKEN", "t"),
                ("VLLM_PROXY_ADMISSION_MAX_INFLIGHT", "48"),
                ("VLLM_PROXY_ADMISSION_START_INFLIGHT", "32"),
                ("VLLM_PROXY_ADMISSION_RAMP_STEP", "8"),
                ("VLLM_PROXY_ADMISSION_RAMP_INTERVAL_SECS", "1800"),
                ("VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS", "30000"),
                ("VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS", "10"),
                ("VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS", "2"),
                ("VLLM_BACKEND_CONNECT_FAILOVER", "1"),
            ],
            || {
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.admission(),
                    Some(crate::admission::AdmissionConfig {
                        max_inflight: 48,
                        start_inflight: 32,
                        ramp_step: 8,
                        ramp_interval: std::time::Duration::from_secs(1800),
                        ttft_p95_max: Some(std::time::Duration::from_secs(30)),
                        backpressure_ttl: std::time::Duration::from_secs(10),
                        retry_after: std::time::Duration::from_secs(2),
                    })
                );
                assert!(config.backend_connect_failover);

                // No TTFT check when the bound is 0; no ramp when start is omitted.
                env::set_var("VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS", "0");
                env::remove_var("VLLM_PROXY_ADMISSION_START_INFLIGHT");
                let config = Config::from_env().unwrap();
                let admission = config.admission().unwrap();
                assert_eq!(admission.ttft_p95_max, None);
                assert_eq!(admission.start_inflight, 48);

                // Validation.
                env::set_var("VLLM_PROXY_ADMISSION_START_INFLIGHT", "64");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("VLLM_PROXY_ADMISSION_START_INFLIGHT"), "{err}");
                env::set_var("VLLM_PROXY_ADMISSION_START_INFLIGHT", "32");
                env::set_var("VLLM_PROXY_ADMISSION_RAMP_STEP", "0");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("VLLM_PROXY_ADMISSION_RAMP_STEP"), "{err}");
                env::set_var("VLLM_PROXY_ADMISSION_RAMP_STEP", "8");
                env::set_var("VLLM_PROXY_ADMISSION_MAX_INFLIGHT", "lots");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("VLLM_PROXY_ADMISSION_MAX_INFLIGHT"), "{err}");
                env::set_var("VLLM_PROXY_ADMISSION_MAX_INFLIGHT", "48");
                env::set_var("VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS", "0");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(
                    err.contains("VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS"),
                    "{err}"
                );
                env::remove_var("VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS");
                env::set_var("VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS", "0");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(
                    err.contains("VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS"),
                    "{err}"
                );
                env::set_var("VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS", "2");
                // Unbudgeted execution modes cannot coexist with admission
                // (each mode is otherwise fully configured, so this is the
                // only reason the config can fail).
                env::set_var("FUSION_ENABLED", "1");
                env::set_var("FUSION_INTERNAL_BEARER_TOKEN", "fusion-secret");
                env::set_var("FUSION_ENDPOINTS_URL", "https://fusion.example/endpoints");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("FUSION_ENABLED"), "{err}");
                env::remove_var("FUSION_ENABLED");
                env::remove_var("FUSION_INTERNAL_BEARER_TOKEN");
                env::remove_var("FUSION_ENDPOINTS_URL");
                env::set_var("WEB_CONTEXT_SEARCH_URL", "https://search.example");
                env::set_var("WEB_CONTEXT_SEARCH_API_KEY", "search-secret");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("WEB_CONTEXT_SEARCH_URL"), "{err}");
                env::remove_var("WEB_CONTEXT_SEARCH_URL");
                env::remove_var("WEB_CONTEXT_SEARCH_API_KEY");
                // Probe URLs pair with the backend URLs one to one.
                env::set_var("VLLM_BACKEND_URLS", "https://a.example,https://b.example");
                env::set_var("VLLM_BACKEND_PROBE_URLS", "http://10.0.0.1:8000/");
                let err = Config::from_env().unwrap_err().to_string();
                assert!(err.contains("VLLM_BACKEND_PROBE_URLS"), "{err}");
                env::set_var(
                    "VLLM_BACKEND_PROBE_URLS",
                    "http://10.0.0.1:8000/, http://10.0.0.2:8000",
                );
                let config = Config::from_env().unwrap();
                assert_eq!(
                    config.backend_probe_urls,
                    vec!["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
                );
                env::remove_var("VLLM_BACKEND_PROBE_URLS");
                env::remove_var("VLLM_BACKEND_URLS");
                env::remove_var("VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS");
            },
        );
    }
}
