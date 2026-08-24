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
    /// Maximum idle time between upstream SSE chunks. Zero disables the
    /// watchdog. This is separate from `timeout_secs`, which bounds the total
    /// reqwest request lifetime rather than token cadence.
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
            health_check_interval_secs: env_int("HEALTH_CHECK_INTERVAL_SECS", 5) as u64,
            health_check_max_failures: env_int("HEALTH_CHECK_MAX_FAILURES", 3) as u32,
            health_check_timeout_secs: env_int("HEALTH_CHECK_TIMEOUT_SECS", 3) as u64,
            ohttp_enabled: env_bool("OHTTP_ENABLED"),
            listen_port,
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

        Ok(config)
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
}
