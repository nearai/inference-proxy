use std::env;

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_int(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str) -> bool {
    env::var(name)
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub struct Config {
    pub model_name: String,
    pub token: String,

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

    // Request size limits
    pub max_request_size: usize,
    pub max_image_request_size: usize,
    pub max_audio_request_size: usize,

    // Cache
    pub chat_cache_expiration_secs: u64,

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
    pub timeout_tokenize_secs: u64,

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
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let model_name = env::var("MODEL_NAME")
            .map_err(|_| anyhow::anyhow!("MODEL_NAME environment variable is required"))?;
        let token = env::var("TOKEN")
            .map_err(|_| anyhow::anyhow!("TOKEN environment variable is required"))?;
        if token.is_empty() {
            anyhow::bail!("TOKEN must not be empty");
        }

        let vllm_base_url = env_or("VLLM_BASE_URL", "http://localhost:8000");
        let base = vllm_base_url.trim_end_matches('/');

        let images_url = env_or("VLLM_IMAGES_URL", &format!("{base}/v1/images/generations"));
        let images_edits_url = env_or("VLLM_IMAGES_EDITS_URL", &format!("{base}/v1/images/edits"));
        let transcriptions_url = env_or(
            "VLLM_TRANSCRIPTIONS_URL",
            &format!("{base}/v1/audio/transcriptions"),
        );
        let rerank_url = env_or("VLLM_RERANK_URL", &format!("{base}/v1/rerank"));
        let score_url = env_or("VLLM_SCORE_URL", &format!("{base}/v1/score"));

        let git_rev = std::fs::read_to_string("/etc/.GIT_REV")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let config = Config {
            model_name,
            token,
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
            tls_cert_path: env::var("TLS_CERT_PATH")
                .ok()
                .filter(|s| !s.is_empty())
                .filter(|s| std::path::Path::new(s).exists()),
            max_keepalive: env_int("VLLM_PROXY_MAX_KEEPALIVE", 100),
            max_request_size: env_int("VLLM_PROXY_MAX_REQUEST_SIZE", 10 * 1024 * 1024),
            max_image_request_size: env_int("VLLM_PROXY_MAX_IMAGE_REQUEST_SIZE", 50 * 1024 * 1024),
            max_audio_request_size: env_int("VLLM_PROXY_MAX_AUDIO_REQUEST_SIZE", 100 * 1024 * 1024),
            chat_cache_expiration_secs: env_int("CHAT_CACHE_EXPIRATION", 1200) as u64,
            dev_mode: env_bool("DEV"),
            gpu_no_hw_mode: env_bool("GPU_NO_HW_MODE"),
            git_rev,
            rate_limit_per_second: env_int("RATE_LIMIT_PER_SECOND", 100) as u64,
            rate_limit_burst_size: env_int("RATE_LIMIT_BURST_SIZE", 200) as u32,
            rate_limit_trust_proxy_headers: !env_bool("RATE_LIMIT_NO_TRUST_PROXY"),
            timeout_secs: 600,
            timeout_tokenize_secs: 10,
            openai_chat_compatibility_check_enabled: env_bool("OPENAI_CHAT_COMPATIBILITY_CHECK"),
            startup_check_retries: env_int("STARTUP_CHECK_RETRIES", 3),
            startup_check_retry_delay_secs: env_int("STARTUP_CHECK_RETRY_DELAY_SECS", 5) as u64,
            startup_check_timeout_secs: env_int("STARTUP_CHECK_TIMEOUT_SECS", 30) as u64,
        };

        // Validate startup check configuration
        if config.startup_check_retries == 0 {
            anyhow::bail!("STARTUP_CHECK_RETRIES must be at least 1");
        }
        if config.startup_check_timeout_secs == 0 {
            anyhow::bail!("STARTUP_CHECK_TIMEOUT_SECS must be greater than 0");
        }

        Ok(config)
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
                .contains("must not be empty"));
        });
    }

    #[test]
    fn test_config_default_values() {
        with_env_vars(&[("MODEL_NAME", "my-model"), ("TOKEN", "secret")], || {
            // Remove optional vars to test defaults
            env::remove_var("VLLM_BASE_URL");
            env::remove_var("DEV");
            env::remove_var("GPU_NO_HW_MODE");
            env::remove_var("CHAT_CACHE_EXPIRATION");

            let config = Config::from_env().unwrap();

            assert_eq!(config.model_name, "my-model");
            assert_eq!(config.token, "secret");
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
        });
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
