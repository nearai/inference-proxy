use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use rand::Rng;
use subtle::ConstantTimeEq;
use tracing::warn;

use crate::config::Config;
use crate::error::AppError;
use crate::AppState;

/// Extractor that validates Bearer token authentication.
/// Use as a handler parameter to require auth on a route.
///
/// If authentication was via a cloud API key (`sk-` prefix),
/// `cloud_api_key` contains the key for downstream usage reporting.
pub struct RequireAuth {
    pub cloud_api_key: Option<String>,
}

/// Constant-time token comparison to prevent timing attacks.
/// Returns true if `a` and `b` are equal, using a fixed-time algorithm
/// that does not short-circuit on the first mismatched byte.
pub(crate) fn token_eq(a: &str, b: &str) -> bool {
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

/// Classify a `reqwest::Error` into a coarse bucket so failure modes are
/// distinguishable in logs and metrics. The raw `Display` is the same string
/// (`"error sending request for url ..."`) for connect/RST/EOF/etc, which is
/// not enough to diagnose intermittent transport failures.
fn classify_reqwest_error(e: &reqwest::Error) -> &'static str {
    if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connect"
    } else if e.is_body() {
        "body"
    } else if e.is_decode() {
        "decode"
    } else if e.is_request() {
        "request"
    } else {
        "other"
    }
}

/// Render the full error chain (`source()` → `source()` → ...) so we don't
/// lose the underlying hyper / rustls / io::Error message that reqwest hides.
fn error_chain(e: &(dyn std::error::Error + 'static)) -> String {
    let mut out = e.to_string();
    let mut src = e.source();
    while let Some(inner) = src {
        out.push_str(" | ");
        out.push_str(&inner.to_string());
        src = inner.source();
    }
    out
}

/// Validate an `sk-` prefixed API key against the cloud API.
///
/// Retries on transport errors and 5xx responses with exponential backoff and
/// full jitter. Terminal responses (200/402/429 and other 4xx) return
/// immediately.
async fn check_cloud_api_key(
    http_client: &reqwest::Client,
    config: &Config,
    cloud_api_url: &str,
    token: &str,
) -> Result<(), AppError> {
    let url = format!("{cloud_api_url}/v1/check_api_key");
    let max_attempts = config.cloud_api_auth_max_attempts.max(1);
    let per_attempt_timeout = std::time::Duration::from_secs(config.cloud_api_auth_timeout_secs);
    let initial_backoff_ms = config.cloud_api_auth_initial_backoff_ms;

    for attempt in 1..=max_attempts {
        let result = http_client
            .post(&url)
            .header("authorization", format!("Bearer {token}"))
            .timeout(per_attempt_timeout)
            .send()
            .await;

        let status = match result {
            Ok(response) => response.status().as_u16(),
            Err(e) => {
                let kind = classify_reqwest_error(&e);
                let chain = error_chain(&e);
                metrics::counter!(
                    "cloud_api_auth_attempts_total",
                    "outcome" => "transport_error",
                    "kind" => kind,
                )
                .increment(1);
                if attempt < max_attempts {
                    let delay = backoff_delay(initial_backoff_ms, attempt);
                    warn!(
                        error = %e,
                        error_chain = %chain,
                        error_kind = kind,
                        is_connect = e.is_connect(),
                        is_timeout = e.is_timeout(),
                        is_request = e.is_request(),
                        is_body = e.is_body(),
                        attempt,
                        max_attempts,
                        retry_in_ms = delay.as_millis() as u64,
                        "Cloud API key check transport error, retrying"
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                warn!(
                    error = %e,
                    error_chain = %chain,
                    error_kind = kind,
                    is_connect = e.is_connect(),
                    is_timeout = e.is_timeout(),
                    is_request = e.is_request(),
                    is_body = e.is_body(),
                    attempt,
                    "Cloud API key check request failed"
                );
                return Err(AppError::Unauthorized);
            }
        };

        match status {
            200 => {
                metrics::counter!("cloud_api_auth_attempts_total", "outcome" => "ok").increment(1);
                return Ok(());
            }
            402 => {
                metrics::counter!(
                    "cloud_api_auth_attempts_total",
                    "outcome" => "insufficient_credits"
                )
                .increment(1);
                return Err(AppError::InsufficientCredits);
            }
            429 => {
                metrics::counter!("cloud_api_auth_attempts_total", "outcome" => "rate_limited")
                    .increment(1);
                return Err(AppError::RateLimited);
            }
            s if (500..600).contains(&s) => {
                metrics::counter!("cloud_api_auth_attempts_total", "outcome" => "upstream_5xx")
                    .increment(1);
                if attempt < max_attempts {
                    let delay = backoff_delay(initial_backoff_ms, attempt);
                    warn!(
                        status = s,
                        attempt,
                        max_attempts,
                        retry_in_ms = delay.as_millis() as u64,
                        "Cloud API key check returned 5xx, retrying"
                    );
                    tokio::time::sleep(delay).await;
                    continue;
                }
                warn!(status = s, attempt, "Cloud API key check failed (5xx)");
                return Err(AppError::Unauthorized);
            }
            s => {
                metrics::counter!("cloud_api_auth_attempts_total", "outcome" => "rejected")
                    .increment(1);
                warn!(status = s, "Cloud API key check rejected");
                return Err(AppError::Unauthorized);
            }
        }
    }

    // Unreachable: the loop above either returns or `continue`s, and the final
    // attempt always returns. Defensive fallback.
    Err(AppError::Unauthorized)
}

/// Full-jitter exponential backoff: random uniform between 0 and
/// `initial_ms * 2^(attempt-1)`, capped at 5s.
///
/// Full jitter (rather than fixed exponential) avoids retry stampedes when
/// many in-flight requests fail simultaneously due to a shared upstream blip.
fn backoff_delay(initial_ms: u64, attempt: usize) -> std::time::Duration {
    if initial_ms == 0 {
        return std::time::Duration::from_millis(0);
    }
    let exp = (attempt as u32).saturating_sub(1).min(10);
    let upper = initial_ms.saturating_mul(1u64 << exp).min(5_000);
    let jitter = rand::rng().random_range(0..=upper);
    std::time::Duration::from_millis(jitter)
}

impl FromRequestParts<AppState> for RequireAuth {
    type Rejection = AppError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let auth_header = parts
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok());

        match auth_header {
            Some(header) if header.starts_with("Bearer ") => {
                let token = &header[7..];
                if token_eq(token, &state.config.token) {
                    return Ok(RequireAuth {
                        cloud_api_key: None,
                    });
                }

                // Fallback: validate sk- tokens against cloud API
                if token.starts_with("sk-") {
                    if let Some(cloud_api_url) = &state.config.cloud_api_url {
                        check_cloud_api_key(
                            &state.http_client,
                            &state.config,
                            cloud_api_url,
                            token,
                        )
                        .await?;
                        return Ok(RequireAuth {
                            cloud_api_key: Some(token.to_string()),
                        });
                    }
                }

                Err(AppError::Unauthorized)
            }
            _ => Err(AppError::Unauthorized),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// Measure the median duration (in nanoseconds) of `iterations` calls to `compare_fn(a, b)`.
    fn median_nanos(
        a: &str,
        b: &str,
        compare_fn: fn(&str, &str) -> bool,
        iterations: usize,
    ) -> u64 {
        let mut durations = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let a = black_box(a);
            let b = black_box(b);
            let start = Instant::now();
            let _ = black_box(compare_fn(a, b));
            durations.push(start.elapsed().as_nanos() as u64);
        }
        durations.sort_unstable();
        durations[durations.len() / 2]
    }

    #[test]
    fn test_error_chain_walks_source_chain() {
        #[derive(Debug)]
        struct Wrap {
            msg: &'static str,
            inner: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
        }
        impl std::fmt::Display for Wrap {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.msg)
            }
        }
        impl std::error::Error for Wrap {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.inner.as_deref().map(|e| e as _)
            }
        }

        let inner = Wrap {
            msg: "peer reset",
            inner: None,
        };
        let middle = Wrap {
            msg: "tls write",
            inner: Some(Box::new(inner)),
        };
        let outer = Wrap {
            msg: "send failed",
            inner: Some(Box::new(middle)),
        };

        let s = error_chain(&outer);
        assert!(s.contains("send failed"), "must include outer: {s}");
        assert!(s.contains("tls write"), "must include middle: {s}");
        assert!(s.contains("peer reset"), "must include innermost: {s}");
        assert_eq!(
            s.matches('|').count(),
            2,
            "two separators for three frames: {s}"
        );
    }

    #[tokio::test]
    async fn test_classify_reqwest_error_connect() {
        // Hitting a closed loopback port produces a connect error.
        let client = reqwest::Client::new();
        let err = client
            .get("http://127.0.0.1:1") // reserved, nothing listens
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .expect_err("must fail to connect");
        assert_eq!(classify_reqwest_error(&err), "connect");
        assert!(err.is_connect());
    }

    #[test]
    fn test_backoff_delay_zero_initial_returns_zero() {
        for attempt in 1..=5 {
            assert_eq!(backoff_delay(0, attempt), std::time::Duration::ZERO);
        }
    }

    #[test]
    fn test_backoff_delay_bounded_by_exponential_cap() {
        // Full jitter draws uniformly in [0, initial * 2^(attempt-1)], capped at 5s.
        // Sample many times to confirm the upper bound holds and that the
        // distribution actually grows with attempt (not always 0).
        let initial = 100u64;
        for attempt in 1..=4 {
            let cap_ms = initial * (1u64 << (attempt as u32 - 1));
            let mut max_seen = 0u64;
            for _ in 0..200 {
                let d = backoff_delay(initial, attempt).as_millis() as u64;
                assert!(
                    d <= cap_ms,
                    "attempt {attempt}: {d}ms exceeds cap {cap_ms}ms"
                );
                max_seen = max_seen.max(d);
            }
            // With 200 draws, the max should land somewhere in the upper half
            // of the window for any non-trivial cap. Use a loose bound to
            // avoid flakiness while still catching a stuck-at-zero regression.
            if cap_ms >= 4 {
                assert!(
                    max_seen > 0,
                    "attempt {attempt}: backoff never produced a non-zero delay (cap {cap_ms}ms)"
                );
            }
        }
    }

    #[test]
    fn test_backoff_delay_overall_cap() {
        // A huge initial value must still saturate at 5s.
        for _ in 0..50 {
            assert!(backoff_delay(10_000, 5).as_millis() as u64 <= 5_000);
        }
    }

    #[test]
    fn test_token_eq_correctness() {
        assert!(token_eq("secret-token-123", "secret-token-123"));
        assert!(!token_eq("secret-token-123", "wrong-token-456"));
        assert!(!token_eq("secret-token-123", "secret-token-124"));
        assert!(!token_eq("short", "short-but-longer"));
        assert!(!token_eq("", "notempty"));
        assert!(token_eq("", ""));
    }

    /// Verifies that `token_eq` (constant-time) takes the same time regardless of
    /// where the mismatch occurs. The ratio of late_mismatch / early_mismatch
    /// should be close to 1.0, indicating no timing leak.
    #[test]
    fn test_constant_time_comparison_no_timing_discrepancy() {
        let secret = "a]9$kL2#mP7!xR4&wQ8*nJ5^tY1+hF3@vB6%cD0".repeat(8);
        let early_mismatch = format!("X{}", &secret[1..]);
        let late_mismatch = format!("{}X", &secret[..secret.len() - 1]);

        let iterations = 50_000;

        // Warm up
        median_nanos(&secret, &early_mismatch, token_eq, 1_000);
        median_nanos(&secret, &late_mismatch, token_eq, 1_000);

        let t_early = median_nanos(&secret, &early_mismatch, token_eq, iterations);
        let t_late = median_nanos(&secret, &late_mismatch, token_eq, iterations);

        let ratio = t_late as f64 / t_early.max(1) as f64;
        eprintln!("Constant-time:  early={t_early}ns  late={t_late}ns  ratio={ratio:.2}");

        // Constant-time comparison should have a ratio very close to 1.0.
        // A real timing leak (using ==) would show ratios of 5–50×.
        // We use a generous threshold to tolerate noise on shared CI runners
        // while still catching real timing side-channels.
        assert!(
            ratio < 2.0,
            "Constant-time comparison should not leak timing (ratio {ratio:.2} >= 2.0)"
        );
    }
}
