use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use subtle::ConstantTimeEq;
use tracing::warn;

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

/// Validate an `sk-` prefixed API key against the cloud API.
async fn check_cloud_api_key(
    http_client: &reqwest::Client,
    cloud_api_url: &str,
    token: &str,
) -> Result<(), AppError> {
    let url = format!("{cloud_api_url}/v1/check_api_key");

    let response = http_client
        .post(&url)
        .header("authorization", format!("Bearer {token}"))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| {
            warn!(error = %e, "Cloud API key check request failed");
            AppError::Unauthorized
        })?;

    match response.status().as_u16() {
        200 => Ok(()),
        402 => Err(AppError::InsufficientCredits),
        429 => Err(AppError::RateLimited),
        status => {
            warn!(status, "Cloud API key check rejected");
            Err(AppError::Unauthorized)
        }
    }
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
                        check_cloud_api_key(&state.http_client, cloud_api_url, token).await?;
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
