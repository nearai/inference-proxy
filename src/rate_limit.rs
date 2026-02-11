use std::net::IpAddr;
use std::num::NonZeroU32;
use std::sync::Arc;

use axum::extract::ConnectInfo;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use governor::clock::DefaultClock;
use governor::state::keyed::DashMapStateStore;
use governor::{Quota, RateLimiter};
use tracing::warn;

use crate::error::AppError;

/// Keyed rate limiter: one bucket per client IP address.
pub type KeyedRateLimiter = RateLimiter<IpAddr, DashMapStateStore<IpAddr>, DefaultClock>;

/// Build a rate limiter from config values.
pub fn build_rate_limiter(per_second: u64, burst_size: u32) -> Arc<KeyedRateLimiter> {
    let quota = Quota::per_second(NonZeroU32::new(per_second as u32).unwrap_or(NonZeroU32::MIN))
        .allow_burst(NonZeroU32::new(burst_size).unwrap_or(NonZeroU32::MIN));
    Arc::new(RateLimiter::dashmap(quota))
}

/// Extract the client IP from the request.
///
/// Checks `X-Forwarded-For` first (first IP in the chain), then `X-Real-IP`,
/// then falls back to the peer address from `ConnectInfo`.
fn client_ip<B>(request: &Request<B>) -> Option<IpAddr> {
    // X-Forwarded-For: client, proxy1, proxy2
    if let Some(xff) = request.headers().get("x-forwarded-for") {
        if let Ok(val) = xff.to_str() {
            if let Some(first) = val.split(',').next() {
                if let Ok(ip) = first.trim().parse::<IpAddr>() {
                    return Some(ip);
                }
            }
        }
    }

    // X-Real-IP
    if let Some(real_ip) = request.headers().get("x-real-ip") {
        if let Ok(val) = real_ip.to_str() {
            if let Ok(ip) = val.trim().parse::<IpAddr>() {
                return Some(ip);
            }
        }
    }

    // Peer address
    request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip())
}

/// Rate limiting middleware. Returns 429 Too Many Requests when the client
/// exceeds the configured rate.
pub async fn rate_limit_middleware(request: Request<axum::body::Body>, next: Next) -> Response {
    let limiter = match request.extensions().get::<Arc<KeyedRateLimiter>>() {
        Some(l) => l.clone(),
        None => return next.run(request).await, // No limiter configured
    };

    let ip = client_ip(&request).unwrap_or(IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));

    match limiter.check_key(&ip) {
        Ok(_) => next.run(request).await,
        Err(_not_until) => {
            warn!(client_ip = %ip, "Rate limit exceeded");
            AppError::RateLimited.into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_rate_limiter_defaults() {
        let limiter = build_rate_limiter(100, 200);
        // Should allow at least one request
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(limiter.check_key(&ip).is_ok());
    }

    #[test]
    fn test_rate_limiter_burst_then_reject() {
        // Allow 1 per second with burst of 2
        let limiter = build_rate_limiter(1, 2);
        let ip: IpAddr = "10.0.0.1".parse().unwrap();

        // First two should succeed (burst)
        assert!(limiter.check_key(&ip).is_ok(), "1st request should pass");
        assert!(
            limiter.check_key(&ip).is_ok(),
            "2nd request should pass (burst)"
        );

        // Third should be rejected
        assert!(
            limiter.check_key(&ip).is_err(),
            "3rd request should be rate limited"
        );
    }

    #[test]
    fn test_rate_limiter_per_ip_isolation() {
        let limiter = build_rate_limiter(1, 1);
        let ip1: IpAddr = "10.0.0.1".parse().unwrap();
        let ip2: IpAddr = "10.0.0.2".parse().unwrap();

        // Exhaust ip1's quota
        assert!(limiter.check_key(&ip1).is_ok());
        assert!(limiter.check_key(&ip1).is_err());

        // ip2 should still have quota
        assert!(limiter.check_key(&ip2).is_ok());
    }
}
