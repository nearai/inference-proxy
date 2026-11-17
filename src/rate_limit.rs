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

/// Shared state for the rate-limiting middleware.
#[derive(Clone)]
pub struct RateLimitState {
    pub limiter: Arc<KeyedRateLimiter>,
    pub trusted_proxies: Arc<[ipnet::IpNet]>,
}

/// Build a rate limiter from config values.
pub fn build_rate_limiter(per_second: u64, burst_size: u32) -> Arc<KeyedRateLimiter> {
    // Safely clamp to u32 range to avoid silent overflow on cast.
    let per_second_u32 = per_second.min(u32::MAX as u64) as u32;
    let quota = Quota::per_second(NonZeroU32::new(per_second_u32).unwrap_or(NonZeroU32::MIN))
        .allow_burst(NonZeroU32::new(burst_size).unwrap_or(NonZeroU32::MIN));
    Arc::new(RateLimiter::dashmap(quota))
}

/// Walk from the socket peer toward the client, accepting one address per trusted hop.
fn client_ip<B>(request: &Request<B>, trusted_proxies: &[ipnet::IpNet]) -> Option<IpAddr> {
    let peer = request
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()?
        .0
        .ip()
        .to_canonical();
    let trusted = |ip: IpAddr| trusted_proxies.iter().any(|net| net.contains(&ip));
    if !trusted(peer) {
        return Some(peer);
    }

    let headers = request.headers();
    let mut current = peer;
    if headers.contains_key("x-forwarded-for") {
        for line in headers.get_all("x-forwarded-for").iter().rev() {
            for item in line.as_bytes().rsplit(|b| *b == b',') {
                if !trusted(current) {
                    return Some(current);
                }
                let Some(ip) = std::str::from_utf8(item.trim_ascii())
                    .ok()
                    .and_then(|v| v.parse::<IpAddr>().ok())
                else {
                    return Some(peer);
                };
                current = ip.to_canonical();
            }
        }
    } else {
        let mut values = headers.get_all("x-real-ip").iter();
        if let (Some(value), None) = (values.next(), values.next()) {
            if let Some(ip) = value
                .to_str()
                .ok()
                .and_then(|v| v.trim().parse::<IpAddr>().ok())
            {
                current = ip.to_canonical();
            }
        }
    }
    Some(current)
}

/// Rate limiting middleware. Returns 429 Too Many Requests when the client
/// exceeds the configured rate.
pub async fn rate_limit_middleware(request: Request<axum::body::Body>, next: Next) -> Response {
    let state = match request.extensions().get::<RateLimitState>() {
        Some(s) => s.clone(),
        None => return next.run(request).await, // No limiter configured
    };

    let ip = client_ip(&request, &state.trusted_proxies)
        .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));

    match state.limiter.check_key(&ip) {
        Ok(_) => next.run(request).await,
        Err(_not_until) => {
            warn!(
                client_ip_present = !ip.is_unspecified(),
                "Rate limit exceeded"
            );
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

    #[test]
    fn test_build_rate_limiter_u64_overflow_clamped() {
        // Should not panic — clamps to u32::MAX instead of overflowing
        let limiter = build_rate_limiter(u64::MAX, 1);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(limiter.check_key(&ip).is_ok());
    }

    #[test]
    fn client_ip_uses_only_verified_hops() {
        let trusted = ["192.0.2.0/24".parse().unwrap()];
        let mut req = Request::builder()
            .header("x-forwarded-for", "2001:db8::1, 198.51.100.9")
            .header("x-forwarded-for", "192.0.2.2")
            .body(())
            .unwrap();
        assert_eq!(client_ip(&req, &trusted), None);
        req.extensions_mut().insert(ConnectInfo(
            "[::ffff:192.0.2.1]:123"
                .parse::<std::net::SocketAddr>()
                .unwrap(),
        ));
        assert_eq!(client_ip(&req, &[]).unwrap().to_string(), "192.0.2.1");
        assert_eq!(
            client_ip(&req, &trusted).unwrap().to_string(),
            "198.51.100.9"
        );
        req.headers_mut()
            .insert("x-forwarded-for", "invalid".parse().unwrap());
        req.headers_mut()
            .insert("x-real-ip", "2001:db8::2".parse().unwrap());
        assert_eq!(client_ip(&req, &trusted).unwrap().to_string(), "192.0.2.1");
        req.headers_mut().remove("x-forwarded-for");
        assert_eq!(
            client_ip(&req, &trusted).unwrap().to_string(),
            "2001:db8::2"
        );
    }

    #[tokio::test]
    async fn trusted_peer_separates_clients_over_tcp() {
        use axum::http::StatusCode;
        use axum::{middleware, routing::get, Extension, Router};
        use std::net::SocketAddr;
        let app = Router::new()
            .route("/", get(|| async { StatusCode::OK }))
            .layer(middleware::from_fn(rate_limit_middleware))
            .layer(Extension(RateLimitState {
                limiter: build_rate_limiter(1, 1),
                trusted_proxies: vec!["127.0.0.1/32".parse().unwrap()].into(),
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        for (chain, expected) in [
            ("198.51.100.1", StatusCode::OK),
            ("203.0.113.1, 198.51.100.1", StatusCode::TOO_MANY_REQUESTS),
            ("198.51.100.2", StatusCode::OK),
        ] {
            assert_eq!(
                client
                    .get(&url)
                    .header("x-forwarded-for", chain)
                    .send()
                    .await
                    .unwrap()
                    .status(),
                expected
            );
        }
        server.abort();
    }
}
