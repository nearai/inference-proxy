//! Deprecation notice on direct endpoint responses.
//!
//! A request that authenticates with a Cloud API key (`sk-`) on a model's own
//! `*.completions.near.ai` domain is direct access. Its response carries
//! `Deprecation` (RFC 9745), `Sunset` (RFC 8594) and a `Link` to the
//! migration guide. Nothing else gets them: callers with a config `TOKEN`, a
//! gateway lane (`NON_TEE_DEPLOYMENT`), and OHTTP inner requests, which reach
//! this router over loopback and so carry no model domain.
//!
//! Only headers are added. The body, and so the signed response hash, is
//! unchanged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::Request;
use axum::http::header::{HeaderName, HeaderValue, LINK};
use axum::http::request::Parts;
use axum::middleware::Next;
use axum::response::Response;

use crate::auth::{AuthPath, IngressRouteKind, RequestSource};

/// When direct access was deprecated: 2026-11-20T00:00:00Z, as an RFC 9745
/// date (`@` and Unix seconds).
pub const DEPRECATION: &str = "@1795132800";
/// The last day direct access is supported, 2026-12-06, as an HTTP-date.
pub const SUNSET: &str = "Sun, 06 Dec 2026 23:59:59 GMT";
/// The migration guide.
pub const DEPRECATION_LINK: &str =
    "<https://docs.near.ai/cloud/guides/direct-to-gateway>; rel=\"deprecation\"; type=\"text/html\"";

const DEPRECATION_HEADER: HeaderName = HeaderName::from_static("deprecation");
const SUNSET_HEADER: HeaderName = HeaderName::from_static("sunset");

/// Set by `RequireAuth` for a direct request, read by [`notice_middleware`]
/// on the way out.
#[derive(Clone, Default)]
struct DirectAccess(Arc<AtomicBool>);

/// A Cloud API key on a model domain, outside a gateway lane.
pub fn is_direct_access(source: &RequestSource, non_tee_deployment: bool) -> bool {
    !non_tee_deployment
        && source.auth_path == AuthPath::CloudApiKey
        && matches!(
            source.ingress_route,
            IngressRouteKind::Canonical
                | IngressRouteKind::Indexed
                | IngressRouteKind::Long
                | IngressRouteKind::LongIndexed
        )
}

/// Called by `RequireAuth` once the request has authenticated.
pub(crate) fn mark(parts: &Parts, source: &RequestSource, non_tee_deployment: bool) {
    if !is_direct_access(source, non_tee_deployment) {
        return;
    }
    if let Some(flag) = parts.extensions.get::<DirectAccess>() {
        flag.0.store(true, Ordering::Relaxed);
    }
}

/// Adds the deprecation headers to the response of a request `RequireAuth`
/// marked as direct access.
pub async fn notice_middleware(mut request: Request, next: Next) -> Response {
    let flag = DirectAccess::default();
    request.extensions_mut().insert(flag.clone());
    let mut response = next.run(request).await;
    if flag.0.load(Ordering::Relaxed) {
        let headers = response.headers_mut();
        headers.insert(DEPRECATION_HEADER, HeaderValue::from_static(DEPRECATION));
        headers.insert(SUNSET_HEADER, HeaderValue::from_static(SUNSET));
        headers.append(LINK, HeaderValue::from_static(DEPRECATION_LINK));
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(auth_path: AuthPath, ingress_route: IngressRouteKind) -> RequestSource {
        RequestSource {
            auth_path,
            ingress_route,
            caller: "test",
        }
    }

    #[test]
    fn dates_decode_to_the_announced_days() {
        let seconds: i64 = DEPRECATION.strip_prefix('@').unwrap().parse().unwrap();
        assert_eq!(
            jiff::Timestamp::from_second(seconds).unwrap().to_string(),
            "2026-11-20T00:00:00Z"
        );
        // The RFC 2822 parser also checks the weekday against the date.
        let sunset = jiff::fmt::rfc2822::parse(SUNSET).unwrap();
        assert_eq!(sunset.timestamp().to_string(), "2026-12-06T23:59:59Z");
    }

    #[test]
    fn only_a_cloud_api_key_on_a_model_domain_is_direct() {
        for route in [
            IngressRouteKind::Canonical,
            IngressRouteKind::Indexed,
            IngressRouteKind::Long,
            IngressRouteKind::LongIndexed,
        ] {
            assert!(is_direct_access(
                &source(AuthPath::CloudApiKey, route),
                false
            ));
            assert!(!is_direct_access(
                &source(AuthPath::CloudApiKey, route),
                true
            ));
            assert!(!is_direct_access(
                &source(AuthPath::TrustedConfigToken, route),
                false
            ));
        }
        for route in [IngressRouteKind::Other, IngressRouteKind::Missing] {
            assert!(!is_direct_access(
                &source(AuthPath::CloudApiKey, route),
                false
            ));
        }
    }
}
