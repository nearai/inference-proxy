//! Request priority for the engine's scheduler, decided by the proxy.
//!
//! SGLang's priority scheduling (`--enable-priority-scheduling`) orders the
//! waiting queue by the request's integer `priority` (higher first by default)
//! and, when the queue is full, evicts the lowest-priority queued request in
//! favour of a higher-priority arrival. A request without a `priority` gets the
//! *lowest* possible value, so once scheduling is on every request must be
//! tagged, and a client must never be able to pick its own value.
//!
//! The rule, with no configuration on the CVM side:
//!
//! - Every chat/completions body gets `priority` set by the proxy; whatever the
//!   client sent is discarded.
//! - A caller authenticated with the proxy's own config token (cloud-api, or a
//!   gateway such as the OpenRouter lane) may pick the value through
//!   `X-NearAI-Priority`. A gateway sets it from `VLLM_BACKEND_PRIORITY`.
//! - Everyone else, and trusted callers without the header, get
//!   [`DEFAULT_PRIORITY`] (0). cloud-api never forwards customer headers, so a
//!   customer cannot reach the header through it, and a direct customer's
//!   header is ignored because they authenticate with an `sk-` key.
//!
//! `0` is also vLLM's default value for the field, and SGLang ignores
//! `priority` while its scheduling flag is off (`abort_on_priority_when_disabled`
//! defaults to false), so the proxy can be rolled before the engines.

use axum::http::HeaderMap;
use serde_json::Value;

/// Header a trusted caller uses to set the priority of its requests.
pub const PRIORITY_HEADER: &str = "x-nearai-priority";

/// Priority given to every request that does not carry a trusted header.
pub const DEFAULT_PRIORITY: i64 = 0;

/// Accepted magnitude for header values; anything else falls back to the
/// default. Keeps the metric label bounded to operator-chosen values.
pub const MAX_ABS_PRIORITY: i64 = 1_000;

/// Validate an operator-configured priority (`VLLM_BACKEND_PRIORITY`).
pub fn validate_priority(raw: &str) -> Result<i64, String> {
    let value: i64 = raw
        .trim()
        .parse()
        .map_err(|_| "priority must be an integer".to_string())?;
    if !(-MAX_ABS_PRIORITY..=MAX_ABS_PRIORITY).contains(&value) {
        return Err(format!("priority must be within ±{MAX_ABS_PRIORITY}"));
    }
    Ok(value)
}

/// Priority for a request: the header value when the caller is trusted and
/// sent a valid one, the default otherwise.
pub fn resolve_priority(headers: &HeaderMap, trusted_caller: bool) -> i64 {
    if !trusted_caller {
        return DEFAULT_PRIORITY;
    }
    headers
        .get(PRIORITY_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| validate_priority(v).ok())
        .unwrap_or(DEFAULT_PRIORITY)
}

/// Set `priority` on a chat/completions request body, overwriting any
/// client-supplied value.
pub fn apply_priority(request_json: &mut Value, headers: &HeaderMap, trusted_caller: bool) {
    let priority = resolve_priority(headers, trusted_caller);
    if let Some(body) = request_json.as_object_mut() {
        body.insert("priority".to_string(), Value::from(priority));
    }
    metrics::counter!("request_priority_total", "priority" => priority.to_string()).increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use serde_json::json;

    fn headers(value: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(value) = value {
            h.insert(PRIORITY_HEADER, HeaderValue::from_str(value).unwrap());
        }
        h
    }

    #[test]
    fn validates_operator_values() {
        assert_eq!(validate_priority(" -1 ").unwrap(), -1);
        assert_eq!(validate_priority("1000").unwrap(), 1000);
        for bad in ["", "high", "1.5", "1001", "-1001", "-9223372036854775808"] {
            assert!(validate_priority(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn trusted_header_sets_the_value_everything_else_is_default() {
        assert_eq!(resolve_priority(&headers(Some("-1")), true), -1);
        assert_eq!(resolve_priority(&headers(Some(" 7 ")), true), 7);
        assert_eq!(resolve_priority(&headers(None), true), 0);
        assert_eq!(resolve_priority(&headers(Some("abc")), true), 0);
        assert_eq!(resolve_priority(&headers(Some("5000")), true), 0);
        // An untrusted caller's header is ignored, whichever way it points.
        assert_eq!(resolve_priority(&headers(Some("-1")), false), 0);
        assert_eq!(resolve_priority(&headers(Some("999")), false), 0);
    }

    #[test]
    fn client_supplied_priority_is_overwritten() {
        let mut body = json!({"model": "m", "priority": 999, "messages": []});
        apply_priority(&mut body, &headers(None), false);
        assert_eq!(body["priority"], 0);
        let mut body = json!({"model": "m", "priority": 999});
        apply_priority(&mut body, &headers(Some("-1")), true);
        assert_eq!(body["priority"], -1);
        let mut body = json!({"model": "m"});
        apply_priority(&mut body, &headers(None), true);
        assert_eq!(body["priority"], 0);
    }
}
