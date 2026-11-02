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
//! vLLM's `--scheduling-policy priority` runs the *lowest* value first. A
//! proxy in front of vLLM sets `VLLM_PROXY_PRIORITY_LOWER_FIRST`, and the
//! engine then gets the negated value: the header keeps one meaning (higher =
//! more important) on both engines, so the gateway's `-1` and a reseller
//! org's `-2` still run after cloud-api's default 0.
//!
//! Optionally (`VLLM_PROXY_PRIORITY_SIZE_TIERS`, CVM proxy mode only) a size
//! bonus is added on top: `engine = caller * CLASS_SCALE + bonus`, bonus in
//! `0..CLASS_SCALE`, picked from the request's estimated prompt tokens
//! (`context_tier` estimators), so within one caller class shorter prompts
//! run ahead of long prefills and no size can lift a request over a higher
//! class. The `lower_first` negation applies to the combined value. Unset =
//! off = the values above, untouched.
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
/// default. Keeps the metric label bounded to operator-chosen values. With
/// size tiers the engine sees up to `1000 * CLASS_SCALE + MAX_SIZE_BONUS`.
pub const MAX_ABS_PRIORITY: i64 = 1_000;

/// Weight of one caller-class step. Larger than any size bonus, so a bonus
/// never reorders caller classes.
pub const CLASS_SCALE: i64 = 100;

/// Largest size bonus (`CLASS_SCALE - 1`).
pub const MAX_SIZE_BONUS: i64 = CLASS_SCALE - 1;

/// Size tiers: `(inclusive upper bound in estimated tokens, bonus)`, ascending,
/// plus the bonus of the final `*` entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SizeTiers {
    bounded: Vec<(u64, i64)>,
    rest: i64,
}

impl SizeTiers {
    /// Parse `T1:B1,T2:B2,...,*:Bn` (see `VLLM_PROXY_PRIORITY_SIZE_TIERS`).
    pub fn parse(raw: &str) -> Result<Self, String> {
        let mut bounded: Vec<(u64, i64)> = Vec::new();
        let entries: Vec<&str> = raw.split(',').map(str::trim).collect();
        for (index, entry) in entries.iter().enumerate() {
            let (limit, bonus) = entry
                .split_once(':')
                .ok_or_else(|| format!("entry {entry:?} must look like MAX_TOKENS:BONUS"))?;
            let (limit, bonus) = (limit.trim(), bonus.trim());
            let bonus: i64 = bonus
                .parse()
                .map_err(|_| format!("bonus {bonus:?} must be an integer"))?;
            if !(0..=MAX_SIZE_BONUS).contains(&bonus) {
                return Err(format!("bonus {bonus} must be within 0..={MAX_SIZE_BONUS}"));
            }
            if limit == "*" {
                if index + 1 != entries.len() {
                    return Err("`*` must be the last entry".to_string());
                }
                return Ok(Self {
                    bounded,
                    rest: bonus,
                });
            }
            let limit: u64 = limit
                .parse()
                .ok()
                .filter(|limit| *limit > 0)
                .ok_or_else(|| format!("threshold {limit:?} must be a positive integer or `*`"))?;
            if bounded.last().is_some_and(|(last, _)| limit <= *last) {
                return Err(format!("threshold {limit} must be above the previous one"));
            }
            bounded.push((limit, bonus));
        }
        Err("the last entry must be `*:BONUS`".to_string())
    }

    /// Number of tiers, `*` included.
    pub fn len(&self) -> usize {
        self.bounded.len() + 1
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// `(tier index, bonus)` for an estimated prompt size; the bound is inclusive.
    pub fn classify(&self, tokens: u64) -> (usize, i64) {
        match self.bounded.iter().position(|(limit, _)| tokens <= *limit) {
            Some(index) => (index, self.bounded[index].1),
            None => (self.bounded.len(), self.rest),
        }
    }
}

/// Which estimator reads the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Chat,
    Completion,
}

/// Fold the size bonus into the `priority` that [`apply_priority`] wrote.
/// Run after E2EE decryption, so the estimate reads plaintext, and before any
/// branch that forwards the body. `lower_first` must match the one given to
/// `apply_priority`: the value is un-negated, combined, and negated again.
pub fn apply_size_bonus(
    request_json: &mut Value,
    tiers: &SizeTiers,
    kind: BodyKind,
    lower_first: bool,
) {
    let Some(written) = request_json.get("priority").and_then(Value::as_i64) else {
        return;
    };
    let tokens = match kind {
        BodyKind::Chat => crate::context_tier::chat_estimate(request_json),
        BodyKind::Completion => crate::context_tier::completion_estimate(request_json),
    }
    .tokens();
    let (tier, bonus) = tiers.classify(tokens);
    let caller = if lower_first { -written } else { written };
    let combined = caller * CLASS_SCALE + bonus;
    let engine_priority = if lower_first { -combined } else { combined };
    request_json["priority"] = Value::from(engine_priority);
    metrics::counter!("request_priority_size_tier_total", "tier" => tier.to_string()).increment(1);
}

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
/// client-supplied value. `lower_first`: the engine runs the lowest value
/// first (vLLM), so the body gets the negated value. The metric keeps the
/// caller's value either way.
pub fn apply_priority(
    request_json: &mut Value,
    headers: &HeaderMap,
    trusted_caller: bool,
    lower_first: bool,
) {
    let priority = resolve_priority(headers, trusted_caller);
    let engine_priority = if lower_first { -priority } else { priority };
    if let Some(body) = request_json.as_object_mut() {
        body.insert("priority".to_string(), Value::from(engine_priority));
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
        apply_priority(&mut body, &headers(None), false, false);
        assert_eq!(body["priority"], 0);
        let mut body = json!({"model": "m", "priority": 999});
        apply_priority(&mut body, &headers(Some("-1")), true, false);
        assert_eq!(body["priority"], -1);
        let mut body = json!({"model": "m"});
        apply_priority(&mut body, &headers(None), true, false);
        assert_eq!(body["priority"], 0);
    }

    #[test]
    fn lower_first_engine_gets_the_negated_value() {
        // vLLM runs the lowest value first: the gateway's -1 and a reseller's
        // -2 must land behind cloud-api's 0, a +5 org ahead of it.
        for (header, engine) in [(Some("-1"), 1), (Some("-2"), 2), (Some("5"), -5), (None, 0)] {
            let mut body = json!({"model": "m", "priority": -999});
            apply_priority(&mut body, &headers(header), true, true);
            assert_eq!(body["priority"], engine, "{header:?}");
        }
        // Untrusted callers still get the default, whatever they send.
        let mut body = json!({"model": "m", "priority": -999});
        apply_priority(&mut body, &headers(Some("1000")), false, true);
        assert_eq!(body["priority"], 0);
    }

    #[test]
    fn parses_size_tiers() {
        let tiers = SizeTiers::parse(" 16000:10, 65536 : 5 ,*:0 ").unwrap();
        assert_eq!(tiers.len(), 3);
        assert_eq!(tiers.classify(0), (0, 10));
        assert_eq!(tiers.classify(16_000), (0, 10), "inclusive bound");
        assert_eq!(tiers.classify(16_001), (1, 5));
        assert_eq!(tiers.classify(65_536), (1, 5));
        assert_eq!(tiers.classify(65_537), (2, 0));
        assert_eq!(SizeTiers::parse("*:7").unwrap().classify(u64::MAX), (0, 7));
        assert_eq!(SizeTiers::parse("1:99,*:0").unwrap().classify(1), (0, 99));
    }

    #[test]
    fn rejects_malformed_size_tiers() {
        for bad in [
            "",
            "16000:10",
            "16000:10,*",
            "*:0,16000:10",
            "16000:10,*:0,*:0",
            "0:10,*:0",
            "-5:10,*:0",
            "abc:10,*:0",
            "16000:x,*:0",
            "16000:-1,*:0",
            "16000:100,*:0",
            "*:100",
            "16000:10,16000:5,*:0",
            "65536:10,16000:5,*:0",
            "16000:10,,*:0",
            "16000:10,*:0,",
            "1.5:10,*:0",
        ] {
            assert!(SizeTiers::parse(bad).is_err(), "{bad:?}");
        }
    }

    fn tiers() -> SizeTiers {
        SizeTiers::parse("100:10,1000:5,*:0").unwrap()
    }

    fn chat_of(bytes: usize) -> Value {
        json!({"model": "m", "messages": [{"role": "user", "content": "x".repeat(bytes)}]})
    }

    fn sized(mut body: Value, header: Option<&str>, lower_first: bool) -> i64 {
        apply_priority(&mut body, &headers(header), true, lower_first);
        apply_size_bonus(&mut body, &tiers(), BodyKind::Chat, lower_first);
        body["priority"].as_i64().unwrap()
    }

    #[test]
    fn size_bonus_orders_within_a_class_and_never_across_classes() {
        // 4 bytes/token + 4 for the message: 100 bytes → 29, 2000 → 504, 8000 → 2004.
        let (short, mid, long) = (chat_of(100), chat_of(2_000), chat_of(8_000));
        assert_eq!(sized(short.clone(), None, false), 10);
        assert_eq!(sized(mid.clone(), None, false), 5);
        assert_eq!(sized(long.clone(), None, false), 0);
        // Negative classes: -1 → -100 + bonus, still below every class-0 value.
        assert_eq!(sized(short.clone(), Some("-1"), false), -90);
        assert_eq!(sized(long.clone(), Some("-1"), false), -100);
        assert_eq!(sized(short.clone(), Some("-2"), false), -190);
        assert!(sized(short.clone(), Some("-1"), false) < sized(long.clone(), None, false));
        assert_eq!(sized(short.clone(), Some("1"), false), 110);
        assert!(sized(long.clone(), Some("1"), false) > sized(short.clone(), None, false));
    }

    #[test]
    fn size_bonus_with_lower_first_negates_the_combined_value() {
        let (short, long) = (chat_of(100), chat_of(8_000));
        assert_eq!(sized(short.clone(), None, true), -10);
        assert_eq!(sized(long.clone(), None, true), 0);
        assert_eq!(sized(short.clone(), Some("-1"), true), 90);
        assert_eq!(sized(long.clone(), Some("-1"), true), 100);
        // Lowest runs first: class 0 short < class 0 long < class -1 short.
        assert!(sized(short.clone(), None, true) < sized(long, None, true));
        assert!(sized(chat_of(8_000), None, true) < sized(short, Some("-1"), true));
    }

    #[test]
    fn size_bonus_discards_client_priority_and_counts_the_tier() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let mut body = chat_of(100);
        body["priority"] = json!(999_999);
        let got = metrics::with_local_recorder(&recorder, || sized(body, None, false));
        assert_eq!(got, 10);
        let rendered = handle.render();
        assert!(
            rendered.contains("request_priority_size_tier_total{tier=\"0\"} 1"),
            "{rendered}"
        );
        assert!(rendered.contains("request_priority_total{priority=\"0\"} 1"));
    }

    #[test]
    fn size_estimate_covers_chat_completions_and_odd_bodies() {
        let t = tiers();
        let mut chat = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "x".repeat(4_000)},
            {"type": "image_url", "image_url": {"url": "data:"}},
        ]}], "priority": 0});
        apply_size_bonus(&mut chat, &t, BodyKind::Chat, false);
        assert_eq!(
            chat["priority"], 0,
            "1000 + 1024 + 4 tokens is the last tier"
        );

        let mut comp = json!({"prompt": ["x".repeat(200), "y".repeat(204)], "priority": 0});
        apply_size_bonus(&mut comp, &t, BodyKind::Completion, false);
        assert_eq!(comp["priority"], 5, "101 tokens is past the first bound");
        let mut ids = json!({"prompt": [1, 2, 3], "priority": 0});
        apply_size_bonus(&mut ids, &t, BodyKind::Completion, false);
        assert_eq!(ids["priority"], 10);

        // Odd shapes never panic and count as empty.
        for odd in [
            json!({"priority": 0}),
            json!({"priority": 0, "messages": "nope"}),
            json!({"priority": 0, "messages": [null, 1, {"content": 5}]}),
            json!({"priority": 0, "prompt": {"a": 1}}),
        ] {
            for kind in [BodyKind::Chat, BodyKind::Completion] {
                let mut body = odd.clone();
                apply_size_bonus(&mut body, &t, kind, false);
                assert!(body["priority"].as_i64().unwrap() >= 0);
            }
        }
        // Without a priority (non-object body) nothing is written.
        let mut not_object = json!([1, 2]);
        apply_size_bonus(&mut not_object, &t, BodyKind::Chat, false);
        assert_eq!(not_object, json!([1, 2]));
    }
}
