//! Long-context tier for gateway mode (`VLLM_BACKEND_LONG_CONTEXT_*`).
//!
//! A model can serve oversized prompts from dedicated hosts registered under a
//! second model-proxy domain, so a 200k-token prefill does not sit in front of
//! the short requests on the base fleet. The hosts are ordinary backends here:
//! their handle URLs are the same digests under the long domain
//! (`https://<model>-long-b<handle>.completions.near.ai`), and the forwarded
//! body is unchanged.
//!
//! cloud-api routes to that tier from the model row's `long_context`
//! providerConfig. The gateway bypasses cloud-api, so it makes the same
//! decision itself: a request whose estimated input is strictly above
//! `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS` is placed on the long-context
//! backends, everything else on the base ones.
//!
//! By default an empty tier is not a hard failure: `restriction` and
//! `recheck_restriction` lift the restriction and the request runs on the
//! other tier instead, since both run the same engine with the same context
//! length. `VLLM_BACKEND_TIER_STRICT` (bool, default off) turns that off for
//! deployments that isolate the tiers on purpose — the OpenRouter lane keeps
//! a 300k-token prefill off the base fleet's short-request hosts, and keeps
//! short requests off the long host when the base fleet is down, rather than
//! trading one kind of head-of-line blocking for the other. Strict mode pins
//! the restriction even once the tier is empty; callers refuse the request
//! instead (`RejectReason::TierUnavailable` in `admission.rs`, or a 503 with
//! `error_type: "tier_unavailable"` when admission is off). Only meaningful
//! with a long-context tier configured; set without one, it is ignored (a
//! startup warning says so).
//!
//! The estimate mirrors the one cloud-api routes with,
//! `inference_provider_pool::context_routing::estimate_input` plus the
//! `required` computation in `inference_provider_pool/mod.rs`:
//!
//! ```text
//! countable = (message text + serialized tool_calls + serialized tools) / 4
//! uncounted = media parts × 1024 + messages × 4
//! required  = ceil(countable × safety_factor) + uncounted
//!             + (if count_output_reserve { output reserve } else { 0 })
//! ```
//!
//! so the safety factor applies to the byte-estimated text only — everything
//! else is already a token count. Tool definitions and tool-call arguments
//! are counted because the lane's dominant shape is agentic, where they are
//! most of the prompt. cloud-api additionally refines the decision with an
//! exact `POST /v1/tokenize` near the boundary; the gateway does not — that
//! is a tokenizer dependency and an extra upstream round trip for a placement
//! that is a preference, not a correctness rule (both tiers run the same
//! engine with the same context length, so the "wrong" tier still answers).
//!
//! `safety_factor` (`VLLM_BACKEND_LONG_CONTEXT_SAFETY_FACTOR`, default `1.2`)
//! stays at cloud-api's own value deliberately: an over-estimate here only
//! means a request prefills on the long tier instead of the base one, and
//! both run the same engine with the same 1M context, so guessing high is
//! cheap and guessing low sends a genuinely oversized prefill in front of
//! short requests on the base fleet.
//!
//! `count_output_reserve` (`VLLM_BACKEND_LONG_CONTEXT_COUNT_OUTPUT_RESERVE`,
//! default `false`) is where the gateway now deliberately diverges from
//! cloud-api. cloud-api counts the reserved output window because there it
//! guards a real per-tier capacity limit; on this lane both tiers are the
//! same engine, so the output window has no bearing on prefill cost, and
//! counting it by default was placing requests with a small prompt but a
//! large `max_tokens` on the long tier for no reason. Measured on 2026-09-25,
//! 58% of gpu02's hourly gateway traffic (873 of 1,508 requests) had under
//! 71k actual input tokens but landed on the long tier by `max_tokens` alone.
//! Set to `1` to reproduce the old behaviour exactly — the rollback, since
//! the safety factor's default is unchanged.
//!
//! Two small differences from cloud-api remain by construction: the gateway
//! serializes the incoming `tool_calls`/`tools` values where cloud-api
//! serializes its own typed structs, and it measures after
//! `tool_calls::normalize_tool_call_arguments` has repaired the history, so
//! byte counts can differ slightly at the boundary — far less than the
//! tokenize refinement the gateway skips anyway.

use serde_json::Value;
use tracing::debug;

use crate::backend_pool::BackendPool;

/// cloud-api's `CONTEXT_ROUTE_MEDIA_PART_TOKENS` (default 1024), the flat cost
/// of a non-text content part: byte-counting base64 media would read a single
/// image as a ~250k-token prompt.
const MEDIA_PART_TOKENS: u64 = 1024;
/// Chat-template overhead cloud-api adds per message.
const MESSAGE_TOKENS: u64 = 4;

/// Which half of the pool a backend belongs to, and which one a request wants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextTier {
    Base,
    Long,
}

impl ContextTier {
    pub fn as_str(self) -> &'static str {
        match self {
            ContextTier::Base => "base",
            ContextTier::Long => "long",
        }
    }
}

/// A request's estimated demand on the context window, decomposed the way
/// cloud-api decomposes it: `text` is byte-estimated and carries the safety
/// factor, `exact` is already a token count (media parts, template overhead,
/// token ids a caller sent), `reserve` is the output window it asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Estimate {
    text: u64,
    exact: u64,
    reserve: u64,
}

impl Estimate {
    /// Estimated prompt tokens, as logged and measured (cloud-api's
    /// pre-factor input estimate: the reserved output is not prompt).
    pub fn tokens(self) -> u64 {
        self.text + self.exact
    }

    /// Strictly above `above_tokens` — with `policy.safety_factor` applied to
    /// the byte-estimated part, and `reserve` counted only when
    /// `policy.count_output_reserve` — means the long-context tier.
    fn tier(self, above_tokens: u64, policy: EstimatePolicy) -> ContextTier {
        let reserve = if policy.count_output_reserve {
            self.reserve
        } else {
            0
        };
        let required =
            (self.text as f64 * policy.safety_factor).ceil() as u64 + self.exact + reserve;
        if required > above_tokens {
            ContextTier::Long
        } else {
            ContextTier::Base
        }
    }
}

/// The two knobs `Estimate::tier` applies on top of the raw `Estimate`. See
/// the module docs for the formula, the defaults and why they diverge from
/// cloud-api.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimatePolicy {
    /// Multiplies `text` before it is added to the demand
    /// (`VLLM_BACKEND_LONG_CONTEXT_SAFETY_FACTOR`).
    pub safety_factor: f64,
    /// Whether `reserve` (the caller's `max_completion_tokens` /
    /// `max_tokens`) counts toward the demand at all
    /// (`VLLM_BACKEND_LONG_CONTEXT_COUNT_OUTPUT_RESERVE`).
    pub count_output_reserve: bool,
}

impl Default for EstimatePolicy {
    /// The gateway's own default: cloud-api's `1.2` safety factor kept
    /// deliberately, the output reserve not counted. See the module docs.
    fn default() -> Self {
        EstimatePolicy {
            safety_factor: 1.2,
            count_output_reserve: false,
        }
    }
}

impl EstimatePolicy {
    /// cloud-api's own policy, and this gateway's behaviour before
    /// `count_output_reserve` existed as a setting. The rollback
    /// (`VLLM_BACKEND_LONG_CONTEXT_COUNT_OUTPUT_RESERVE=1`) reproduces it
    /// exactly, since the default safety factor is already `1.2`.
    pub const LEGACY_CLOUD_API: EstimatePolicy = EstimatePolicy {
        safety_factor: 1.2,
        count_output_reserve: true,
    };
}

/// `/v1/chat/completions`, as cloud-api counts it: the text of every message,
/// the serialized tool calls in its history and the serialized tool
/// definitions; a content part without `text` is media at a flat cost, and
/// every message adds the chat template's overhead.
pub fn chat_estimate(request: &Value) -> Estimate {
    let mut bytes = 0u64;
    let mut media_parts = 0u64;
    let mut messages = 0u64;
    if let Some(list) = request.get("messages").and_then(Value::as_array) {
        messages = list.len() as u64;
        for message in list {
            match message.get("content") {
                Some(Value::String(text)) => bytes += text.len() as u64,
                Some(Value::Array(parts)) => {
                    for part in parts {
                        match part.get("text").and_then(Value::as_str) {
                            Some(text) => bytes += text.len() as u64,
                            None => media_parts += 1,
                        }
                    }
                }
                _ => {}
            }
            bytes += serialized_len(message.get("tool_calls"));
        }
    }
    bytes += serialized_len(request.get("tools"));
    Estimate {
        text: bytes / 4,
        exact: media_parts * MEDIA_PART_TOKENS + messages * MESSAGE_TOKENS,
        reserve: output_reserve(request),
    }
}

/// `/v1/completions`: a string `prompt` is estimated like chat text; token
/// ids — a flat array, or one array per prompt — are the tokens themselves.
pub fn completion_estimate(request: &Value) -> Estimate {
    let mut bytes = 0u64;
    let mut ids = 0u64;
    match request.get("prompt") {
        Some(Value::String(prompt)) => bytes += prompt.len() as u64,
        Some(Value::Array(items)) => {
            for item in items {
                match item {
                    Value::String(text) => bytes += text.len() as u64,
                    Value::Array(tokens) => {
                        ids += tokens.iter().filter(|t| is_token_id(t)).count() as u64
                    }
                    token => ids += u64::from(is_token_id(token)),
                }
            }
        }
        _ => {}
    }
    Estimate {
        text: bytes / 4,
        exact: ids,
        reserve: output_reserve(request),
    }
}

/// Serialized length of a tool-call list or tool definition block, which is
/// what the chat template renders into the prompt. Absent or null: nothing.
fn serialized_len(value: Option<&Value>) -> u64 {
    value
        .filter(|value| !value.is_null())
        .and_then(|value| serde_json::to_string(value).ok())
        .map_or(0, |text| text.len() as u64)
}

fn is_token_id(value: &Value) -> bool {
    value.as_i64().is_some() || value.as_u64().is_some()
}

/// The output window the caller reserved; cloud-api adds it to the demand
/// before comparing against a tier's capacity.
fn output_reserve(request: &Value) -> u64 {
    ["max_completion_tokens", "max_tokens"]
        .iter()
        .find_map(|field| request.get(field).and_then(Value::as_i64))
        .unwrap_or(0)
        .max(0) as u64
}

/// What the tier decision produced for one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierDecision {
    /// The tier the request's estimated size asks for. Fixed for the whole
    /// request: admission keeps a long request's wait out of the lane's
    /// time-to-first-generation window wherever it ends up running, because
    /// the prefill takes tens of seconds on either tier.
    pub estimated: ContextTier,
    /// The tier candidate selection is restricted to — placement, connection
    /// fail-over and the fleet-wide saturation checks. `None` once that tier
    /// has no healthy backend and restriction lifts (non-strict); in strict
    /// mode this is always `Some(estimated)`, even on an empty tier, and the
    /// caller refuses the request instead of widening the search.
    pub restrict: Option<ContextTier>,
}

/// Decide a request's tier. `None` when the feature is off; nothing is
/// estimated then. `strict` is `Config::backend_tier_strict`; `policy` is
/// `Config::context_tier_policy()`.
pub fn decide(
    pool: &BackendPool,
    above_tokens: u64,
    strict: bool,
    policy: EstimatePolicy,
    estimate: impl FnOnce() -> Estimate,
) -> Option<TierDecision> {
    if above_tokens == 0 {
        return None;
    }
    let estimate = estimate();
    let estimated = estimate.tier(above_tokens, policy);
    let restrict = restriction(pool, estimated, strict);
    // `restrict` alone cannot tell empty from full any more once strict mode
    // pins it either way, so the metric is computed straight off the pool.
    let empty = pool.healthy_count_in(Some(estimated)) == 0;
    let outcome = match (empty, strict) {
        (false, _) => "routed",
        (true, true) => "refused",
        (true, false) => "fallback",
    };
    metrics::histogram!("request_estimated_prompt_tokens").record(estimate.tokens() as f64);
    metrics::counter!(
        "backend_tier_requests_total",
        "tier" => estimated.as_str(),
        "outcome" => outcome
    )
    .increment(1);
    debug!(
        estimated_tokens = estimate.tokens(),
        tier = estimated.as_str(),
        outcome,
        "Context tier decided"
    );
    Some(TierDecision {
        estimated,
        restrict,
    })
}

/// The restriction to apply: `tier` while it still has a healthy backend,
/// `None` once it has none — an empty tier falls back to the other one rather
/// than refusing, since both run the same engine. In strict mode the
/// restriction never lifts: always `Some(tier)`, whether or not it currently
/// has a healthy backend, so an empty tier is a refusal rather than a spill
/// onto the other one (see the module docs).
pub fn restriction(pool: &BackendPool, tier: ContextTier, strict: bool) -> Option<ContextTier> {
    if strict {
        return Some(tier);
    }
    (pool.healthy_count_in(Some(tier)) > 0).then_some(tier)
}

/// The same, re-resolved after the pool may have changed under a request that
/// was already routed: a connection fail-over marks a host unreachable, or a
/// placement loses the race with one. Non-strict: a tier that emptied in the
/// meantime is counted as `fallback_late`, apart from the one outcome
/// `decide` records per request, and the restriction lifts. Strict: the
/// restriction never lifts, so `restrict` stays `Some` even on an empty tier
/// — counted `refused_late` instead — and the caller (placement, connection
/// fail-over) is left to refuse once it finds the pinned tier has no
/// eligible backend.
pub fn recheck_restriction(
    pool: &BackendPool,
    tier: ContextTier,
    strict: bool,
) -> Option<ContextTier> {
    let restrict = restriction(pool, tier, strict);
    match restrict {
        None => {
            metrics::counter!(
                "backend_tier_requests_total",
                "tier" => tier.as_str(),
                "outcome" => "fallback_late"
            )
            .increment(1);
        }
        Some(_) if strict && pool.healthy_count_in(Some(tier)) == 0 => {
            metrics::counter!(
                "backend_tier_requests_total",
                "tier" => tier.as_str(),
                "outcome" => "refused_late"
            )
            .increment(1);
        }
        Some(_) => {}
    }
    restrict
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(messages: Value) -> Value {
        json!({"model": "m", "messages": messages})
    }

    #[test]
    fn chat_counts_text_tool_calls_and_tools_with_media_and_template_cost() {
        // 8 bytes of text → 2 tokens, plus the template's 4 per message.
        let estimate = chat_estimate(&chat(json!([{"role": "user", "content": "12345678"}])));
        assert_eq!(estimate.tokens(), 2 + MESSAGE_TOKENS);

        // Array parts: `text` counts by bytes, every other part is media at a
        // flat cost (byte-counting base64 would read one image as ~250k).
        let estimate = chat_estimate(&chat(json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "1234"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "input_audio", "input_audio": {"data": "AAAA"}},
            ]
        }])));
        assert_eq!(
            estimate.tokens(),
            1 + 2 * MEDIA_PART_TOKENS + MESSAGE_TOKENS
        );

        // Tool definitions and the tool calls in the history occupy the window
        // as the JSON the template renders: `[{"f":"1234"}]` is 14 bytes and
        // `[{"n":"12345678901234"}]` is 24. Agentic bodies are mostly this.
        let estimate = chat_estimate(&json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"f": "1234"}]},
                {"role": "tool", "content": "1234"},
            ],
            "tools": [{"n": "12345678901234"}]
        }));
        assert_eq!(estimate.tokens(), (14 + 4 + 24) / 4 + 2 * MESSAGE_TOKENS);

        // No messages at all: nothing to count.
        assert_eq!(chat_estimate(&json!({"model": "m"})).tokens(), 0);
    }

    #[test]
    fn completions_count_strings_by_bytes_and_token_ids_exactly() {
        let estimate = completion_estimate(&json!({"prompt": "12345678"}));
        assert_eq!(estimate.tokens(), 2);

        // An array of strings sums their bytes.
        let estimate = completion_estimate(&json!({"prompt": ["1234", "12345678"]}));
        assert_eq!(estimate.tokens(), 3);

        // A flat array of token ids is exact, and carries no safety factor.
        let estimate = completion_estimate(&json!({"prompt": [1, 2, 3, 4, 5]}));
        assert_eq!(estimate.tokens(), 5);
        assert_eq!(
            estimate.tier(5, EstimatePolicy::default()),
            ContextTier::Base
        );
        assert_eq!(
            estimate.tier(4, EstimatePolicy::default()),
            ContextTier::Long
        );

        // One array of ids per prompt sums their lengths; anything that is not
        // an integer is not a token.
        let estimate = completion_estimate(&json!({"prompt": [[1, 2, 3], [4, 5]]}));
        assert_eq!(estimate.tokens(), 5);
        let estimate = completion_estimate(&json!({"prompt": [1, null, {"a": 1}, 2.5, 2]}));
        assert_eq!(estimate.tokens(), 2);

        // No prompt: nothing to estimate.
        assert_eq!(completion_estimate(&json!({"model": "m"})).tokens(), 0);
    }

    #[test]
    fn the_safety_factor_applies_to_estimated_text_and_the_bound_is_strict() {
        // Under the legacy (cloud-api) policy, which today's default
        // reproduces exactly: 400 bytes → 100 tokens → 120 with the factor,
        // plus 4 for the message.
        let estimate = chat_estimate(&chat(json!([{"role": "user", "content": "x".repeat(400)}])));
        assert_eq!(estimate.tokens(), 100 + MESSAGE_TOKENS);
        assert_eq!(
            estimate.tier(124, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Base,
            "strictly greater"
        );
        assert_eq!(
            estimate.tier(123, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Long
        );
        // Without the factor the demand would be 104 and this would be Base.
        assert_eq!(
            estimate.tier(110, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Long
        );
    }

    #[test]
    fn the_reserved_output_window_counts_toward_the_decision_only_under_the_legacy_policy() {
        let body = |field: &str, value: Value| json!({"messages": [{"role": "user", "content": "12345678"}], field.to_string(): value});
        // Text 2 → 3 with the factor, plus 4 for the message: 4000 more of
        // reserved output decides the tier without being prompt — but only
        // under the legacy policy, which counts it; the default does not
        // (see `default_policy_does_not_count_the_reserved_output_window`).
        let estimate = chat_estimate(&body("max_tokens", json!(4_000)));
        assert_eq!(estimate.tokens(), 2 + MESSAGE_TOKENS);
        assert_eq!(
            estimate.tier(4_007, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Base
        );
        assert_eq!(
            estimate.tier(4_006, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Long
        );
        // `max_completion_tokens` wins over `max_tokens`, and a negative or
        // null value reserves nothing.
        let mut both = body("max_tokens", json!(4_000));
        both["max_completion_tokens"] = json!(8);
        assert_eq!(
            chat_estimate(&both).tier(15, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Base
        );
        assert_eq!(
            chat_estimate(&body("max_tokens", json!(-5))).tier(7, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Base
        );
        assert_eq!(
            completion_estimate(&json!({"prompt": [1, 2, 3], "max_tokens": 10}))
                .tier(13, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Base
        );
    }

    #[test]
    fn default_policy_does_not_count_the_reserved_output_window() {
        // ~500 tokens of prompt text (2000 bytes) with a huge reserve: the
        // default policy (no output reserve counted) stays on the base tier;
        // the legacy cloud-api policy counts the reserve and the same request
        // crosses onto the long one.
        let body = json!({
            "messages": [{"role": "user", "content": "x".repeat(2_000)}],
            "max_tokens": 131_072,
        });
        let estimate = chat_estimate(&body);
        assert_eq!(estimate.tokens(), 500 + MESSAGE_TOKENS);
        assert_eq!(
            estimate.tier(100_000, EstimatePolicy::default()),
            ContextTier::Base
        );
        assert_eq!(
            estimate.tier(100_000, EstimatePolicy::LEGACY_CLOUD_API),
            ContextTier::Long
        );
    }

    #[test]
    fn default_policy_still_applies_the_safety_factor_to_the_byte_estimate() {
        // The default keeps cloud-api's 1.2 factor: bytes/4 = 90,000 inflates
        // to 108,000, above the 100k threshold, while 83,000 inflates to
        // 99,600, just under it. Only the output reserve stopped counting by
        // default — not the safety factor.
        let estimate = |tokens: u64| {
            chat_estimate(&chat(json!([{
                "role": "user",
                "content": "x".repeat((tokens * 4) as usize)
            }])))
        };
        assert_eq!(
            estimate(90_000).tier(100_000, EstimatePolicy::default()),
            ContextTier::Long
        );
        assert_eq!(
            estimate(83_000).tier(100_000, EstimatePolicy::default()),
            ContextTier::Base
        );
    }

    #[test]
    fn default_policy_bound_stays_strict_above_the_threshold() {
        // bytes/4 = 100,001: one token over the threshold before the safety
        // factor is even applied is still the long tier under the default
        // policy.
        let estimate = chat_estimate(&chat(json!([{
            "role": "user",
            "content": "x".repeat(400_004)
        }])));
        assert_eq!(estimate.tokens(), 100_001 + MESSAGE_TOKENS);
        assert_eq!(
            estimate.tier(100_000, EstimatePolicy::default()),
            ContextTier::Long
        );
    }

    #[test]
    fn media_and_message_overhead_count_under_either_policy() {
        // Media parts and the chat-template's per-message overhead are
        // `exact`, not `text`: neither the safety factor nor the
        // output-reserve flag ever touches them, under either policy.
        let estimate = chat_estimate(&chat(json!([{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        }])));
        let threshold = MEDIA_PART_TOKENS + MESSAGE_TOKENS;
        assert_eq!(estimate.tokens(), threshold);
        for policy in [EstimatePolicy::default(), EstimatePolicy::LEGACY_CLOUD_API] {
            assert_eq!(estimate.tier(threshold, policy), ContextTier::Base);
            assert_eq!(estimate.tier(threshold - 1, policy), ContextTier::Long);
        }
    }

    #[test]
    fn the_empty_tier_fallback_is_counted() {
        let pool = BackendPool::with_long_context(
            vec!["http://base:8000".to_string()],
            vec!["http://long:8000".to_string()],
        );
        pool.backends()[1]
            .healthy
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            decide(&pool, 100_000, false, EstimatePolicy::default(), || {
                Estimate {
                    text: 1_000_000,
                    ..Estimate::default()
                }
            })
        });
        let rendered = handle.render();
        assert!(
            rendered.contains("backend_tier_requests_total{tier=\"long\",outcome=\"fallback\"} 1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("request_estimated_prompt_tokens_count 1"),
            "{rendered}"
        );
    }

    #[test]
    fn the_feature_is_off_at_threshold_zero_and_falls_back_to_a_live_tier() {
        let pool = BackendPool::with_long_context(
            vec!["http://base:8000".to_string()],
            vec!["http://long:8000".to_string()],
        );
        let huge = || Estimate {
            text: 1_000_000,
            ..Estimate::default()
        };
        let health = |index: usize, healthy: bool| {
            pool.backends()[index]
                .healthy
                .store(healthy, std::sync::atomic::Ordering::Relaxed)
        };
        let decided = |estimated, restrict| {
            Some(TierDecision {
                estimated,
                restrict,
            })
        };
        assert_eq!(
            decide(&pool, 0, false, EstimatePolicy::default(), || panic!(
                "not estimated when off"
            )),
            None
        );
        assert_eq!(
            decide(&pool, 100_000, false, EstimatePolicy::default(), huge),
            decided(ContextTier::Long, Some(ContextTier::Long))
        );
        // The long tier is down: place it on the base fleet rather than
        // refuse, but the request stays a long one for the breaker.
        health(1, false);
        assert_eq!(
            decide(&pool, 100_000, false, EstimatePolicy::default(), huge),
            decided(ContextTier::Long, None)
        );
        // And the other way around.
        health(1, true);
        health(0, false);
        assert_eq!(
            decide(
                &pool,
                100_000,
                false,
                EstimatePolicy::default(),
                Estimate::default
            ),
            decided(ContextTier::Base, None)
        );
    }

    #[test]
    fn strict_mode_never_lifts_the_restriction_and_counts_refused() {
        let pool = BackendPool::with_long_context(
            vec!["http://base:8000".to_string()],
            vec!["http://long:8000".to_string()],
        );
        pool.backends()[1]
            .healthy
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let huge = || Estimate {
            text: 1_000_000,
            ..Estimate::default()
        };
        // Non-strict would lift this to `None` (see `the_empty_tier_fallback_is_counted`);
        // strict keeps the request pinned to its empty tier so the caller
        // refuses it instead of spilling onto the base fleet.
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let decision = metrics::with_local_recorder(&recorder, || {
            decide(&pool, 100_000, true, EstimatePolicy::default(), huge)
        });
        assert_eq!(
            decision,
            Some(TierDecision {
                estimated: ContextTier::Long,
                restrict: Some(ContextTier::Long),
            })
        );
        let rendered = handle.render();
        assert!(
            rendered.contains("backend_tier_requests_total{tier=\"long\",outcome=\"refused\"} 1"),
            "{rendered}"
        );
        // A tier that does have a healthy backend is unaffected by strict mode.
        pool.backends()[1]
            .healthy
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            decide(&pool, 100_000, true, EstimatePolicy::default(), huge),
            Some(TierDecision {
                estimated: ContextTier::Long,
                restrict: Some(ContextTier::Long),
            })
        );
        // `restriction`/`recheck_restriction` never lift in strict mode, even
        // for a tier that never had a backend at all.
        pool.backends()[1]
            .healthy
            .store(false, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            restriction(&pool, ContextTier::Long, true),
            Some(ContextTier::Long)
        );
        assert_eq!(
            recheck_restriction(&pool, ContextTier::Long, true),
            Some(ContextTier::Long)
        );
        assert_eq!(restriction(&pool, ContextTier::Long, false), None);
    }

    #[test]
    fn strict_mode_recheck_restriction_counts_refused_late_while_the_tier_stays_empty() {
        // Mirrors `fallback_late` (non-strict): a request already routed to a
        // tier that then emptied under it. Strict mode's `recheck_restriction`
        // still returns `Some(tier)` (never lifts), so the empty-vs-full
        // distinction has to come from a separate metric rather than the
        // return value.
        let pool = BackendPool::with_long_context(
            vec!["http://base:8000".to_string()],
            vec!["http://long:8000".to_string()],
        );
        pool.backends()[1]
            .healthy
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let restrict = metrics::with_local_recorder(&recorder, || {
            recheck_restriction(&pool, ContextTier::Long, true)
        });
        assert_eq!(restrict, Some(ContextTier::Long));
        let rendered = handle.render();
        assert!(
            rendered
                .contains("backend_tier_requests_total{tier=\"long\",outcome=\"refused_late\"} 1"),
            "{rendered}"
        );
        // A tier that still has a healthy backend records nothing: the
        // metric only fires when the pinned tier is actually empty.
        pool.backends()[1]
            .healthy
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            recheck_restriction(&pool, ContextTier::Long, true)
        });
        assert!(
            !handle.render().contains("refused_late"),
            "{}",
            handle.render()
        );
    }
}
