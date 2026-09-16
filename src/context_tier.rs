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
//! The estimate mirrors cloud-api's `estimate_input_tokens`
//! (`crates/services/src/completions/mod.rs`): the byte length of each
//! message's text divided by four, at least one, compared against the
//! threshold with cloud-api's `CONTEXT_ROUTE_SAFETY_FACTOR` on top. cloud-api
//! additionally refines the decision with an exact `POST /v1/tokenize` near
//! the boundary; the gateway does not — that is a tokenizer dependency and an
//! extra upstream round trip for a placement that is a preference, not a
//! correctness rule (both tiers run the same engine with the same context
//! length, so the "wrong" tier still answers).

use serde_json::Value;
use tracing::debug;

use crate::backend_pool::BackendPool;

/// cloud-api's `CONTEXT_ROUTE_SAFETY_FACTOR` (default 1.2): bytes/4
/// underestimates code- and CJK-heavy prompts by roughly a quarter.
const SAFETY_FACTOR: f64 = 1.2;

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

/// A request's estimated input size: `text` comes from content bytes and
/// carries the safety factor, `ids` are token ids a `/v1/completions` caller
/// sent, which are exact and need no margin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Estimate {
    text: u64,
    ids: u64,
}

impl Estimate {
    /// cloud-api's count of text: bytes / 4, never zero.
    fn from_text_bytes(bytes: u64) -> Self {
        Self {
            text: (bytes / 4).max(1),
            ids: 0,
        }
    }

    /// Estimated prompt tokens, as logged and measured.
    pub fn tokens(self) -> u64 {
        self.text + self.ids
    }

    /// Strictly above `above_tokens` — with the safety factor on the estimated
    /// part — means the long-context tier.
    fn tier(self, above_tokens: u64) -> ContextTier {
        let routed = (self.text as f64 * SAFETY_FACTOR).ceil() as u64 + self.ids;
        if routed > above_tokens {
            ContextTier::Long
        } else {
            ContextTier::Base
        }
    }
}

/// `/v1/chat/completions`: the text of every message, exactly as cloud-api
/// counts it — a string `content`, or the `text` of each part of an array
/// `content`. Tools, images and other part types are not counted.
pub fn chat_estimate(request: &Value) -> Estimate {
    let bytes = request
        .get("messages")
        .and_then(Value::as_array)
        .map_or(0, |messages| messages.iter().map(message_text_bytes).sum());
    Estimate::from_text_bytes(bytes)
}

fn message_text_bytes(message: &Value) -> u64 {
    match message.get("content") {
        Some(Value::String(text)) => text.len() as u64,
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .map(|text| text.len() as u64)
            .sum(),
        _ => 0,
    }
}

/// `/v1/completions`: a string `prompt` is estimated like chat text; token
/// ids — a flat array, or one array per prompt — are the tokens themselves.
pub fn completion_estimate(request: &Value) -> Estimate {
    match request.get("prompt") {
        Some(Value::String(prompt)) => Estimate::from_text_bytes(prompt.len() as u64),
        Some(Value::Array(items)) => {
            let mut bytes = 0u64;
            let mut ids = 0u64;
            for item in items {
                match item {
                    Value::String(text) => bytes += text.len() as u64,
                    Value::Array(tokens) => ids += tokens.len() as u64,
                    _ => ids += 1,
                }
            }
            if ids == 0 {
                Estimate::from_text_bytes(bytes)
            } else {
                Estimate {
                    text: bytes / 4,
                    ids,
                }
            }
        }
        _ => Estimate::default(),
    }
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
    /// fail-over and the fleet-wide saturation checks — or `None` once that
    /// tier has no healthy backend.
    pub restrict: Option<ContextTier>,
}

/// Decide a request's tier. `None` when the feature is off; nothing is
/// estimated then.
pub fn decide(
    pool: &BackendPool,
    above_tokens: u64,
    estimate: impl FnOnce() -> Estimate,
) -> Option<TierDecision> {
    if above_tokens == 0 {
        return None;
    }
    let estimate = estimate();
    let estimated = estimate.tier(above_tokens);
    let restrict = restriction(pool, estimated);
    metrics::histogram!("request_estimated_prompt_tokens").record(estimate.tokens() as f64);
    if restrict.is_some() {
        metrics::counter!(
            "backend_tier_requests_total",
            "tier" => estimated.as_str(),
            "outcome" => "routed"
        )
        .increment(1);
    }
    debug!(
        estimated_tokens = estimate.tokens(),
        tier = estimated.as_str(),
        fallback = restrict.is_none(),
        "Context tier decided"
    );
    Some(TierDecision {
        estimated,
        restrict,
    })
}

/// The restriction to apply right now: `tier` while it still has a healthy
/// backend, `None` once it has none — an empty tier falls back to the other
/// one rather than refusing, since both run the same engine. Re-resolved
/// whenever the pool may have changed under the request (a connection
/// fail-over takes a host out of the rotation); the fallback is counted here.
pub fn restriction(pool: &BackendPool, tier: ContextTier) -> Option<ContextTier> {
    if pool.healthy_count_in(Some(tier)) > 0 {
        return Some(tier);
    }
    metrics::counter!(
        "backend_tier_requests_total",
        "tier" => tier.as_str(),
        "outcome" => "fallback"
    )
    .increment(1);
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(messages: Value) -> Value {
        json!({"model": "m", "messages": messages})
    }

    #[test]
    fn chat_counts_string_content_and_text_parts_only() {
        // 8 bytes of text → 2 tokens.
        let estimate = chat_estimate(&chat(json!([{"role": "user", "content": "12345678"}])));
        assert_eq!(estimate.tokens(), 2);

        // Array parts: only `text` counts, whatever else the part carries.
        let estimate = chat_estimate(&chat(json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "1234"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "input_audio", "input_audio": {"data": "AAAA"}},
            ]
        }])));
        assert_eq!(estimate.tokens(), 1);

        // Mixed messages sum; tools and other fields are invisible.
        let estimate = chat_estimate(&json!({
            "messages": [
                {"role": "system", "content": "1234"},
                {"role": "user", "content": [{"type": "text", "text": "12345678"}]},
                {"role": "assistant", "content": null},
                {"role": "tool", "tool_calls": [{"function": {"arguments": "123456789012"}}]},
            ],
            "tools": [{"function": {"description": "1234567890123456"}}]
        }));
        assert_eq!(estimate.tokens(), 3);

        // No messages at all still estimates one token (cloud-api's floor).
        assert_eq!(chat_estimate(&json!({"model": "m"})).tokens(), 1);
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
        assert_eq!(estimate.tier(5), ContextTier::Base);
        assert_eq!(estimate.tier(4), ContextTier::Long);

        // One array of ids per prompt sums their lengths.
        let estimate = completion_estimate(&json!({"prompt": [[1, 2, 3], [4, 5]]}));
        assert_eq!(estimate.tokens(), 5);

        // No prompt: nothing to estimate.
        assert_eq!(completion_estimate(&json!({"model": "m"})).tokens(), 0);
    }

    #[test]
    fn the_safety_factor_applies_to_estimated_text_and_the_bound_is_strict() {
        // 400 bytes → 100 tokens → 120 with the factor.
        let estimate = chat_estimate(&chat(json!([{"role": "user", "content": "x".repeat(400)}])));
        assert_eq!(estimate.tokens(), 100);
        assert_eq!(estimate.tier(120), ContextTier::Base, "strictly greater");
        assert_eq!(estimate.tier(119), ContextTier::Long);
        // Without the factor 100 tokens would still be under 110.
        assert_eq!(estimate.tier(110), ContextTier::Long);
    }

    #[test]
    fn the_feature_is_off_at_threshold_zero_and_falls_back_to_a_live_tier() {
        let pool = BackendPool::with_long_context(
            vec!["http://base:8000".to_string()],
            vec!["http://long:8000".to_string()],
        );
        let huge = || Estimate::from_text_bytes(4_000_000);
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
        assert_eq!(decide(&pool, 0, || panic!("not estimated when off")), None);
        assert_eq!(
            decide(&pool, 100_000, huge),
            decided(ContextTier::Long, Some(ContextTier::Long))
        );
        // The long tier is down: place it on the base fleet rather than
        // refuse, but the request stays a long one for the breaker.
        health(1, false);
        assert_eq!(
            decide(&pool, 100_000, huge),
            decided(ContextTier::Long, None)
        );
        // And the other way around.
        health(1, true);
        health(0, false);
        assert_eq!(
            decide(&pool, 100_000, || Estimate::from_text_bytes(8)),
            decided(ContextTier::Base, None)
        );
    }
}
