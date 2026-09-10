use serde_json::Value;

/// Chat counters retain total input; cached input is a subset, not a deduction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_read_tokens: i64,
}

impl ChatUsage {
    pub(crate) fn from_usage(usage: &Value) -> Self {
        Self::from_cumulative_usage(usage, &mut CachedTokens::default())
    }

    pub(crate) fn from_cumulative_usage(usage: &Value, cached: &mut CachedTokens) -> Self {
        let input_tokens = usage
            .get("prompt_tokens")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let output_tokens = usage
            .get("completion_tokens")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        cached.update(usage, input_tokens);
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens: cached.observed().unwrap_or(0),
        }
    }
}

/// Missing or noninteger cache details are not a new cumulative observation.
/// Integer values outside Cloud API's i32 counter range are invalid zeroes.
pub(crate) fn cached_tokens(usage: &Value) -> Option<i64> {
    let value = usage.get("prompt_tokens_details")?.get("cached_tokens")?;
    if !value.is_i64() && !value.is_u64() {
        return None;
    }
    Some(
        value
            .as_i64()
            .and_then(|count| i32::try_from(count).ok())
            .map(i64::from)
            .unwrap_or(0),
    )
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CachedTokens(Option<i64>);

impl CachedTokens {
    pub(crate) fn update(&mut self, usage: &Value, input_tokens: i64) {
        self.0 = cached_tokens(usage)
            .or(self.0)
            .map(|count| count.clamp(0, input_tokens.max(0)));
    }

    pub(crate) const fn observed(&self) -> Option<i64> {
        self.0
    }
}
