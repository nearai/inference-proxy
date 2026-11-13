//! Request field guard: bounds client request fields before dispatch.
//!
//! Sampling fields, image fields and media URLs are checked against limits,
//! and engine-internal keys are removed from client bodies.
//!
//! Two actions:
//! - **Reject** (400 `invalid_request_error`, `param` set): a value outside
//!   the bounds a well-behaved client never exceeds. Refused, not clamped, so
//!   the caller learns the request was not served as sent.
//! - **Strip**: engine-internal keys a client has no business sending are
//!   removed from the top level of the body (or dropped as multipart form
//!   fields), silently, like the client `priority`.
//!
//! On by default (`VLLM_PROXY_REQUEST_GUARD=0` turns all of it off, including
//! the `/v1/images/generations` key list). Every
//! numeric limit has its own env var; `0` turns that one rule off. Error
//! messages never echo the caller's values.

use serde_json::{Map, Value};

use crate::error::AppError;

/// Engine-internal top-level keys removed from client bodies by default
/// (`VLLM_PROXY_STRIPPED_REQUEST_FIELDS` replaces the list; empty = none).
pub const DEFAULT_STRIPPED_FIELDS: &[&str] = &[
    // vLLM: KV and encoder-cache transfer metadata and the `extra_args` map.
    "kv_transfer_params",
    "ec_transfer_params",
    "vllm_xargs",
    // SGLang: native-only session, cache-key and debug fields.
    "session_params",
    "mm_hashes",
    "return_hidden_states",
    "custom_logit_processor",
];

/// Image request fields with bounds (`check_image_field`).
pub const IMAGE_BOUNDED_FIELDS: &[&str] = &["n", "size", "num_inference_steps", "guidance_scale"];

/// Media URL schemes the engine may receive. Everything else (`file:`, a bare
/// path, `ftp:`, ...) is refused.
const ALLOWED_MEDIA_SCHEMES: &[&str] = &["http", "https", "data"];

/// Content-part keys that carry a media URL.
const MEDIA_PART_FIELDS: &[&str] = &["image_url", "video_url", "audio_url", "input_image"];

/// Sampling floats that must be finite numbers. JSON has no NaN/Infinity, but
/// the engines' pydantic models coerce strings such as `"nan"` and `"inf"`.
const FINITE_FLOAT_FIELDS: &[&str] = &[
    "temperature",
    "top_p",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
];

/// Upper bound for token ids when the vocabulary size is not configured.
const DEFAULT_TOKEN_ID_BOUND: u64 = 1 << 31;

#[derive(Debug, Clone, PartialEq)]
pub struct RequestGuardConfig {
    /// Master switch (`VLLM_PROXY_REQUEST_GUARD`, default on).
    pub enabled: bool,
    /// `top_logprobs`, and `logprobs` when it is a number
    /// (`VLLM_PROXY_MAX_TOP_LOGPROBS`, default 20).
    pub max_top_logprobs: u64,
    /// `n` (`VLLM_PROXY_MAX_N`, default 8).
    pub max_n: u64,
    /// Entries in `logit_bias` (`VLLM_PROXY_MAX_LOGIT_BIAS_ENTRIES`, default 300).
    pub max_logit_bias_entries: usize,
    /// Largest `|logit_bias|` value (`VLLM_PROXY_MAX_LOGIT_BIAS_VALUE`,
    /// default 100, the OpenAI range).
    pub max_logit_bias_value: f64,
    /// Entries in `stop` (`VLLM_PROXY_MAX_STOP_SEQUENCES`, default 16).
    pub max_stop_sequences: usize,
    /// Characters per `stop` string (`VLLM_PROXY_MAX_STOP_SEQUENCE_CHARS`, default 1024).
    pub max_stop_chars: usize,
    /// Entries in `stop_token_ids` (`VLLM_PROXY_MAX_STOP_TOKEN_IDS`, default 64).
    pub max_stop_token_ids: usize,
    /// Token ids in `logit_bias` and `stop_token_ids` must be below this
    /// (`VLLM_PROXY_VOCAB_SIZE`; 0 = unset, ids only need to be in `[0, 2^31)`).
    pub vocab_size: u64,
    /// Characters in `model` (`VLLM_PROXY_MAX_MODEL_NAME_CHARS`, default 256).
    pub max_model_chars: usize,
    /// Characters in a content part's `uuid`
    /// (`VLLM_PROXY_MAX_CONTENT_PART_UUID_CHARS`, default 256).
    pub max_content_part_uuid_chars: usize,
    /// Images per request on the image routes (`VLLM_PROXY_MAX_IMAGE_N`, default 10).
    pub max_image_n: u64,
    /// Longest side in pixels of an image `size` (`VLLM_PROXY_MAX_IMAGE_SIDE`, default 2048).
    pub max_image_side: u64,
    /// `num_inference_steps` on the image routes (`VLLM_PROXY_MAX_IMAGE_STEPS`, default 100).
    pub max_image_steps: u64,
    /// Limits on structured-output regexes (`structured_regex.rs`).
    pub regex: crate::structured_regex::RegexLimits,
    /// Forward EBNF grammars (`structured_outputs.grammar`, `ebnf`,
    /// `guided_grammar`); off by default (`VLLM_PROXY_ALLOW_GRAMMAR`).
    pub allow_grammar: bool,
    /// Refuse media URLs whose scheme is not http, https or data
    /// (`VLLM_PROXY_REJECT_LOCAL_MEDIA_URLS`, default on).
    pub reject_local_media_urls: bool,
    /// Top-level keys removed from client bodies
    /// (`VLLM_PROXY_STRIPPED_REQUEST_FIELDS`).
    pub stripped_fields: Vec<String>,
}

impl Default for RequestGuardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_top_logprobs: 20,
            max_n: 8,
            max_logit_bias_entries: 300,
            max_logit_bias_value: 100.0,
            max_stop_sequences: 16,
            max_stop_chars: 1024,
            max_stop_token_ids: 64,
            vocab_size: 0,
            max_model_chars: 256,
            max_content_part_uuid_chars: 256,
            max_image_n: 10,
            max_image_side: 2048,
            max_image_steps: 100,
            regex: crate::structured_regex::RegexLimits::default(),
            allow_grammar: false,
            reject_local_media_urls: true,
            stripped_fields: DEFAULT_STRIPPED_FIELDS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

/// Parse `VLLM_PROXY_STRIPPED_REQUEST_FIELDS`: comma-separated, trimmed.
pub fn parse_field_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn count(rule: &'static str, action: &'static str) {
    metrics::counter!("request_guard_total", "rule" => rule, "action" => action).increment(1);
}

fn refuse(rule: &'static str, param: &str, message: String) -> AppError {
    count(rule, "rejected");
    AppError::InvalidParam {
        param: param.to_string(),
        message,
    }
}

/// A number, or a string the engine's pydantic model would coerce to one.
/// `"nan"` and `"inf"` parse to non-finite values on purpose.
fn numeric(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

impl RequestGuardConfig {
    fn token_id_bound(&self) -> u64 {
        if self.vocab_size == 0 {
            DEFAULT_TOKEN_ID_BOUND
        } else {
            self.vocab_size
        }
    }

    fn valid_token_id(&self, id: f64) -> bool {
        id.is_finite() && id >= 0.0 && id.fract() == 0.0 && (id as u64) < self.token_id_bound()
    }

    /// Refuses a `model` longer than `max_model_chars`.
    pub fn check_model_name(&self, model: &str) -> Result<(), AppError> {
        if self.enabled && self.max_model_chars > 0 && model.chars().count() > self.max_model_chars
        {
            return Err(refuse(
                "model_name_length",
                "model",
                format!(
                    "`model` must be at most {} characters.",
                    self.max_model_chars
                ),
            ));
        }
        Ok(())
    }

    fn check_model(&self, body: &Map<String, Value>) -> Result<(), AppError> {
        match body.get("model") {
            Some(Value::String(model)) => self.check_model_name(model),
            _ => Ok(()),
        }
    }

    /// Chat and text completions: every rule. Returns whether the body was
    /// changed (a field stripped). Runs on the body as it will be forwarded.
    pub fn apply_completion_body(&self, body: &mut Value) -> Result<bool, AppError> {
        if !self.enabled {
            return Ok(false);
        }
        let Some(obj) = body.as_object() else {
            return Ok(false);
        };
        self.check_model(obj)?;
        self.check_sampling(obj)?;
        self.check_messages(obj)?;
        self.check_structured(obj)?;
        Ok(self.strip(body))
    }

    /// Structured-output constraints anywhere outside the prompt: regexes,
    /// JSON-schema `pattern` / `patternProperties`, and grammars.
    fn check_structured(&self, body: &Map<String, Value>) -> Result<(), AppError> {
        for (key, value) in body {
            if matches!(key.as_str(), "messages" | "prompt" | "input") {
                continue;
            }
            self.walk_structured(key, value, 0)?;
        }
        Ok(())
    }

    fn walk_structured(&self, key: &str, value: &Value, depth: usize) -> Result<(), AppError> {
        if depth > 64 {
            return Err(refuse(
                "structured_depth",
                key,
                "The structured-output schema is nested too deeply.".to_string(),
            ));
        }
        match (key, value) {
            ("regex" | "guided_regex" | "pattern", Value::String(p)) => self.check_pattern(key, p),
            ("grammar" | "ebnf" | "guided_grammar", Value::String(_)) if !self.allow_grammar => {
                Err(refuse(
                    "structured_grammar",
                    key,
                    "Grammar-constrained output is not supported; use a JSON schema or a regex."
                        .to_string(),
                ))
            }
            ("json_schema" | "json" | "guided_json" | "schema", Value::String(s)) => {
                match serde_json::from_str::<Value>(s) {
                    Ok(v) => self.walk_structured("schema", &v, depth + 1),
                    Err(_) => Ok(()),
                }
            }
            ("patternProperties", Value::Object(m)) => {
                for (pattern, sub) in m {
                    self.check_pattern(key, pattern)?;
                    self.walk_structured(pattern, sub, depth + 1)?;
                }
                Ok(())
            }
            (_, Value::Object(m)) => {
                for (k, v) in m {
                    self.walk_structured(k, v, depth + 1)?;
                }
                Ok(())
            }
            (_, Value::Array(a)) => {
                for v in a {
                    self.walk_structured(key, v, depth + 1)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn check_pattern(&self, key: &str, pattern: &str) -> Result<(), AppError> {
        crate::structured_regex::check_regex(pattern, &self.regex).map_err(|reason| {
            refuse(
                "structured_regex",
                key,
                format!("The structured-output pattern is not supported: {reason}."),
            )
        })
    }

    /// Embeddings, rerank, score and image generation: the `model` cap and
    /// the stripped fields (the sampling rules do not apply to these bodies).
    pub fn apply_passthrough_body(&self, body: &mut Value) -> Result<bool, AppError> {
        if !self.enabled {
            return Ok(false);
        }
        let Some(obj) = body.as_object() else {
            return Ok(false);
        };
        self.check_model(obj)?;
        Ok(self.strip(body))
    }

    /// Bounds on the image request fields of a JSON body (images/generations).
    pub fn check_image_body(&self, body: &Value) -> Result<(), AppError> {
        let Some(obj) = body.as_object() else {
            return Ok(());
        };
        for name in IMAGE_BOUNDED_FIELDS {
            match obj.get(*name) {
                None | Some(Value::Null) => {}
                Some(Value::String(s)) => self.check_image_field(name, s)?,
                Some(other) => self.check_image_field(name, &other.to_string())?,
            }
        }
        Ok(())
    }

    /// Bounds on one image request field (`n`, `size`, `num_inference_steps`,
    /// `guidance_scale`), as JSON or as a form value. Other names pass.
    pub fn check_image_field(&self, name: &str, value: &str) -> Result<(), AppError> {
        if !self.enabled {
            return Ok(());
        }
        let value = value.trim();
        let int_in = |max: u64| -> bool {
            max == 0 || value.parse::<u64>().is_ok_and(|v| (1..=max).contains(&v))
        };
        match name {
            "n" if !int_in(self.max_image_n) => Err(refuse(
                "image_n",
                name,
                format!("`n` must be an integer from 1 to {}", self.max_image_n),
            )),
            "num_inference_steps" if !int_in(self.max_image_steps) => Err(refuse(
                "image_steps",
                name,
                format!(
                    "`num_inference_steps` must be an integer from 1 to {}",
                    self.max_image_steps
                ),
            )),
            "size" if !self.image_size_ok(value) => Err(refuse(
                "image_size",
                name,
                format!(
                    "`size` must be `auto` or `WIDTHxHEIGHT` with each side from 1 to {}",
                    self.max_image_side
                ),
            )),
            "guidance_scale" if !value.parse::<f64>().is_ok_and(f64::is_finite) => Err(refuse(
                "image_guidance",
                name,
                "`guidance_scale` must be a finite number".to_string(),
            )),
            _ => Ok(()),
        }
    }

    fn image_size_ok(&self, value: &str) -> bool {
        if self.max_image_side == 0 || value == "auto" {
            return true;
        }
        let Some((w, h)) = value.split_once('x') else {
            return false;
        };
        [w, h].iter().all(|side| {
            side.parse::<u64>()
                .is_ok_and(|v| (1..=self.max_image_side).contains(&v))
        })
    }

    /// One multipart form field (images/edits, audio/transcriptions).
    /// `Ok(false)` = drop the field, `Ok(true)` = forward it.
    pub fn keep_form_field(&self, name: &str, value: &[u8]) -> Result<bool, AppError> {
        if !self.enabled {
            return Ok(true);
        }
        if name == "model" {
            self.check_model_name(&String::from_utf8_lossy(value))?;
        }
        if self.stripped_fields.iter().any(|f| f == name) {
            count("stripped_field", "stripped");
            return Ok(false);
        }
        Ok(true)
    }

    fn strip(&self, body: &mut Value) -> bool {
        let Some(obj) = body.as_object_mut() else {
            return false;
        };
        let mut changed = false;
        for field in &self.stripped_fields {
            if obj.shift_remove(field).is_some() {
                count("stripped_field", "stripped");
                changed = true;
            }
        }
        changed
    }

    fn check_sampling(&self, body: &Map<String, Value>) -> Result<(), AppError> {
        // top_logprobs, and `logprobs` as a number (text completions).
        if self.max_top_logprobs > 0 {
            let max = self.max_top_logprobs as f64;
            for key in ["top_logprobs", "logprobs"] {
                if let Some(x) = body.get(key).and_then(numeric) {
                    if x.is_nan() || x > max {
                        return Err(refuse(
                            "top_logprobs",
                            key,
                            format!("`{key}` must be at most {}.", self.max_top_logprobs),
                        ));
                    }
                }
            }
        }

        if self.max_n > 0 {
            if let Some(x) = body.get("n").and_then(numeric) {
                if x.is_nan() || x > self.max_n as f64 {
                    return Err(refuse(
                        "n",
                        "n",
                        format!("`n` must be at most {}.", self.max_n),
                    ));
                }
            }
        }

        if let Some(Value::Object(bias)) = body.get("logit_bias") {
            if self.max_logit_bias_entries > 0 && bias.len() > self.max_logit_bias_entries {
                return Err(refuse(
                    "logit_bias",
                    "logit_bias",
                    format!(
                        "`logit_bias` may have at most {} entries.",
                        self.max_logit_bias_entries
                    ),
                ));
            }
            for (token, value) in bias {
                let id_ok = token
                    .trim()
                    .parse::<u64>()
                    .is_ok_and(|id| id < self.token_id_bound());
                if !id_ok {
                    return Err(refuse(
                        "logit_bias",
                        "logit_bias",
                        "`logit_bias` keys must be token ids within the model vocabulary."
                            .to_string(),
                    ));
                }
                let value_ok = numeric(value).is_some_and(|v| {
                    v.is_finite()
                        && (self.max_logit_bias_value <= 0.0
                            || v.abs() <= self.max_logit_bias_value)
                });
                if !value_ok {
                    return Err(refuse(
                        "logit_bias",
                        "logit_bias",
                        format!(
                            "`logit_bias` values must be finite numbers between -{0} and {0}.",
                            self.max_logit_bias_value
                        ),
                    ));
                }
            }
        }

        match body.get("stop") {
            Some(Value::String(s)) => self.check_stop_string(s)?,
            Some(Value::Array(list)) => {
                if self.max_stop_sequences > 0 && list.len() > self.max_stop_sequences {
                    return Err(refuse(
                        "stop",
                        "stop",
                        format!(
                            "`stop` may have at most {} sequences.",
                            self.max_stop_sequences
                        ),
                    ));
                }
                for s in list.iter().filter_map(Value::as_str) {
                    self.check_stop_string(s)?;
                }
            }
            _ => {}
        }

        if let Some(Value::Array(ids)) = body.get("stop_token_ids") {
            if self.max_stop_token_ids > 0 && ids.len() > self.max_stop_token_ids {
                return Err(refuse(
                    "stop_token_ids",
                    "stop_token_ids",
                    format!(
                        "`stop_token_ids` may have at most {} entries.",
                        self.max_stop_token_ids
                    ),
                ));
            }
            if !ids
                .iter()
                .all(|id| numeric(id).is_some_and(|x| self.valid_token_id(x)))
            {
                return Err(refuse(
                    "stop_token_ids",
                    "stop_token_ids",
                    "`stop_token_ids` must be non-negative token ids within the model vocabulary."
                        .to_string(),
                ));
            }
        }

        if let Some(x) = body.get("logprob_start_len").and_then(numeric) {
            if x.is_nan() || x < -1.0 {
                return Err(refuse(
                    "logprob_start_len",
                    "logprob_start_len",
                    "`logprob_start_len` must be -1 or greater.".to_string(),
                ));
            }
        }

        // The engines take `max_completion_tokens` over `max_tokens`.
        let max_tokens = ["max_completion_tokens", "max_tokens"]
            .iter()
            .find_map(|k| body.get(*k).and_then(numeric));
        if let (Some(min), Some(max)) = (body.get("min_tokens").and_then(numeric), max_tokens) {
            if min > max {
                return Err(refuse(
                    "min_tokens",
                    "min_tokens",
                    "`min_tokens` must not exceed `max_tokens`.".to_string(),
                ));
            }
        }

        for key in FINITE_FLOAT_FIELDS {
            let Some(v) = body.get(*key) else { continue };
            let bad = match v {
                Value::Null | Value::Bool(_) => false,
                _ => !numeric(v).is_some_and(f64::is_finite),
            };
            if bad {
                return Err(refuse(
                    "non_finite_sampling",
                    key,
                    format!("`{key}` must be a finite number."),
                ));
            }
        }
        Ok(())
    }

    fn check_stop_string(&self, s: &str) -> Result<(), AppError> {
        if self.max_stop_chars > 0 && s.chars().count() > self.max_stop_chars {
            return Err(refuse(
                "stop",
                "stop",
                format!(
                    "Each `stop` sequence must be at most {} characters.",
                    self.max_stop_chars
                ),
            ));
        }
        Ok(())
    }

    /// Content parts: the `uuid` cap and the media URL scheme.
    fn check_messages(&self, body: &Map<String, Value>) -> Result<(), AppError> {
        let Some(messages) = body.get("messages").and_then(Value::as_array) else {
            return Ok(());
        };
        let parts = messages
            .iter()
            .filter_map(|m| m.get("content").and_then(Value::as_array))
            .flatten()
            .filter_map(Value::as_object);
        for part in parts {
            self.check_uuid(part)?;
            for field in MEDIA_PART_FIELDS {
                let url = match part.get(*field) {
                    Some(Value::String(url)) => Some(url.as_str()),
                    Some(Value::Object(media)) => {
                        self.check_uuid(media)?;
                        media.get("url").and_then(Value::as_str)
                    }
                    _ => None,
                };
                if let Some(url) = url {
                    self.check_media_url(field, url)?;
                }
            }
        }
        Ok(())
    }

    fn check_uuid(&self, obj: &Map<String, Value>) -> Result<(), AppError> {
        if self.max_content_part_uuid_chars == 0 {
            return Ok(());
        }
        if let Some(Value::String(uuid)) = obj.get("uuid") {
            if uuid.chars().count() > self.max_content_part_uuid_chars {
                return Err(refuse(
                    "content_part_uuid",
                    "uuid",
                    format!(
                        "Content part `uuid` must be at most {} characters.",
                        self.max_content_part_uuid_chars
                    ),
                ));
            }
        }
        Ok(())
    }

    fn check_media_url(&self, field: &str, url: &str) -> Result<(), AppError> {
        if !self.reject_local_media_urls {
            return Ok(());
        }
        let url = crate::image_validation::normalize_image_url_input(url);
        let scheme = url
            .split_once(':')
            .map(|(s, _)| s.to_ascii_lowercase())
            .unwrap_or_default();
        if ALLOWED_MEDIA_SCHEMES.contains(&scheme.as_str()) {
            return Ok(());
        }
        Err(refuse(
            "media_url_scheme",
            field,
            format!("`{field}` must be an http, https or data URL."),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn guard() -> RequestGuardConfig {
        RequestGuardConfig::default()
    }

    /// The `param` of a rejection, or a panic if the body passed.
    fn rejected_param(mut body: Value) -> String {
        match guard().apply_completion_body(&mut body) {
            Err(AppError::InvalidParam { param, message }) => {
                assert!(!message.is_empty());
                param
            }
            other => panic!("expected InvalidParam, got {other:?}"),
        }
    }

    fn passes(mut body: Value) {
        guard()
            .apply_completion_body(&mut body)
            .unwrap_or_else(|e| panic!("expected pass, got {e:?}"));
    }

    #[test]
    fn ordinary_requests_pass_unchanged() {
        let mut body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}, "uuid": "abc"},
                {"type": "image_url", "image_url": "data:image/png;base64,AAAA"}
            ]}],
            "n": 2, "logprobs": true, "top_logprobs": 20,
            "logit_bias": {"50256": -100, "11": 5.5},
            "stop": ["\n\n", "END"], "stop_token_ids": [0, 151643],
            "temperature": 0.7, "top_p": 1, "min_tokens": 4, "max_tokens": 16,
            "logprob_start_len": -1, "presence_penalty": null
        });
        let before = body.clone();
        assert!(!guard().apply_completion_body(&mut body).unwrap());
        assert_eq!(body, before);
    }

    #[test]
    fn top_logprobs_and_numeric_logprobs_are_capped() {
        assert_eq!(rejected_param(json!({"top_logprobs": 21})), "top_logprobs");
        assert_eq!(
            rejected_param(json!({"top_logprobs": 1000000})),
            "top_logprobs"
        );
        // pydantic coerces numeric strings.
        assert_eq!(
            rejected_param(json!({"top_logprobs": "1000000"})),
            "top_logprobs"
        );
        assert_eq!(
            rejected_param(json!({"prompt": "x", "logprobs": 21})),
            "logprobs"
        );
        passes(json!({"prompt": "x", "logprobs": 5}));
        passes(json!({"logprobs": true, "top_logprobs": 0}));
    }

    #[test]
    fn n_is_capped() {
        assert_eq!(rejected_param(json!({"n": 9})), "n");
        assert_eq!(rejected_param(json!({"n": 1e12})), "n");
        passes(json!({"n": 8}));
    }

    #[test]
    fn logit_bias_size_ids_and_values() {
        let big: Map<String, Value> = (0..301).map(|i| (i.to_string(), json!(1))).collect();
        assert_eq!(rejected_param(json!({"logit_bias": big})), "logit_bias");
        assert_eq!(
            rejected_param(json!({"logit_bias": {"-1": 1}})),
            "logit_bias"
        );
        assert_eq!(
            rejected_param(json!({"logit_bias": {"abc": 1}})),
            "logit_bias"
        );
        assert_eq!(
            rejected_param(json!({"logit_bias": {"99999999999": 1}})),
            "logit_bias"
        );
        assert_eq!(
            rejected_param(json!({"logit_bias": {"5": 1e39}})),
            "logit_bias"
        );
        assert_eq!(
            rejected_param(json!({"logit_bias": {"5": "nan"}})),
            "logit_bias"
        );
        assert_eq!(
            rejected_param(json!({"logit_bias": {"5": 101}})),
            "logit_bias"
        );
        let ok: Map<String, Value> = (0..300).map(|i| (i.to_string(), json!(-100))).collect();
        passes(json!({ "logit_bias": ok }));
    }

    #[test]
    fn vocab_size_bounds_token_ids() {
        let cfg = RequestGuardConfig {
            vocab_size: 1000,
            ..guard()
        };
        let mut body = json!({"logit_bias": {"1000": 1}});
        assert!(cfg.apply_completion_body(&mut body).is_err());
        let mut body = json!({"stop_token_ids": [999, 1000]});
        assert!(cfg.apply_completion_body(&mut body).is_err());
        let mut body = json!({"logit_bias": {"999": 1}, "stop_token_ids": [999]});
        assert!(cfg.apply_completion_body(&mut body).is_ok());
    }

    #[test]
    fn stop_strings_are_bounded() {
        let many: Vec<String> = (0..17).map(|i| i.to_string()).collect();
        assert_eq!(rejected_param(json!({ "stop": many })), "stop");
        let long = "x".repeat(1025);
        assert_eq!(rejected_param(json!({ "stop": long.clone() })), "stop");
        assert_eq!(rejected_param(json!({ "stop": ["ok", long] })), "stop");
        passes(json!({ "stop": "x".repeat(1024) }));
    }

    #[test]
    fn stop_token_ids_are_bounded() {
        let many: Vec<u32> = (0..65).collect();
        assert_eq!(
            rejected_param(json!({ "stop_token_ids": many })),
            "stop_token_ids"
        );
        assert_eq!(
            rejected_param(json!({"stop_token_ids": [1, -5]})),
            "stop_token_ids"
        );
        assert_eq!(
            rejected_param(json!({"stop_token_ids": [1.5]})),
            "stop_token_ids"
        );
        assert_eq!(
            rejected_param(json!({"stop_token_ids": ["x"]})),
            "stop_token_ids"
        );
        let max: Vec<u32> = (0..64).collect();
        passes(json!({ "stop_token_ids": max }));
    }

    #[test]
    fn model_name_length_is_capped() {
        assert_eq!(rejected_param(json!({"model": "a".repeat(257)})), "model");
        passes(json!({"model": "a".repeat(256)}));
        // Multipart and passthrough share the rule.
        assert!(guard()
            .keep_form_field("model", "a".repeat(257).as_bytes())
            .is_err());
        let mut body = json!({"model": "/a".repeat(5000), "input": "x"});
        assert!(guard().apply_passthrough_body(&mut body).is_err());
    }

    #[test]
    fn logprob_start_len_below_minus_one_is_rejected() {
        assert_eq!(
            rejected_param(json!({"logprob_start_len": -2})),
            "logprob_start_len"
        );
        passes(json!({"logprob_start_len": 0}));
    }

    #[test]
    fn min_tokens_must_not_exceed_max_tokens() {
        assert_eq!(
            rejected_param(json!({"min_tokens": 10, "max_tokens": 5})),
            "min_tokens"
        );
        assert_eq!(
            rejected_param(json!({"min_tokens": 10, "max_completion_tokens": 5, "max_tokens": 50})),
            "min_tokens"
        );
        passes(json!({"min_tokens": 10}));
        passes(json!({"min_tokens": 5, "max_tokens": 5}));
    }

    #[test]
    fn sampling_floats_must_be_finite() {
        assert_eq!(rejected_param(json!({"temperature": "nan"})), "temperature");
        assert_eq!(rejected_param(json!({"top_p": "Infinity"})), "top_p");
        assert_eq!(rejected_param(json!({"top_p": "-inf"})), "top_p");
        assert_eq!(rejected_param(json!({"temperature": [1]})), "temperature");
        passes(json!({"temperature": "0.5", "top_p": 0.9, "min_p": null}));
    }

    #[test]
    fn content_part_uuid_is_capped() {
        let long = "u".repeat(257);
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}, "uuid": long}
        ]}]});
        assert_eq!(rejected_param(body), "uuid");
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "uuid": "u".repeat(257)}}
        ]}]});
        assert_eq!(rejected_param(body), "uuid");
    }

    #[test]
    fn local_media_urls_are_rejected() {
        for url in [
            "file:///etc/passwd",
            "/etc/passwd",
            "  FILE:///x",
            "ftp://example.com/a.png",
            "C:\\x.png",
        ] {
            for field in ["image_url", "video_url", "audio_url"] {
                let body = json!({"messages": [{"role": "user", "content": [
                    {"type": field, field: {"url": url}}
                ]}]});
                assert_eq!(rejected_param(body), field, "{field} {url}");
            }
        }
        let bare = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": "/etc/passwd"}
        ]}]});
        assert_eq!(rejected_param(bare), "image_url");
        passes(json!({"messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "HTTPS://example.com/v.mp4"}}
        ]}]}));
    }

    #[test]
    fn engine_internal_fields_are_stripped() {
        let mut body = json!({
            "model": "m", "messages": [],
            "kv_transfer_params": {"do_remote_prefill": true, "remote_prefill_cached_tokens": 1},
            "ec_transfer_params": {}, "vllm_xargs": {"kv_transfer_params": "x"},
            "session_params": {"id": 1}, "mm_hashes": ["h"], "return_hidden_states": true,
            "custom_logit_processor": "x"
        });
        assert!(guard().apply_completion_body(&mut body).unwrap());
        assert_eq!(body, json!({"model": "m", "messages": []}));

        let mut body = json!({"input": "cat", "vllm_xargs": {}, "mm_hashes": ["h"]});
        assert!(guard().apply_passthrough_body(&mut body).unwrap());
        assert_eq!(body, json!({"input": "cat"}));
    }

    #[test]
    fn form_fields() {
        let g = guard();
        assert!(!g.keep_form_field("vllm_xargs", b"{}").unwrap());
        assert!(g.keep_form_field("prompt", b"cat").unwrap());
        assert!(g.keep_form_field("model", &[b'm'; 300]).is_err());
    }

    #[test]
    fn zero_limits_and_master_switch_turn_rules_off() {
        let off = RequestGuardConfig {
            enabled: false,
            ..guard()
        };
        let mut body = json!({"n": 1000, "top_logprobs": 1000, "vllm_xargs": {}});
        assert!(!off.apply_completion_body(&mut body).unwrap());
        assert!(body.get("vllm_xargs").is_some());
        assert!(off.keep_form_field("vllm_xargs", b"x").unwrap());

        let no_n = RequestGuardConfig {
            max_n: 0,
            stripped_fields: Vec::new(),
            ..guard()
        };
        let mut body = json!({"n": 1000, "vllm_xargs": {}});
        assert!(!no_n.apply_completion_body(&mut body).unwrap());
    }

    #[test]
    fn image_fields_are_bounded() {
        let g = guard();
        for (name, ok) in [
            ("n", "10"),
            ("size", "1024x1024"),
            ("size", "auto"),
            ("num_inference_steps", "100"),
            ("guidance_scale", "3.5"),
            ("prompt", "anything"),
        ] {
            assert!(g.check_image_field(name, ok).is_ok(), "{name}={ok}");
        }
        for (name, bad) in [
            ("n", "11"),
            ("n", "0"),
            ("n", "-1"),
            ("size", "4096x1024"),
            ("size", "0x512"),
            ("size", "huge"),
            ("num_inference_steps", "100000"),
            ("num_inference_steps", "1.5"),
            ("guidance_scale", "NaN"),
        ] {
            assert!(g.check_image_field(name, bad).is_err(), "{name}={bad}");
        }
        assert!(g
            .check_image_body(&json!({"n": 2, "size": "512x512", "num_inference_steps": 30}))
            .is_ok());
        assert!(g.check_image_body(&json!({"n": 50})).is_err());
        assert!(g
            .check_image_body(&json!({"n": null, "size": null}))
            .is_ok());
        let off = RequestGuardConfig {
            max_image_n: 0,
            ..guard()
        };
        assert!(off.check_image_field("n", "50").is_ok());
    }

    #[test]
    fn structured_output_constraints_are_bounded() {
        let g = guard();
        let slow = "^(.{0,500}){0,500}$";
        for body in [
            json!({"structured_outputs": {"regex": slow}}),
            json!({"regex": slow}),
            json!({"guided_regex": slow}),
            json!({"response_format": {"type": "json_schema", "json_schema": {"schema":
                {"type": "object", "properties": {"a": {"type": "string", "pattern": slow}}}}}}),
            json!({"json_schema": format!(r#"{{"type":"string","pattern":"{slow}"}}"#)}),
            json!({"response_format": {"type": "json_schema", "json_schema": {"schema":
                {"type": "object", "patternProperties": {slow: {"type": "string"}}}}}}),
            json!({"tools": [{"type": "function", "function": {"name": "f", "parameters":
                {"type": "object", "properties": {"a": {"type": "string", "pattern": slow}}}}}]}),
            json!({"structured_outputs": {"grammar": "root ::= \"a\""}}),
            json!({"ebnf": "root ::= \"a\""}),
        ] {
            let mut b = body.clone();
            assert!(g.apply_completion_body(&mut b).is_err(), "{body}");
        }
        for body in [
            json!({"structured_outputs": {"regex": "[a-z]{1,64}"}}),
            json!({"response_format": {"type": "json_schema", "json_schema": {"schema":
                {"type": "object", "properties": {"d": {"type": "string",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}$"}}}}}}),
            json!({"messages": [{"role": "user", "content": "regex: (.{0,500}){0,500}"}]}),
        ] {
            let mut b = body.clone();
            assert!(g.apply_completion_body(&mut b).is_ok(), "{body}");
        }
        let allow = RequestGuardConfig {
            allow_grammar: true,
            ..guard()
        };
        assert!(allow
            .apply_completion_body(&mut json!({"ebnf": "root ::= \"a\""}))
            .is_ok());
    }

    #[test]
    fn parses_field_list() {
        assert_eq!(parse_field_list(" a, b ,,c"), vec!["a", "b", "c"]);
        assert!(parse_field_list("").is_empty());
    }
}
