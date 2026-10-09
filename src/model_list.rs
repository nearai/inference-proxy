//! Gateway list mode: one process serving several models
//! (`VLLM_PROXY_MODEL_LIST_FILE`).
//!
//! A gateway used to be model-scoped: one `MODEL_NAME`, one backend pool, one
//! admission budget, one billing identity, and a request body whose `model`
//! was never read. With a model list the process holds one such bundle per
//! model and picks it from the body's `model` — an exact match against the
//! configured ids, after authentication. Everything that isolates one lane
//! from another stays per model: the pool and its health checker, conversation
//! affinity, the admission controller (budget, per-host share, TTFT breaker,
//! back-pressure marks), the engine-load poller, the long-context tier and the
//! HTTP client that carries the model's backend bearer. What is one per
//! process is what was never about the model: the API key check, the
//! organization allowlist, usage reporting to cloud-api, the content policy,
//! the models document source, the stream timings and the health-check
//! timings.
//!
//! The list is a JSON file rendered by deploy tooling. It holds no secrets: a
//! model's backend token is named by the environment variable that carries it.
//! Every key except `id` and `backend_urls` is optional and falls back to the
//! process-level variable of the same name, so one model written out in full
//! and one written as `id` plus `backend_urls` under the same environment are
//! the same model (`Config::single_model`). Two keys have no variable and so
//! no fallback, `reasoning_effort_map` and `merge_system_messages`: they
//! exist for a list's models only.
//!
//! Without the variable nothing here runs: the process serves one model from
//! its environment, as every CVM proxy and single-model gateway does.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;
use tracing::info;

use crate::admission::AdmissionController;
use crate::backend_affinity::BackendConversationAffinity;
use crate::backend_pool::BackendPool;
use crate::config::{
    check_admission, check_long_context_tier, check_probe_urls, validate_discount_to_user,
    AdmissionKnobs, Config, TierKnobs,
};
use crate::engine_load::{self, EngineLoad};
use crate::error::AppError;
use crate::model_metrics::ModelLabel;
use crate::reasoning::EffortMap;
use crate::AppState;

#[cfg(test)]
#[path = "model_list_tests.rs"]
mod model_list_tests;

/// Path of the model list file. Set = list mode.
pub const MODEL_LIST_FILE_ENV: &str = "VLLM_PROXY_MODEL_LIST_FILE";

/// The single-model backend of a process in list mode. Nothing is routed
/// there: the routes that use the single-model pool are not served, and a
/// `.invalid` name (RFC 6761) cannot resolve if one ever were.
pub const UNROUTED_BACKEND_URL: &str = "http://model-list.invalid";

/// Variables that name the one model of a single-model process and its
/// backends. A list entry has no fallback for these, so next to a list they
/// are refused instead of ignored.
pub(crate) const SINGLE_MODEL_ENV: [&str; 5] = [
    "MODEL_NAME",
    "VLLM_BACKEND_URLS",
    "VLLM_BACKEND_PROBE_URLS",
    "VLLM_BACKEND_LONG_CONTEXT_URLS",
    "VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS",
];

/// A `backend_token_env` must name a variable of this family, so a list can
/// never point a model's backend bearer at an unrelated secret of the process.
const BACKEND_TOKEN_ENV_PREFIX: &str = "VLLM_BACKEND_TOKEN";

/// One served model's effective configuration: a list entry with every
/// omitted key filled in, or the single model of a process without a list.
/// The field names are `Config`'s.
#[derive(Clone, PartialEq)]
pub struct ModelConfig {
    /// The exact cloud-api model name: what requests select by and what usage
    /// is billed under.
    pub id: String,
    pub backend_urls: Vec<String>,
    pub backend_probe_urls: Vec<String>,
    pub backend_long_context_urls: Vec<String>,
    pub backend_long_context_probe_urls: Vec<String>,
    pub long_context_above_tokens: u64,
    pub backend_tier_strict: bool,
    pub admission_max_inflight: u32,
    pub admission_start_inflight: u32,
    pub admission_queue_saturated_at: u32,
    pub admission_tier_borrowing: bool,
    pub admission_long_max_inflight_per_host: u32,
    pub admission_long_reserved_inflight: u32,
    pub capacity_requests_per_minute: u64,
    pub discount_to_user: Option<f64>,
    pub reasoning_off_effort: String,
    /// Efforts this model's engine refuses, and what is sent in their place
    /// (`reasoning_effort_map`). A list key only: empty for the one model of
    /// a process without a list.
    pub reasoning_effort_map: EffortMap,
    /// This model's chat template takes one `system` message, first: a
    /// request with one anywhere else gets them merged into that
    /// (`merge_system_messages`, `system_messages.rs`). A list key only:
    /// `false` for the one model of a process without a list.
    pub merge_system_messages: bool,
    pub backend_token: Option<String>,
    pub backend_priority: Option<i64>,
}

impl ModelConfig {
    /// Engine-load probe URLs in pool order: the base tier, then the
    /// long-context one (`Config::pool_probe_urls`).
    pub fn pool_probe_urls(&self) -> Vec<String> {
        self.backend_probe_urls
            .iter()
            .chain(&self.backend_long_context_probe_urls)
            .cloned()
            .collect()
    }
}

impl std::fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelConfig")
            .field("id", &self.id)
            .field("backend_urls", &self.backend_urls)
            .field("backend_probe_urls", &self.backend_probe_urls)
            .field("backend_long_context_urls", &self.backend_long_context_urls)
            .field(
                "backend_long_context_probe_urls",
                &self.backend_long_context_probe_urls,
            )
            .field("long_context_above_tokens", &self.long_context_above_tokens)
            .field("backend_tier_strict", &self.backend_tier_strict)
            .field("admission_max_inflight", &self.admission_max_inflight)
            .field("admission_start_inflight", &self.admission_start_inflight)
            .field(
                "admission_queue_saturated_at",
                &self.admission_queue_saturated_at,
            )
            .field("admission_tier_borrowing", &self.admission_tier_borrowing)
            .field(
                "admission_long_max_inflight_per_host",
                &self.admission_long_max_inflight_per_host,
            )
            .field(
                "admission_long_reserved_inflight",
                &self.admission_long_reserved_inflight,
            )
            .field(
                "capacity_requests_per_minute",
                &self.capacity_requests_per_minute,
            )
            .field("discount_to_user", &self.discount_to_user)
            .field("reasoning_off_effort", &self.reasoning_off_effort)
            .field("reasoning_effort_map", &self.reasoning_effort_map)
            .field("merge_system_messages", &self.merge_system_messages)
            .field(
                "backend_token",
                &self.backend_token.as_ref().map(|_| "<set>"),
            )
            .field("backend_priority", &self.backend_priority)
            .finish()
    }
}

/// The parsed list and where it came from (`Config::model_list`).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelListConfig {
    pub path: String,
    pub models: Vec<ModelConfig>,
}

/// The process-level values of the per-model variables: what a list entry
/// gets for a key it leaves out. Not `Debug`: it holds the process-level
/// backend token.
#[derive(Clone, Default)]
pub struct ModelDefaults {
    pub long_context_above_tokens: u64,
    /// `VLLM_BACKEND_TIER_STRICT` as written; it applies to entries with a
    /// tier only.
    pub backend_tier_strict: bool,
    pub admission_max_inflight: u32,
    /// `None` when `VLLM_PROXY_ADMISSION_START_INFLIGHT` is not set: the
    /// entry then starts at its own maximum, as a single model does.
    pub admission_start_inflight: Option<u32>,
    pub admission_queue_saturated_at: u32,
    pub admission_tier_borrowing: bool,
    pub admission_long_max_inflight_per_host: u32,
    pub admission_long_reserved_inflight: u32,
    pub capacity_requests_per_minute: u64,
    pub discount_to_user: Option<f64>,
    pub reasoning_off_effort: String,
    pub backend_token: Option<String>,
    pub backend_priority: Option<i64>,
}

/// What an entry is validated against besides its own keys: the half of the
/// admission settings that stays process-level, and whether the process can
/// bill.
#[derive(Clone, Debug)]
pub struct ProcessSettings {
    pub admission_ramp_step: u32,
    pub admission_ramp_interval_secs: u64,
    pub admission_backpressure_secs: u64,
    pub admission_retry_after_secs: u64,
    /// `CLOUD_API_URL` and `CLOUD_API_USAGE_TOKEN` are both set.
    pub bills_usage: bool,
}

/// The file: `{"models": [ … ]}`. Unknown keys are refused so a typo cannot
/// silently fall back to a process-level default.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelListFile {
    models: Vec<ModelEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelEntry {
    id: String,
    backend_urls: Vec<String>,
    #[serde(default)]
    backend_probe_urls: Vec<String>,
    long_context: Option<LongContextEntry>,
    admission_max_inflight: Option<u32>,
    admission_start_inflight: Option<u32>,
    admission_queue_saturated_at: Option<u32>,
    capacity_requests_per_minute: Option<u64>,
    discount_to_user: Option<f64>,
    reasoning_off_effort: Option<String>,
    reasoning_effort_map: Option<EffortPairs>,
    /// `true` or `false`. Not an `Option`: `null` is not a boolean either,
    /// and is refused like any other value that is not one.
    #[serde(default)]
    merge_system_messages: bool,
    /// Name of the environment variable holding the backend bearer.
    backend_token_env: Option<String>,
    backend_priority: Option<i64>,
}

/// `reasoning_effort_map` as the file writes it: a JSON object of string to
/// string, kept as its pairs, in file order and with a key written twice
/// still there twice. A map type would settle that by which came last;
/// `EffortMap::new` decides what the pairs may say, duplicates included.
struct EffortPairs(Vec<(String, String)>);

impl<'de> Deserialize<'de> for EffortPairs {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Pairs;

        impl<'de> serde::de::Visitor<'de> for Pairs {
            type Value = EffortPairs;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an object mapping string keys to string values")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut entries: A,
            ) -> Result<Self::Value, A::Error> {
                let mut pairs = Vec::new();
                while let Some(pair) = entries.next_entry::<String, String>()? {
                    pairs.push(pair);
                }
                Ok(EffortPairs(pairs))
            }
        }

        deserializer.deserialize_map(Pairs)
    }
}

/// A model's long-context tier. The tier variables are defaults for the
/// models that have this block and mean nothing for the ones that do not.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LongContextEntry {
    backend_urls: Vec<String>,
    #[serde(default)]
    backend_probe_urls: Vec<String>,
    above_tokens: Option<u64>,
    strict: Option<bool>,
    borrowing: Option<bool>,
    max_inflight_per_host: Option<u32>,
    reserved_inflight: Option<u32>,
}

/// What list mode needs from the rest of the configuration, and the features
/// it cannot run next to because they assume one model per process.
pub(crate) fn check_process(config: &Config) -> anyhow::Result<()> {
    let list = MODEL_LIST_FILE_ENV;
    if !config.non_tee_deployment {
        anyhow::bail!(
            "{list} requires NON_TEE_DEPLOYMENT=1: a process serving several models has no single model to attest or sign for"
        );
    }
    if config.models_document_url.is_none() {
        anyhow::bail!(
            "{list} requires VLLM_PROXY_MODELS_DOCUMENT_URL: with several models `/v1/models` has no single engine list to pass through"
        );
    }
    let single_model_features = [
        (config.fusion_enabled, "FUSION_ENABLED"),
        (
            config.web_context_search_url.is_some(),
            "WEB_CONTEXT_SEARCH_URL",
        ),
        (config.ohttp_enabled, "OHTTP_ENABLED"),
        (
            config.vllm_data_parallel_size.is_some(),
            "VLLM_DATA_PARALLEL_SIZE",
        ),
        (
            config.openai_chat_compatibility_check_enabled,
            "OPENAI_CHAT_COMPATIBILITY_CHECK",
        ),
        (config.replica_state.is_some(), "REPLICA_STATE_REDIS_URL"),
    ];
    for (enabled, name) in single_model_features {
        if enabled {
            anyhow::bail!(
                "{list} cannot be combined with {name}: that feature assumes one model per process"
            );
        }
    }
    Ok(())
}

/// Read, resolve and validate the list. `lookup` reads an environment
/// variable (a model's `backend_token_env`).
pub(crate) fn load(
    path: &str,
    defaults: &ModelDefaults,
    process: &ProcessSettings,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> anyhow::Result<Vec<ModelConfig>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{MODEL_LIST_FILE_ENV}: cannot read {path}: {e}"))?;
    parse(&text, defaults, process, lookup)
        .map_err(|e| anyhow::anyhow!("{MODEL_LIST_FILE_ENV} ({path}): {e}"))
}

/// `load` without the file. Any failure is a startup failure.
pub(crate) fn parse(
    text: &str,
    defaults: &ModelDefaults,
    process: &ProcessSettings,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> anyhow::Result<Vec<ModelConfig>> {
    let file: ModelListFile =
        serde_json::from_str(text).map_err(|e| anyhow::anyhow!("not a valid model list: {e}"))?;
    if file.models.is_empty() {
        anyhow::bail!("`models` is empty: list at least one model");
    }
    let mut models = Vec::with_capacity(file.models.len());
    for entry in file.models {
        let id = entry.id.clone();
        let model = resolve(entry, defaults, lookup)
            .and_then(|model| validate(&model, process).map(|()| model))
            .map_err(|e| anyhow::anyhow!("model {id:?}: {e}"))?;
        models.push(model);
    }
    check_across_models(&models)?;
    Ok(models)
}

/// Fill in what the entry leaves out from the process-level defaults.
fn resolve(
    entry: ModelEntry,
    defaults: &ModelDefaults,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> anyhow::Result<ModelConfig> {
    // The id is compared byte for byte with the request's `model` and sent to
    // cloud-api as the billing key, so it has to be written exactly.
    if entry.id.is_empty() || entry.id.trim() != entry.id || entry.id.chars().any(char::is_control)
    {
        anyhow::bail!(
            "`id` must be the exact cloud-api model name: not empty, no surrounding whitespace"
        );
    }
    let backend_urls = urls("backend_urls", entry.backend_urls)?;
    if backend_urls.is_empty() {
        anyhow::bail!("`backend_urls` is empty: a model needs at least one backend");
    }
    let backend_probe_urls = urls("backend_probe_urls", entry.backend_probe_urls)?;

    let admission_max_inflight = entry
        .admission_max_inflight
        .unwrap_or(defaults.admission_max_inflight);
    let reasoning_off_effort = match entry.reasoning_off_effort {
        Some(effort) if effort.trim().is_empty() => {
            anyhow::bail!("`reasoning_off_effort` is empty")
        }
        Some(effort) => effort.trim().to_string(),
        None => defaults.reasoning_off_effort.clone(),
    };
    // Checked against the off effort the model ends up with, its own or the
    // process-level one: the two must not contradict each other.
    let reasoning_effort_map = EffortMap::new(
        entry
            .reasoning_effort_map
            .map_or_else(Vec::new, |map| map.0),
        &reasoning_off_effort,
    )?;
    let mut model = ModelConfig {
        id: entry.id,
        backend_urls,
        backend_probe_urls,
        backend_long_context_urls: Vec::new(),
        backend_long_context_probe_urls: Vec::new(),
        long_context_above_tokens: 0,
        backend_tier_strict: false,
        admission_max_inflight,
        admission_start_inflight: entry
            .admission_start_inflight
            .or(defaults.admission_start_inflight)
            .unwrap_or(admission_max_inflight),
        admission_queue_saturated_at: entry
            .admission_queue_saturated_at
            .unwrap_or(defaults.admission_queue_saturated_at),
        admission_tier_borrowing: false,
        admission_long_max_inflight_per_host: 0,
        admission_long_reserved_inflight: 0,
        capacity_requests_per_minute: entry
            .capacity_requests_per_minute
            .unwrap_or(defaults.capacity_requests_per_minute),
        discount_to_user: match entry.discount_to_user {
            Some(discount) => validate_discount_to_user(discount, &discount.to_string())?,
            None => defaults.discount_to_user,
        },
        reasoning_off_effort,
        reasoning_effort_map,
        merge_system_messages: entry.merge_system_messages,
        backend_token: match entry.backend_token_env {
            Some(name) => Some(backend_token_from_env(&name, lookup)?),
            None => defaults.backend_token.clone(),
        },
        backend_priority: match entry.backend_priority {
            Some(priority) => Some(
                crate::priority::validate_priority(&priority.to_string())
                    .map_err(|e| anyhow::anyhow!("VLLM_BACKEND_PRIORITY: {e}"))?,
            ),
            None => defaults.backend_priority,
        },
    };
    if let Some(tier) = entry.long_context {
        model.backend_long_context_urls = urls("long_context.backend_urls", tier.backend_urls)?;
        if model.backend_long_context_urls.is_empty() {
            anyhow::bail!(
                "`long_context.backend_urls` is empty: leave `long_context` out for a model without a tier"
            );
        }
        model.backend_long_context_probe_urls =
            urls("long_context.backend_probe_urls", tier.backend_probe_urls)?;
        model.long_context_above_tokens = tier
            .above_tokens
            .unwrap_or(defaults.long_context_above_tokens);
        model.backend_tier_strict = tier.strict.unwrap_or(defaults.backend_tier_strict);
        model.admission_tier_borrowing =
            tier.borrowing.unwrap_or(defaults.admission_tier_borrowing);
        model.admission_long_max_inflight_per_host = tier
            .max_inflight_per_host
            .unwrap_or(defaults.admission_long_max_inflight_per_host);
        model.admission_long_reserved_inflight = tier
            .reserved_inflight
            .unwrap_or(defaults.admission_long_reserved_inflight);
    }
    Ok(model)
}

/// Base URLs as the variables hold them: trimmed, no trailing slash. A blank
/// or unparsable entry is a mistake in a rendered file, not something to skip,
/// and so is anything that is more than a base: a query or fragment (paths
/// are appended to these), or credentials, which do not belong in the file.
fn urls(key: &str, raw: Vec<String>) -> anyhow::Result<Vec<String>> {
    raw.into_iter()
        .map(|url| {
            let url = url.trim().trim_end_matches('/').to_string();
            let parsed = match reqwest::Url::parse(&url) {
                Ok(parsed) if matches!(parsed.scheme(), "http" | "https") => parsed,
                _ => anyhow::bail!("`{key}`: {url:?} is not an http(s) URL"),
            };
            // Not echoed: the URL holds the secret.
            if !parsed.username().is_empty() || parsed.password().is_some() {
                anyhow::bail!("`{key}`: a URL must not carry credentials");
            }
            if parsed.query().is_some() || parsed.fragment().is_some() {
                anyhow::bail!("`{key}`: {url:?} must be a base URL, without a query or fragment");
            }
            Ok(url)
        })
        .collect()
}

/// Where a base URL points, for telling whether two entries are one backend:
/// scheme, host, effective port and path. The parser lower-cases the host,
/// drops a default port, resolves `.`/`..` segments and writes an IP address
/// one way; a trailing dot on the host and a trailing slash on the path are
/// dropped here. So `https://HOST`, `https://host:443` and
/// `https://host/v1/..` are the same endpoint, however they are spelled.
fn endpoint(url: &str) -> anyhow::Result<String> {
    let parsed =
        reqwest::Url::parse(url).map_err(|_| anyhow::anyhow!("{url:?} is not an http(s) URL"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("{url:?} has no host"))?
        .trim_end_matches('.');
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| anyhow::anyhow!("{url:?} has no port"))?;
    Ok(format!(
        "{}://{host}:{port}{}",
        parsed.scheme(),
        parsed.path().trim_end_matches('/')
    ))
}

fn endpoints(urls: &[String]) -> anyhow::Result<Vec<String>> {
    urls.iter().map(|url| endpoint(url)).collect()
}

/// The backend bearer named by `backend_token_env`. The name is not echoed
/// when it is refused: someone may have pasted the token itself there.
fn backend_token_from_env(
    name: &str,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> anyhow::Result<String> {
    let well_formed = name.starts_with(BACKEND_TOKEN_ENV_PREFIX)
        && name
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_');
    if !well_formed {
        anyhow::bail!(
            "`backend_token_env` must be the name of an environment variable starting with {BACKEND_TOKEN_ENV_PREFIX} (the file never holds the token itself)"
        );
    }
    lookup(name)
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .ok_or_else(|| anyhow::anyhow!("`backend_token_env` names {name}, which is not set"))
}

/// The rules a single model's variables are held to, on what the entry
/// resolved to. The messages name the variables each key is named after.
fn validate(model: &ModelConfig, process: &ProcessSettings) -> anyhow::Result<()> {
    check_admission(&AdmissionKnobs {
        max_inflight: model.admission_max_inflight,
        start_inflight: model.admission_start_inflight,
        ramp_step: process.admission_ramp_step,
        ramp_interval_secs: process.admission_ramp_interval_secs,
        backpressure_secs: process.admission_backpressure_secs,
        queue_saturated_at: model.admission_queue_saturated_at,
        retry_after_secs: process.admission_retry_after_secs,
    })?;
    check_probe_urls(&model.backend_urls, &model.backend_probe_urls)?;
    // By endpoint, so the rules that one backend, and one engine, serves one
    // tier hold however a host is spelled (`endpoint`).
    check_long_context_tier(&TierKnobs {
        backend_urls: &endpoints(&model.backend_urls)?,
        backend_probe_urls: &endpoints(&model.backend_probe_urls)?,
        backend_long_context_urls: &endpoints(&model.backend_long_context_urls)?,
        backend_long_context_probe_urls: &endpoints(&model.backend_long_context_probe_urls)?,
        long_context_above_tokens: model.long_context_above_tokens,
        admission_max_inflight: model.admission_max_inflight,
        admission_start_inflight: model.admission_start_inflight,
        admission_tier_borrowing: model.admission_tier_borrowing,
        admission_long_max_inflight_per_host: model.admission_long_max_inflight_per_host,
        admission_long_reserved_inflight: model.admission_long_reserved_inflight,
    })?;
    // Same fail-closed rule as the variable: backends do not bill a request
    // that carries a trusted token, so this process must be able to.
    if model.backend_token.is_some() && !process.bills_usage {
        anyhow::bail!(
            "VLLM_BACKEND_TOKEN requires CLOUD_API_URL and CLOUD_API_USAGE_TOKEN: backends do not bill trusted-token requests"
        );
    }
    // The reasoning handling runs for a model whose backends trust this
    // process (`ModelView::trusted_by_backends`), and the map is part of it.
    // On any other model it would be written down and never applied.
    if !model.reasoning_effort_map.is_empty() && model.backend_token.is_none() {
        anyhow::bail!(
            "`reasoning_effort_map` requires a backend token (`backend_token_env` or VLLM_BACKEND_TOKEN): reasoning controls are only rewritten for a model that has one"
        );
    }
    Ok(())
}

/// What only shows across entries: an id listed twice, and a backend or an
/// engine probe listed under two models. A backend serves one model; a
/// request for another one sent there would be answered, and billed, as the
/// wrong model, and a shared probe would count one engine's load twice.
/// Backends and probes are compared by `endpoint`, not as strings: two
/// spellings of one host are still one backend.
fn check_across_models(models: &[ModelConfig]) -> anyhow::Result<()> {
    let mut ids = std::collections::HashSet::new();
    for model in models {
        if !ids.insert(model.id.as_str()) {
            anyhow::bail!("model {:?} is listed more than once", model.id);
        }
    }
    let (mut backends, mut probes) = (HashMap::new(), HashMap::new());
    for model in models {
        claim(
            &mut backends,
            "backend URL",
            model,
            model
                .backend_urls
                .iter()
                .chain(&model.backend_long_context_urls),
        )?;
        claim(
            &mut probes,
            "probe URL",
            model,
            model
                .backend_probe_urls
                .iter()
                .chain(&model.backend_long_context_probe_urls),
        )?;
    }
    Ok(())
}

/// Record `model` as the owner of the endpoints of `urls`, refusing one that
/// another model owns. `owners` maps an endpoint to the model that listed it
/// first and the URL it wrote.
fn claim<'a>(
    owners: &mut HashMap<String, (&'a str, &'a str)>,
    what: &str,
    model: &'a ModelConfig,
    urls: impl Iterator<Item = &'a String>,
) -> anyhow::Result<()> {
    for url in urls {
        match owners.insert(endpoint(url)?, (&model.id, url)) {
            Some((other, _)) if other == model.id => {}
            Some((other, theirs)) if theirs == url => anyhow::bail!(
                "{what} {url} is listed under both {other:?} and {:?}: a backend serves one model",
                model.id
            ),
            Some((other, theirs)) => anyhow::bail!(
                "{what} {url} is the same endpoint as {theirs}, and is listed under both {other:?} and {:?}: a backend serves one model",
                model.id
            ),
            None => {}
        }
    }
    Ok(())
}

/// One model of the list at runtime: its configuration and everything that
/// keeps its lane apart from the others.
pub struct ServedModel {
    pub config: ModelConfig,
    /// `config.id` as the `model` metric label, leaked once at startup so it
    /// is `'static` (`model_metrics::ModelLabel`).
    label: &'static str,
    /// Carries this model's backend bearer and priority header, and reaches
    /// only its backends.
    pub backend_client: reqwest::Client,
    pub backend_pool: Arc<BackendPool>,
    pub backend_affinity: Arc<BackendConversationAffinity>,
    pub admission: Arc<AdmissionController>,
    /// The last read of the models document did not list this model
    /// (`note_listed`).
    unlisted: AtomicBool,
}

impl ServedModel {
    pub fn id(&self) -> &str {
        &self.config.id
    }

    /// Record whether the models document lists this model, and say whether
    /// that changed since the last read: `/v1/models` is polled, so the
    /// change is worth a log line and the steady state is not.
    pub(crate) fn note_listed(&self, listed: bool) -> bool {
        self.unlisted.swap(!listed, Ordering::Relaxed) == listed
    }

    pub fn view(&self) -> ModelView<'_> {
        ModelView {
            id: &self.config.id,
            label: Some(self.label),
            backend_client: &self.backend_client,
            backend_pool: &self.backend_pool,
            backend_affinity: &self.backend_affinity,
            admission: &self.admission,
            long_context_above_tokens: self.config.long_context_above_tokens,
            backend_tier_strict: self.config.backend_tier_strict,
            reasoning_off_effort: &self.config.reasoning_off_effort,
            reasoning_effort_map: (!self.config.reasoning_effort_map.is_empty())
                .then_some(&self.config.reasoning_effort_map),
            merge_system_messages: self.config.merge_system_messages,
            trusted_by_backends: self.config.backend_token.is_some(),
            discount_to_user: self.config.discount_to_user,
        }
    }
}

/// The model a chat/completions request is served as: the single model of the
/// process, or the list entry its `model` selected. The routes read
/// everything model-specific from here, so both modes run the same code.
#[derive(Clone, Copy)]
pub struct ModelView<'a> {
    /// The billing key, and the name in the signed text.
    pub id: &'a str,
    /// `None` for the single model: its series carry no `model` label.
    pub label: ModelLabel,
    pub backend_client: &'a reqwest::Client,
    pub backend_pool: &'a Arc<BackendPool>,
    pub backend_affinity: &'a Arc<BackendConversationAffinity>,
    pub admission: &'a Arc<AdmissionController>,
    pub long_context_above_tokens: u64,
    pub backend_tier_strict: bool,
    pub reasoning_off_effort: &'a str,
    /// The model's `reasoning_effort_map` when it has entries. Always `None`
    /// for the single model: the key exists in a list only.
    pub reasoning_effort_map: Option<&'a EffortMap>,
    /// The model's `merge_system_messages`. Always `false` for the single
    /// model: the key exists in a list only.
    pub merge_system_messages: bool,
    /// A backend token is configured: the backends are inference-proxies
    /// that trust this process, i.e. it runs as a gateway lane.
    pub trusted_by_backends: bool,
    pub discount_to_user: Option<f64>,
}

impl<'a> ModelView<'a> {
    fn single(state: &'a AppState) -> Self {
        Self {
            id: &state.config.model_name,
            label: None,
            backend_client: &state.backend_client,
            backend_pool: &state.backend_pool,
            backend_affinity: &state.backend_affinity,
            admission: &state.admission,
            long_context_above_tokens: state.config.long_context_above_tokens,
            backend_tier_strict: state.config.backend_tier_strict,
            reasoning_off_effort: &state.config.reasoning_off_effort,
            reasoning_effort_map: None,
            merge_system_messages: false,
            trusted_by_backends: state.config.backend_token.is_some(),
            discount_to_user: state.config.discount_to_user,
        }
    }
}

/// The served models (`AppState::models`).
pub struct ModelList {
    models: Vec<ServedModel>,
    by_id: HashMap<String, usize>,
}

impl ModelList {
    /// Build every model's bundle and start its background tasks: the
    /// engine-load poller when it has probes, and the pool health checker.
    /// `backend_client` builds the HTTP client for a set of default headers,
    /// with the process's pool and timeout settings. Must run inside a Tokio
    /// runtime.
    ///
    /// The health checker runs for a one-backend pool too, which a
    /// single-model process skips: `/healthz` here reports each model from
    /// its pool's view, and a host taken out after a failed connect only
    /// comes back through a probe.
    fn start(
        config: &Config,
        http_client: &reqwest::Client,
        backend_client: &dyn Fn(reqwest::header::HeaderMap) -> anyhow::Result<reqwest::Client>,
    ) -> anyhow::Result<Self> {
        let list = config
            .model_list
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("{MODEL_LIST_FILE_ENV} is not configured"))?;
        let probe_interval = Duration::from_secs(config.backend_probe_interval_secs);
        let mut models = Vec::with_capacity(list.models.len());
        for model in &list.models {
            let label: &'static str = Box::leak(model.id.clone().into_boxed_str());
            let headers = backend_headers(model)?;
            let client = if headers.is_empty() {
                http_client.clone()
            } else {
                backend_client(headers)?
            };
            let pool = Arc::new(
                BackendPool::with_long_context(
                    model.backend_urls.clone(),
                    model.backend_long_context_urls.clone(),
                )
                .with_model(Some(label)),
            );
            // A sample older than three intervals counts as unknown.
            let engine_load =
                Arc::new(EngineLoad::new(pool.len(), probe_interval * 3).with_model(Some(label)));
            let probe_urls = model.pool_probe_urls();
            let engine_probes = !probe_urls.is_empty();
            if engine_probes {
                engine_load::spawn_engine_load_poller(
                    engine_load.clone(),
                    http_client.clone(),
                    probe_urls,
                    probe_interval,
                );
            }
            let admission = Arc::new(AdmissionController::for_model(
                Some(label),
                config.admission_for(model),
                pool.len(),
                engine_load,
            ));
            let affinity = Arc::new(BackendConversationAffinity::new(
                config.backend_conversation_affinity,
                pool.len(),
                config.backend_affinity_max_imbalance,
                config.chat_cache_expiration_secs,
            ));
            crate::backend_pool::spawn_health_check(
                pool.clone(),
                http_client.clone(),
                Duration::from_secs(config.health_check_interval_secs),
                Duration::from_secs(config.health_check_timeout_secs),
                config.health_check_max_failures,
                &config.backend_health_path,
            );
            info!(
                model = %model.id,
                backends = ?model.backend_urls,
                long_context_backends = ?model.backend_long_context_urls,
                long_context_above_tokens = model.long_context_above_tokens,
                tier_strict = model.backend_tier_strict,
                engine_probes,
                conversation_affinity = affinity.is_active(),
                admission_max_inflight = model.admission_max_inflight,
                admission_start_inflight = model.admission_start_inflight,
                admission_queue_saturated_at = model.admission_queue_saturated_at,
                admission_tier_borrowing = model.admission_tier_borrowing,
                admission_long_max_inflight_per_host = model.admission_long_max_inflight_per_host,
                admission_long_reserved_inflight = model.admission_long_reserved_inflight,
                capacity_requests_per_minute = model.capacity_requests_per_minute,
                discount_to_user = model.discount_to_user,
                reasoning_off_effort = %model.reasoning_off_effort,
                reasoning_effort_map = ?model.reasoning_effort_map,
                merge_system_messages = model.merge_system_messages,
                backend_token = model.backend_token.is_some(),
                backend_priority = model.backend_priority,
                "Serving model"
            );
            models.push(ServedModel {
                config: model.clone(),
                label,
                backend_client: client,
                backend_pool: pool,
                backend_affinity: affinity,
                admission,
                unlisted: AtomicBool::new(false),
            });
        }
        let by_id = models
            .iter()
            .enumerate()
            .map(|(index, model)| (model.config.id.clone(), index))
            .collect();
        Ok(Self { models, by_id })
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Always false: a list is refused at startup when it is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// The models in the order the file lists them.
    pub fn iter(&self) -> impl Iterator<Item = &ServedModel> {
        self.models.iter()
    }

    pub fn get(&self, id: &str) -> Option<&ServedModel> {
        self.by_id.get(id).map(|index| &self.models[*index])
    }

    /// The model the request body names: an exact match of its `model`
    /// against the configured ids. Anything else — another model, a
    /// different case, no `model`, a `model` that is not a string — is
    /// OpenAI's 404 `model_not_found`, with nothing dispatched or billed.
    pub fn select(&self, request: &Value) -> Result<&ServedModel, AppError> {
        let requested = requested_model(request);
        match requested.and_then(|id| self.get(id)) {
            Some(model) => {
                count_model_match(ModelMatch::Exact);
                Ok(model)
            }
            None => {
                count_model_match(classify(
                    requested,
                    self.models.iter().map(|model| model.config.id.as_str()),
                ));
                Err(AppError::ModelNotFound {
                    model: requested.map(str::to_string),
                })
            }
        }
    }
}

/// What `app_state` takes from the process that calls it: `main`, or a test
/// that starts a gateway.
pub struct Process<'a> {
    pub config: Arc<Config>,
    pub signing: crate::signing::SigningPair,
    /// The general client: cloud-api, the models document, the health and
    /// engine probes. It never carries a backend bearer.
    pub http_client: reqwest::Client,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    /// Builds a client with these default headers and the process's pool and
    /// timeout settings: one for each model that has a bearer or a priority.
    pub backend_client: &'a dyn Fn(reqwest::header::HeaderMap) -> anyhow::Result<reqwest::Client>,
}

/// The whole state of a process in list mode, with every model's background
/// tasks started. `main` builds its state here and so do the tests, so what
/// is tested is what runs. Must run inside a Tokio runtime.
///
/// The single-model fields of `AppState` are set to serve nothing: no backend
/// bearer, one backend that cannot resolve, admission off, nothing to pin.
/// The routes a list serves do not read them (`model_for`), and the ones that
/// do are not served (`routes::build_router_for`).
pub fn app_state(process: Process<'_>) -> anyhow::Result<AppState> {
    let Process {
        config,
        signing,
        http_client,
        metrics_handle,
        backend_client,
    } = process;
    let models = ModelList::start(&config, &http_client, backend_client)?;
    // What every model's lane shares; each model's own numbers are on its
    // "Serving model" line.
    info!(
        models = models.len(),
        admission_ramp_step = config.admission_ramp_step,
        admission_ramp_interval_secs = config.admission_ramp_interval_secs,
        admission_ttft_p95_max_ms = config.admission_ttft_p95_max_ms,
        admission_backpressure_secs = config.admission_backpressure_secs,
        admission_retry_after_secs = config.admission_retry_after_secs,
        probe_interval_secs = config.backend_probe_interval_secs,
        health_check_interval_secs = config.health_check_interval_secs,
        health_check_max_failures = config.health_check_max_failures,
        health_path = %config.backend_health_path,
        conversation_affinity = config.backend_conversation_affinity,
        connect_failover = config.backend_connect_failover,
        first_token_deadline_ms = config.first_token_deadline_ms,
        first_token_deadline_per_1k_tokens_ms = config.first_token_deadline_per_1k_tokens_ms,
        first_token_deadline_max_ms = config.first_token_deadline_max_ms,
        "Model list enabled"
    );
    Ok(AppState {
        usage_report_delivery: crate::usage_report::UsageReportDelivery::from_config(
            &config,
            &http_client,
        ),
        signing: Arc::new(signing),
        cache: Arc::new(crate::cache::ChatCache::new(
            &config.model_name,
            config.chat_cache_expiration_secs,
        )),
        attestation_cache: Arc::new(crate::attestation::AttestationCache::new(
            config.attestation_cache_ttl_secs,
        )),
        backend_client: http_client.clone(),
        http_client,
        metrics_handle,
        tls_cert_fingerprint: Arc::new(crate::attestation::TlsCertTracker::new(
            config.tls_cert_path.clone(),
        )?),
        backend_pool: Arc::new(BackendPool::new(vec![UNROUTED_BACKEND_URL.to_string()])),
        ohttp_gateway: None,
        ohttp_attestation_ed25519: None,
        fusion_caches: Arc::new(crate::fusion::FusionCaches::default()),
        vllm_dp_affinity: Arc::new(crate::vllm_dp_affinity::VllmDpAffinity::new(
            None,
            config.chat_cache_expiration_secs,
        )),
        backend_affinity: Arc::new(BackendConversationAffinity::new(
            false,
            1,
            config.backend_affinity_max_imbalance,
            config.chat_cache_expiration_secs,
        )),
        admission: Arc::new(AdmissionController::disabled()),
        models: Some(Arc::new(models)),
        config,
    })
}

/// The default headers of a model's backend client: its bearer and its
/// priority, so neither can be attached to a request to cloud-api or to
/// another model's backends.
fn backend_headers(model: &ModelConfig) -> anyhow::Result<reqwest::header::HeaderMap> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(token) = &model.backend_token {
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "model {:?}: the backend token is not a valid header value",
                    model.id
                )
            })?;
        value.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    if let Some(priority) = model.backend_priority {
        headers.insert(
            crate::priority::PRIORITY_HEADER,
            reqwest::header::HeaderValue::from(priority),
        );
    }
    Ok(headers)
}

/// How a request's `model` compares with what the process serves. The label
/// of `request_model_match_total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelMatch {
    /// Byte for byte a configured id.
    Exact,
    /// A configured id in another ASCII case.
    CaseDiffers,
    /// Some other string.
    Other,
    /// Absent, or not a string.
    Missing,
}

impl ModelMatch {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelMatch::Exact => "exact",
            ModelMatch::CaseDiffers => "case_differs",
            ModelMatch::Other => "other",
            ModelMatch::Missing => "missing",
        }
    }
}

fn requested_model(request: &Value) -> Option<&str> {
    request.get("model").and_then(Value::as_str)
}

fn classify<'a>(requested: Option<&str>, configured: impl Iterator<Item = &'a str>) -> ModelMatch {
    let Some(requested) = requested else {
        return ModelMatch::Missing;
    };
    let mut case_differs = false;
    for id in configured {
        if id == requested {
            return ModelMatch::Exact;
        }
        case_differs |= id.eq_ignore_ascii_case(requested);
    }
    if case_differs {
        ModelMatch::CaseDiffers
    } else {
        ModelMatch::Other
    }
}

/// One low-cardinality counter. The request's model string is the caller's:
/// it goes neither into a label nor into a log line.
fn count_model_match(result: ModelMatch) {
    metrics::counter!("request_model_match_total", "result" => result.as_str()).increment(1);
}

/// The model a chat/completions request is served as. Called after
/// authentication, so an unauthenticated caller cannot use the 404 to
/// enumerate models.
///
/// A single-model process serves its one model whatever the body says, as it
/// always has. When it runs as a gateway lane — outside a TEE
/// (`NON_TEE_DEPLOYMENT`) and with a backend token, which together no proxy
/// inside a CVM has — it also counts how the body's `model` compares with
/// `MODEL_NAME`, so the effect of exact matching can be read from production
/// before a list switches it on.
pub fn model_for<'a>(state: &'a AppState, request: &Value) -> Result<ModelView<'a>, AppError> {
    match &state.models {
        Some(models) => models.select(request).map(ServedModel::view),
        None => {
            if state.config.non_tee_deployment && state.config.backend_token.is_some() {
                count_model_match(classify(
                    requested_model(request),
                    std::iter::once(state.config.model_name.as_str()),
                ));
            }
            Ok(ModelView::single(state))
        }
    }
}
