use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use bytes::{Bytes, BytesMut};
use futures_util::{future::join_all, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, error, info};

use crate::agent_loop;
use crate::auth::RequireAuth;
use crate::encryption::{ChunkTransform, ResponseTransform};
use crate::error::AppError;
use crate::proxy::{self, make_usage_reporter, ProxyOpts, ResponseShape, UsageType};
use crate::{AppState, TracingIds};

const FUSION_TYPES: [&str; 2] = ["openrouter:fusion", "nearai:fusion"];
const INTERNAL_FUNCTION_NAME: &str = "__nearai_fusion";

pub struct FusionCaches {
    endpoint: Mutex<Option<EndpointSnapshot>>,
    endpoint_fetch: AsyncMutex<()>,
    attestation_capability: Mutex<HashMap<String, (Instant, bool)>>,
}

impl FusionCaches {
    pub fn new() -> Self {
        Self {
            endpoint: Mutex::new(None),
            endpoint_fetch: AsyncMutex::new(()),
            attestation_capability: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for FusionCaches {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
struct EndpointSnapshot {
    fetched_at: Instant,
    source_url: String,
    by_model: HashMap<String, EndpointInfo>,
}

pub struct ChatCompletionContext {
    pub state: AppState,
    pub auth: RequireAuth,
    pub tracing_ids: TracingIds,
    pub request_hash: String,
    pub is_stream: bool,
    pub response_transform: Option<ResponseTransform>,
    pub chunk_transform: Option<ChunkTransform>,
}

#[derive(Clone, Debug)]
struct EndpointInfo {
    domain: String,
    base_url: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    prompt_tokens: i64,
    completion_tokens: i64,
}

impl Usage {
    fn add(&mut self, other: &Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
    }

    fn total_tokens(&self) -> i64 {
        self.prompt_tokens + self.completion_tokens
    }

    fn to_json(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens(),
        })
    }

    fn from_response(value: &Value) -> Usage {
        let usage = value.get("usage").unwrap_or(&Value::Null);
        Usage {
            prompt_tokens: usage
                .get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            completion_tokens: usage
                .get("completion_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
        }
    }
}

#[derive(Clone, Debug)]
struct FusionConfig {
    analysis_models: Vec<String>,
    judge_model: Option<String>,
    max_tool_calls: u32,
    max_completion_tokens: Option<u64>,
    temperature: Option<f64>,
    reasoning: Option<Value>,
}

#[derive(Clone, Debug)]
struct PanelOutcome {
    model: String,
    domain: Option<String>,
    status: String,
    verifiable: bool,
    usage: Usage,
    web_tool_calls: u32,
    answer: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug)]
struct JudgeOutcome {
    model: String,
    domain: Option<String>,
    status: String,
    verifiable: bool,
    usage: Usage,
    web_tool_calls: u32,
    summary: Option<String>,
    error: Option<String>,
}

struct InternalCallOutcome {
    response: Value,
    usage: Usage,
    web_tool_calls: u32,
}

#[derive(Debug, Deserialize)]
struct EndpointsPayload {
    endpoints: Vec<EndpointEntry>,
}

#[derive(Debug, Deserialize)]
struct EndpointEntry {
    domain: String,
    models: Vec<String>,
}

/// True when the request advertises a Fusion tool entry.
pub fn has_fusion_tool(request: &Value) -> bool {
    request
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|tools| tools.iter().any(is_fusion_tool))
}

/// Return the maximum trusted Fusion depth seen in the request headers.
///
/// Depth headers are accepted only from internal Fusion calls authenticated
/// with `FUSION_INTERNAL_BEARER_TOKEN`; regular clients cannot accidentally
/// self-deny Fusion by sending these headers.
pub fn trusted_request_depth(headers: &axum::http::HeaderMap, state: &AppState) -> u32 {
    let Some(internal_token) = state.config.fusion_internal_bearer_token.as_deref() else {
        return 0;
    };
    let Some(token) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
    else {
        return 0;
    };
    if !crate::auth::token_eq(token, internal_token) {
        return 0;
    }

    ["x-openrouter-fusion-depth", "x-nearai-fusion-depth"]
        .iter()
        .filter_map(|name| headers.get(*name))
        .filter_map(|v| v.to_str().ok())
        .filter_map(|s| s.parse::<u32>().ok())
        .max()
        .unwrap_or(0)
}

pub async fn run_chat_completion(
    ctx: ChatCompletionContext,
    mut request_json: Value,
) -> Result<Response, AppError> {
    let started = Instant::now();
    let fusion_tool = remove_fusion_tool(&mut request_json).ok_or_else(|| {
        AppError::BadRequest("Fusion was enabled but no Fusion tool was found".to_string())
    })?;
    let config = parse_config(&fusion_tool, &ctx.state)?;
    let forced = is_required_tool_choice(&request_json);

    let mut aggregate_usage = Usage::default();
    let mut preflight_status = "skipped".to_string();

    if !forced {
        let mut advertised = request_json.clone();
        append_internal_function_tool(&mut advertised);
        remove_streaming_for_internal_call(&mut advertised);
        let preflight = call_local_chat_json(&ctx.state, &ctx.tracing_ids, advertised).await?;
        let preflight_usage = Usage::from_response(&preflight);
        aggregate_usage.add(&preflight_usage);
        if !contains_fusion_tool_call(&preflight) {
            preflight_status = "no_invocation".to_string();
            let mut response = preflight;
            response["usage"] = aggregate_usage.to_json();
            response["nearai_fusion"] = json!({
                "status": "skipped",
                "reason": "outer_model_did_not_call_fusion",
                "forced": false,
                "preflight_status": preflight_status,
                "aggregate_usage": aggregate_usage.to_json(),
            });
            return finish_response(ctx, response).await;
        }
        preflight_status = "invoked".to_string();
    }

    let panel_results = run_panels(&ctx.state, &ctx.tracing_ids, &request_json, &config).await;
    for panel in &panel_results {
        aggregate_usage.add(&panel.usage);
    }
    let successful_panels: Vec<_> = panel_results
        .iter()
        .filter(|p| p.status == "ok")
        .cloned()
        .collect();
    if successful_panels.is_empty() {
        return Err(AppError::UpstreamParsed {
            status: StatusCode::BAD_GATEWAY,
            message: "all_panels_failed".to_string(),
            error_type: "fusion_error".to_string(),
        });
    }

    let judge = run_judge(
        &ctx.state,
        &ctx.tracing_ids,
        &request_json,
        &config,
        &successful_panels,
    )
    .await;
    if let Some(judge) = &judge {
        aggregate_usage.add(&judge.usage);
    }

    let synthesis_request =
        build_synthesis_request(&request_json, &successful_panels, judge.as_ref());
    // TODO(billing): if synthesis fails here, panel and judge usage gathered
    // above is not reported. V1 assumes local synthesis failures are rare; V2
    // should report partial aggregate usage on this error path.
    let mut synthesis =
        call_local_chat_json(&ctx.state, &ctx.tracing_ids, synthesis_request).await?;
    let synthesis_usage = Usage::from_response(&synthesis);
    aggregate_usage.add(&synthesis_usage);
    synthesis["usage"] = aggregate_usage.to_json();
    synthesis["nearai_fusion"] = metadata_json(MetadataJson {
        forced,
        preflight_status: &preflight_status,
        panels: &panel_results,
        judge: judge.as_ref(),
        synthesis_usage: &synthesis_usage,
        aggregate_usage: &aggregate_usage,
        elapsed: started.elapsed(),
        max_tool_calls: config.max_tool_calls,
    });

    finish_response(ctx, synthesis).await
}

fn is_fusion_tool(tool: &Value) -> bool {
    tool.get("type")
        .and_then(|v| v.as_str())
        .is_some_and(|t| FUSION_TYPES.contains(&t))
}

fn remove_fusion_tool(request: &mut Value) -> Option<Value> {
    let tools = request.get_mut("tools")?.as_array_mut()?;
    let index = tools.iter().position(is_fusion_tool)?;
    let tool = tools.remove(index);
    if tools.is_empty() {
        request.as_object_mut()?.remove("tools");
    }
    Some(tool)
}

fn parse_config(tool: &Value, state: &AppState) -> Result<FusionConfig, AppError> {
    let mut analysis_models = read_string_array(tool, "analysis_models")
        .or_else(|| {
            tool.get("config")
                .and_then(|v| read_string_array(v, "analysis_models"))
        })
        .unwrap_or_else(|| state.config.fusion_default_analysis_models.clone());
    analysis_models = normalize_unique_models(analysis_models);

    if analysis_models.is_empty() {
        return Err(AppError::BadRequest(
            "Fusion requires analysis_models or FUSION_DEFAULT_ANALYSIS_MODELS".to_string(),
        ));
    }
    if analysis_models.len() > state.config.fusion_max_panel_models {
        return Err(AppError::BadRequest(format!(
            "Fusion analysis_models exceeds FUSION_MAX_PANEL_MODELS ({})",
            state.config.fusion_max_panel_models
        )));
    }

    Ok(FusionConfig {
        analysis_models,
        judge_model: read_string(tool, "model").map(|m| normalize_model_name(&m)),
        max_tool_calls: bounded_max_tool_calls(
            read_u32(tool, "max_tool_calls"),
            state.config.agent_loop_max_iterations,
        ),
        max_completion_tokens: read_u64(tool, "max_completion_tokens"),
        temperature: read_f64(tool, "temperature"),
        reasoning: tool.get("reasoning").cloned(),
    })
}

fn read_string_array(value: &Value, key: &str) -> Option<Vec<String>> {
    value.get(key)?.as_array().map(|items| {
        items
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect()
    })
}

fn read_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn read_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
}

fn read_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|v| v.as_u64())
}

fn read_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(|v| v.as_f64())
}

fn normalize_model_name(model: &str) -> String {
    model.trim().trim_start_matches('~').to_string()
}

fn normalize_unique_models(models: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for model in models {
        let model = normalize_model_name(&model);
        if !model.is_empty() && seen.insert(model.clone()) {
            unique.push(model);
        }
    }
    unique
}

fn bounded_max_tool_calls(requested: Option<u32>, server_cap: u32) -> u32 {
    requested.unwrap_or(server_cap).min(server_cap)
}

fn is_required_tool_choice(request: &Value) -> bool {
    match request.get("tool_choice") {
        Some(Value::String(s)) => s == "required",
        Some(Value::Object(obj)) => obj
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .is_some_and(|name| name == INTERNAL_FUNCTION_NAME),
        _ => false,
    }
}

fn append_internal_function_tool(request: &mut Value) {
    let internal_tool = json!({
        "type": "function",
        "function": {
            "name": INTERNAL_FUNCTION_NAME,
            "description": "Run server-side multi-model deliberation and return a synthesized answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "Optional focus for the deliberation."
                    }
                },
                "additionalProperties": false
            }
        }
    });

    if let Some(tools) = request.get_mut("tools").and_then(|v| v.as_array_mut()) {
        tools.retain(|tool| {
            tool.get("type").and_then(|v| v.as_str())
                != Some(agent_loop::WEB_CONTEXT_SEARCH_TOOL_NAME)
        });
        tools.push(internal_tool);
    } else {
        request["tools"] = json!([internal_tool]);
    }
    request["parallel_tool_calls"] = false.into();
}

fn remove_streaming_for_internal_call(request: &mut Value) {
    request["stream"] = Value::Bool(false);
    if let Some(obj) = request.as_object_mut() {
        obj.remove("stream_options");
    }
}

fn strip_tools_for_internal_call(request: &mut Value) {
    if let Some(obj) = request.as_object_mut() {
        obj.remove("tools");
        obj.remove("tool_choice");
        obj.remove("stream_options");
    }
    request["stream"] = Value::Bool(false);
}

fn contains_fusion_tool_call(response: &Value) -> bool {
    response
        .get("choices")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|choice| choice.get("message"))
        .filter_map(|message| message.get("tool_calls"))
        .filter_map(|v| v.as_array())
        .flatten()
        .any(|call| {
            call.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .is_some_and(|name| name == INTERNAL_FUNCTION_NAME)
        })
}

async fn run_panels(
    state: &AppState,
    tracing_ids: &TracingIds,
    request_json: &Value,
    config: &FusionConfig,
) -> Vec<PanelOutcome> {
    let futures = config.analysis_models.iter().cloned().map(|model| {
        let state = state.clone();
        let tracing_ids = tracing_ids.clone();
        let request_json = request_json.clone();
        let config = config.clone();
        async move { call_panel(&state, &tracing_ids, &request_json, &config, model).await }
    });
    join_all(futures).await
}

async fn call_panel(
    state: &AppState,
    tracing_ids: &TracingIds,
    request_json: &Value,
    config: &FusionConfig,
    model: String,
) -> PanelOutcome {
    match endpoint_for_model(state, &model).await {
        Ok(endpoint) => {
            let verifiable = probe_attestation(state, &endpoint).await;
            let body = build_panel_request(request_json, config, &model);
            match call_direct_chat_with_web_tools(
                state,
                tracing_ids,
                &endpoint,
                body,
                config.max_tool_calls,
            )
            .await
            {
                Ok(outcome) => PanelOutcome {
                    model,
                    domain: Some(endpoint.domain),
                    status: "ok".to_string(),
                    verifiable,
                    usage: outcome.usage,
                    web_tool_calls: outcome.web_tool_calls,
                    answer: extract_message_content(&outcome.response),
                    error: None,
                },
                Err(e) => PanelOutcome {
                    model,
                    domain: Some(endpoint.domain),
                    status: "failed".to_string(),
                    verifiable,
                    usage: Usage::default(),
                    web_tool_calls: 0,
                    answer: None,
                    error: Some(e.to_string()),
                },
            }
        }
        Err(e) => PanelOutcome {
            model,
            domain: None,
            status: "failed".to_string(),
            verifiable: false,
            usage: Usage::default(),
            web_tool_calls: 0,
            answer: None,
            error: Some(e.to_string()),
        },
    }
}

async fn run_judge(
    state: &AppState,
    tracing_ids: &TracingIds,
    request_json: &Value,
    config: &FusionConfig,
    panels: &[PanelOutcome],
) -> Option<JudgeOutcome> {
    let model = config
        .judge_model
        .clone()
        .unwrap_or_else(|| panels[0].model.clone());
    let endpoint = match endpoint_for_model(state, &model).await {
        Ok(endpoint) => endpoint,
        Err(e) => {
            return Some(JudgeOutcome {
                model,
                domain: None,
                status: "failed".to_string(),
                verifiable: false,
                usage: Usage::default(),
                web_tool_calls: 0,
                summary: None,
                error: Some(e.to_string()),
            });
        }
    };
    let verifiable = probe_attestation(state, &endpoint).await;
    let body = build_judge_request(request_json, config, panels, &model);
    match call_direct_chat_with_web_tools(
        state,
        tracing_ids,
        &endpoint,
        body,
        config.max_tool_calls,
    )
    .await
    {
        Ok(outcome) => {
            let content = extract_message_content(&outcome.response);
            match content.as_deref().and_then(parse_judge_summary) {
                Some(summary) => Some(JudgeOutcome {
                    model,
                    domain: Some(endpoint.domain),
                    status: "ok".to_string(),
                    verifiable,
                    usage: outcome.usage,
                    web_tool_calls: outcome.web_tool_calls,
                    summary: Some(summary),
                    error: None,
                }),
                None => Some(JudgeOutcome {
                    model,
                    domain: Some(endpoint.domain),
                    status: "invalid_json_degraded".to_string(),
                    verifiable,
                    usage: outcome.usage,
                    web_tool_calls: outcome.web_tool_calls,
                    summary: None,
                    error: Some("judge returned invalid JSON".to_string()),
                }),
            }
        }
        Err(e) => Some(JudgeOutcome {
            model,
            domain: Some(endpoint.domain),
            status: "failed".to_string(),
            verifiable,
            usage: Usage::default(),
            web_tool_calls: 0,
            summary: None,
            error: Some(e.to_string()),
        }),
    }
}

fn build_panel_request(request_json: &Value, config: &FusionConfig, model: &str) -> Value {
    let mut request = request_json.clone();
    request["model"] = Value::String(model.to_string());
    request["messages"] = augmented_messages(
        request_json,
        "You are one member of a private multi-model panel. Answer the user's request independently, with concise reasoning and a concrete final answer.",
        None,
    );
    apply_generation_config(&mut request, config);
    request
}

fn build_judge_request(
    request_json: &Value,
    config: &FusionConfig,
    panels: &[PanelOutcome],
    model: &str,
) -> Value {
    let panel_payload: Vec<Value> = panels
        .iter()
        .map(|p| {
            json!({
                "model": p.model,
                "answer": p.answer.as_deref().unwrap_or(""),
            })
        })
        .collect();
    let prompt = json!({
        "original_messages": request_json.get("messages").cloned().unwrap_or_else(|| json!([])),
        "panel_answers": panel_payload,
        "instruction": "Return JSON with keys: consensus, disagreements, strengths, risks, synthesis_guidance. Do not include markdown fences."
    });
    let mut request = json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a judge for private multi-model deliberation. Compare panel answers and return strict JSON only."
            },
            {
                "role": "user",
                "content": prompt.to_string()
            }
        ],
        "response_format": {"type": "json_object"},
        "stream": false
    });
    if has_web_context_search_tool(request_json) {
        request["tools"] = json!([{
            "type": agent_loop::WEB_CONTEXT_SEARCH_TOOL_NAME
        }]);
    }
    apply_generation_config(&mut request, config);
    request
}

fn build_synthesis_request(
    request_json: &Value,
    panels: &[PanelOutcome],
    judge: Option<&JudgeOutcome>,
) -> Value {
    let mut request = request_json.clone();
    strip_tools_for_internal_call(&mut request);
    let mut summaries = String::new();
    if let Some(judge_summary) = judge.and_then(|j| j.summary.as_deref()) {
        summaries.push_str("Judge summary:\n");
        summaries.push_str(judge_summary);
        summaries.push_str("\n\n");
    } else {
        summaries.push_str("Panel summaries:\n");
        for panel in panels {
            summaries.push_str("- ");
            summaries.push_str(&panel.model);
            summaries.push_str(": ");
            summaries.push_str(panel.answer.as_deref().unwrap_or(""));
            summaries.push('\n');
        }
    }
    request["messages"] = augmented_messages(
        request_json,
        "You are the final synthesis model for private multi-model deliberation. Use the deliberation summary to answer the original user directly. Do not mention hidden panel details unless the user explicitly asks how the answer was produced.",
        Some(&summaries),
    );
    request
}

fn augmented_messages(request_json: &Value, system: &str, deliberation: Option<&str>) -> Value {
    let mut messages = Vec::new();
    messages.push(json!({"role": "system", "content": system}));
    if let Some(original) = request_json.get("messages").and_then(|v| v.as_array()) {
        messages.extend(original.iter().cloned());
    }
    if let Some(summary) = deliberation {
        messages.push(json!({
            "role": "user",
            "content": format!("Private deliberation summary:\n{summary}\n\nProduce the final answer now.")
        }));
    }
    Value::Array(messages)
}

fn apply_generation_config(request: &mut Value, config: &FusionConfig) {
    if let Some(max_tokens) = config.max_completion_tokens {
        request["max_completion_tokens"] = Value::Number(max_tokens.into());
    }
    if let Some(temp) = config.temperature {
        if let Some(num) = serde_json::Number::from_f64(temp) {
            request["temperature"] = Value::Number(num);
        }
    }
    if let Some(reasoning) = &config.reasoning {
        request["reasoning"] = reasoning.clone();
    }
}

async fn call_local_chat_json(
    state: &AppState,
    tracing_ids: &TracingIds,
    mut body: Value,
) -> Result<Value, AppError> {
    strip_fusion_tools_if_any(&mut body);
    remove_streaming_for_internal_call(&mut body);
    let body = serde_json::to_vec(&body).map_err(|e| AppError::Internal(e.into()))?;
    let (url, _guard) = state.backend_pool.select_url("/v1/chat/completions");
    post_chat_json(state, &url, None, tracing_ids, body, "local_synthesis", 1).await
}

async fn call_direct_chat_json(
    state: &AppState,
    tracing_ids: &TracingIds,
    endpoint: &EndpointInfo,
    body: Value,
) -> Result<Value, AppError> {
    // SECURITY(V1): E2EE is terminated at this orchestrating proxy. Panel and
    // judge calls forward plaintext over TLS to trusted direct TEE endpoints;
    // V2 should support per-hop encryption or cryptographic panel binding.
    let token = state
        .config
        .fusion_internal_bearer_token
        .as_deref()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("Fusion internal token missing")))?;
    let url = format!(
        "{}/chat/completions",
        endpoint.base_url.trim_end_matches('/')
    );
    let body = serde_json::to_vec(&body).map_err(|e| AppError::Internal(e.into()))?;
    post_chat_json(
        state,
        &url,
        Some(token),
        tracing_ids,
        body,
        "fusion_direct",
        state.config.fusion_internal_max_attempts,
    )
    .await
}

async fn call_direct_chat_with_web_tools(
    state: &AppState,
    tracing_ids: &TracingIds,
    endpoint: &EndpointInfo,
    mut body: Value,
    max_tool_calls: u32,
) -> Result<InternalCallOutcome, AppError> {
    if !has_web_context_search_tool(&body) || max_tool_calls == 0 {
        strip_tools_for_internal_call(&mut body);
        let response = call_direct_chat_json(state, tracing_ids, endpoint, body).await?;
        let usage = Usage::from_response(&response);
        return Ok(InternalCallOutcome {
            response,
            usage,
            web_tool_calls: 0,
        });
    }

    let brave_url = state
        .config
        .web_context_search_url
        .as_deref()
        .ok_or_else(|| {
            AppError::BadRequest(
                "web_context_search is not configured on this deployment".to_string(),
            )
        })?;
    let brave_key = state
        .config
        .web_context_search_api_key
        .as_deref()
        .ok_or_else(|| {
            AppError::BadRequest(
                "web_context_search is not configured on this deployment".to_string(),
            )
        })?;
    let tool_timeout = Duration::from_secs(state.config.web_context_search_timeout_secs);

    strip_fusion_tools_if_any(&mut body);
    retain_only_web_context_search_tool(&mut body);
    agent_loop::rewrite_tool_for_upstream(&mut body);
    remove_streaming_for_internal_call(&mut body);

    let mut aggregate_usage = Usage::default();
    let mut executed_tool_calls = 0u32;

    loop {
        let response = call_direct_chat_json(state, tracing_ids, endpoint, body.clone()).await?;
        aggregate_usage.add(&Usage::from_response(&response));

        let tool_calls = extract_web_context_tool_calls(&response);
        if tool_calls.is_empty() {
            return Ok(InternalCallOutcome {
                response,
                usage: aggregate_usage,
                web_tool_calls: executed_tool_calls,
            });
        }

        let mut tool_results = Vec::with_capacity(tool_calls.len());
        for tool_call in &tool_calls {
            let tool_call_id = tool_call
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_default();
            let remaining = max_tool_calls.saturating_sub(executed_tool_calls);
            if remaining == 0 {
                tool_results.push((
                    tool_call_id,
                    "limit".to_string(),
                    "Web search tool-call limit reached. Produce the best answer from the context already available.".to_string(),
                ));
                continue;
            }

            let args = tool_call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let query = parse_web_context_query(args);
            let (status, output) = match query {
                Some(query) if !query.is_empty() => {
                    executed_tool_calls = executed_tool_calls.saturating_add(1);
                    match agent_loop::brave_llm_context_search_text(
                        &state.http_client,
                        brave_url,
                        brave_key,
                        &query,
                        tool_timeout,
                        tracing_ids,
                    )
                    .await
                    {
                        Ok(output) => ("ok".to_string(), output),
                        Err(err) if err == "timeout" => (
                            "timeout".to_string(),
                            "Web search timed out. Answer using the available context.".to_string(),
                        ),
                        Err(_) => (
                            "error".to_string(),
                            "Web search failed. Answer using the available context.".to_string(),
                        ),
                    }
                }
                _ => {
                    executed_tool_calls = executed_tool_calls.saturating_add(1);
                    (
                        "error".to_string(),
                        "Tool arguments were missing or invalid; no search performed.".to_string(),
                    )
                }
            };
            tool_results.push((tool_call_id, status, output));
        }

        append_tool_call_results(&mut body, tool_calls, tool_results)?;
        if executed_tool_calls >= max_tool_calls {
            if let Some(obj) = body.as_object_mut() {
                obj.remove("tools");
                obj.remove("tool_choice");
                obj.remove("parallel_tool_calls");
            }
        }
        remove_streaming_for_internal_call(&mut body);
    }
}

async fn post_chat_json(
    state: &AppState,
    url: &str,
    bearer: Option<&str>,
    tracing_ids: &TracingIds,
    body: Vec<u8>,
    label: &'static str,
    max_attempts: usize,
) -> Result<Value, AppError> {
    let body = Bytes::from(body);
    let attempts = max_attempts.max(1);
    for attempt in 1..=attempts {
        let attempt_started = Instant::now();
        let mut req = state
            .http_client
            .post(url)
            .timeout(Duration::from_secs(state.config.fusion_panel_timeout_secs))
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .header("x-openrouter-fusion-depth", "1")
            .header("x-nearai-fusion-depth", "1");
        if let Some(token) = bearer {
            req = req.bearer_auth(token);
        }
        for (k, v) in tracing_ids.upstream_headers() {
            req = req.header(k, v);
        }

        let response = match req.body(body.clone()).send().await {
            Ok(response) => response,
            Err(e) => {
                record_upstream_attempt(label, attempt_started);
                let retry = attempt < attempts && is_retryable_transport_error(&e);
                if retry {
                    sleep_before_fusion_retry(state, attempt).await;
                    continue;
                }
                return Err(map_fusion_transport_error(e));
            }
        };

        let status = response.status();
        if !status.is_success() && attempt < attempts && status.is_server_error() {
            record_upstream_attempt(label, attempt_started);
            sleep_before_fusion_retry(state, attempt).await;
            continue;
        }

        let bytes =
            match read_response_limited(response, state.config.fusion_max_response_bytes).await {
                Ok(bytes) => bytes,
                Err(e) => {
                    record_upstream_attempt(label, attempt_started);
                    if attempt < attempts && e.is_retryable() {
                        sleep_before_fusion_retry(state, attempt).await;
                        continue;
                    }
                    return Err(e.into_app_error());
                }
            };

        record_upstream_attempt(label, attempt_started);
        if !status.is_success() {
            return Err(AppError::Upstream {
                status,
                body: bytes,
            });
        }
        return serde_json::from_slice(&bytes).map_err(|e| AppError::Internal(e.into()));
    }

    Err(AppError::Internal(anyhow::anyhow!(
        "Fusion retry loop exhausted unexpectedly"
    )))
}

fn record_upstream_attempt(label: &'static str, started: Instant) {
    metrics::histogram!("upstream_request_duration_seconds", "endpoint" => label)
        .record(started.elapsed().as_secs_f64());
}

fn is_retryable_transport_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_request() || error.is_body()
}

fn map_fusion_transport_error(error: reqwest::Error) -> AppError {
    if error.is_timeout() {
        AppError::UpstreamParsed {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: "fusion_panel_timeout".to_string(),
            error_type: "fusion_error".to_string(),
        }
    } else {
        AppError::Internal(error.into())
    }
}

async fn sleep_before_fusion_retry(state: &AppState, attempt: usize) {
    let shift = u32::try_from(attempt.saturating_sub(1)).unwrap_or(u32::MAX);
    let multiplier = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
    let upper_ms = state
        .config
        .fusion_internal_retry_initial_backoff_ms
        .saturating_mul(multiplier)
        .min(5_000);
    let delay_ms = rand::random_range(0..=upper_ms);
    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
}

async fn read_response_limited(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<Bytes, FusionBodyReadError> {
    if response
        .content_length()
        .is_some_and(|len| len > max_bytes as u64)
    {
        return Err(FusionBodyReadError::TooLarge);
    }

    let mut body = BytesMut::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(FusionBodyReadError::Transport)?;
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(FusionBodyReadError::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

enum FusionBodyReadError {
    TooLarge,
    Transport(reqwest::Error),
}

impl FusionBodyReadError {
    fn is_retryable(&self) -> bool {
        match self {
            FusionBodyReadError::TooLarge => false,
            FusionBodyReadError::Transport(error) => is_retryable_transport_error(error),
        }
    }

    fn into_app_error(self) -> AppError {
        match self {
            FusionBodyReadError::TooLarge => fusion_response_too_large(),
            FusionBodyReadError::Transport(error) => map_fusion_transport_error(error),
        }
    }
}

fn fusion_response_too_large() -> AppError {
    AppError::UpstreamParsed {
        status: StatusCode::BAD_GATEWAY,
        message: "fusion_response_too_large".to_string(),
        error_type: "fusion_error".to_string(),
    }
}

fn strip_fusion_tools_if_any(request: &mut Value) {
    if let Some(tools) = request.get_mut("tools").and_then(|v| v.as_array_mut()) {
        tools.retain(|tool| !is_fusion_tool(tool));
    }
    if request
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|tools| tools.is_empty())
    {
        if let Some(obj) = request.as_object_mut() {
            obj.remove("tools");
        }
    }
}

fn has_web_context_search_tool(request: &Value) -> bool {
    request
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|tools| {
            tools.iter().any(|tool| {
                tool.get("type").and_then(|v| v.as_str())
                    == Some(agent_loop::WEB_CONTEXT_SEARCH_TOOL_NAME)
            })
        })
}

fn retain_only_web_context_search_tool(request: &mut Value) {
    let Some(tools) = request.get_mut("tools").and_then(|v| v.as_array_mut()) else {
        return;
    };
    tools.retain(|tool| {
        tool.get("type").and_then(|v| v.as_str()) == Some(agent_loop::WEB_CONTEXT_SEARCH_TOOL_NAME)
    });
    if tools.is_empty() {
        if let Some(obj) = request.as_object_mut() {
            obj.remove("tools");
        }
    }
}

fn extract_web_context_tool_calls(response: &Value) -> Vec<Value> {
    let finish_reason = response
        .pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str());
    if finish_reason != Some("tool_calls") {
        return Vec::new();
    }
    let Some(tool_calls) = response
        .pointer("/choices/0/message/tool_calls")
        .and_then(|v| v.as_array())
    else {
        return Vec::new();
    };
    if tool_calls.is_empty()
        || !tool_calls.iter().all(|call| {
            call.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                == Some(agent_loop::WEB_CONTEXT_SEARCH_TOOL_NAME)
        })
    {
        return Vec::new();
    }
    tool_calls.clone()
}

fn parse_web_context_query(arguments: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(arguments).ok()?;
    parsed
        .get("query")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
}

fn append_tool_call_results(
    request: &mut Value,
    tool_calls: Vec<Value>,
    tool_results: Vec<(String, String, String)>,
) -> Result<(), AppError> {
    let messages = request
        .get_mut("messages")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| AppError::BadRequest("Fusion request requires messages".to_string()))?;
    messages.push(json!({
        "role": "assistant",
        "tool_calls": tool_calls,
    }));
    for (tool_call_id, status, output) in tool_results {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json!({
                "status": status,
                "name": agent_loop::WEB_CONTEXT_SEARCH_TOOL_NAME,
                "output": output,
            }).to_string(),
        }));
    }
    Ok(())
}

async fn endpoint_for_model(state: &AppState, model: &str) -> Result<EndpointInfo, AppError> {
    let snapshot = endpoint_snapshot(state).await?;
    snapshot
        .by_model
        .get(model)
        .cloned()
        .ok_or_else(|| AppError::BadRequest(format!("Fusion model is not discoverable: {model}")))
}

async fn endpoint_snapshot(state: &AppState) -> Result<EndpointSnapshot, AppError> {
    {
        let guard = state.fusion_caches.endpoint.lock().unwrap();
        if let Some(snapshot) = guard.as_ref() {
            if snapshot.source_url == state.config.fusion_endpoints_url
                && snapshot.fetched_at.elapsed()
                    < Duration::from_secs(state.config.fusion_endpoints_ttl_secs)
            {
                return Ok(snapshot.clone());
            }
        }
    }

    let _fetch_guard = state.fusion_caches.endpoint_fetch.lock().await;
    {
        let guard = state.fusion_caches.endpoint.lock().unwrap();
        if let Some(snapshot) = guard.as_ref() {
            if snapshot.source_url == state.config.fusion_endpoints_url
                && snapshot.fetched_at.elapsed()
                    < Duration::from_secs(state.config.fusion_endpoints_ttl_secs)
            {
                return Ok(snapshot.clone());
            }
        }
    }

    let response = state
        .http_client
        .get(&state.config.fusion_endpoints_url)
        .timeout(Duration::from_secs(state.config.fusion_panel_timeout_secs))
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                AppError::UpstreamParsed {
                    status: StatusCode::GATEWAY_TIMEOUT,
                    message: "fusion_endpoint_timeout".to_string(),
                    error_type: "fusion_error".to_string(),
                }
            } else {
                AppError::Internal(e.into())
            }
        })?;
    let status = response.status();
    let bytes = read_response_limited(response, state.config.fusion_max_response_bytes)
        .await
        .map_err(FusionBodyReadError::into_app_error)?;
    if !status.is_success() {
        return Err(AppError::Upstream {
            status,
            body: bytes,
        });
    }
    let payload: EndpointsPayload =
        serde_json::from_slice(&bytes).map_err(|e| AppError::Internal(e.into()))?;
    let mut by_model = HashMap::new();
    for endpoint in payload.endpoints {
        let base_url = endpoint_base_url(&endpoint.domain);
        for model in endpoint.models {
            by_model.insert(
                normalize_model_name(&model),
                EndpointInfo {
                    domain: endpoint.domain.clone(),
                    base_url: base_url.clone(),
                },
            );
        }
    }
    let snapshot = EndpointSnapshot {
        fetched_at: Instant::now(),
        source_url: state.config.fusion_endpoints_url.clone(),
        by_model,
    };
    let mut guard = state.fusion_caches.endpoint.lock().unwrap();
    *guard = Some(snapshot.clone());
    Ok(snapshot)
}

fn endpoint_base_url(domain: &str) -> String {
    let trimmed = domain.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        if trimmed.ends_with("/v1") {
            trimmed.to_string()
        } else {
            format!("{trimmed}/v1")
        }
    } else {
        format!("https://{trimmed}/v1")
    }
}

async fn probe_attestation(state: &AppState, endpoint: &EndpointInfo) -> bool {
    let cache_ttl = Duration::from_secs(state.config.fusion_endpoints_ttl_secs);
    {
        let guard = state.fusion_caches.attestation_capability.lock().unwrap();
        if let Some((fetched_at, verifiable)) = guard.get(&endpoint.base_url) {
            if fetched_at.elapsed() < cache_ttl {
                return *verifiable;
            }
        }
    }

    let url = format!(
        "{}/attestation/report",
        endpoint.base_url.trim_end_matches('/')
    );
    let verifiable = match state
        .http_client
        .get(url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => true,
        Ok(response) => {
            debug!(
                domain = %endpoint.domain,
                status = %response.status(),
                "Fusion attestation probe not supported"
            );
            false
        }
        Err(e) => {
            debug!(domain = %endpoint.domain, error = %e, "Fusion attestation probe failed");
            false
        }
    };
    // V1 records endpoint verifiability as metadata only; panel outputs are
    // still usable when the direct endpoint does not expose an attestation
    // report. TODO(V2): make panel admission policy configurable.
    let mut guard = state.fusion_caches.attestation_capability.lock().unwrap();
    guard.insert(endpoint.base_url.clone(), (Instant::now(), verifiable));
    verifiable
}

fn extract_message_content(response: &Value) -> Option<String> {
    response
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .map(|s| s.to_string())
}

fn parse_judge_summary(content: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(content).ok()?;
    Some(parsed.to_string())
}

struct MetadataJson<'a> {
    forced: bool,
    preflight_status: &'a str,
    panels: &'a [PanelOutcome],
    judge: Option<&'a JudgeOutcome>,
    synthesis_usage: &'a Usage,
    aggregate_usage: &'a Usage,
    elapsed: Duration,
    max_tool_calls: u32,
}

fn metadata_json(input: MetadataJson<'_>) -> Value {
    // Attestation V1 covers only the final synthesis response. Panel and judge
    // attestation is an informational liveness probe and is not
    // cryptographically bound into the final response signature.
    json!({
        "status": "invoked",
        "forced": input.forced,
        "preflight_status": input.preflight_status,
        "max_tool_calls": input.max_tool_calls,
        "panel": input.panels.iter().map(|p| {
            json!({
                "model": p.model,
                "status": p.status,
                "domain": p.domain,
                "verifiable": p.verifiable,
                "usage": p.usage.to_json(),
                "web_tool_calls": p.web_tool_calls,
                "error": p.error,
            })
        }).collect::<Vec<_>>(),
        "judge": input.judge.map(|j| {
            json!({
                "model": j.model,
                "status": j.status,
                "domain": j.domain,
                "verifiable": j.verifiable,
                "usage": j.usage.to_json(),
                "web_tool_calls": j.web_tool_calls,
                "error": j.error,
            })
        }),
        "synthesis": {
            "status": "ok",
            "domain": "local",
            "usage": input.synthesis_usage.to_json(),
        },
        "aggregate_usage": input.aggregate_usage.to_json(),
        "attestation": {
            "per_member_signatures": false,
            "final_signed_by": "inference-proxy",
            "verifiable_members": input.panels.iter().filter(|p| p.verifiable).count()
                + input.judge.filter(|j| j.verifiable).map(|_| 1).unwrap_or(0),
        },
        "duration_ms": input.elapsed.as_millis(),
    })
}

async fn finish_response(
    ctx: ChatCompletionContext,
    response_json: Value,
) -> Result<Response, AppError> {
    let opts = ProxyOpts {
        signing: ctx.state.signing.clone(),
        cache: ctx.state.cache.clone(),
        id_prefix: "chatcmpl".to_string(),
        model_name: ctx.state.config.model_name.clone(),
        usage_reporter: make_usage_reporter(&ctx.auth, &ctx.state),
        usage_type: UsageType::ChatCompletion,
        request_hash: Some(ctx.request_hash.clone()),
        response_transform: ctx.response_transform,
        chunk_transform: ctx.chunk_transform,
        backend_guard: None,
        response_shape: ResponseShape::ChatCompletion,
        tracing_ids: Some(ctx.tracing_ids),
    };

    if ctx.is_stream {
        stream_final_response(response_json, ctx.request_hash, opts).await
    } else {
        let body = serde_json::to_vec(&response_json).map_err(|e| AppError::Internal(e.into()))?;
        proxy::sign_and_cache_json_response(&body, &ctx.request_hash, opts, StatusCode::OK).await
    }
}

async fn stream_final_response(
    response_json: Value,
    request_sha256: String,
    opts: ProxyOpts,
) -> Result<Response, AppError> {
    let id = response_json
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            format!(
                "chatcmpl-{}",
                &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
            )
        });
    let model = response_json
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&opts.model_name)
        .to_string();
    let created = response_json
        .get("created")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64
        });
    let content = extract_message_content(&response_json).unwrap_or_default();
    let usage = response_json
        .get("usage")
        .cloned()
        .unwrap_or_else(|| Usage::default().to_json());
    let metadata = response_json.get("nearai_fusion").cloned();
    let finish_reason = response_json
        .pointer("/choices/0/finish_reason")
        .cloned()
        .unwrap_or_else(|| Value::String("stop".to_string()));

    let mut chunks = Vec::new();
    chunks.push(json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}],
    }));
    if !content.is_empty() {
        chunks.push(json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}],
        }));
    }
    chunks.push(json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }));
    let mut usage_chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
    });
    if let Some(metadata) = metadata {
        usage_chunk["nearai_fusion"] = metadata;
    }
    chunks.push(usage_chunk);

    let mut hasher = Sha256::new();
    let mut bytes = Vec::new();
    for mut chunk in chunks {
        if let Some(transform) = &opts.chunk_transform {
            transform(&mut chunk)?;
        }
        let serialized = serde_json::to_string(&chunk).map_err(|e| AppError::Internal(e.into()))?;
        let frame = format!("data: {serialized}\n\n");
        hasher.update(frame.as_bytes());
        bytes.push(Ok::<Bytes, std::io::Error>(Bytes::from(frame)));
    }
    hasher.update(b"data: [DONE]\n\n");
    bytes.push(Ok(Bytes::from_static(b"data: [DONE]\n\n")));

    let response_sha256 = hex::encode(hasher.finalize());
    let text = format!("{}:{request_sha256}:{response_sha256}", opts.model_name);
    let signed = opts.signing.sign_chat(&text).map_err(|e| {
        error!(error = %e, "Signing failed for Fusion stream");
        AppError::Internal(e)
    })?;
    let signed_json = serde_json::to_string(&signed).map_err(|e| AppError::Internal(e.into()))?;
    opts.cache.set_chat(&id, &signed_json);
    proxy::try_report_usage(&response_json, &id, &opts);

    info!(
        chat_id = %id,
        model = %opts.model_name.to_lowercase(),
        "Fusion streaming request completed"
    );

    let stream = tokio_stream::iter(bytes);
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_and_removes_fusion_tool() {
        let mut request = json!({
            "model": "outer",
            "tools": [
                {"type": "function", "function": {"name": "other"}},
                {"type": "openrouter:fusion", "analysis_models": ["~a", "b"]}
            ]
        });
        assert!(has_fusion_tool(&request));
        let removed = remove_fusion_tool(&mut request).unwrap();
        assert_eq!(removed["type"], "openrouter:fusion");
        assert!(!has_fusion_tool(&request));
        assert_eq!(request["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn endpoint_base_url_accepts_domains_and_urls() {
        assert_eq!(
            endpoint_base_url("glm.completions.near.ai"),
            "https://glm.completions.near.ai/v1"
        );
        assert_eq!(
            endpoint_base_url("http://127.0.0.1:1234"),
            "http://127.0.0.1:1234/v1"
        );
        assert_eq!(
            endpoint_base_url("http://127.0.0.1:1234/v1"),
            "http://127.0.0.1:1234/v1"
        );
    }

    #[test]
    fn usage_adds_prompt_completion_and_total() {
        let mut usage = Usage {
            prompt_tokens: 2,
            completion_tokens: 3,
        };
        usage.add(&Usage {
            prompt_tokens: 5,
            completion_tokens: 7,
        });
        assert_eq!(usage.to_json()["prompt_tokens"], 7);
        assert_eq!(usage.to_json()["completion_tokens"], 10);
        assert_eq!(usage.to_json()["total_tokens"], 17);
    }

    #[test]
    fn strips_leading_tilde_from_models() {
        assert_eq!(normalize_model_name("~Qwen/Qwen3"), "Qwen/Qwen3");
        assert_eq!(normalize_model_name(" Qwen/Qwen3 "), "Qwen/Qwen3");
    }

    #[test]
    fn normalizes_and_dedupes_models_preserving_order() {
        assert_eq!(
            normalize_unique_models(vec![
                "~panel-a".to_string(),
                "panel-b".to_string(),
                " panel-a ".to_string(),
                "".to_string(),
                "~panel-b".to_string(),
                "panel-c".to_string(),
            ]),
            vec![
                "panel-a".to_string(),
                "panel-b".to_string(),
                "panel-c".to_string(),
            ]
        );
    }

    #[test]
    fn max_tool_calls_is_capped_by_server_limit() {
        assert_eq!(bounded_max_tool_calls(None, 5), 5);
        assert_eq!(bounded_max_tool_calls(Some(0), 5), 0);
        assert_eq!(bounded_max_tool_calls(Some(3), 5), 3);
        assert_eq!(bounded_max_tool_calls(Some(50), 5), 5);
    }
}
