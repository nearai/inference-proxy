//! Pre-execution input counting. Decrypted content stays in this process and the
//! selected inference backend; the monetary API receives only the integer count.
use crate::{
    auth::RequireTrustedAuth,
    encryption::{self, Endpoint},
    error::AppError,
    proxy::UsageType,
    AppState,
};
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use serde_json::{json, Value};

pub(crate) fn unavailable() -> AppError {
    AppError::UpstreamParsed {
        status: StatusCode::SERVICE_UNAVAILABLE,
        message: "Input token counting unavailable".into(),
        error_type: "input_token_count_error".into(),
    }
}

/// The backend owns its tokenizer and chat template. Count only the finalized body.
pub(crate) async fn count(
    client: &reqwest::Client,
    inference_url: &str,
    kind: &UsageType,
    body: &Value,
) -> Result<i32, AppError> {
    count_with_bearer(client, inference_url, kind, body, None).await
}
pub(crate) async fn count_with_bearer(
    client: &reqwest::Client,
    inference_url: &str,
    kind: &UsageType,
    body: &Value,
    bearer: Option<&str>,
) -> Result<i32, AppError> {
    tokio::time::timeout(
        std::time::Duration::from_secs(15),
        count_inner(client, inference_url, kind, body, bearer),
    )
    .await
    .map_err(|_| unavailable())?
}
async fn count_inner(
    client: &reqwest::Client,
    inference_url: &str,
    kind: &UsageType,
    body: &Value,
    bearer: Option<&str>,
) -> Result<i32, AppError> {
    if matches!(kind, UsageType::ImageGeneration) {
        return Ok(0);
    }
    if matches!(kind, UsageType::AudioTranscription) {
        return body["duration_seconds"]
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or_else(unavailable);
    }
    let mut url = reqwest::Url::parse(inference_url).map_err(|_| unavailable())?;
    let path = url.path().split("/v1/").next().unwrap_or("");
    let path = if path == url.path() { "" } else { path }.to_owned();
    url.set_path(&format!("{path}/tokenize"));
    url.set_query(None);
    if let Some(token) = bearer {
        // Fusion calls a sibling trusted gateway, which owns normalization and E2EE.
        // Do not assume its public tokenizer is the same as its selected engine's.
        url.set_path(&format!("{path}/v1/internal/input-tokens"));
        let response = client
            .post(url)
            .bearer_auth(token)
            .json(&json!({"kind":"chat", "request":body}))
            .send()
            .await
            .map_err(|_| unavailable())?;
        if !response.status().is_success() {
            return Err(unavailable());
        }
        let mut response = response;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| unavailable())? {
            if bytes.len().saturating_add(chunk.len()) > 16_384 {
                return Err(unavailable());
            }
            bytes.extend_from_slice(&chunk);
        }
        return serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|v| v["input_tokens"].as_i64())
            .and_then(|n| i32::try_from(n).ok())
            .filter(|n| *n >= 0)
            .ok_or_else(unavailable);
    }
    let model = body.get("model").cloned().unwrap_or(Value::Null);
    let mut requests = Vec::new();
    match kind {
        UsageType::ChatCompletion if body.get("messages").is_some() => {
            let mut request = body.clone();
            request
                .as_object_mut()
                .ok_or_else(unavailable)?
                .retain(|key, _| {
                    matches!(
                        key.as_str(),
                        "model"
                            | "messages"
                            | "tools"
                            | "tool_choice"
                            | "response_format"
                            | "chat_template"
                            | "chat_template_kwargs"
                            | "reasoning_effort"
                            | "lora_path"
                            | "mm_processor_kwargs"
                            | "continue_final_message"
                            | "add_special_tokens"
                    )
                });
            request["add_generation_prompt"] = json!(true);
            requests.push(request);
        }
        UsageType::Rerank => {
            let query = body["query"].as_str().ok_or_else(unavailable)?;
            for doc in body["documents"].as_array().ok_or_else(unavailable)? {
                let text = doc
                    .as_str()
                    .or_else(|| doc.get("text").and_then(Value::as_str))
                    .ok_or_else(unavailable)?;
                requests.push(json!({"model":model, "prompt":query, "add_special_tokens":true}));
                requests.push(json!({"model":model, "prompt":text, "add_special_tokens":true}));
            }
        }
        UsageType::Score => {
            // Score uses pairs: a scalar on one side broadcasts to the other side.
            let left = items(&body["text_1"])?;
            let right = items(&body["text_2"])?;
            let len = left.len().max(right.len());
            if !(left.len() == right.len() || left.len() == 1 || right.len() == 1) {
                return Err(unavailable());
            }
            for i in 0..len {
                for input in [left[i % left.len()].clone(), right[i % right.len()].clone()] {
                    requests.push(json!({"model":model,"prompt":input,"add_special_tokens":true}));
                }
            }
        }
        _ => {
            let input = body
                .get("prompt")
                .or_else(|| body.get("input"))
                .ok_or_else(unavailable)?;
            let mut direct = 0i32;
            for input in items(input)? {
                if let Some(ids) = input
                    .as_array()
                    .filter(|a| a.iter().all(|v| v.as_u64().is_some()))
                {
                    direct = direct
                        .checked_add(i32::try_from(ids.len()).map_err(|_| unavailable())?)
                        .ok_or_else(unavailable)?;
                } else {
                    requests.push(json!({"model":model,"prompt":input,"add_special_tokens":true}));
                }
            }
            if requests.is_empty() {
                return Ok(direct);
            }
            if direct != 0 {
                return Err(unavailable());
            }
        }
    }
    if requests.is_empty() || requests.len() > 2048 {
        return Err(unavailable());
    }
    let mut total = 0i32;
    for request in requests {
        let mut call = client
            .post(url.clone())
            .timeout(std::time::Duration::from_secs(10))
            .json(&request);
        if let Some(token) = bearer {
            call = call.bearer_auth(token);
        }
        let mut response = call.send().await.map_err(|_| unavailable())?;
        if !response.status().is_success() {
            return Err(unavailable());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| unavailable())? {
            // /tokenize may return token ids as well as the count. Bound the response,
            // discard ids, and never include the body in an error or log.
            if bytes.len().saturating_add(chunk.len()) > 16 * 1024 * 1024 {
                return Err(unavailable());
            }
            bytes.extend_from_slice(&chunk);
        }
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| unavailable())?;
        let count = value["count"]
            .as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .filter(|n| *n >= 0)
            .ok_or_else(unavailable)?;
        total = total.checked_add(count).ok_or_else(unavailable)?;
    }
    Ok(total)
}

fn items(value: &Value) -> Result<Vec<Value>, AppError> {
    if value.is_string() {
        return Ok(vec![value.clone()]);
    }
    let array = value
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(unavailable)?;
    if array.iter().all(|v| v.as_u64().is_some()) {
        return Ok(vec![value.clone()]);
    }
    Ok(array.clone())
}

/// Private count-only endpoint for Cloud API's trusted provider connection. It performs
/// the same decryption and chat normalization as execution and returns no token ids.
pub async fn internal_count(
    State(state): State<AppState>,
    _auth: RequireTrustedAuth,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<Value>, AppError> {
    let bytes =
        crate::routes::chat::read_body_with_limit(body, state.config.max_request_size).await?;
    let envelope: Value = serde_json::from_slice(&bytes)
        .map_err(|_| AppError::BadRequest("Invalid counting request".into()))?;
    let mut request = envelope["request"].clone();
    if !request.is_object() {
        return Err(AppError::BadRequest("Invalid counting request".into()));
    }
    if envelope["billing_probe"] == "per_attempt_v1" {
        if state.config.cloud_api_url.is_none() || state.config.cloud_api_usage_token.is_none() {
            return Err(unavailable());
        }
        return Ok(Json(json!({"billing":"per_attempt_v1"})));
    }
    let (endpoint, kind, path) = match envelope["kind"].as_str() {
        Some("chat") => (
            Endpoint::ChatCompletions,
            UsageType::ChatCompletion,
            "/v1/chat/completions",
        ),
        Some("completion") => (
            Endpoint::Completions,
            UsageType::ChatCompletion,
            "/v1/completions",
        ),
        Some("embedding") => (Endpoint::Embeddings, UsageType::Embedding, "/v1/embeddings"),
        Some("privacy") => (
            Endpoint::Embeddings,
            UsageType::PrivacyClassify,
            "/v1/privacy/classify",
        ),
        Some("rerank") => (Endpoint::Rerank, UsageType::Rerank, "/v1/rerank"),
        Some("score") => (Endpoint::Score, UsageType::Score, "/v1/score"),
        _ => {
            return Err(AppError::BadRequest(
                "Unsupported counting operation".into(),
            ))
        }
    };
    if let Some(ctx) = encryption::extract_encryption_context(&headers)? {
        encryption::decrypt_request_fields(&mut request, endpoint, &ctx, &state.signing)?;
    }
    let model = crate::model_list::model_for(&state, &request)?;
    crate::model_list::guard_request_body(
        &request,
        crate::model_list::checked_model(&state, model.id),
    )?;
    crate::model_list::apply_backend_lora(&state, &mut request);
    if matches!(endpoint, Endpoint::ChatCompletions) {
        crate::routes::chat::strip_empty_tool_calls(&mut request);
        crate::tool_calls::normalize_tool_call_arguments(&mut request);
        crate::response_format::repair_json_schema_response_format(&mut request);
        // Tokenization can fetch media too. Apply the execution path's policy
        // before sending any plaintext or URL to the engine.
        crate::content_policy::reject_unsupported_content_parts(
            &request,
            &state.config.rejected_content_part_types,
        )?;
        crate::image_validation::reject_invalid_images(&request, &state.config.image_validation())
            .await?;
        if model.trusted_by_backends {
            crate::reasoning::apply_reasoning_switch(&mut request, model.reasoning_off_effort);
            if let Some(map) = model.reasoning_effort_map {
                crate::reasoning::apply_effort_map(&mut request, map, model.label);
            }
        }
        if model.merge_system_messages {
            crate::system_messages::merge_system_messages(&mut request, model.label);
        }
    }
    let override_url = match kind {
        UsageType::Rerank => state.config.rerank_url_override.as_deref(),
        UsageType::Score => state.config.score_url_override.as_deref(),
        _ => None,
    };
    let count = if let Some(url) = override_url {
        count(&state.http_client, url, &kind, &request).await?
    } else {
        let (url, _guard) = model.backend_pool.select_url(path);
        count(model.backend_client, &url, &kind, &request).await?
    };
    Ok(Json(json!({"input_tokens": count})))
}
