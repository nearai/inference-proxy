//! Synchronous monetary admission for direct cloud-key inference. No prompts or outputs are
//! sent to the accounting service: only authenticated identity and executable usage bounds.
use crate::error::AppError;
use crate::proxy::{UsageReporter, UsageType};
use axum::http::StatusCode;
use serde_json::{json, Value};
use std::time::Duration;

pub(crate) async fn admit(
    reporter: &mut UsageReporter,
    kind: &UsageType,
    body: &mut Value,
    executions: i32,
) -> Result<(), AppError> {
    admit_for_model(reporter, kind, body, executions, None).await
}

pub(crate) async fn admit_for_model(
    reporter: &mut UsageReporter,
    kind: &UsageType,
    body: &mut Value,
    executions: i32,
    execution_model: Option<&str>,
) -> Result<(), AppError> {
    let unavailable = || AppError::UpstreamParsed {
        status: StatusCode::SERVICE_UNAVAILABLE,
        message: "Monetary admission unavailable".into(),
        error_type: "budget_admission_error".into(),
    };
    let base = reporter.cloud_api_url.as_deref().ok_or_else(unavailable)?;
    let token = reporter
        .cloud_api_usage_token
        .as_deref()
        .ok_or_else(unavailable)?;
    let (org, workspace, key) = match (
        &reporter.org_id,
        &reporter.workspace_id,
        &reporter.api_key_id,
    ) {
        (Some(org), Some(workspace), Some(key)) => (org, workspace, key),
        _ => return Err(unavailable()),
    };
    let id = uuid::Uuid::new_v4();
    let number = |field: &str, default: i64| -> Result<i64, AppError> {
        match body.get(field).filter(|v| !v.is_null()) {
            None => Ok(default),
            Some(value) => value
                .as_i64()
                .filter(|n| *n > 0)
                .ok_or_else(|| AppError::BadRequest(format!("{field} must be a positive integer"))),
        }
    };
    let text = matches!(kind, UsageType::ChatCompletion);
    let max_output = if text {
        match body
            .get("max_completion_tokens")
            .or_else(|| body.get("max_tokens"))
        {
            Some(value) if !value.is_null() => Some(
                value
                    .as_i64()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| AppError::BadRequest("Invalid output limit".into()))?,
            ),
            _ => None,
        }
    } else {
        None
    };
    let (inference_type, images) = match kind {
        UsageType::ChatCompletion => ("chat_completion", 0),
        UsageType::AudioTranscription => ("audio_transcription", 0),
        UsageType::ImageGeneration => ("image_generation", number("n", 1)?),
        UsageType::Embedding => ("embedding", 0),
        UsageType::Rerank => ("rerank", 0),
        UsageType::Score => ("score", 0),
        UsageType::PrivacyClassify => ("privacy_classify", 0),
    };
    let batch_items = if text {
        // A flat token-id list is one prompt; strings/nested token lists are a batch.
        body.get("prompt")
            .and_then(Value::as_array)
            .filter(|items| !items.iter().all(Value::is_number))
            .map_or(1, Vec::len)
            .max(1)
    } else if images > 0 {
        1
    } else {
        body.get("input")
            .or_else(|| body.get("documents"))
            .and_then(Value::as_array)
            .map_or(1, Vec::len)
            .max(1)
    };
    let input_units = if matches!(kind, UsageType::AudioTranscription) {
        Some(number("duration_seconds", 0)?)
    } else {
        None
    };
    let response = reporter.http_client.post(format!("{}/v1/internal/budget", base.trim_end_matches('/')))
        .bearer_auth(token).timeout(Duration::from_secs(12)).json(&json!({
            "id": id.to_string(), "organization_id": org, "workspace_id": workspace, "api_key_id": key,
            "model": reporter.model_name, "execution_model": execution_model, "inference_type": inference_type,
            "input_units": input_units, "batch_items": batch_items, "max_output_tokens": max_output,
            "choices": if text { number("n", 1)?.max(number("best_of", 1)?) } else { 1 }, "executions": executions, "images": images,
        })).send().await.map_err(|_| unavailable())?;
    if response.status() == StatusCode::PAYMENT_REQUIRED {
        return Err(AppError::InsufficientCredits);
    }
    if !response.status().is_success() {
        return Err(unavailable());
    }
    let admitted: Value = response.json().await.map_err(|_| unavailable())?;
    if admitted["id"].as_str() != Some(id.to_string().as_str()) {
        return Err(unavailable());
    }
    let canonical = admitted["model"]
        .as_str()
        .ok_or_else(unavailable)?
        .to_string();
    if text {
        let limit = admitted["max_output_tokens"]
            .as_i64()
            .filter(|n| *n > 0)
            .ok_or_else(unavailable)?;
        // Both names are normalized to the same enforced ceiling before any backend sees it.
        body["max_tokens"] = json!(limit);
        if body.get("max_completion_tokens").is_some() {
            body["max_completion_tokens"] = json!(limit);
        }
    }
    reporter.model_name = canonical;
    reporter.budget_report_id = Some(id.to_string());
    Ok(())
}

// Fusion fans out in one task. Each actual HTTP attempt gets a fresh reservation and
// final receipt, so a failed panel or synthesis cannot discard another panel's cost.
tokio::task_local! { static FUSION_REPORTER: Option<UsageReporter>; }
pub(crate) async fn scope<F: std::future::Future>(
    reporter: Option<UsageReporter>,
    future: F,
) -> F::Output {
    FUSION_REPORTER.scope(reporter, future).await
}
pub(crate) fn scoped_reporter() -> Option<UsageReporter> {
    FUSION_REPORTER.try_with(Clone::clone).ok().flatten()
}
