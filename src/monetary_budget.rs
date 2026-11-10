//! Synchronous monetary admission for direct cloud-key inference. No prompts or outputs are
//! sent to the accounting service: only authenticated identity and locally measured numeric features.
use crate::error::AppError;
use crate::proxy::{UsageReporter, UsageType};
use axum::http::StatusCode;
use serde_json::{json, Value};
use std::time::Duration;

pub(crate) async fn admit(
    reporter: &mut UsageReporter,
    kind: &UsageType,
    body: &Value,
    executions: i32,
) -> Result<(), AppError> {
    admit_for_model(reporter, kind, body, executions, None).await
}

pub(crate) async fn admit_for_model(
    reporter: &mut UsageReporter,
    kind: &UsageType,
    body: &Value,
    executions: i32,
    execution_model: Option<&str>,
) -> Result<(), AppError> {
    let unavailable = || AppError::UpstreamParsed {
        status: StatusCode::SERVICE_UNAVAILABLE,
        message: "Monetary admission unavailable".into(),
        error_type: "budget_admission_error".into(),
    };
    if !reporter.delivery.admission_ready() {
        return Err(unavailable());
    }
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
    let label = match kind {
        UsageType::ChatCompletion => "chat_completion",
        UsageType::ImageGeneration => "image_generation",
        UsageType::AudioTranscription { .. } => "audio_transcription",
        _ => {
            return Err(AppError::BadRequest(
                "Admission is unavailable for this inference type".into(),
            ))
        }
    };
    let mut features = crate::monetary_features::UsageFeatures::from_json(label.into(), body)?;
    if let UsageType::AudioTranscription { duration_seconds } = kind {
        features.audio_duration_seconds = duration_seconds.filter(|n| *n > 0).map(|n| n as u64);
    }
    features.executions = u64::try_from(executions)
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(unavailable)?;
    features.media_bytes = body.get("media_bytes").and_then(Value::as_u64).unwrap_or(0);
    let response = reporter.http_client.post(format!("{}/v1/internal/budget", base.trim_end_matches('/')))
        .bearer_auth(token).timeout(Duration::from_secs(12)).json(&json!({
            "id": id.to_string(), "organization_id": org, "workspace_id": workspace, "api_key_id": key,
            "model": reporter.model_name, "execution_model": execution_model, "features": features,
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
