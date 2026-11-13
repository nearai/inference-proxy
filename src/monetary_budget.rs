//! Synchronous monetary admission for direct cloud-key inference. No prompts or outputs are
//! sent to the accounting service: only authenticated identity and locally measured numeric features.
use crate::error::AppError;
use crate::proxy::{UsageReporter, UsageType};
use axum::http::StatusCode;
use serde_json::{json, Value};
use std::time::Duration;

mod metrics;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionMode {
    #[default]
    Enforce,
    Audit,
    Off,
}

impl std::str::FromStr for AdmissionMode {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "enforce" => Ok(Self::Enforce),
            "audit" => Ok(Self::Audit),
            "off" => Ok(Self::Off),
            _ => anyhow::bail!("MONETARY_ADMISSION must be enforce, audit, or off"),
        }
    }
}

pub(crate) fn features(
    kind: &UsageType,
    body: &Value,
    executions: i32,
) -> Result<crate::monetary_features::UsageFeatures, AppError> {
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
        .ok_or_else(|| AppError::BadRequest("Invalid execution count".into()))?;
    features.media_bytes = body.get("media_bytes").and_then(Value::as_u64).unwrap_or(0);
    Ok(features)
}

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
    if reporter.admission_mode == AdmissionMode::Off {
        return Ok(());
    }
    let measured = features(kind, body, executions);
    match measured {
        Ok(features) => admit_features(reporter, &features, execution_model).await,
        Err(_) if reporter.admission_mode == AdmissionMode::Audit => {
            tracing::warn!("Admission audit could not measure request");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

pub(crate) async fn admit_features(
    reporter: &mut UsageReporter,
    features: &crate::monetary_features::UsageFeatures,
    execution_model: Option<&str>,
) -> Result<(), AppError> {
    // Never carry an earlier attempt's ID into a new attempt or an audit-only receipt.
    reporter.budget_report_id = None;
    if reporter.admission_mode == AdmissionMode::Off {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let result = request_admission(reporter, features, execution_model).await;
    metrics::rpc_result(reporter.admission_mode, &result, started.elapsed());
    if result.is_err() {
        reporter.budget_report_id = None;
    }
    if reporter.admission_mode == AdmissionMode::Audit && result.is_err() {
        tracing::warn!("Admission audit unavailable; continuing with normal usage reporting");
        Ok(())
    } else {
        result.map(|_| ())
    }
}

#[derive(serde::Deserialize)]
struct AdmissionResponse {
    id: Option<String>,
    model: String,
    #[serde(default)]
    mode: AdmissionMode,
}

async fn request_admission(
    reporter: &mut UsageReporter,
    features: &crate::monetary_features::UsageFeatures,
    execution_model: Option<&str>,
) -> Result<AdmissionMode, AppError> {
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
    // No inference has started while waiting for admission. A lost acknowledgement can be
    // cleaned up by the generated ID; cancellation races still have the database TTL fallback.
    let mut cleanup_reporter = reporter.clone();
    cleanup_reporter.budget_report_id = Some(id.to_string());
    let mut cleanup = AttemptCleanup::new(Some(&cleanup_reporter), false);
    let response = reporter.http_client.post(format!("{}/v1/internal/budget", base.trim_end_matches('/')))
        .bearer_auth(token).timeout(Duration::from_millis(reporter.admission_timeout_ms)).json(&json!({
            "id": id.to_string(), "organization_id": org, "workspace_id": workspace, "api_key_id": key,
            "model": reporter.model_name, "execution_model": execution_model, "features": features,
            "mode": reporter.admission_mode,
        })).send().await.map_err(|_| unavailable())?;
    let status = response.status();
    if !status.is_success() {
        // Explicit rejection never authorizes inference. Ambiguous server failures may have
        // happened after a commit, so retain the cancellation cleanup for those.
        if status.is_client_error() {
            cleanup.disarm();
        }
        return Err(match status {
            StatusCode::PAYMENT_REQUIRED => AppError::InsufficientCredits,
            StatusCode::TOO_MANY_REQUESTS => AppError::Overloaded {
                reason: "usage_capacity",
                retry_after_secs: 2,
            },
            StatusCode::BAD_REQUEST => {
                AppError::BadRequest("Invalid admission metadata or estimate".into())
            }
            StatusCode::FORBIDDEN => AppError::Forbidden,
            _ => unavailable(),
        });
    }
    let admitted: AdmissionResponse = response.json().await.map_err(|_| unavailable())?;
    match (admitted.mode, admitted.id) {
        (AdmissionMode::Enforce, Some(returned)) if returned == id.to_string() => {
            if reporter.admission_mode != AdmissionMode::Enforce {
                return Err(unavailable());
            }
            reporter.budget_report_id = Some(id.to_string());
        }
        (AdmissionMode::Audit | AdmissionMode::Off, None) => {}
        _ => return Err(unavailable()),
    }
    if admitted.model.is_empty() {
        return Err(unavailable());
    }
    reporter.model_name = admitted.model;
    cleanup.disarm();
    Ok(admitted.mode)
}

pub(crate) struct AttemptCleanup {
    reporter: Option<UsageReporter>,
    unknown: bool,
}
impl AttemptCleanup {
    fn new(reporter: Option<&UsageReporter>, unknown: bool) -> Self {
        Self {
            reporter: reporter.cloned(),
            unknown,
        }
    }
    pub(crate) fn disarm(&mut self) {
        self.reporter = None;
    }
}
impl Drop for AttemptCleanup {
    fn drop(&mut self) {
        if let (Some(reporter), Ok(runtime)) =
            (self.reporter.take(), tokio::runtime::Handle::try_current())
        {
            let unknown = self.unknown;
            runtime.spawn(async move {
                finish_attempt(&reporter, unknown).await;
            });
        }
    }
}

/// Keep cleanup armed until the caller hands usage to the existing reporting path.
/// Only a connect/build failure proves no execution.
pub(crate) async fn send(
    request: reqwest::RequestBuilder,
    reporter: Option<&UsageReporter>,
) -> Result<(reqwest::Response, AttemptCleanup), reqwest::Error> {
    let mut cleanup = AttemptCleanup::new(reporter, true);
    let response = request.send().await;
    match &response {
        Ok(response) if response.status().is_success() => {}
        _ => {
            cleanup.unknown = !response
                .as_ref()
                .is_err_and(|error| error.is_connect() || error.is_builder());
            if let Some(reporter) = reporter {
                finish_attempt(reporter, cleanup.unknown).await;
            }
            cleanup.disarm();
        }
    }
    response.map(|response| (response, cleanup))
}

pub(crate) async fn unknown_outcome(reporter: &UsageReporter) {
    finish_attempt(reporter, true).await;
}

async fn finish_attempt(reporter: &UsageReporter, unknown: bool) {
    let (Some(base), Some(token), Some(org), Some(id)) = (
        reporter.cloud_api_url.as_deref(),
        reporter.cloud_api_usage_token.as_deref(),
        reporter.org_id.as_deref(),
        reporter.budget_report_id.as_deref(),
    ) else {
        return;
    };
    let operation = if unknown { "unknown" } else { "release" };
    let result = reporter
        .http_client
        .post(format!(
            "{}/v1/internal/budget/{operation}",
            base.trim_end_matches('/')
        ))
        .bearer_auth(token)
        .timeout(Duration::from_secs(2))
        .json(&json!({"organization_id":org,"id":id}))
        .send()
        .await;
    if !result.is_ok_and(|r| r.status().is_success()) {
        tracing::warn!(operation, "Budget lifecycle update failed; awaiting expiry");
    }
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
