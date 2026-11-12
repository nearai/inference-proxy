//! Sample metadata only; delivery and accounting do not depend on observation.
use super::AdmissionMode;
use crate::{error::AppError, proxy::UsageReporter};
use serde_json::Value;
use uuid::Uuid;

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub id: Uuid,
    pub estimated_amount: i64,
}

impl Observation {
    pub fn new(id: Uuid, estimated_amount: Option<i64>) -> Option<Self> {
        // Same deterministic one-percent sample as the admission service.
        estimated_amount
            .filter(|amount| *amount >= 0 && id.as_u128().is_multiple_of(100))
            .map(|estimated_amount| Self {
                id,
                estimated_amount,
            })
    }

    pub fn usage(&self, reporter: &UsageReporter, body: &Value) {
        if let (Some(org), Some(provider_id)) = (
            reporter.org_id.as_deref(),
            body.get("id").and_then(Value::as_str),
        ) {
            tracing::info!(event = "usage_admission_usage", attempt_id = %self.id,
                organization_id = org, provider_request_id = provider_id,
                admission_id = reporter.budget_report_id.as_deref().unwrap_or(""),
                estimated_nano_usd = self.estimated_amount,
                "Admission sample usage association");
        }
    }

    pub fn end(&self, outcome: &'static str) {
        tracing::info!(event = "usage_admission_attempt_end", attempt_id = %self.id,
            outcome, "Admission sample outcome");
    }
}

pub fn mode_label(mode: AdmissionMode) -> &'static str {
    match mode {
        AdmissionMode::Off => "off",
        AdmissionMode::Audit => "audit",
        AdmissionMode::Enforce => "enforce",
    }
}

pub fn rpc_result(
    requested: AdmissionMode,
    result: &Result<AdmissionMode, AppError>,
    elapsed: std::time::Duration,
) {
    let outcome = match result {
        Ok(AdmissionMode::Off) => "off",
        Ok(AdmissionMode::Audit) => "audit",
        Ok(AdmissionMode::Enforce) => "admitted",
        Err(AppError::InsufficientCredits) => "budget_exceeded",
        Err(AppError::Overloaded { .. } | AppError::RateLimited) => "unsettled_limit",
        Err(AppError::BadRequest(_)) => "invalid_estimate",
        Err(AppError::Forbidden | AppError::Unauthorized) => "unauthorized",
        Err(_) => "unavailable",
    };
    let mode = result
        .as_ref()
        .map(|mode| mode_label(*mode))
        .unwrap_or("unknown");
    metrics::counter!("inference_proxy_admission_requests_total",
        "requested_mode" => mode_label(requested), "mode" => mode, "outcome" => outcome)
    .increment(1);
    metrics::histogram!("inference_proxy_admission_duration_seconds", "stage" => "rpc",
        "mode" => mode)
    .record(elapsed.as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_invalid_estimates_are_not_samples() {
        let id = Uuid::from_u128(100);
        assert!(Observation::new(id, None).is_none());
        assert!(Observation::new(id, Some(-1)).is_none());
        assert_eq!(Observation::new(id, Some(0)).unwrap().estimated_amount, 0);
        assert!(Observation::new(Uuid::from_u128(101), Some(10)).is_none());
    }

    #[test]
    fn rpc_metrics_distinguish_policy_denial_from_unavailability() {
        let recorder = crate::metrics_middleware::recorder_builder().build_recorder();
        metrics::with_local_recorder(&recorder, || {
            rpc_result(
                AdmissionMode::Enforce,
                &Err(AppError::InsufficientCredits),
                std::time::Duration::from_millis(5),
            );
            rpc_result(
                AdmissionMode::Enforce,
                &Err(AppError::Internal(anyhow::anyhow!("private detail"))),
                std::time::Duration::from_millis(9),
            );
            rpc_result(
                AdmissionMode::Enforce,
                &Ok(AdmissionMode::Audit),
                std::time::Duration::from_millis(2),
            );
        });
        let rendered = recorder.handle().render();
        for outcome in ["budget_exceeded", "unavailable", "audit"] {
            assert!(rendered.contains(&format!("outcome=\"{outcome}\"")));
        }
        assert!(!rendered.contains("private detail"));
        assert!(rendered.contains("inference_proxy_admission_duration_seconds_bucket"));
        assert!(rendered.contains("le=\"0.005\""));
    }
}
