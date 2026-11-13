use super::AdmissionMode;
use crate::error::AppError;

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
