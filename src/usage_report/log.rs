//! How a delivery reports what it does: the line and the drop counter of a
//! report given up on, the shutdown line, and the metrics of the queue.

use std::time::Duration;

use tracing::{info, warn};

use super::{Drained, Job, Refused, UsageReportDelivery, FINAL_OUTCOME_TARGET};
use crate::model_metrics::{model_gauge, ModelLabel};
use crate::proxy::{record_usage_report_outcome, UsageReportOutcome, UsageReporter};

/// Why a report was given up on without a final answer from the billing API.
pub(super) enum GiveUp {
    /// Pushed out of a full queue; it was never sent.
    QueueFull,
    /// Refused by the transport before it was queued; it was never sent.
    Refused(Refused),
    /// No attempt fits before its deadline any more, after this many
    /// attempts, the last of which ended like this and took this long.
    Deadline {
        attempts: u32,
        last_attempt: Option<(UsageReportOutcome, Duration)>,
    },
}

#[derive(Clone, Copy)]
pub(super) enum Step {
    Up,
    Down,
    None,
}

impl Step {
    pub(super) fn apply(self, gauge: metrics::Gauge) {
        match self {
            Step::Up => gauge.increment(1.0),
            Step::Down => gauge.decrement(1.0),
            Step::None => gauge.increment(0.0),
        }
    }
}

/// `label`, the request source and, in list mode, the model: the label set of
/// the per-report delivery series, in the order of the existing ones.
pub(super) fn source_labels(
    reporter: &UsageReporter,
    label: Option<(&'static str, &'static str)>,
) -> Vec<metrics::Label> {
    let source = [
        ("auth_path", reporter.request_source.auth_path.as_label()),
        (
            "ingress_route",
            reporter.request_source.ingress_route.as_label(),
        ),
    ];
    label
        .into_iter()
        .chain(source)
        .map(|(key, value)| metrics::Label::new(key, value))
        .chain(
            reporter
                .model_label
                .map(|model| metrics::Label::new("model", model)),
        )
        .collect()
}

impl UsageReportDelivery {
    pub(super) fn give_up(&self, job: &Job, why: GiveUp) {
        let reporter = &job.reporter;
        let since_completion_ms = job.completed_at.elapsed().as_millis() as u64;
        match why {
            GiveUp::QueueFull => {
                let outcome = UsageReportOutcome::QueueFull;
                record_usage_report_outcome(reporter, outcome, None);
                self.count(
                    "inference_proxy_usage_reports_dropped_total",
                    reporter,
                    ("reason", "queue_full"),
                );
                warn!(
                    request_id = %reporter.request_id.as_deref().unwrap_or(""),
                    org_id = %reporter.org_id.as_deref().unwrap_or(""),
                    workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                    api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                    model = %reporter.model_name,
                    auth_path = reporter.request_source.auth_path.as_label(),
                    ingress_route = reporter.request_source.ingress_route.as_label(),
                    outcome = outcome.as_label(),
                    since_completion_ms,
                    max_queued = self.policy.max_queued,
                    "Usage report dropped: the waiting queue is full and this report waited \
                     longest — usage NOT billed"
                );
            }
            GiveUp::Refused(Refused {
                reason,
                outcome,
                error,
            }) => {
                record_usage_report_outcome(reporter, outcome, None);
                self.count(
                    "inference_proxy_usage_reports_dropped_total",
                    reporter,
                    ("reason", reason),
                );
                warn!(
                    target: FINAL_OUTCOME_TARGET,
                    request_id = %reporter.request_id.as_deref().unwrap_or(""),
                    org_id = %reporter.org_id.as_deref().unwrap_or(""),
                    workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                    api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                    model = %reporter.model_name,
                    error = %error,
                    auth_path = reporter.request_source.auth_path.as_label(),
                    ingress_route = reporter.request_source.ingress_route.as_label(),
                    outcome = outcome.as_label(),
                    since_completion_ms,
                    "Usage report refused before it was queued — usage NOT billed"
                );
            }
            GiveUp::Deadline {
                attempts,
                last_attempt,
            } => {
                let outcome = UsageReportOutcome::DeadlineExceeded;
                // With the duration of its last attempt, when it had one.
                record_usage_report_outcome(
                    reporter,
                    outcome,
                    last_attempt.map(|(_, elapsed)| elapsed),
                );
                self.count(
                    "inference_proxy_usage_reports_dropped_total",
                    reporter,
                    ("reason", "deadline"),
                );
                warn!(
                    request_id = %reporter.request_id.as_deref().unwrap_or(""),
                    org_id = %reporter.org_id.as_deref().unwrap_or(""),
                    workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                    api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                    model = %reporter.model_name,
                    auth_path = reporter.request_source.auth_path.as_label(),
                    ingress_route = reporter.request_source.ingress_route.as_label(),
                    outcome = outcome.as_label(),
                    attempts,
                    last_attempt = last_attempt.map(|(outcome, _)| outcome.as_label()),
                    since_completion_ms,
                    "Usage report dropped: not accepted before its deadline — usage NOT billed"
                );
            }
        }
    }

    pub async fn drain_at_shutdown(&self) -> Option<Drained> {
        if !self.extended {
            return None;
        }
        let drained = self.drain(self.policy.shutdown_drain).await;
        if drained.left() == 0 {
            info!(
                pending = drained.pending,
                waited_ms = drained.waited.as_millis() as u64,
                "Usage reports drained before shutdown"
            );
        } else {
            warn!(
                pending = drained.pending,
                left_waiting = drained.left_waiting,
                left_in_flight = drained.left_in_flight,
                waited_ms = drained.waited.as_millis() as u64,
                "Usage reports left undelivered at shutdown — usage NOT billed"
            );
        }
        Some(drained)
    }

    pub(super) fn count(
        &self,
        name: &'static str,
        reporter: &UsageReporter,
        label: (&'static str, &'static str),
    ) {
        if self.extended {
            metrics::counter!(name, source_labels(reporter, Some(label))).increment(1);
        }
    }

    pub(super) fn waiting_gauge(&self, model: ModelLabel, step: Step) {
        if self.extended {
            step.apply(model_gauge!(
                model,
                "inference_proxy_usage_report_queue_depth"
            ));
        }
    }

    pub(super) fn in_flight_gauge(&self, model: ModelLabel, step: Step) {
        if self.extended {
            step.apply(model_gauge!(
                model,
                "inference_proxy_usage_reports_in_flight"
            ));
        }
    }
}
