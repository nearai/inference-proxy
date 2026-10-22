//! The HTTP sender: one `POST` of a report to cloud-api's
//! `/v1/internal/usage`, with the usage token as the Bearer credential. This is
//! the request the delivery has always made, moved behind [`UsageSender`]; the
//! request, its headers and its classification are unchanged.

use std::time::{Duration, Instant};

use futures_util::future::BoxFuture;

use super::{AttemptLog, Job, SendOutcome, UsageSender};
use crate::proxy::{classify_usage_http_status, classify_usage_request_error, UsageReportOutcome};

/// Sends reports to cloud-api over HTTP. It holds nothing: the URL, the usage
/// token and the client come with each report's reporter, as they always have.
pub(crate) struct HttpSender;

impl UsageSender for HttpSender {
    fn send<'a>(
        &'a self,
        job: &'a Job,
        timeout: Duration,
    ) -> BoxFuture<'a, (SendOutcome, AttemptLog)> {
        Box::pin(async move {
            let reporter = &job.reporter;
            // `spawn_usage_report` checks the token before it hands a report
            // over, so this is not reached in practice. Should it be, the
            // report is refused as final: no panic on a production path.
            let Some(token) = reporter.cloud_api_usage_token.as_deref() else {
                return (
                    SendOutcome::Final("missing_usage_token"),
                    AttemptLog {
                        answer: Err("no usage token configured".to_string()),
                        outcome: UsageReportOutcome::MissingUsageToken,
                        duration: Duration::ZERO,
                    },
                );
            };
            let url = format!("{}/v1/internal/usage", reporter.cloud_api_url);
            let authorization = format!("Bearer {token}");
            let started_at = Instant::now();
            let mut request = reporter
                .http_client
                .post(&url)
                .header("authorization", &authorization)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(job.body.clone())
                .timeout(timeout);
            if let Some(request_id) = reporter.request_id.as_deref() {
                request = request.header("x-request-id", request_id);
            }
            // The body is not read, as before: only the status decides.
            let answer = request.send().await.map(|response| response.status());
            let duration = started_at.elapsed();
            let outcome = match &answer {
                Ok(status) => classify_usage_http_status(*status),
                Err(error) => classify_usage_request_error(error),
            };
            let decision = match retry_reason(outcome, &answer) {
                Some(reason) => SendOutcome::Retryable(reason),
                None if outcome == UsageReportOutcome::Accepted => SendOutcome::Accepted,
                None => SendOutcome::Final("rejected"),
            };
            let answer = answer.map_err(|error| error.to_string());
            (
                decision,
                AttemptLog {
                    answer,
                    outcome,
                    duration,
                },
            )
        })
    }
}

/// Why an attempt with this result may be made again, `None` when the result
/// is final: a 4xx other than 429 means the billing API refused the report
/// itself, and sending the same bytes again would get the same answer.
fn retry_reason(
    outcome: UsageReportOutcome,
    answer: &Result<reqwest::StatusCode, reqwest::Error>,
) -> Option<&'static str> {
    match (outcome, answer) {
        (UsageReportOutcome::Timeout, _) => Some("timeout"),
        (UsageReportOutcome::ConnectError, _) => Some("connect_error"),
        // A request that could not even be built fails the same way each time.
        (UsageReportOutcome::TransportError, Err(error)) if error.is_builder() => None,
        (UsageReportOutcome::TransportError, _) => Some("transport_error"),
        (UsageReportOutcome::Http5xx, _) => Some("http_5xx"),
        (UsageReportOutcome::Http4xx, Ok(reqwest::StatusCode::TOO_MANY_REQUESTS)) => {
            Some("http_429")
        }
        _ => None,
    }
}
