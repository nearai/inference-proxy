//! Delivery of usage reports, over one of two transports: HTTP to cloud-api's
//! `POST /v1/internal/usage` (the default), or `SendMessage` to the usage queue
//! (`VLLM_PROXY_USAGE_REPORT_TRANSPORT=sqs`). The transport is a `UsageSender`.
//!
//! `proxy::spawn_usage_report` checks and completes a report and hands it over
//! here. `submit` first has the transport prepare the report once (HTTP: nothing;
//! SQS: validation and serialization of the envelope), then takes a lock for a
//! queue operation and returns. Nothing in this module is ever awaited by a
//! handler. Sending, and the line of a report dropped from a full queue, happen
//! in tasks of this module's own, outside the span of the request that happened
//! to be completing.
//!
//! With the default [`UsageReportPolicy`] delivery is what it has always been:
//! one task per report, one attempt, a 5 second timeout, and the same log lines
//! and metric series. The `VLLM_PROXY_USAGE_REPORT_*` settings make it
//!
//! - patient: a longer per-attempt timeout, retries with backoff and jitter
//!   for the failures that can pass (timeout, connection, transport, 5xx,
//!   429), and an overall deadline per report;
//! - polite: a cap on reports in flight. A report beyond the cap waits in a
//!   bounded queue, and when that is full the report that has waited longest
//!   is dropped to make room, logged and counted;
//! - careful at shutdown: pending reports get a bounded time to finish.
//!
//! The billing API deduplicates on the completion id, so sending a report
//! again is safe. A report is still only ever sent by one task, one attempt
//! at a time, and every attempt carries the bytes of the first.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::future::BoxFuture;
use tracing::{info, warn};

use crate::proxy::{record_usage_report_outcome, UsageReportOutcome, UsageReporter};

use self::log::{source_labels, GiveUp};
use self::queue::State;
mod http_sender;
mod log;
mod queue;
mod sqs_sender;

/// Ceiling of the backoff between two attempts, whatever the attempt number.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// The three lines a report has always ended with were written by `proxy.rs`
/// and keep its log target, so a filter or a query on it still finds them.
const FINAL_OUTCOME_TARGET: &str = "vllm_proxy_rs::proxy";

/// How many reports dropped from a full queue may have their line still to
/// be written, each in a task of its own. Past it the line is written where
/// the drop happens, so these tasks cannot pile up without bound either.
const MAX_DROPS_BEING_LOGGED: usize = 1_000;

/// How usage reports are delivered. `Default` is the delivery every
/// deployment had before these settings existed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageReportPolicy {
    /// Timeout of one attempt (`VLLM_PROXY_USAGE_REPORT_TIMEOUT_SECS`,
    /// default 5).
    pub attempt_timeout: Duration,
    /// Attempts per report, the first one included
    /// (`VLLM_PROXY_USAGE_REPORT_MAX_ATTEMPTS`, default 1: never retried).
    pub max_attempts: u32,
    /// Backoff before the first retry; it doubles for each later one, up to
    /// 30 seconds (`VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS`, default 500).
    pub initial_backoff: Duration,
    /// How long after its request completed a report may still be sent
    /// (`VLLM_PROXY_USAGE_REPORT_DEADLINE_SECS`, default 0 = `None`, no
    /// deadline). An attempt is only started while its whole timeout still
    /// fits before the deadline; a report that can no longer get one is
    /// dropped and counted.
    pub deadline: Option<Duration>,
    /// Reports in flight at once, across the process
    /// (`VLLM_PROXY_USAGE_REPORT_MAX_IN_FLIGHT`, default 0 = no cap).
    pub max_in_flight: usize,
    /// Reports that may wait for a place once the cap is reached
    /// (`VLLM_PROXY_USAGE_REPORT_MAX_QUEUED`, default 10000). Unused without
    /// a cap.
    pub max_queued: usize,
    /// How long shutdown waits for pending reports
    /// (`VLLM_PROXY_USAGE_REPORT_SHUTDOWN_DRAIN_SECS`, default 0: it does
    /// not wait).
    pub shutdown_drain: Duration,
}

impl Default for UsageReportPolicy {
    fn default() -> Self {
        Self {
            attempt_timeout: Duration::from_secs(5),
            max_attempts: 1,
            initial_backoff: Duration::from_millis(500),
            deadline: None,
            max_in_flight: 0,
            max_queued: 10_000,
            shutdown_drain: Duration::ZERO,
        }
    }
}

/// Where a report is sent (`VLLM_PROXY_USAGE_REPORT_TRANSPORT`). `Default` is
/// HTTP to cloud-api, the transport every deployment had before this.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum UsageReportTransport {
    /// `POST /v1/internal/usage` on cloud-api.
    #[default]
    Http,
    /// `SendMessage` to the usage queue (`VLLM_PROXY_USAGE_REPORT_SQS_*`).
    Sqs(SqsSettings),
}

/// The transport a delivery sends over, by name. The labels are the
/// `transport` label of `inference_proxy_usage_report_transport_info`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransportKind {
    Http,
    Sqs,
}

impl TransportKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Sqs => "sqs",
        }
    }
}

/// Settings of the SQS transport. The credentials themselves are read from
/// the environment when the sender is built (`aws_credentials.rs`); they are
/// not part of the configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqsSettings {
    /// The usage queue (`VLLM_PROXY_USAGE_REPORT_SQS_URL`, required).
    pub queue_url: String,
    /// The queue's region (`VLLM_PROXY_USAGE_REPORT_SQS_REGION`, default `us-east-1`).
    pub region: String,
    /// An endpoint in place of the regional one (`VLLM_PROXY_USAGE_REPORT_SQS_ENDPOINT`,
    /// a local emulator).
    pub endpoint: Option<String>,
    /// Base URL of the IMDSv2 endpoint that serves session credentials (`VLLM_PROXY_AWS_CREDENTIALS_URL`,
    /// required: there are no static keys).
    pub credentials_url: String,
}

/// A report on its way to the billing API: what `proxy::spawn_usage_report`
/// hands over. The body is the report as cloud-api's usage body; the HTTP
/// sender posts it as it is. The SQS sender's `prepare` replaces it, once at
/// submit, with the `SendMessage` request, so every attempt sends the same bytes.
/// The reporter still carries the HTTP client and the cloud-api URL and usage
/// token, which the HTTP sender reads.
#[derive(Clone)]
pub(crate) struct Job {
    /// Who the report is for and where it came from: ids and labels for the
    /// log lines and the metrics, and the HTTP client.
    pub reporter: UsageReporter,
    /// The bytes every attempt sends: the report as serialized by the proxy,
    /// or, after the transport's `prepare`, what that transport sends.
    pub body: bytes::Bytes,
    /// When the request completed and its report was handed over.
    pub completed_at: Instant,
    /// The same instant on the wall clock: the envelope's `emitted_at`, set
    /// once so that every attempt carries the same message.
    pub emitted_at: chrono::DateTime<chrono::Utc>,
}

/// What a sender made of one attempt, in the terms of the shared delivery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SendOutcome {
    /// Accepted: the report is done.
    Accepted,
    /// Not accepted, and another attempt may pass. The label is the retry's
    /// reason in `inference_proxy_usage_report_retries_total`.
    Retryable(&'static str),
    /// Not accepted, and sending the same report again would not change that.
    /// The label is the drop's reason in `inference_proxy_usage_reports_dropped_total`.
    Final(&'static str),
}

/// A report a transport cannot send at all, refused before it is queued. It
/// is final, counted under `reason` and logged with `error`, which names
/// nothing of the report but its size and its missing field names.
#[derive(Debug)]
pub(crate) struct Refused {
    pub reason: &'static str,
    pub outcome: UsageReportOutcome,
    pub error: String,
}

/// What one attempt looked like, for its log line and the metrics.
pub(crate) struct AttemptLog {
    /// The status the billing API answered, or the error that stood in the way.
    pub answer: Result<reqwest::StatusCode, String>,
    /// The attempt's outcome label, as the report's final outcome would carry it.
    pub outcome: UsageReportOutcome,
    /// How long the attempt took.
    pub duration: Duration,
}

/// One way of getting a report to the billing API. The shared delivery decides
/// about attempts, retries, deadlines, the queue and the drain; a sender makes
/// one attempt and says what it means in those terms.
pub(crate) trait UsageSender: Send + Sync {
    /// The transport this sender sends over.
    fn kind(&self) -> TransportKind;

    /// Turns a submitted report into the bytes its attempts send. Called once,
    /// when the report is submitted, so validation and serialization are not
    /// repeated per attempt. Refusing a report here means it is never queued.
    fn prepare(&self, _job: &mut Job) -> Result<(), Refused> {
        Ok(())
    }

    /// One bounded check, before serving, that reports can be sent. The
    /// default has none (HTTP); the error is the attempt's outcome and its text.
    fn startup_check<'a>(
        &'a self,
        _client: &'a reqwest::Client,
        _timeout: Duration,
    ) -> BoxFuture<'a, Result<(), (UsageReportOutcome, String)>> {
        Box::pin(async { Ok(()) })
    }

    /// One attempt, which gives up after `timeout`.
    ///
    /// `SendOutcome::Accepted` means the answer is `Ok` with a success status:
    /// the delivery logs and counts a report as accepted on that alone. A
    /// sender that says accepted with any other answer breaks this contract;
    /// the delivery then counts the report as rejected instead of accepting it.
    fn send<'a>(
        &'a self,
        job: &'a Job,
        timeout: Duration,
    ) -> BoxFuture<'a, (SendOutcome, AttemptLog)>;
}

/// What was pending when `drain` started and what it left behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Drained {
    pub pending: usize,
    pub left_waiting: usize,
    pub left_in_flight: usize,
    pub waited: Duration,
}

impl Drained {
    pub fn left(&self) -> usize {
        self.left_waiting + self.left_in_flight
    }
}

/// Tries of the startup check of the usage queue.
const STARTUP_CHECK_ATTEMPTS: u32 = 4;

/// The process's usage-report delivery: one per process, shared by every
/// model of a list (the billing API and its capacity are shared too).
pub struct UsageReportDelivery {
    policy: UsageReportPolicy,
    /// False for the default policy over HTTP, however it was arrived at. The delivery
    /// series, the extra log fields and the shutdown line exist only for a
    /// process whose policy differs from the default, so one that changed
    /// nothing keeps its `/metrics` and its logs.
    extended: bool,
    /// How an attempt is made: over HTTP to cloud-api, or onto the usage queue.
    sender: Arc<dyn UsageSender>,
    state: Mutex<State>,
    /// Notified whenever nothing is pending any more.
    idle: tokio::sync::Notify,
}

impl Default for UsageReportDelivery {
    fn default() -> Self {
        Self::with(UsageReportPolicy::default())
    }
}

impl UsageReportDelivery {
    pub fn new(policy: UsageReportPolicy) -> Arc<Self> {
        Self::with_sender(policy, Arc::new(http_sender::HttpSender))
    }

    /// The delivery for the configured transport. Fails when the transport
    /// cannot send (no credentials for SQS): that is a startup error.
    pub fn for_transport(
        policy: UsageReportPolicy,
        transport: &UsageReportTransport,
    ) -> anyhow::Result<Arc<Self>> {
        let delivery = match transport {
            UsageReportTransport::Http => Self::new(policy),
            UsageReportTransport::Sqs(settings) => {
                let sender = sqs_sender::SqsSender::new(settings);
                Self::with_sender(policy, Arc::new(sender))
            }
        };
        // Names the transport when it is not HTTP: the HTTP startup output is
        // what it was. No endpoint, queue URL or credential is logged.
        if delivery.sender.kind() != TransportKind::Http {
            info!(
                transport = delivery.sender.kind().label(),
                "Usage report transport configured"
            );
        }
        Ok(delivery)
    }

    /// Startup check of the transport (SQS only; HTTP makes no call and logs
    /// nothing): `STARTUP_CHECK_ATTEMPTS` tries, each within the policy's
    /// attempt timeout, backing off from its initial backoff. An answer no retry
    /// can change (a missing queue) stops at once. A failure is a startup error
    /// naming the outcome category, never a secret.
    pub async fn check_startup(&self, client: &reqwest::Client) -> anyhow::Result<()> {
        if self.sender.kind() == TransportKind::Http {
            return Ok(());
        }
        let mut backoff = self.policy.initial_backoff;
        for attempt in 1..=STARTUP_CHECK_ATTEMPTS {
            let Err((outcome, error)) = self
                .sender
                .startup_check(client, self.policy.attempt_timeout)
                .await
            else {
                info!(
                    transport = self.sender.kind().label(),
                    "Usage queue reachable"
                );
                return Ok(());
            };
            if attempt == STARTUP_CHECK_ATTEMPTS || !outcome.is_retryable() {
                anyhow::bail!("usage queue check failed ({}): {error}", outcome.as_label());
            }
            tracing::warn!(
                attempt,
                outcome = outcome.as_label(),
                "Usage queue check failed; retrying"
            );
            tokio::time::sleep(backoff).await;
            backoff *= 2;
        }
        unreachable!("the last attempt returns")
    }

    /// The transport this process runs on, decided once at startup. With
    /// `sqs`, the startup check of the usage queue decides: if it fails and
    /// the HTTP transport is fully configured (`http_configured`: cloud-api's
    /// URL and usage token), the process uses HTTP for its whole life and says
    /// so loudly; if HTTP is not configured the failure is a startup error, so
    /// that no process runs without a billing path. A process that starts on
    /// SQS stays on SQS: there is no switching afterwards.
    pub async fn resolve_transport(
        policy: UsageReportPolicy,
        transport: UsageReportTransport,
        http_configured: bool,
    ) -> anyhow::Result<UsageReportTransport> {
        if transport == UsageReportTransport::Http {
            return Ok(transport);
        }
        let delivery = Self::for_transport(policy, &transport)?;
        delivery
            .resolve_after_check(transport, http_configured)
            .await
    }

    async fn resolve_after_check(
        &self,
        transport: UsageReportTransport,
        http_configured: bool,
    ) -> anyhow::Result<UsageReportTransport> {
        let error = match self.check_startup(&reqwest::Client::new()).await {
            Ok(()) => return Ok(transport),
            Err(error) => error,
        };
        if !http_configured {
            return Err(error.context(
                "the sqs usage transport cannot be used and the http transport is not \
                 configured (CLOUD_API_URL and CLOUD_API_USAGE_TOKEN): refusing to start \
                 without a billing path",
            ));
        }
        tracing::error!(
            error = %error,
            "Usage queue check failed: falling back to the HTTP usage transport (cloud-api) \
             for the life of this process"
        );
        // The active transport as a pair of gauges: alert on `transport="http"` being 1.
        metrics::gauge!("inference_proxy_usage_report_transport_info", "transport" => "sqs")
            .set(0.0);
        metrics::gauge!("inference_proxy_usage_report_transport_info", "transport" => "http")
            .set(1.0);
        Ok(UsageReportTransport::Http)
    }

    /// The delivery with `sender` making its attempts. `new` is this with the
    /// HTTP sender; the startup line is the same whichever sender is given.
    pub(crate) fn with_sender(
        policy: UsageReportPolicy,
        sender: Arc<dyn UsageSender>,
    ) -> Arc<Self> {
        let delivery = Self::build(policy, sender);
        if delivery.extended {
            let policy = &delivery.policy;
            info!(
                attempt_timeout_secs = policy.attempt_timeout.as_secs(),
                max_attempts = policy.max_attempts,
                initial_backoff_ms = policy.initial_backoff.as_millis() as u64,
                deadline_secs = policy.deadline.map_or(0, |deadline| deadline.as_secs()),
                max_in_flight = policy.max_in_flight,
                max_queued = policy.max_queued,
                shutdown_drain_secs = policy.shutdown_drain.as_secs(),
                "Usage report delivery configured"
            );
        }
        Arc::new(delivery)
    }

    fn with(policy: UsageReportPolicy) -> Self {
        Self::build(policy, Arc::new(http_sender::HttpSender))
    }

    fn build(policy: UsageReportPolicy, sender: Arc<dyn UsageSender>) -> Self {
        // A transport other than the default is named by a gauge of 1. The
        // default (HTTP) adds no series: its `/metrics` stay as they were.
        if sender.kind() != TransportKind::Http {
            metrics::gauge!(
                "inference_proxy_usage_report_transport_info",
                "transport" => sender.kind().label()
            )
            .set(1.0);
        }
        Self {
            // A non-default transport is new, so its drops are always counted:
            // the HTTP delivery keeps its series unless the policy changes.
            extended: policy != UsageReportPolicy::default()
                || sender.kind() != TransportKind::Http,
            policy,
            sender,
            state: Mutex::default(),
            idle: tokio::sync::Notify::new(),
        }
    }

    pub fn policy(&self) -> &UsageReportPolicy {
        &self.policy
    }

    /// Whether reports go to cloud-api over HTTP, the transport with no queue.
    pub(crate) fn is_http(&self) -> bool {
        self.sender.kind() == TransportKind::Http
    }

    /// Time left before `job`'s deadline, `None` without one.
    fn time_left(&self, job: &Job) -> Option<Duration> {
        self.policy
            .deadline
            .map(|deadline| deadline.saturating_sub(job.completed_at.elapsed()))
    }

    /// Backoff after `attempts` failed attempts: half of
    /// `initial_backoff * 2^(attempts - 1)` (at most 30 seconds) plus a random
    /// share of the other half. The fixed half keeps a retry from being
    /// immediate, the random half keeps the reports that failed together from
    /// retrying together.
    fn backoff(&self, attempts: u32) -> Duration {
        let initial_ms = u64::try_from(self.policy.initial_backoff.as_millis()).unwrap_or(u64::MAX);
        let exponent = attempts.saturating_sub(1).min(20);
        let ceiling_ms = initial_ms
            .saturating_mul(1 << exponent)
            .min(MAX_BACKOFF.as_millis() as u64);
        let fixed_ms = ceiling_ms.div_ceil(2);
        Duration::from_millis(fixed_ms + rand::random_range(0..=ceiling_ms - fixed_ms))
    }

    /// Send `job` until the billing API accepts it, answers something final,
    /// the attempts run out or no attempt fits before its deadline any more.
    async fn deliver(&self, job: &Job) {
        let reporter = &job.reporter;
        let timeout = self.policy.attempt_timeout;
        let mut attempts: u32 = 0;
        let mut last_attempt = None;
        // Why the attempt about to be made is a retry.
        let mut retrying: Option<&'static str> = None;
        loop {
            // An attempt is only started while its whole timeout fits before
            // the deadline. Cutting the timeout short instead would, under a
            // backlog as old as the deadline, send every report with almost
            // no time left and get none of them accepted.
            if self.time_left(job).is_some_and(|left| left < timeout) {
                return self.give_up(
                    job,
                    GiveUp::Deadline {
                        attempts,
                        last_attempt,
                    },
                );
            }
            if let Some(reason) = retrying.take() {
                self.count(
                    "inference_proxy_usage_report_retries_total",
                    reporter,
                    ("reason", reason),
                );
            }
            attempts += 1;
            let (decision, attempt) = self.sender.send(job, timeout).await;
            last_attempt = Some((attempt.outcome, attempt.duration));
            self.count(
                "inference_proxy_usage_report_attempts_total",
                reporter,
                ("outcome", attempt.outcome.as_label()),
            );

            if let (SendOutcome::Retryable(reason), true) =
                (decision, attempts < self.policy.max_attempts)
            {
                let delay = self.backoff(attempts);
                // No use backing off for an attempt that could not start.
                if self
                    .time_left(job)
                    .is_some_and(|left| left < delay + timeout)
                {
                    return self.give_up(
                        job,
                        GiveUp::Deadline {
                            attempts,
                            last_attempt,
                        },
                    );
                }
                warn!(
                    request_id = %reporter.request_id.as_deref().unwrap_or(""),
                    org_id = %reporter.org_id.as_deref().unwrap_or(""),
                    workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                    api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                    model = %reporter.model_name,
                    status = attempt.answer.as_ref().ok().map(tracing::field::display),
                    error = attempt.answer.as_ref().err().map(tracing::field::display),
                    duration_ms = attempt.duration.as_millis() as u64,
                    auth_path = reporter.request_source.auth_path.as_label(),
                    ingress_route = reporter.request_source.ingress_route.as_label(),
                    outcome = attempt.outcome.as_label(),
                    attempt = attempts,
                    max_attempts = self.policy.max_attempts,
                    retry_in_ms = delay.as_millis() as u64,
                    "Usage report attempt failed, retrying"
                );
                retrying = Some(reason);
                tokio::time::sleep(delay).await;
                continue;
            }

            // The report's final outcome: the series and the log lines a
            // report has always ended with. `attempts` and
            // `since_completion_ms` are absent under the default policy.
            let outcome = attempt.outcome;
            let elapsed = attempt.duration;
            let answer = &attempt.answer;
            record_usage_report_outcome(reporter, outcome, Some(elapsed));
            let attempts = self.extended.then_some(attempts);
            let since_completion_ms = self
                .extended
                .then(|| job.completed_at.elapsed().as_millis() as u64);
            match answer {
                Ok(status) if decision == SendOutcome::Accepted => {
                    // Named by transport: "by Cloud API" is only true over HTTP.
                    // The HTTP text is the one it has always had.
                    let accepted = if self.is_http() {
                        "Direct-key usage report accepted by Cloud API"
                    } else {
                        "Usage report accepted by the usage queue"
                    };
                    if self.extended {
                        metrics::histogram!(
                            "inference_proxy_usage_report_time_to_accepted_seconds",
                            source_labels(reporter, None)
                        )
                        .record(job.completed_at.elapsed().as_secs_f64());
                    }
                    info!(
                        target: FINAL_OUTCOME_TARGET,
                        request_id = %reporter.request_id.as_deref().unwrap_or(""),
                        org_id = %reporter.org_id.as_deref().unwrap_or(""),
                        workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                        api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                        model = %reporter.model_name,
                        status = %status,
                        duration_ms = elapsed.as_millis() as u64,
                        auth_path = reporter.request_source.auth_path.as_label(),
                        ingress_route = reporter.request_source.ingress_route.as_label(),
                        attempts,
                        since_completion_ms,
                        "{}",
                        accepted
                    );
                    return;
                }
                Ok(status) => {
                    warn!(
                        target: FINAL_OUTCOME_TARGET,
                        request_id = %reporter.request_id.as_deref().unwrap_or(""),
                        org_id = %reporter.org_id.as_deref().unwrap_or(""),
                        workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                        api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                        model = %reporter.model_name,
                        status = %status,
                        duration_ms = elapsed.as_millis() as u64,
                        auth_path = reporter.request_source.auth_path.as_label(),
                        ingress_route = reporter.request_source.ingress_route.as_label(),
                        outcome = outcome.as_label(),
                        attempts,
                        since_completion_ms,
                        "Usage reporting returned non-success"
                    );
                }
                Err(error) => {
                    warn!(
                        target: FINAL_OUTCOME_TARGET,
                        request_id = %reporter.request_id.as_deref().unwrap_or(""),
                        org_id = %reporter.org_id.as_deref().unwrap_or(""),
                        workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                        api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                        model = %reporter.model_name,
                        error = %error,
                        duration_ms = elapsed.as_millis() as u64,
                        auth_path = reporter.request_source.auth_path.as_label(),
                        ingress_route = reporter.request_source.ingress_route.as_label(),
                        outcome = outcome.as_label(),
                        attempts,
                        since_completion_ms,
                        "Usage reporting failed"
                    );
                }
            }
            // Not accepted, and no attempt follows: either the answer is
            // final, or it could have passed but the attempts are used up.
            let reason = match decision {
                SendOutcome::Final(reason) => reason,
                SendOutcome::Retryable(_) => "attempts_exhausted",
                // Accepted with an answer that is not a success breaks the
                // sender's contract (see `UsageSender::send`). Not billed by
                // this delivery, so it is a rejection, not a panic.
                SendOutcome::Accepted => "rejected",
            };
            self.count(
                "inference_proxy_usage_reports_dropped_total",
                reporter,
                ("reason", reason),
            );
            return;
        }
    }
}

#[cfg(test)]
#[path = "usage_report_tests.rs"]
mod tests;
