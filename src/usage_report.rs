//! Delivery of usage reports to the billing API (cloud-api's
//! `POST /v1/internal/usage`).
//!
//! `proxy::spawn_usage_report` checks and completes a report, serializes it
//! once and hands it over here. From then on the report is off the request
//! path: `submit` takes a lock for a queue operation and returns, and nothing
//! in this module is ever awaited by a handler. Everything else, the log line
//! of a dropped report included, happens in tasks of this module's own,
//! outside the span of the request that happened to be completing.
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

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::model_metrics::{model_gauge, ModelLabel};
use crate::proxy::{
    classify_usage_http_status, classify_usage_request_error, record_usage_report_outcome,
    UsageReportOutcome, UsageReporter,
};

/// Ceiling of the backoff between two attempts, whatever the attempt number.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// The three lines a report has always ended with were written by `proxy.rs`
/// and keep its log target, so a filter or a query on it still finds them.
const FINAL_OUTCOME_TARGET: &str = "vllm_proxy_rs::proxy";

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

/// A report on its way to the billing API: what `proxy::spawn_usage_report`
/// hands over.
pub(crate) struct Job {
    /// Who the report is for and where it came from: ids and labels for the
    /// log lines and the metrics, and the HTTP client.
    pub reporter: UsageReporter,
    pub url: String,
    pub authorization: String,
    /// The serialized report. Every attempt sends these bytes.
    pub body: bytes::Bytes,
    /// When the request completed and its report was handed over.
    pub completed_at: Instant,
}

#[derive(Default)]
struct State {
    /// Reports waiting for a place, oldest first. (A waiting report refers
    /// back to its delivery through its reporter, until it leaves the queue.)
    waiting: VecDeque<Job>,
    /// Places taken: reports being sent or backing off before a retry.
    in_flight: usize,
}

/// Why a report was given up on without a final answer from the billing API.
#[derive(Clone, Copy)]
enum GiveUp {
    /// Pushed out of a full queue; it was never sent.
    QueueFull,
    /// No attempt fits before its deadline any more, after this many
    /// attempts, the last of which ended like this.
    Deadline {
        attempts: u32,
        last_attempt: Option<UsageReportOutcome>,
    },
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

/// The process's usage-report delivery: one per process, shared by every
/// model of a list (the billing API and its capacity are shared too).
pub struct UsageReportDelivery {
    policy: UsageReportPolicy,
    /// False for the default policy. The delivery series, the extra log
    /// fields and the shutdown line exist only for a process that changed a
    /// setting, so one that changed none keeps its `/metrics` and its logs.
    extended: bool,
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
        let delivery = Self::with(policy);
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
        Self {
            extended: policy != UsageReportPolicy::default(),
            policy,
            state: Mutex::default(),
            idle: tokio::sync::Notify::new(),
        }
    }

    pub fn policy(&self) -> &UsageReportPolicy {
        &self.policy
    }

    /// Reports waiting for a place, and reports holding one.
    pub fn pending(&self) -> (usize, usize) {
        let state = self.state();
        (state.waiting.len(), state.in_flight)
    }

    fn state(&self) -> MutexGuard<'_, State> {
        // Nothing panics while holding the lock; if something ever does, the
        // counts are still the best there is.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Take a report for delivery. Never blocks and never waits: the report
    /// starts at once while there is a place, waits in the queue otherwise.
    /// Must be called inside a Tokio runtime.
    pub(crate) fn submit(self: &Arc<Self>, job: Job) {
        let model = job.reporter.model_label;
        let evicted = {
            let mut state = self.state();
            if self.policy.max_in_flight == 0 || state.in_flight < self.policy.max_in_flight {
                state.in_flight += 1;
                drop(state);
                // Held from here on, whatever becomes of the task.
                let place = Place {
                    delivery: Arc::clone(self),
                    model,
                    held: true,
                };
                self.in_flight_gauge(model, Step::Up);
                // Nothing waits; said once so the series is there to read
                // before the first report ever has to.
                self.waiting_gauge(model, Step::None);
                tokio::spawn(place.work(job));
                return;
            }
            // The queue is full: the report that has waited longest goes. It
            // has the least time left before its deadline.
            let evicted = if state.waiting.len() >= self.policy.max_queued {
                state.waiting.pop_front()
            } else {
                None
            };
            state.waiting.push_back(job);
            evicted
        };
        self.waiting_gauge(model, Step::Up);
        if let Some(evicted) = evicted {
            self.waiting_gauge(evicted.reporter.model_label, Step::Down);
            // Logged and counted in a task of its own: the line is about
            // another request's report, so it belongs neither on this
            // caller's path nor in its request span.
            let delivery = Arc::clone(self);
            tokio::spawn(async move { delivery.give_up(&evicted, GiveUp::QueueFull) });
        }
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
            let started_at = Instant::now();
            let mut request = reporter
                .http_client
                .post(&job.url)
                .header("authorization", &job.authorization)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(job.body.clone())
                .timeout(timeout);
            if let Some(request_id) = reporter.request_id.as_deref() {
                request = request.header("x-request-id", request_id);
            }
            // The body is not read, as before: only the status decides.
            let answer = request.send().await.map(|response| response.status());
            let elapsed = started_at.elapsed();
            let outcome = match &answer {
                Ok(status) => classify_usage_http_status(*status),
                Err(error) => classify_usage_request_error(error),
            };
            last_attempt = Some(outcome);
            self.count(
                "inference_proxy_usage_report_attempts_total",
                reporter,
                ("outcome", outcome.as_label()),
            );

            let retry_reason = retry_reason(outcome, &answer);
            if let (Some(reason), true) = (retry_reason, attempts < self.policy.max_attempts) {
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
                    status = answer.as_ref().ok().map(tracing::field::display),
                    error = answer.as_ref().err().map(tracing::field::display),
                    duration_ms = elapsed.as_millis() as u64,
                    auth_path = reporter.request_source.auth_path.as_label(),
                    ingress_route = reporter.request_source.ingress_route.as_label(),
                    outcome = outcome.as_label(),
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
            record_usage_report_outcome(reporter, outcome, Some(elapsed));
            let attempts = self.extended.then_some(attempts);
            let since_completion_ms = self
                .extended
                .then(|| job.completed_at.elapsed().as_millis() as u64);
            match &answer {
                Ok(status) if outcome == UsageReportOutcome::Accepted => {
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
                        "Direct-key usage report accepted by Cloud API"
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
            let reason = if retry_reason.is_some() {
                "attempts_exhausted"
            } else {
                "rejected"
            };
            self.count(
                "inference_proxy_usage_reports_dropped_total",
                reporter,
                ("reason", reason),
            );
            return;
        }
    }

    /// Drop `job` without a (further) attempt: counted as the report's final
    /// outcome and as a drop, and logged with its ids.
    fn give_up(&self, job: &Job, why: GiveUp) {
        let reporter = &job.reporter;
        let (outcome, reason) = match why {
            GiveUp::QueueFull => (UsageReportOutcome::QueueFull, "queue_full"),
            GiveUp::Deadline { .. } => (UsageReportOutcome::DeadlineExceeded, "deadline"),
        };
        record_usage_report_outcome(reporter, outcome, None);
        self.count(
            "inference_proxy_usage_reports_dropped_total",
            reporter,
            ("reason", reason),
        );
        let since_completion_ms = job.completed_at.elapsed().as_millis() as u64;
        match why {
            GiveUp::QueueFull => warn!(
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
            ),
            GiveUp::Deadline {
                attempts,
                last_attempt,
            } => warn!(
                request_id = %reporter.request_id.as_deref().unwrap_or(""),
                org_id = %reporter.org_id.as_deref().unwrap_or(""),
                workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                model = %reporter.model_name,
                auth_path = reporter.request_source.auth_path.as_label(),
                ingress_route = reporter.request_source.ingress_route.as_label(),
                outcome = outcome.as_label(),
                attempts,
                last_attempt = last_attempt.map(UsageReportOutcome::as_label),
                since_completion_ms,
                "Usage report dropped: not accepted before its deadline — usage NOT billed"
            ),
        }
    }

    /// Wait until nothing is pending, for at most `limit`.
    pub async fn drain(&self, limit: Duration) -> Drained {
        let started_at = Instant::now();
        let give_up_at = tokio::time::Instant::now() + limit.min(Duration::from_secs(86_400));
        let (waiting, in_flight) = self.pending();
        let pending = waiting + in_flight;
        loop {
            // Registered before the count is read, so the notification of a
            // report that ends in between is not missed.
            let idle = self.idle.notified();
            tokio::pin!(idle);
            idle.as_mut().enable();
            let (waiting, in_flight) = self.pending();
            if waiting + in_flight == 0 || tokio::time::timeout_at(give_up_at, idle).await.is_err()
            {
                let (left_waiting, left_in_flight) = self.pending();
                return Drained {
                    pending,
                    left_waiting,
                    left_in_flight,
                    waited: started_at.elapsed(),
                };
            }
        }
    }

    /// Shutdown, once the server has stopped serving: give pending reports
    /// `shutdown_drain` to finish and say how many were left. A process with
    /// the default policy exits as it always has, without waiting or logging
    /// (`None`).
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

    /// One more on a per-report delivery counter.
    fn count(
        &self,
        name: &'static str,
        reporter: &UsageReporter,
        label: (&'static str, &'static str),
    ) {
        if self.extended {
            metrics::counter!(name, source_labels(reporter, Some(label))).increment(1);
        }
    }

    /// One more (`Step::Up`) or one fewer report of `model` waiting for a
    /// place; `Step::None` only makes the series exist.
    fn waiting_gauge(&self, model: ModelLabel, step: Step) {
        if self.extended {
            step.apply(model_gauge!(
                model,
                "inference_proxy_usage_report_queue_depth"
            ));
        }
    }

    /// One more or one fewer report of `model` holding a place.
    fn in_flight_gauge(&self, model: ModelLabel, step: Step) {
        if self.extended {
            step.apply(model_gauge!(
                model,
                "inference_proxy_usage_reports_in_flight"
            ));
        }
    }
}

#[derive(Clone, Copy)]
enum Step {
    Up,
    Down,
    None,
}

impl Step {
    fn apply(self, gauge: metrics::Gauge) {
        match self {
            Step::Up => gauge.increment(1.0),
            Step::Down => gauge.decrement(1.0),
            Step::None => gauge.increment(0.0),
        }
    }
}

/// A place among the `max_in_flight`, held by one worker task.
struct Place {
    delivery: Arc<UsageReportDelivery>,
    /// The model of the report in the place, for the gauge.
    model: ModelLabel,
    held: bool,
}

impl Place {
    /// Deliver `first`, then whatever waits, and give the place back when
    /// nothing does.
    async fn work(mut self, first: Job) {
        let mut job = first;
        loop {
            self.delivery.deliver(&job).await;
            match self.next() {
                Some(next) => job = next,
                None => return,
            }
            // A backlog of reports past their deadline is dropped without a
            // single await; other tasks get their turn between two of them.
            tokio::task::yield_now().await;
        }
    }

    /// The next report to deliver in this place, or `None` after giving the
    /// place back because nothing waits. One lock for both, so a report
    /// queued in between cannot be left waiting with a place free.
    fn next(&mut self) -> Option<Job> {
        let (next, idle) = {
            let mut state = self.delivery.state();
            let next = state.waiting.pop_front();
            if next.is_none() {
                state.in_flight -= 1;
                self.held = false;
            }
            (next, state.in_flight == 0)
        };
        self.delivery.in_flight_gauge(self.model, Step::Down);
        if let Some(next) = &next {
            self.model = next.reporter.model_label;
            self.delivery.waiting_gauge(self.model, Step::Down);
            self.delivery.in_flight_gauge(self.model, Step::Up);
        }
        if idle {
            self.delivery.idle.notify_waiters();
        }
        next
    }
}

impl Drop for Place {
    /// A worker that ends without giving its place back (it panicked, or the
    /// runtime is going away) must not take the place with it: under a cap
    /// that would stop delivery for good once every place was lost. Reports
    /// still waiting are picked up by the worker of the next report.
    fn drop(&mut self) {
        if self.held {
            let idle = {
                let mut state = self.delivery.state();
                state.in_flight -= 1;
                state.in_flight == 0 && state.waiting.is_empty()
            };
            self.delivery.in_flight_gauge(self.model, Step::Down);
            if idle {
                self.delivery.idle.notify_waiters();
            }
        }
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

/// `label`, the request source and, in list mode, the model: the label set of
/// the per-report delivery series, in the order of the existing ones.
fn source_labels(
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

#[cfg(test)]
#[path = "usage_report_tests.rs"]
mod tests;
