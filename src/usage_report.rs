//! Delivery of usage reports to the billing API (cloud-api's
//! `POST /v1/internal/usage`).
//!
//! `proxy::spawn_usage_report` checks and completes a report, serializes it
//! once and hands it over here. From then on the report is off the request
//! path: `submit` takes a lock for a queue operation and returns, and nothing
//! in this module is ever awaited by a handler. Sending, and the line of a
//! report dropped from a full queue, happen in tasks of this module's own,
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
//!
//! All of that is in memory: a report dropped from the queue, past its
//! deadline or held by a process that dies is usage served and never billed.
//! `VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH` adds an outbox on disk
//! (`usage_outbox.rs`). A report is then written to it first and delivered
//! from it: it stays there until the billing API accepts it or refuses it for
//! good, across restarts, a billing API outage of any length and a second
//! process on the same file. `submit` is still a lock and a push; the file is
//! only ever touched by the outbox's own thread. When the file cannot be
//! used, reports are delivered from memory as above until it can.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, watch};
use tracing::{debug, error, info, warn};

use crate::auth::{AuthPath, IngressRouteKind, RequestSource};
use crate::model_metrics::{model_gauge, ModelLabel};
use crate::proxy::{
    classify_usage_http_status, classify_usage_request_error, record_usage_report_outcome,
    UsageReportOutcome, UsageReporter,
};
use crate::usage_outbox::{self, Persist, Refused, Settle, Store, UsageOutboxConfig};

/// Ceiling of the backoff between two attempts, whatever the attempt number.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// The three lines a report has always ended with were written by `proxy.rs`
/// and keep its log target, so a filter or a query on it still finds them.
const FINAL_OUTCOME_TARGET: &str = "vllm_proxy_rs::proxy";

/// How many reports dropped from a full queue may have their line still to
/// be written, each in a task of its own. Past it the line is written where
/// the drop happens, so these tasks cannot pile up without bound either.
const MAX_DROPS_BEING_LOGGED: usize = 1_000;

/// How long shutdown waits for the outbox to write what is left and give its
/// leases back. The disk may be the reason the process is being stopped.
const OUTBOX_CLOSE_TIMEOUT: Duration = Duration::from_secs(10);

/// Model labels read back from the outbox are kept for the life of the
/// process, like those of a model list. There are as many as models were ever
/// served from the file; a file that claims more gets no label for the rest.
const MAX_STORED_MODEL_LABELS: usize = 256;

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

impl UsageReportPolicy {
    /// `max_attempts` of a report that is sent until it is accepted: what a
    /// process with an outbox gets when
    /// `VLLM_PROXY_USAGE_REPORT_MAX_ATTEMPTS` is not set.
    pub const UNTIL_ACCEPTED: u32 = u32::MAX;

    /// `max_in_flight` of a process with an outbox when
    /// `VLLM_PROXY_USAGE_REPORT_MAX_IN_FLIGHT` is not set. An outbox is
    /// always delivered under a cap: without one, the whole backlog of an
    /// outage would be sent at once when the billing API came back.
    pub const OUTBOX_MAX_IN_FLIGHT: usize = 8;

    /// The policy of a process with an outbox and no other setting: a report
    /// is sent until it is accepted, `OUTBOX_MAX_IN_FLIGHT` at a time.
    pub fn durable() -> Self {
        Self {
            max_attempts: Self::UNTIL_ACCEPTED,
            max_in_flight: Self::OUTBOX_MAX_IN_FLIGHT,
            ..Self::default()
        }
    }

    /// The attempts a report gets, `None` when it is sent until accepted.
    pub fn attempt_cap(&self) -> Option<u32> {
        (self.max_attempts != Self::UNTIL_ACCEPTED).then_some(self.max_attempts)
    }
}

/// A report on its way to the billing API: what `proxy::spawn_usage_report`
/// hands over, or a row of the outbox made into one (`job_from`).
pub(crate) struct Job {
    /// Who the report is for and where it came from: ids and labels for the
    /// log lines and the metrics, and the HTTP client.
    pub reporter: UsageReporter,
    pub url: String,
    pub authorization: String,
    /// The serialized report. Every attempt sends these bytes.
    pub body: bytes::Bytes,
    /// When the request completed and its report was handed over. For a
    /// report taken from the outbox, when it was taken: its request
    /// completed `Stored::age` before that.
    pub completed_at: Instant,
    /// The row this report is, when it was taken from the outbox.
    pub stored: Option<Stored>,
}

/// What the outbox knows about a report it leased to this process.
pub(crate) struct Stored {
    id: i64,
    /// Attempts made before this lease, by any process.
    attempts: u32,
    /// Why the last of them failed, when there was one.
    last_failure: Option<(&'static str, UsageReportOutcome)>,
    /// How long before the lease its request completed.
    age: Duration,
    /// Until when the report is this process's to send.
    lease_until: Instant,
}

impl Job {
    /// Time since the request completed, whichever process served it.
    fn since_completion(&self) -> Duration {
        let before = self.stored.as_ref().map_or(Duration::ZERO, |row| row.age);
        self.completed_at.elapsed().saturating_add(before)
    }
}

impl Persist for Job {
    fn row(&self) -> usage_outbox::NewRow<'_> {
        let reporter = &self.reporter;
        usage_outbox::NewRow {
            body: &self.body,
            request_id: reporter.request_id.as_deref(),
            model_label: reporter.model_label,
            auth_path: reporter.request_source.auth_path.as_label(),
            ingress_route: reporter.request_source.ingress_route.as_label(),
            age: self.since_completion(),
        }
    }
}

#[derive(Default)]
struct State {
    /// Reports waiting for a place, oldest first. (A waiting report refers
    /// back to its delivery through its reporter, until it leaves the queue.)
    waiting: VecDeque<Job>,
    /// Places taken: reports being sent or backing off before a retry.
    in_flight: usize,
    /// Reports dropped from a full queue whose line is still to be written.
    drops_being_logged: usize,
    /// `None` without an outbox.
    outbox: Option<OutboxState>,
}

impl State {
    fn is_idle(&self) -> bool {
        self.waiting.is_empty()
            && self.in_flight == 0
            && self.drops_being_logged == 0
            && self.outbox.as_ref().is_none_or(OutboxState::is_empty)
    }
}

/// What a process with an outbox keeps in memory about it.
struct OutboxState {
    /// Reports leased to this process and waiting for a place, longest due
    /// first.
    claimed: VecDeque<Job>,
    /// How many reports the claim on its way to the store asked for.
    asked: Option<usize>,
    /// The file may hold a report that is due: worth a claim.
    look: bool,
    /// When that becomes true by itself: the retry of a report is due.
    look_at: Option<Instant>,
    /// Attempts in a row, across the process, that failed in a way that can
    /// pass, since the billing API last gave a report an answer of its own.
    failing_streak: u32,
    /// Places taken by reports from the file.
    sending: usize,
    /// Shutdown has begun: no report is claimed or started any more.
    stopping: bool,
    /// The last thing the store tried worked.
    available: bool,
    /// The store has said what the file holds.
    synced: bool,
    /// Rows of the file per model, waiting or being sent by anyone.
    stored: Vec<(ModelLabel, u64)>,
    /// When the request of the oldest of them completed.
    oldest_completed_at_ms: Option<i64>,
    rejected: u64,
    /// Per model, what `queue_depth` is beside the rows of the file: the
    /// reports waiting in memory, less the rows this process is sending.
    beside: HashMap<ModelLabel, i64>,
}

impl OutboxState {
    fn new() -> Self {
        Self {
            claimed: VecDeque::new(),
            asked: None,
            look: true,
            look_at: None,
            failing_streak: 0,
            sending: 0,
            stopping: false,
            available: true,
            synced: false,
            stored: Vec::new(),
            oldest_completed_at_ms: None,
            rejected: 0,
            beside: HashMap::new(),
        }
    }

    fn stored_total(&self) -> u64 {
        self.stored.iter().map(|(_, rows)| rows).sum()
    }

    /// Nothing is known to wait in the file or on its way from it.
    fn is_empty(&self) -> bool {
        self.synced && self.claimed.is_empty() && self.stored_total() == 0
    }

    /// The next leased report, for a place that is free. The file counts its
    /// row among the pending ones until the outcome is written, so from here
    /// until `sent` it is taken off what `queue_depth` and `pending` call
    /// waiting.
    fn start_next(&mut self) -> Option<Job> {
        let job = self.claimed.pop_front()?;
        self.sending += 1;
        *self.beside.entry(job.reporter.model_label).or_default() -= 1;
        Some(job)
    }

    /// The attempt at a report of `model` taken with `start_next` has ended.
    fn sent(&mut self, model: ModelLabel) {
        self.sending = self.sending.saturating_sub(1);
        *self.beside.entry(model).or_default() += 1;
    }
}

/// Why a report was given up on without a final answer from the billing API.
#[derive(Clone, Copy)]
enum GiveUp {
    /// Pushed out of a full queue; it was never sent.
    QueueFull,
    /// Removed from an outbox at its bound, where it had waited longest.
    OutboxFull,
    /// No attempt fits before its deadline any more, after this many
    /// attempts, the last of which ended like this and took this long (a
    /// report taken from the outbox past its deadline had its last attempt
    /// under an earlier lease, and how long that took is not kept).
    Deadline {
        attempts: u32,
        last_attempt: Option<(UsageReportOutcome, Option<Duration>)>,
    },
}

/// One attempt at a report and what it met.
struct Attempt {
    answer: Result<reqwest::StatusCode, reqwest::Error>,
    outcome: UsageReportOutcome,
    elapsed: Duration,
}

/// Where the reports of the outbox are sent. Read from the configuration of
/// the running process, never from the file: neither the URL nor the bearer
/// is written to disk.
struct Endpoint {
    client: reqwest::Client,
    cloud_api_url: String,
    url: String,
    authorization: String,
}

/// The durable outbox of a delivery and what goes with it.
struct Outbox {
    store: Store<Job>,
    endpoint: Endpoint,
    /// How long a claimed report is this process's: two attempt timeouts
    /// (waiting for a place, then the attempt) and a margin.
    lease: Duration,
    /// What must be left of a lease, beyond the attempt timeout, for the
    /// attempt to start.
    lease_slack: Duration,
    /// Reports handed to the store whose transaction has not been heard of.
    in_transit: AtomicUsize,
    /// Wakes the dispatcher: a place is free, or a report was settled.
    wake: tokio::sync::Notify,
    /// True from shutdown on. Ends the pauses of the places.
    stopping: watch::Sender<bool>,
    /// The model labels read back from the file (`MAX_STORED_MODEL_LABELS`).
    labels: Mutex<HashSet<&'static str>>,
}

impl Outbox {
    /// Hand `job` to the store. `None` when the store took it; otherwise the
    /// job back, and why.
    fn keep(&self, job: Job, max_buffered: usize) -> Option<(Job, Refused)> {
        // Counted before the store has it, so that it is never out of sight.
        self.in_transit.fetch_add(1, Ordering::SeqCst);
        let refused = self.store.offer(job, max_buffered).err();
        if refused.is_some() {
            self.in_transit.fetch_sub(1, Ordering::SeqCst);
        }
        refused
    }

    /// `label` as a `ModelLabel`. A label is leaked once and reused.
    fn label(&self, label: Option<String>) -> ModelLabel {
        let label = label?;
        let mut labels = self.labels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(known) = labels.get(label.as_str()) {
            return Some(known);
        }
        if labels.len() >= MAX_STORED_MODEL_LABELS {
            return None;
        }
        let leaked: &'static str = Box::leak(label.into_boxed_str());
        labels.insert(leaked);
        Some(leaked)
    }
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
    /// False for the default policy, however it was arrived at. The delivery
    /// series, the extra log fields and the shutdown line exist only for a
    /// process whose policy differs from the default, so one that changed
    /// nothing keeps its `/metrics` and its logs.
    extended: bool,
    state: Mutex<State>,
    /// Notified whenever nothing is pending any more.
    idle: tokio::sync::Notify,
    /// The outbox on disk (`VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH`). `None`:
    /// reports are delivered from memory and nothing below it is used.
    outbox: Option<Outbox>,
}

impl Default for UsageReportDelivery {
    fn default() -> Self {
        Self::with(UsageReportPolicy::default())
    }
}

impl UsageReportDelivery {
    pub fn new(policy: UsageReportPolicy) -> Arc<Self> {
        let delivery = Self::with(policy);
        delivery.say_configured();
        Arc::new(delivery)
    }

    /// The delivery of a process as configured: with the outbox when
    /// `VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH` is set, `new` otherwise.
    /// `http_client` is the client that talks to the billing API. With an
    /// outbox this must be called inside a Tokio runtime.
    pub fn from_config(config: &crate::config::Config, http_client: &reqwest::Client) -> Arc<Self> {
        let policy = config.usage_report.clone();
        let Some(outbox) = &config.usage_report_outbox else {
            return Self::new(policy);
        };
        match (&config.cloud_api_url, &config.cloud_api_usage_token) {
            (Some(url), Some(token)) => {
                Self::with_outbox(policy, outbox.clone(), http_client.clone(), url, token)
            }
            // Nothing is reported without them, so there is nothing to keep,
            // and what an earlier process left in the file could not be sent.
            // The file is left as it is.
            _ => {
                error!(
                    path = %outbox.path.display(),
                    "Usage report outbox not opened: CLOUD_API_URL and CLOUD_API_USAGE_TOKEN \
                     are both needed to send what it holds"
                );
                metrics::gauge!("inference_proxy_usage_report_outbox_available").set(0.0);
                Self::new(policy)
            }
        }
    }

    /// A delivery that writes every report to the outbox at `outbox.path`
    /// and sends it from there, to `cloud_api_url` with `usage_token` as its
    /// bearer. Opening the file is left to the outbox's own thread and never
    /// fails here: while the file cannot be used, reports are delivered from
    /// memory. Must be called inside a Tokio runtime.
    ///
    /// The outbox is always delivered under a cap, so a `max_in_flight` of 0
    /// becomes `UsageReportPolicy::OUTBOX_MAX_IN_FLIGHT`.
    pub fn with_outbox(
        mut policy: UsageReportPolicy,
        outbox: UsageOutboxConfig,
        http_client: reqwest::Client,
        cloud_api_url: &str,
        usage_token: &str,
    ) -> Arc<Self> {
        if policy.max_in_flight == 0 {
            policy.max_in_flight = UsageReportPolicy::OUTBOX_MAX_IN_FLIGHT;
        }
        let lease = policy.attempt_timeout * 2 + outbox.lease_margin;
        info!(
            path = %outbox.path.display(),
            max_pending = outbox.max_pending,
            max_rejected = outbox.max_rejected,
            commit_interval_ms = outbox.commit_interval.as_millis() as u64,
            lease_secs = lease.as_secs(),
            until_accepted = policy.attempt_cap().is_none(),
            "Usage report outbox enabled: a report is kept on disk until the billing API \
             accepts it or refuses it for good"
        );
        let (events, from_store) = mpsc::unbounded_channel();
        let mut delivery = Self::with(policy);
        delivery.state = Mutex::new(State {
            outbox: Some(OutboxState::new()),
            ..State::default()
        });
        delivery.outbox = Some(Outbox {
            lease,
            lease_slack: outbox.lease_margin / 4,
            store: Store::start(outbox, events),
            endpoint: Endpoint {
                client: http_client,
                cloud_api_url: cloud_api_url.to_string(),
                url: format!("{cloud_api_url}/v1/internal/usage"),
                authorization: format!("Bearer {usage_token}"),
            },
            in_transit: AtomicUsize::new(0),
            wake: tokio::sync::Notify::new(),
            stopping: watch::Sender::new(false),
            labels: Mutex::default(),
        });
        delivery.say_configured();
        let delivery = Arc::new(delivery);
        tokio::spawn(Arc::clone(&delivery).dispatch(from_store));
        delivery
    }

    fn with(policy: UsageReportPolicy) -> Self {
        Self {
            extended: policy != UsageReportPolicy::default(),
            policy,
            state: Mutex::default(),
            idle: tokio::sync::Notify::new(),
            outbox: None,
        }
    }

    fn say_configured(&self) {
        if self.extended {
            let policy = &self.policy;
            info!(
                attempt_timeout_secs = policy.attempt_timeout.as_secs(),
                // Absent for a report that is sent until it is accepted.
                max_attempts = policy.attempt_cap(),
                initial_backoff_ms = policy.initial_backoff.as_millis() as u64,
                deadline_secs = policy.deadline.map_or(0, |deadline| deadline.as_secs()),
                max_in_flight = policy.max_in_flight,
                max_queued = policy.max_queued,
                shutdown_drain_secs = policy.shutdown_drain.as_secs(),
                "Usage report delivery configured"
            );
        }
    }

    pub fn policy(&self) -> &UsageReportPolicy {
        &self.policy
    }

    /// Reports waiting for a place, and reports holding one. With an outbox
    /// the waiting ones are those in memory and the rows of the file nobody
    /// here is sending, whichever process wrote them.
    pub fn pending(&self) -> (usize, usize) {
        let state = self.state();
        let mut waiting = state.waiting.len();
        if let (Some(outbox), Some(kept)) = (&self.outbox, &state.outbox) {
            let stored = usize::try_from(kept.stored_total()).unwrap_or(usize::MAX);
            waiting = waiting
                .saturating_add(stored.saturating_sub(kept.sending))
                .saturating_add(outbox.in_transit.load(Ordering::SeqCst));
        }
        (waiting, state.in_flight)
    }

    /// Nothing waits, nothing is in flight, and every drop has been logged.
    /// With an outbox, the file holds nothing either and nothing is on its
    /// way into it.
    fn is_idle(&self) -> bool {
        self.state().is_idle()
            && self
                .outbox
                .as_ref()
                .is_none_or(|outbox| outbox.in_transit.load(Ordering::SeqCst) == 0)
    }

    fn state(&self) -> MutexGuard<'_, State> {
        // Nothing panics while holding the lock; if something ever does, the
        // counts are still the best there is.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Take a report for delivery. Never blocks and never waits: the report
    /// starts at once while there is a place, waits in the queue otherwise.
    /// With an outbox it is handed to the store instead, which is a lock and
    /// a push as well: the file is written by the store's own thread.
    /// Must be called inside a Tokio runtime.
    pub(crate) fn submit(self: &Arc<Self>, job: Job) {
        let Some(outbox) = &self.outbox else {
            return self.hold(job);
        };
        if let Some((job, why)) = outbox.keep(job, self.policy.max_queued) {
            // The file cannot take it now. It is delivered all the same, from
            // memory, and not kept across a restart.
            self.count(
                "inference_proxy_usage_report_outbox_bypassed_total",
                &job.reporter,
                ("reason", why.as_label()),
            );
            self.hold(job);
        }
    }

    /// Deliver `job` from memory: at once while there is a place, from the
    /// queue otherwise.
    fn hold(self: &Arc<Self>, job: Job) {
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
                state.waiting.pop_front().map(|evicted| {
                    let in_own_task = state.drops_being_logged < MAX_DROPS_BEING_LOGGED;
                    if in_own_task {
                        state.drops_being_logged += 1;
                    }
                    (evicted, in_own_task)
                })
            } else {
                None
            };
            state.waiting.push_back(job);
            evicted
        };
        self.waiting_gauge(model, Step::Up);
        if let Some((evicted, in_own_task)) = evicted {
            self.waiting_gauge(evicted.reporter.model_label, Step::Down);
            if in_own_task {
                // The line is about another request's report: it belongs
                // neither on this caller's path nor in its request span.
                let delivery = Arc::clone(self);
                tokio::spawn(async move {
                    delivery.give_up(&evicted, GiveUp::QueueFull);
                    let idle = {
                        let mut state = delivery.state();
                        state.drops_being_logged -= 1;
                        state.is_idle()
                    };
                    if idle {
                        delivery.idle.notify_waiters();
                    }
                });
            } else {
                self.give_up(&evicted, GiveUp::QueueFull);
            }
        }
    }

    /// Time left before `job`'s deadline, `None` without one.
    fn time_left(&self, job: &Job) -> Option<Duration> {
        self.policy
            .deadline
            .map(|deadline| deadline.saturating_sub(job.since_completion()))
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
    /// (A report taken from the outbox gets one attempt at a time instead,
    /// and what follows is written to its row: `deliver_stored`.)
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
            attempts = attempts.saturating_add(1);
            let attempt = self.attempt(job).await;
            last_attempt = Some((attempt.outcome, Some(attempt.elapsed)));

            let retry_reason = retry_reason(attempt.outcome, &attempt.answer);
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
                self.say_retrying(job, &attempt, attempts, delay);
                retrying = Some(reason);
                tokio::time::sleep(delay).await;
                continue;
            }
            return self.end(job, &attempt, attempts, retry_reason);
        }
    }

    /// One attempt at a report taken from the outbox, and its outcome
    /// written to the row: accepted and the row is deleted; refused for good
    /// and it moves to `rejected`; failed in a way that can pass and it is
    /// due again after the backoff, its lease given back, for whichever
    /// process is next.
    ///
    /// True after such a failure: the place is then to stay taken for a while
    /// (`pause`), so that a billing API that is down or struggling gets fewer
    /// requests, not the whole backlog one report after the other.
    async fn deliver_stored(&self, outbox: &Outbox, job: &Job, row: &Stored) -> bool {
        let reporter = &job.reporter;
        let timeout = self.policy.attempt_timeout;
        // The report waited for its place longer than its lease allows for:
        // another process may take it while this one is still sending. It is
        // given back and claimed again, with a new lease.
        let lease_left = row.lease_until.saturating_duration_since(Instant::now());
        if lease_left < timeout + outbox.lease_slack {
            outbox.store.settle(Settle::Release { id: row.id });
            if let Some(kept) = &mut self.state().outbox {
                kept.look = true;
            }
            return false;
        }
        // With a deadline, a report has a maximum age, whoever kept it.
        if self.time_left(job).is_some_and(|left| left < timeout) {
            self.give_up(
                job,
                GiveUp::Deadline {
                    attempts: row.attempts,
                    last_attempt: row.last_failure.map(|(_, outcome)| (outcome, None)),
                },
            );
            outbox.store.settle(Settle::Delete { id: row.id });
            return false;
        }
        if let Some((reason, _)) = row.last_failure {
            self.count(
                "inference_proxy_usage_report_retries_total",
                reporter,
                ("reason", reason),
            );
        }
        let attempt = self.attempt(job).await;
        let attempts = row.attempts.saturating_add(1);
        let status = attempt.answer.as_ref().ok().map(|status| status.as_u16());

        let Some(reason) = retry_reason(attempt.outcome, &attempt.answer) else {
            // An answer about this report: the billing API is there.
            if let Some(kept) = &mut self.state().outbox {
                kept.failing_streak = 0;
            }
            self.end(job, &attempt, attempts, None);
            outbox
                .store
                .settle(if attempt.outcome == UsageReportOutcome::Accepted {
                    Settle::Delete { id: row.id }
                } else {
                    Settle::Reject {
                        id: row.id,
                        reason: "rejected",
                        status,
                        outcome: attempt.outcome.as_label(),
                        attempts,
                    }
                });
            return false;
        };
        if attempts >= self.policy.max_attempts {
            // An explicit cap is used up. The report is kept where a person
            // can look at it and send it again.
            self.end(job, &attempt, attempts, Some(reason));
            outbox.store.settle(Settle::Reject {
                id: row.id,
                reason: "attempts_exhausted",
                status,
                outcome: attempt.outcome.as_label(),
                attempts,
            });
        } else {
            let delay = self.backoff(attempts);
            if self
                .time_left(job)
                .is_some_and(|left| left < delay + timeout)
            {
                self.give_up(
                    job,
                    GiveUp::Deadline {
                        attempts,
                        last_attempt: Some((attempt.outcome, Some(attempt.elapsed))),
                    },
                );
                outbox.store.settle(Settle::Delete { id: row.id });
            } else {
                self.say_retrying(job, &attempt, attempts, delay);
                outbox.store.settle(Settle::Retry {
                    id: row.id,
                    attempts,
                    next_attempt_at_ms: usage_outbox::now_ms()
                        .saturating_add(delay.as_millis() as i64),
                    last_outcome: reason,
                });
                // The dispatcher looks again when the retry is due.
                let due_at = Instant::now() + delay;
                if let Some(kept) = &mut self.state().outbox {
                    kept.look_at = Some(kept.look_at.map_or(due_at, |at| at.min(due_at)));
                }
            }
        }
        true
    }

    /// Keep a place taken after an attempt failed in a way that can pass.
    /// The pause is the backoff of the number of such failures in a row
    /// across the process, so it doubles while nothing gets through and is
    /// back to the shortest after the first report that does. Shutdown ends
    /// it at once.
    ///
    /// The reports leased ahead and not started are given back: nothing may
    /// start them for a while, and a lease nobody uses only keeps another
    /// process from sending them.
    async fn pause(&self, outbox: &Outbox) {
        let (delay, unstarted) = {
            let mut state = self.state();
            let Some(kept) = &mut state.outbox else {
                return;
            };
            if kept.stopping {
                return;
            }
            kept.failing_streak = kept.failing_streak.saturating_add(1);
            kept.look |= !kept.claimed.is_empty();
            let delay = self
                .backoff(kept.failing_streak)
                .min(outbox.store.config().max_pause);
            (delay, std::mem::take(&mut kept.claimed))
        };
        for job in unstarted {
            if let Some(row) = &job.stored {
                outbox.store.settle(Settle::Release { id: row.id });
            }
        }
        let mut stopping = outbox.stopping.subscribe();
        tokio::select! {
            _ = tokio::time::sleep(delay) => {}
            _ = stopping.wait_for(|stopping| *stopping) => {}
        }
    }

    /// Send `job` once.
    async fn attempt(&self, job: &Job) -> Attempt {
        let reporter = &job.reporter;
        let started_at = Instant::now();
        let mut request = reporter
            .http_client
            .post(&job.url)
            .header("authorization", &job.authorization)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(job.body.clone())
            .timeout(self.policy.attempt_timeout);
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
        self.count(
            "inference_proxy_usage_report_attempts_total",
            reporter,
            ("outcome", outcome.as_label()),
        );
        Attempt {
            answer,
            outcome,
            elapsed,
        }
    }

    /// The line of an attempt that failed and will be made again in `delay`.
    fn say_retrying(&self, job: &Job, attempt: &Attempt, attempts: u32, delay: Duration) {
        let reporter = &job.reporter;
        warn!(
            request_id = %reporter.request_id.as_deref().unwrap_or(""),
            org_id = %reporter.org_id.as_deref().unwrap_or(""),
            workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
            api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
            model = %reporter.model_name,
            status = attempt.answer.as_ref().ok().map(tracing::field::display),
            error = attempt.answer.as_ref().err().map(tracing::field::display),
            duration_ms = attempt.elapsed.as_millis() as u64,
            auth_path = reporter.request_source.auth_path.as_label(),
            ingress_route = reporter.request_source.ingress_route.as_label(),
            outcome = attempt.outcome.as_label(),
            attempt = attempts,
            // Absent for a report that is sent until it is accepted.
            max_attempts = self.policy.attempt_cap(),
            retry_in_ms = delay.as_millis() as u64,
            "Usage report attempt failed, retrying"
        );
    }

    /// The report's final outcome: the series and the log lines a report has
    /// always ended with. `attempts` and `since_completion_ms` are absent
    /// under the default policy. `retry_reason` is why another attempt could
    /// have been made, had any been left.
    fn end(&self, job: &Job, attempt: &Attempt, attempts: u32, retry_reason: Option<&'static str>) {
        let reporter = &job.reporter;
        let (outcome, elapsed) = (attempt.outcome, attempt.elapsed);
        record_usage_report_outcome(reporter, outcome, Some(elapsed));
        let attempts = self.extended.then_some(attempts);
        let since_completion_ms = self
            .extended
            .then(|| job.since_completion().as_millis() as u64);
        match &attempt.answer {
            Ok(status) if outcome == UsageReportOutcome::Accepted => {
                if self.extended {
                    metrics::histogram!(
                        "inference_proxy_usage_report_time_to_accepted_seconds",
                        source_labels(reporter, None)
                    )
                    .record(job.since_completion().as_secs_f64());
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
        // Not accepted, and no attempt follows: either the answer is final,
        // or it could have passed but the attempts are used up.
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
    }

    /// Drop `job` without a (further) attempt: counted as the report's final
    /// outcome and as a drop, and logged with its ids.
    fn give_up(&self, job: &Job, why: GiveUp) {
        let reporter = &job.reporter;
        let since_completion_ms = job.since_completion().as_millis() as u64;
        match why {
            GiveUp::OutboxFull => {
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
                    max_pending = self
                        .outbox
                        .as_ref()
                        .map(|outbox| outbox.store.config().max_pending),
                    "Usage report dropped: the outbox is full and this report waited \
                     longest — usage NOT billed"
                );
            }
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
            GiveUp::Deadline {
                attempts,
                last_attempt,
            } => {
                let outcome = UsageReportOutcome::DeadlineExceeded;
                // With the duration of its last attempt, when it had one.
                record_usage_report_outcome(
                    reporter,
                    outcome,
                    last_attempt.and_then(|(_, elapsed)| elapsed),
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

    /// Wait until nothing is pending, for at most `limit`.
    pub async fn drain(&self, limit: Duration) -> Drained {
        let started_at = Instant::now();
        let give_up_at = tokio::time::Instant::now() + limit.min(Duration::from_secs(86_400));
        let (waiting, in_flight) = self.pending();
        let pending = waiting + in_flight;
        loop {
            // Registered before the state is read, so the notification of a
            // report that ends in between is not missed.
            let idle = self.idle.notified();
            tokio::pin!(idle);
            idle.as_mut().enable();
            if self.is_idle() || tokio::time::timeout_at(give_up_at, idle).await.is_err() {
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
    ///
    /// With an outbox nothing has to be delivered before the process ends:
    /// what is still in memory is written to the file, the attempts in
    /// flight get `shutdown_drain`, and the leases are given back so that the
    /// next process can send what is left at once (`close`).
    pub async fn drain_at_shutdown(&self) -> Option<Drained> {
        if let Some(outbox) = &self.outbox {
            return Some(self.close(outbox).await);
        }
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

    /// Shutdown with an outbox. Reports waiting in memory are handed to the
    /// store, the attempts in flight get `shutdown_drain` to end, and the
    /// store then writes what it was handed, gives back every lease this
    /// process holds and closes the file. `Drained` counts what this process
    /// still held itself; `left_waiting` is what stayed in memory only.
    async fn close(&self, outbox: &Outbox) -> Drained {
        let started_at = Instant::now();
        let (in_memory, unstarted, in_flight) = {
            let mut state = self.state();
            let state = &mut *state;
            let kept = state.outbox.get_or_insert_with(OutboxState::new);
            kept.stopping = true;
            // Leased and not started: given back with every other lease.
            let unstarted = std::mem::take(&mut kept.claimed);
            let in_memory = std::mem::take(&mut state.waiting);
            (in_memory, unstarted, state.in_flight)
        };
        drop(unstarted);
        // Ends the pause of every place, and keeps new ones from starting.
        outbox.stopping.send_replace(true);
        let pending = in_memory.len() + outbox.in_transit.load(Ordering::SeqCst) + in_flight;

        // A report the file cannot take goes back to waiting for a place.
        let mut refused = VecDeque::new();
        for job in in_memory {
            let model = job.reporter.model_label;
            match outbox.keep(job, usize::MAX) {
                None => self.waiting_gauge(model, Step::Down),
                Some((job, _)) => refused.push_back(job),
            }
        }
        if !refused.is_empty() {
            let mut state = self.state();
            refused.append(&mut state.waiting);
            state.waiting = refused;
        }

        let give_up_at = tokio::time::Instant::now() + self.policy.shutdown_drain;
        loop {
            // Registered before the state is read, as in `drain`.
            let ended = self.idle.notified();
            tokio::pin!(ended);
            ended.as_mut().enable();
            let none_left = {
                let state = self.state();
                state.in_flight == 0 && state.waiting.is_empty()
            };
            if none_left || tokio::time::timeout_at(give_up_at, ended).await.is_err() {
                break;
            }
        }
        let (left_waiting, left_in_flight, left_sending) = {
            let state = self.state();
            let sending = state.outbox.as_ref().map_or(0, |kept| kept.sending);
            (state.waiting.len(), state.in_flight, sending)
        };

        // The disk may be why the process is stopping: this is not waited
        // for without end either.
        let closed = tokio::time::timeout(OUTBOX_CLOSE_TIMEOUT, outbox.store.close()).await;
        let waited = started_at.elapsed();
        match closed {
            Ok(Ok(Some(left))) => info!(
                pending,
                in_outbox = left.pending_total(),
                left_in_flight,
                waited_ms = waited.as_millis() as u64,
                "Usage report outbox closed for shutdown: what it holds is sent by the next \
                 process"
            ),
            _ => warn!(
                pending,
                left_in_flight,
                waited_ms = waited.as_millis() as u64,
                "Usage report outbox could not be closed for shutdown: what was not written \
                 is lost, and the leases of this process are left to run out"
            ),
        }
        // Reports that were only ever in memory, because the file could not
        // take them, end with the process as they do without an outbox.
        let left_in_memory = left_waiting + left_in_flight.saturating_sub(left_sending);
        if left_in_memory > 0 {
            warn!(
                pending,
                left_waiting,
                left_in_flight = left_in_flight.saturating_sub(left_sending),
                waited_ms = waited.as_millis() as u64,
                "Usage reports left undelivered at shutdown — usage NOT billed"
            );
        }
        Drained {
            pending,
            left_waiting,
            left_in_flight,
            waited,
        }
    }

    /// The task that moves reports from the store to the places: it hears
    /// what the store did, starts the reports it leased as places come free
    /// and claims more. It also publishes the outbox gauges, here rather than
    /// on the store's thread so that every series of this module is written
    /// on the runtime. One per delivery, for the life of the process.
    async fn dispatch(self: Arc<Self>, mut from_store: mpsc::UnboundedReceiver<StoreEvent>) {
        let Some(outbox) = &self.outbox else {
            return;
        };
        let mut recheck = tokio::time::interval(outbox.store.config().recheck_interval);
        recheck.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let look_at = self.state().outbox.as_ref().and_then(|kept| kept.look_at);
            let retry_due = async {
                match look_at {
                    Some(at) => tokio::time::sleep_until(at.into()).await,
                    None => std::future::pending().await,
                }
            };
            let mut rechecking = false;
            tokio::select! {
                event = from_store.recv() => match event {
                    Some(event) => self.on_store_event(outbox, event),
                    None => return,
                },
                _ = outbox.wake.notified() => {}
                _ = recheck.tick() => rechecking = true,
                _ = retry_due => {}
            }
            while let Ok(event) = from_store.try_recv() {
                self.on_store_event(outbox, event);
            }
            self.advance(outbox, rechecking);
            self.publish(outbox);
            if self.is_idle() {
                self.idle.notify_waiters();
            }
        }
    }

    /// Take in what the store did.
    fn on_store_event(self: &Arc<Self>, outbox: &Outbox, event: StoreEvent) {
        match event {
            usage_outbox::Event::Committed(committed) => {
                let usage_outbox::Committed {
                    stored,
                    evicted,
                    rejected_evicted,
                    claimed,
                    next_due_ms,
                    stats,
                } = committed;
                let (now, now_ms) = (Instant::now(), usage_outbox::now_ms());
                let claimed = claimed.map(|rows| {
                    rows.into_iter()
                        .map(|row| self.job_from(outbox, row, now, now_ms))
                        .collect::<Vec<_>>()
                });
                let pending = stats
                    .pending
                    .into_iter()
                    .map(|(label, rows)| (outbox.label(label), rows))
                    .collect();
                let (back, in_memory) = {
                    let mut state = self.state();
                    let state = &mut *state;
                    let kept = state.outbox.get_or_insert_with(OutboxState::new);
                    let back = !kept.available;
                    kept.available = true;
                    kept.synced = true;
                    kept.stored = pending;
                    kept.oldest_completed_at_ms = stats.oldest_completed_at_ms;
                    kept.rejected = stats.rejected;
                    if stored > 0 {
                        kept.look = true;
                    }
                    if let Some(claimed) = claimed {
                        // As many as asked for: there may be more.
                        let asked = kept.asked.take().unwrap_or(0);
                        kept.look |= claimed.len() >= asked;
                        if let Some(due_ms) = next_due_ms {
                            let wait = u64::try_from(due_ms.saturating_sub(now_ms)).unwrap_or(0);
                            let due_at = now + Duration::from_millis(wait);
                            kept.look_at = Some(kept.look_at.map_or(due_at, |at| at.min(due_at)));
                        }
                        if kept.stopping {
                            // Too late to start them; `close` gives every
                            // lease back.
                            drop(claimed);
                        } else {
                            kept.claimed.extend(claimed);
                        }
                    }
                    // The file is usable again: what waited in memory in the
                    // meantime is written to it after all.
                    let in_memory = if back && !kept.stopping {
                        std::mem::take(&mut state.waiting)
                    } else {
                        VecDeque::new()
                    };
                    (back, in_memory)
                };
                outbox.in_transit.fetch_sub(stored, Ordering::SeqCst);
                if back {
                    info!(
                        path = %outbox.store.config().path.display(),
                        moved_from_memory = in_memory.len(),
                        "Usage report outbox available again"
                    );
                }
                for job in in_memory {
                    self.waiting_gauge(job.reporter.model_label, Step::Down);
                    if let Some((job, _)) = outbox.keep(job, usize::MAX) {
                        self.hold(job);
                    }
                }
                for row in evicted {
                    let job = self.job_from(outbox, row, now, now_ms);
                    self.give_up(&job, GiveUp::OutboxFull);
                }
                if rejected_evicted > 0 {
                    metrics::counter!("inference_proxy_usage_report_outbox_rejected_evicted_total")
                        .increment(rejected_evicted as u64);
                }
            }
            usage_outbox::Event::Failed {
                error,
                reports,
                claim,
            } => {
                let was_available = {
                    let mut state = self.state();
                    let kept = state.outbox.get_or_insert_with(OutboxState::new);
                    kept.synced = true;
                    if claim {
                        kept.asked = None;
                        kept.look = true;
                    }
                    std::mem::replace(&mut kept.available, false)
                };
                if let Some((op, error)) = error {
                    metrics::counter!("inference_proxy_usage_report_outbox_errors_total", "op" => op)
                        .increment(1);
                    let path = outbox.store.config().path.display();
                    if was_available {
                        error!(
                            path = %path,
                            op,
                            error = %error,
                            unstored = reports.len(),
                            "Usage report outbox unavailable: reports are delivered from memory \
                             and not kept across a restart until it is back"
                        );
                    } else {
                        // Said once; the counter has every later try.
                        debug!(path = %path, op, error = %error, "Usage report outbox still unavailable");
                    }
                }
                outbox.in_transit.fetch_sub(reports.len(), Ordering::SeqCst);
                for job in reports {
                    self.count(
                        "inference_proxy_usage_report_outbox_bypassed_total",
                        &job.reporter,
                        ("reason", "write_failed"),
                    );
                    self.hold(job);
                }
            }
            usage_outbox::Event::Warning { op, error } => {
                metrics::counter!("inference_proxy_usage_report_outbox_errors_total", "op" => op)
                    .increment(1);
                warn!(op, error = %error, "Usage report outbox: a maintenance step failed");
            }
        }
    }

    /// Start leased reports in the places that are free, and claim more when
    /// there is room for them. `rechecking`: the file is looked at whether
    /// or not anything here says it changed.
    fn advance(self: &Arc<Self>, outbox: &Outbox, rechecking: bool) {
        let cap = self.policy.max_in_flight.max(1);
        let (start, ask) = {
            let mut state = self.state();
            let state = &mut *state;
            let Some(kept) = &mut state.outbox else {
                return;
            };
            if kept.stopping {
                return;
            }
            let mut start = Vec::new();
            while state.in_flight < cap {
                let Some(job) = kept.start_next() else {
                    break;
                };
                state.in_flight += 1;
                start.push(job);
            }
            if kept.look_at.is_some_and(|at| at <= Instant::now()) {
                kept.look_at = None;
                kept.look = true;
            }
            kept.look |= rechecking;
            // One round of reports is leased ahead, so that a place that
            // comes free finds its next report here. Not while attempts are
            // failing: a leased report nobody starts is a lease running out.
            let free = cap.saturating_sub(state.in_flight);
            let ahead = if kept.failing_streak == 0 { cap } else { 0 };
            let want = (free + ahead).saturating_sub(kept.claimed.len());
            let ask =
                (want > 0 && kept.look && kept.asked.is_none() && kept.available).then_some(want);
            if ask.is_some() {
                kept.asked = ask;
                kept.look = false;
            }
            (start, ask)
        };
        for job in start {
            let model = job.reporter.model_label;
            let place = Place {
                delivery: Arc::clone(self),
                model,
                held: true,
            };
            self.in_flight_gauge(model, Step::Up);
            tokio::spawn(place.work(job));
        }
        match ask {
            Some(want) => outbox.store.claim(want, outbox.lease),
            // Fresh numbers, and the way an unavailable store is tried again.
            None if rechecking => outbox.store.sync(),
            None => {}
        }
    }

    /// The outbox gauges, and `queue_depth`, which with an outbox is set
    /// from the rows of the file rather than counted up and down: the file
    /// is shared, and a row another process sent never passed through here.
    fn publish(&self, outbox: &Outbox) {
        let (available, stored, mut waiting, oldest_completed_at_ms, rejected) = {
            let state = self.state();
            let Some(kept) = &state.outbox else {
                return;
            };
            (
                kept.available && outbox.store.is_available(),
                kept.stored.clone(),
                kept.beside.clone(),
                kept.oldest_completed_at_ms,
                kept.rejected,
            )
        };
        metrics::gauge!("inference_proxy_usage_report_outbox_available").set(if available {
            1.0
        } else {
            0.0
        });
        for (model, rows) in stored {
            model_gauge!(model, "inference_proxy_usage_report_outbox_pending").set(rows as f64);
            *waiting.entry(model).or_default() += i64::try_from(rows).unwrap_or(i64::MAX);
        }
        for (model, waiting) in waiting {
            model_gauge!(model, "inference_proxy_usage_report_queue_depth")
                .set(waiting.max(0) as f64);
        }
        let oldest_age_ms =
            oldest_completed_at_ms.map_or(0, |at| usage_outbox::now_ms().saturating_sub(at).max(0));
        metrics::gauge!("inference_proxy_usage_report_outbox_oldest_pending_age_seconds")
            .set(oldest_age_ms as f64 / 1000.0);
        metrics::gauge!("inference_proxy_usage_report_outbox_rejected").set(rejected as f64);
    }

    /// A row of the outbox as a report to send. Who it is for is read from
    /// the report itself; where it goes and the bearer are the running
    /// process's.
    fn job_from(
        self: &Arc<Self>,
        outbox: &Outbox,
        row: usage_outbox::Row,
        now: Instant,
        now_ms: i64,
    ) -> Job {
        #[derive(Default, serde::Deserialize)]
        struct Subject {
            organization_id: Option<String>,
            workspace_id: Option<String>,
            api_key_id: Option<String>,
            model: Option<String>,
        }
        let subject: Subject = serde_json::from_str(&row.body).unwrap_or_default();
        let age = Duration::from_millis(
            u64::try_from(now_ms.saturating_sub(row.completed_at_ms)).unwrap_or(0),
        );
        Job {
            reporter: UsageReporter {
                http_client: outbox.endpoint.client.clone(),
                cloud_api_url: outbox.endpoint.cloud_api_url.clone(),
                model_name: subject.model.unwrap_or_default(),
                // The bearer is on the job; the reporter of a stored report
                // is only read for ids and labels.
                cloud_api_usage_token: None,
                org_id: subject.organization_id,
                workspace_id: subject.workspace_id,
                api_key_id: subject.api_key_id,
                // Already in the report, when the request had one.
                discount_to_user: None,
                model_label: outbox.label(row.model_label),
                request_id: row.request_id,
                request_source: RequestSource {
                    auth_path: auth_path_from(&row.auth_path),
                    ingress_route: ingress_route_from(&row.ingress_route),
                },
                delivery: Arc::clone(self),
            },
            url: outbox.endpoint.url.clone(),
            authorization: outbox.endpoint.authorization.clone(),
            body: bytes::Bytes::from(row.body),
            completed_at: now,
            stored: Some(Stored {
                id: row.id,
                attempts: row.attempts,
                last_failure: row.last_outcome.as_deref().and_then(passing_failure),
                age,
                lease_until: now + outbox.lease,
            }),
        }
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
    ///
    /// With an outbox the gauge is not stepped: it is set from the rows of
    /// the file (`publish`), and what happens here is kept as the difference
    /// to that count.
    fn waiting_gauge(&self, model: ModelLabel, step: Step) {
        if self.outbox.is_some() {
            if let Some(kept) = &mut self.state().outbox {
                *kept.beside.entry(model).or_default() += match step {
                    Step::Up => 1,
                    Step::Down => -1,
                    Step::None => 0,
                };
            }
            return;
        }
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
            match (&self.delivery.outbox, &job.stored) {
                (Some(outbox), Some(row)) => {
                    let failed = {
                        let _sending = Sending {
                            delivery: &self.delivery,
                            model: self.model,
                        };
                        self.delivery.deliver_stored(outbox, &job, row).await
                    };
                    if failed {
                        self.delivery.pause(outbox).await;
                    }
                }
                _ => self.delivery.deliver(&job).await,
            }
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
    ///
    /// With an outbox, a report waiting in memory goes first (the file could
    /// not take it, so it is not kept anywhere else), then the reports
    /// leased from the file.
    fn next(&mut self) -> Option<Job> {
        let (next, idle) = {
            let mut state = self.delivery.state();
            let state = &mut *state;
            let mut next = state.waiting.pop_front();
            if let (None, Some(kept)) = (&next, &mut state.outbox) {
                if !kept.stopping {
                    next = kept.start_next();
                }
            }
            if next.is_none() {
                state.in_flight -= 1;
                self.held = false;
            }
            // With an outbox, shutdown waits for the places alone.
            let idle = state.is_idle() || (state.outbox.is_some() && state.in_flight == 0);
            (next, idle)
        };
        self.delivery.in_flight_gauge(self.model, Step::Down);
        if let Some(next) = &next {
            self.model = next.reporter.model_label;
            // A report from the file was never counted as waiting in memory
            // (`OutboxState::start_next` did its accounting).
            if next.stored.is_none() {
                self.delivery.waiting_gauge(self.model, Step::Down);
            }
            self.delivery.in_flight_gauge(self.model, Step::Up);
        }
        if idle {
            self.delivery.idle.notify_waiters();
        }
        if let Some(outbox) = &self.delivery.outbox {
            outbox.wake.notify_one();
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
                state.is_idle() || (state.outbox.is_some() && state.in_flight == 0)
            };
            self.delivery.in_flight_gauge(self.model, Step::Down);
            if idle {
                self.delivery.idle.notify_waiters();
            }
            if let Some(outbox) = &self.delivery.outbox {
                outbox.wake.notify_one();
            }
        }
    }
}

/// A report from the outbox while a place is sending it: what
/// `OutboxState::start_next` began ends when this is dropped, with the
/// attempt, however that ends.
struct Sending<'a> {
    delivery: &'a UsageReportDelivery,
    model: ModelLabel,
}

impl Drop for Sending<'_> {
    fn drop(&mut self) {
        if let Some(kept) = &mut self.delivery.state().outbox {
            kept.sent(self.model);
        }
    }
}

/// What the store of a delivery reports back.
type StoreEvent = usage_outbox::Event<Job>;

/// The failures an attempt may be made again after: the `reason` of
/// `inference_proxy_usage_report_retries_total`, which is also what a row of
/// the outbox keeps as its `last_outcome`, and the outcome each stands for.
const PASSING_FAILURES: [(&str, UsageReportOutcome); 5] = [
    ("timeout", UsageReportOutcome::Timeout),
    ("connect_error", UsageReportOutcome::ConnectError),
    ("transport_error", UsageReportOutcome::TransportError),
    ("http_5xx", UsageReportOutcome::Http5xx),
    ("http_429", UsageReportOutcome::Http4xx),
];

/// The entry of `PASSING_FAILURES` a row's `last_outcome` names.
fn passing_failure(label: &str) -> Option<(&'static str, UsageReportOutcome)> {
    PASSING_FAILURES
        .into_iter()
        .find(|(reason, _)| *reason == label)
}

/// The `auth_path` a row was written with. Only requests made with a cloud
/// API key are reported, so that is what an unknown label is read as.
fn auth_path_from(label: &str) -> AuthPath {
    [AuthPath::TrustedConfigToken, AuthPath::CloudApiKey]
        .into_iter()
        .find(|path| path.as_label() == label)
        .unwrap_or(AuthPath::CloudApiKey)
}

/// The `ingress_route` a row was written with; `other` for an unknown label.
fn ingress_route_from(label: &str) -> IngressRouteKind {
    [
        IngressRouteKind::Canonical,
        IngressRouteKind::Indexed,
        IngressRouteKind::Long,
        IngressRouteKind::LongIndexed,
        IngressRouteKind::Other,
        IngressRouteKind::Missing,
    ]
    .into_iter()
    .find(|route| route.as_label() == label)
    .unwrap_or(IngressRouteKind::Other)
}

/// Why an attempt with this result may be made again, `None` when the result
/// is final: a 4xx other than 429 means the billing API refused the report
/// itself, and sending the same bytes again would get the same answer. Every
/// reason given here is in `PASSING_FAILURES`.
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

#[cfg(test)]
#[path = "usage_report_outbox_tests.rs"]
mod outbox_tests;
