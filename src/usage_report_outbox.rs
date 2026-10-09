//! Delivery of usage reports through the outbox on disk
//! (`VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH`): what a process with an outbox
//! does with a report after `submit`. `usage_outbox.rs` is the file; this is
//! who is sent when.
//!
//! A report is handed to the store, which writes it, and is sent from the
//! file: it stays there until the billing API accepts it or refuses it for
//! good, across restarts, an outage of any length and a second process on
//! the same file. When the file cannot take a report (it is unavailable, or
//! full, or its writer does not come back), the report is held in memory and
//! sent from there by the same rules, and written to the file as soon as the
//! file takes it.
//!
//! The rules, for a report in the file and for one in memory alike:
//!
//! - One attempt at a time, in one of `max_in_flight` places. A place is
//!   taken for the attempt and for nothing else: a report that waits for its
//!   next attempt holds none.
//! - A failure that can pass (timeout, connection or transport error, 5xx,
//!   429, 401) makes the report due again after a backoff that doubles up to
//!   `max_backoff`. Any other answer is final: the report moves to the
//!   `rejected` table. So does a report older than `max_age`: nothing is
//!   tried for ever, and nothing is destroyed for its age.
//! - A report that failed never stands in front of one that was never tried.
//!   Retries get `retry_places` of the places and no more; with one place,
//!   one attempt in `RETRY_EVERY`.
//! - When first attempt after first attempt fails and none gets an answer of
//!   its own, the billing API is not answering: from then on one report at a
//!   time probes it, the newest first, at a growing interval (`Breaker`), and
//!   the first report it answers ends that at once. Some reports failing
//!   while others are accepted never starts it.
//!
//! Everything here runs on the runtime: the dispatcher task, one task per
//! attempt, and the metrics and log lines of both. The store's thread writes
//! neither.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use super::{retry_reason, Attempt, Drained, Job, Step, UsageReportDelivery, UsageReportPolicy};
use crate::auth::{AuthPath, IngressRouteKind, RequestSource};
use crate::model_metrics::{model_gauge, ModelLabel};
use crate::proxy::{record_usage_report_outcome, UsageReportOutcome, UsageReporter};
use crate::usage_outbox::{
    Claim, Committed, Event, Health, Outcome, Persist, Rejection, Row, Settle, Store,
    UsageOutboxConfig,
};

/// Model labels read back from the outbox are kept for the life of the
/// process, like those of a model list. There are as many as models were ever
/// served from the file; a file that claims more gets no label for the rest.
const MAX_STORED_MODEL_LABELS: usize = 256;

/// How many reports are leased ahead of the places, in rounds of
/// `max_in_flight`: a place that comes free finds its next report here, and
/// a transaction of the store moves that many at once.
const ROUNDS_AHEAD: usize = 3;

/// With a single place, how many attempts apart a retry may be while reports
/// never tried are waiting.
const RETRY_EVERY: u32 = 4;

/// What `last_outcome` says of a report that grew too old without ever
/// having been sent.
const NEVER_SENT: &str = "never_sent";

/// Why a row is in `rejected`: the billing API refused the report for good,
/// or the report grew older than `MAX_AGE_SECS`, or than `DEADLINE_SECS`,
/// allow.
const REJECTED_REASONS: [&str; 3] = ["rejected", "max_age", "deadline"];

/// The failures an attempt may be made again after: the `reason` of
/// `inference_proxy_usage_report_retries_total`, which is also what a row of
/// the outbox keeps as its `last_outcome`, and the outcome each stands for.
const PASSING_FAILURES: [(&str, UsageReportOutcome); 6] = [
    ("timeout", UsageReportOutcome::Timeout),
    ("connect_error", UsageReportOutcome::ConnectError),
    ("transport_error", UsageReportOutcome::TransportError),
    ("http_5xx", UsageReportOutcome::Http5xx),
    ("http_429", UsageReportOutcome::Http4xx),
    // With an outbox only (`caller_refused`).
    ("http_401", UsageReportOutcome::Http4xx),
];

/// A failure an attempt may be made again after.
type Passing = (&'static str, UsageReportOutcome);

/// A report as the outbox schedules it. Cheap to clone: the report itself is
/// shared.
#[derive(Clone)]
pub(super) struct Item {
    job: Arc<Job>,
    /// Attempts made so far, by any process.
    attempts: u32,
    /// Why the last of them failed, when there was one.
    last_failure: Option<Passing>,
    /// Counted in `outbox_bypassed_total` already: a report is counted once,
    /// however often the file is offered it and does not take it.
    bypassed: bool,
    home: Home,
}

/// Where a report is kept.
#[derive(Clone)]
enum Home {
    /// In the file, leased to this process until `lease_until`.
    File {
        id: i64,
        generation: u64,
        lease_until: Instant,
    },
    /// In memory, because the file could not take it. `ended`: it will not
    /// be sent again and waits for the file to keep it in `rejected`.
    Memory {
        due_at: Instant,
        ended: Option<Rejection>,
    },
}

impl Item {
    fn model(&self) -> ModelLabel {
        self.job.reporter.model_label
    }

    fn in_memory(job: Arc<Job>, attempts: u32, last_failure: Option<Passing>) -> Self {
        Self {
            job,
            attempts,
            last_failure,
            bypassed: false,
            home: Home::Memory {
                due_at: Instant::now(),
                ended: None,
            },
        }
    }

    /// The same report, held in memory from here on: it was a row of a
    /// database that is no longer the one at the path.
    fn out_of_the_file(self) -> Self {
        Self {
            bypassed: self.bypassed,
            ..Self::in_memory(self.job, self.attempts, self.last_failure)
        }
    }
}

impl Persist for Item {
    fn row(&self) -> crate::usage_outbox::NewRow<'_> {
        let reporter = &self.job.reporter;
        let (due_in, rejected) = match &self.home {
            Home::Memory { due_at, ended } => {
                (due_at.saturating_duration_since(Instant::now()), *ended)
            }
            Home::File { .. } => (Duration::ZERO, None),
        };
        crate::usage_outbox::NewRow {
            body: &self.job.body,
            request_id: reporter.request_id.as_deref(),
            model_label: reporter.model_label,
            auth_path: reporter.request_source.auth_path.as_label(),
            ingress_route: reporter.request_source.ingress_route.as_label(),
            age: self.job.since_completion(),
            attempts: self.attempts,
            last_outcome: self.last_failure.map(|(reason, _)| reason),
            due_in,
            rejected,
        }
    }
}

/// Never tried, or tried before: the two are scheduled apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Fresh,
    Retry,
}

/// How an attempt at a report ended.
enum Verdict {
    Accepted(Attempt),
    /// An answer that will not change: the report goes to `rejected`.
    Refused(Attempt),
    /// A failure that can pass: the report is due again later.
    Failed(Attempt, Passing),
    /// Not sent: it is older than a report may grow. To `rejected`, with
    /// this reason.
    TooOld(&'static str),
    /// Not sent: its lease would have run out under the attempt.
    NotSent,
}

/// Why a report was given up on without an answer of its own.
#[derive(Clone, Copy)]
enum GaveUp {
    /// Removed from a file at its bound, where it had waited longest.
    FileFull,
    /// Pushed out of what waits in memory; it is kept nowhere.
    MemoryFull,
    /// Older than a report may grow: in `rejected` now.
    TooOld(&'static str),
    /// Still in memory when the process ended, and the file did not take it.
    Shutdown,
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

/// The billing API is not answering: what tells, and what is done about it.
///
/// What tells is reports failing their first attempt, in a way that can
/// pass, `after` times in a row with no report accepted or refused in
/// between. Reports that failed before say less: they fail again whether or
/// not anything is wrong with the billing API, and there can be many of them
/// while most reports are accepted. So failures that are not all first
/// attempts count only once nothing has been answered for `window` as well,
/// which with any report being accepted never happens. A 401 tells at once:
/// it is about this process, and so about every report.
///
/// While it is engaged one report at a time is sent, `pause` apart, the
/// pause doubling up to `max_pause`. The report is the newest there is: if
/// what failed so far were reports the billing API will not take while it
/// takes others, the newest is the one least like them, and a report still
/// never waits longer than a pause for its first attempt. Any answer about a
/// report ends it at once.
///
/// One report handed over after it engaged does not wait for the pause: when
/// only old reports have been failing, a new one is the best probe there is,
/// and if it is accepted the billing API was never down. One, so that an
/// outage is not probed at the rate requests complete; and one again after
/// every pause in which none was handed over, because the first to come
/// after a quiet spell is news once more.
#[derive(Default)]
struct Breaker {
    /// Failures that can pass since the billing API last answered a report,
    /// how many of them were first attempts, and when the first one was.
    failures: u32,
    fresh_failures: u32,
    since: Option<Instant>,
    engaged: bool,
    /// Probes that failed since it engaged.
    probes_failed: u32,
    /// Not before this is the next probe sent.
    probe_at: Option<Instant>,
    /// The one report that may go ahead of the pause has gone.
    newcomer_tried: bool,
}

/// A report held in memory.
struct Held {
    item: Item,
    /// A place is sending it right now.
    sending: bool,
}

/// What a process with an outbox keeps in memory about it.
struct Kept {
    /// Reports leased from the file and not started, longest waiting first.
    fresh: VecDeque<Item>,
    retries: VecDeque<Item>,
    /// Reports the file does not hold, by the order they came in.
    memory: BTreeMap<u64, Held>,
    next_seq: u64,
    /// Of those, the ones never tried and the ones due again, by when.
    memory_fresh: VecDeque<u64>,
    memory_due: BTreeSet<(Instant, u64)>,
    /// Reports pushed out of `memory`, whose line is still to be written.
    dropped: Vec<(Item, GaveUp)>,
    /// Places taken, by what is in them.
    sending_fresh: usize,
    sending_retries: usize,
    /// Rows of the file among them.
    sending_from_file: usize,
    /// First attempts started since the last retry (for a single place).
    fresh_since_retry: u32,
    breaker: Breaker,
    /// A claim is with the store and has not been answered, and whether it
    /// asks for a probe (the newest report) or for the places to come.
    asked: bool,
    asked_for_a_probe: bool,
    /// The file may hold a report of that kind that is due: worth a claim.
    look_fresh: bool,
    look_retries: bool,
    /// When a report of the file that failed is due again.
    retry_at: Option<Instant>,
    /// What the store last said of the file.
    health: Health,
    /// The file counts as full: from the store saying so until it has room
    /// again and everything memory held meanwhile is written. (While a
    /// backlog goes out of a file at its size, room comes and is used up
    /// many times a minute; that is one state, not many.)
    full: bool,
    /// The store has said what the file holds.
    synced: bool,
    /// The writer has not come back from a transaction: until it does the
    /// file counts as unavailable.
    stalled: bool,
    /// The database the leased rows belong to.
    generation: u64,
    /// Rows of the file per model, waiting or being sent by anyone.
    stored: Vec<(ModelLabel, u64)>,
    /// When the request of the oldest of them completed.
    oldest_completed_at_ms: Option<i64>,
    rejected: Vec<(String, u64)>,
    used_bytes: u64,
    /// The file may grow to less than `max_bytes`: its volume has no room
    /// for that.
    limited_by_volume: bool,
    /// Per model, what `queue_depth` is beside the rows of the file: the
    /// reports waiting in memory, less the rows this process is sending.
    beside: HashMap<ModelLabel, i64>,
    /// Shutdown has begun: no report is claimed any more, and none is
    /// started until the file is closed.
    stopping: bool,
    /// The file is closed. A report handed over after that is not swallowed:
    /// while the process lasts it is sent from memory.
    closed: bool,
}

impl Kept {
    fn stored_total(&self) -> u64 {
        self.stored.iter().map(|(_, rows)| rows).sum()
    }

    fn in_flight(&self) -> usize {
        self.sending_fresh + self.sending_retries
    }

    /// Reports in memory that are still to be sent and that no place has.
    fn waiting_in_memory(&self) -> usize {
        self.memory_fresh.len() + self.memory_due.len()
    }

    fn beside(&mut self, model: ModelLabel, by: i64) {
        *self.beside.entry(model).or_default() += by;
    }

    /// Hold `item` in memory, where the dispatcher finds it. Past
    /// `max_held` the one that came in first is pushed out.
    fn hold(&mut self, item: Item, max_held: usize) {
        let Home::Memory { due_at, ended } = &item.home else {
            return;
        };
        let (seq, model) = (self.next_seq, item.model());
        self.next_seq += 1;
        if ended.is_none() {
            if item.attempts == 0 {
                self.memory_fresh.push_back(seq);
            } else {
                self.memory_due.insert((*due_at, seq));
            }
            self.beside(model, 1);
        }
        self.memory.insert(
            seq,
            Held {
                item,
                sending: false,
            },
        );
        while self.memory.len() > max_held {
            let oldest = self
                .memory
                .iter()
                .find(|(_, held)| !held.sending)
                .map(|(seq, _)| *seq);
            let Some(held) = oldest.and_then(|seq| self.release(seq)) else {
                break;
            };
            self.dropped.push((held.item, GaveUp::MemoryFull));
        }
    }

    /// Take the report `seq` out of memory, wherever it waits there.
    fn release(&mut self, seq: u64) -> Option<Held> {
        let held = self.memory.remove(&seq)?;
        if let Home::Memory { due_at, ended } = &held.item.home {
            if !held.sending && ended.is_none() {
                if held.item.attempts == 0 {
                    self.memory_fresh.retain(|waiting| *waiting != seq);
                } else {
                    self.memory_due.remove(&(*due_at, seq));
                }
                self.beside(held.item.model(), -1);
            }
        }
        Some(held)
    }

    /// The next report of `class` to send, taken from where it waits: one
    /// held in memory first, which is kept nowhere else, then one leased
    /// from the file. `newest`: of those never tried, the one that came in
    /// last (a probe), not the one that waited longest.
    fn next(&mut self, class: Class, now: Instant, newest: bool) -> Option<(Item, Option<u64>)> {
        let seq = match class {
            Class::Fresh if newest => self.memory_fresh.pop_back(),
            Class::Fresh => self.memory_fresh.pop_front(),
            Class::Retry => match self.memory_due.first().copied() {
                Some((due_at, seq)) if due_at <= now => {
                    self.memory_due.remove(&(due_at, seq));
                    Some(seq)
                }
                _ => None,
            },
        };
        if let Some(held) = seq.and_then(|seq| self.memory.get_mut(&seq)) {
            held.sending = true;
            let item = held.item.clone();
            self.beside(item.model(), -1);
            return Some((item, seq));
        }
        let item = match class {
            Class::Fresh if newest => self.fresh.pop_back(),
            Class::Fresh => self.fresh.pop_front(),
            Class::Retry => self.retries.pop_front(),
        }?;
        self.sending_from_file += 1;
        self.beside(item.model(), -1);
        Some((item, None))
    }

    /// Whether a report of `class` is at hand.
    fn ready(&self, class: Class, now: Instant) -> bool {
        match class {
            Class::Fresh => !self.memory_fresh.is_empty() || !self.fresh.is_empty(),
            Class::Retry => {
                !self.retries.is_empty()
                    || self
                        .memory_due
                        .first()
                        .is_some_and(|(due_at, _)| *due_at <= now)
            }
        }
    }
}

/// The outbox of a delivery: the store, where its reports are sent, and what
/// the dispatcher knows.
pub(super) struct Outbox {
    store: Store<Item>,
    endpoint: Endpoint,
    /// How long a claimed report is this process's: two attempt timeouts
    /// (waiting for a place, then the attempt) and a margin.
    lease: Duration,
    /// What must be left of a lease, beyond the attempt timeout, for the
    /// attempt to start.
    lease_slack: Duration,
    /// The places reports that failed before may take.
    retry_places: usize,
    kept: Mutex<Kept>,
    /// Reports handed to the store whose transaction has not been heard of.
    in_transit: AtomicUsize,
    /// `Breaker::engaged`, and whether a report was handed over since: read
    /// and written by `submit`, which takes no lock for it.
    paused: AtomicBool,
    newcomer: AtomicBool,
    /// Wakes the dispatcher: a place is free, or a report came in.
    wake: tokio::sync::Notify,
    /// The model labels read back from the file (`MAX_STORED_MODEL_LABELS`).
    labels: Mutex<HashSet<&'static str>>,
}

impl Outbox {
    /// The outbox at `config.path`, its file being opened by the store's
    /// thread, and what that thread will have to say.
    pub(super) fn start(
        policy: &UsageReportPolicy,
        config: UsageOutboxConfig,
        client: reqwest::Client,
        cloud_api_url: &str,
        usage_token: &str,
    ) -> (Self, mpsc::UnboundedReceiver<Event<Item>>) {
        let lease = policy.attempt_timeout * 2 + config.lease_margin;
        let max_age = max_age(policy, &config);
        info!(
            path = %config.path.display(),
            max_pending = config.max_pending,
            max_rejected = config.max_rejected,
            max_bytes = config.max_bytes,
            max_age_secs = max_age.0.as_secs(),
            max_in_flight = policy.max_in_flight,
            lease_secs = lease.as_secs(),
            "Usage report outbox enabled: a report is kept on disk until the billing API \
             accepts it or refuses it for good"
        );
        warn_about_the_directory(&config);
        let (events, from_store) = mpsc::unbounded_channel();
        let outbox = Self {
            lease,
            lease_slack: config.lease_margin / 4,
            retry_places: (policy.max_in_flight / 4).max(1),
            kept: Mutex::new(Kept {
                fresh: VecDeque::new(),
                retries: VecDeque::new(),
                memory: BTreeMap::new(),
                next_seq: 0,
                memory_fresh: VecDeque::new(),
                memory_due: BTreeSet::new(),
                dropped: Vec::new(),
                sending_fresh: 0,
                sending_retries: 0,
                sending_from_file: 0,
                fresh_since_retry: 0,
                breaker: Breaker::default(),
                asked: false,
                asked_for_a_probe: false,
                look_fresh: true,
                look_retries: true,
                retry_at: None,
                // Until the store says otherwise.
                health: Health {
                    available: true,
                    full: false,
                },
                full: false,
                synced: false,
                stalled: false,
                generation: 0,
                stored: Vec::new(),
                oldest_completed_at_ms: None,
                rejected: Vec::new(),
                used_bytes: 0,
                limited_by_volume: false,
                beside: HashMap::new(),
                stopping: false,
                closed: false,
            }),
            store: Store::start(config, events),
            endpoint: Endpoint {
                client,
                cloud_api_url: cloud_api_url.to_string(),
                url: format!("{cloud_api_url}/v1/internal/usage"),
                authorization: format!("Bearer {usage_token}"),
            },
            in_transit: AtomicUsize::new(0),
            paused: AtomicBool::new(false),
            newcomer: AtomicBool::new(false),
            wake: tokio::sync::Notify::new(),
            labels: Mutex::default(),
        };
        (outbox, from_store)
    }

    fn kept(&self) -> MutexGuard<'_, Kept> {
        // Nothing panics while holding the lock; if something ever does, what
        // is kept is still the best there is.
        self.kept.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn config(&self) -> &UsageOutboxConfig {
        self.store.config()
    }

    /// Take a report: a lock and a push into the store's inbox, or, when the
    /// store is not taking any, a lock and a push into what is held here.
    /// Never the file, never a wait.
    pub(super) fn submit(&self, delivery: &UsageReportDelivery, job: Job) {
        // While the billing API is not answering, a report that comes in is
        // news: the dispatcher is told at once, not when it has been written.
        let news = self.paused.load(Ordering::Relaxed);
        if news {
            self.newcomer.store(true, Ordering::Relaxed);
        }
        let item = Item::in_memory(Arc::new(job), 0, None);
        if let Some((item, why)) = self.keep(item, delivery.policy.max_queued) {
            // The file cannot take it now. It is delivered all the same,
            // from memory, and written to the file when the file takes it.
            let item = delivery.bypassed(item, why);
            self.kept().hold(item, delivery.policy.max_queued);
            self.wake.notify_one();
        } else if news {
            self.wake.notify_one();
        }
    }

    /// Hand `item` to the store. `None` when the store took it; otherwise
    /// the item back, and why.
    fn keep(&self, item: Item, max_buffered: usize) -> Option<(Item, &'static str)> {
        // Counted before the store has it, so that it is never out of sight.
        self.in_transit.fetch_add(1, Ordering::SeqCst);
        let refused = self.store.offer(item, max_buffered).err();
        if refused.is_some() {
            self.in_transit.fetch_sub(1, Ordering::SeqCst);
        }
        refused.map(|(item, why)| (item, why.as_label()))
    }

    /// Reports waiting, and reports being sent.
    pub(super) fn pending(&self) -> (usize, usize) {
        let kept = self.kept();
        let stored = usize::try_from(kept.stored_total()).unwrap_or(usize::MAX);
        let waiting = stored
            .saturating_sub(kept.sending_from_file)
            .saturating_add(kept.waiting_in_memory())
            .saturating_add(self.in_transit.load(Ordering::SeqCst));
        (waiting, kept.in_flight())
    }

    /// The file holds nothing, nothing is on its way into it, nothing waits
    /// in memory and no place is taken.
    pub(super) fn is_idle(&self) -> bool {
        let kept = self.kept();
        kept.synced
            && kept.stored_total() == 0
            && kept.fresh.is_empty()
            && kept.retries.is_empty()
            && kept.waiting_in_memory() == 0
            && kept.in_flight() == 0
            && kept.dropped.is_empty()
            && self.in_transit.load(Ordering::SeqCst) == 0
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

    /// The backoff of a report after `attempts` failed attempts: half of
    /// `initial_backoff * 2^(attempts - 1)`, at most `max_backoff`, plus a
    /// random share of the other half. The ceiling is minutes, not the 30
    /// seconds of a report that waits in a place: a report that keeps
    /// failing then costs an attempt every few minutes and nothing else.
    fn backoff(&self, policy: &UsageReportPolicy, attempts: u32) -> Duration {
        doubling(policy.initial_backoff, attempts, self.config().max_backoff)
    }

    /// The pause before probe number `probes + 1` of a billing API that is
    /// not answering: the same doubling, up to `max_pause`.
    fn pause(&self, policy: &UsageReportPolicy, probes: u32) -> Duration {
        doubling(policy.initial_backoff, probes, self.config().max_pause)
    }
}

/// Half of `initial * 2^(times - 1)`, at most `ceiling`, plus a random share
/// of the other half. The fixed half keeps what follows from being
/// immediate, the random half keeps what failed together from coming back
/// together.
fn doubling(initial: Duration, times: u32, ceiling: Duration) -> Duration {
    let millis = |duration: Duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
    let exponent = times.saturating_sub(1).min(30);
    let ceiling_ms = millis(initial)
        .max(1)
        .saturating_mul(1 << exponent)
        .min(millis(ceiling));
    let fixed_ms = ceiling_ms.div_ceil(2);
    Duration::from_millis(fixed_ms + rand::random_range(0..=ceiling_ms - fixed_ms))
}

/// How old a report may grow before it is moved to `rejected`, and which
/// setting says so. With `DEADLINE_SECS` an attempt only starts while its
/// whole timeout fits before the deadline, as without an outbox.
fn max_age(policy: &UsageReportPolicy, config: &UsageOutboxConfig) -> (Duration, &'static str) {
    match policy.deadline {
        Some(deadline) if deadline.saturating_sub(policy.attempt_timeout) < config.max_age => {
            (deadline.saturating_sub(policy.attempt_timeout), "deadline")
        }
        _ => (config.max_age, "max_age"),
    }
}

/// Whoever can write to the directory can replace the file, and a row of the
/// file is sent with the usage token: say so when others can write to it.
fn warn_about_the_directory(config: &UsageOutboxConfig) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let Some(directory) = config
            .path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        else {
            return;
        };
        if let Ok(metadata) = std::fs::metadata(directory) {
            let mode = metadata.permissions().mode() & 0o777;
            if mode & 0o022 != 0 {
                warn!(
                    directory = %directory.display(),
                    mode = format!("{mode:o}"),
                    "The directory of the usage report outbox can be written to by other \
                     users: whoever can put a row into the file has it sent with the usage \
                     token. Make it 0700"
                );
            }
        }
    }
    #[cfg(not(unix))]
    let _ = config;
}

/// A place while an attempt holds it. Dropped with the attempt, however that
/// ends: the place is free again, and a report of the file that was not
/// settled (the task was cancelled, or panicked) gives its lease back.
struct Place {
    delivery: Arc<UsageReportDelivery>,
    class: Class,
    model: ModelLabel,
    /// The row of the file in the place, until its outcome is recorded.
    unsettled: Option<(i64, u64)>,
    from_file: bool,
    /// The report held in memory in the place, until its outcome is recorded.
    held: Option<u64>,
}

impl Drop for Place {
    fn drop(&mut self) {
        let Some(outbox) = &self.delivery.outbox else {
            return;
        };
        {
            let mut kept = outbox.kept();
            match self.class {
                Class::Fresh => kept.sending_fresh = kept.sending_fresh.saturating_sub(1),
                Class::Retry => kept.sending_retries = kept.sending_retries.saturating_sub(1),
            }
            if self.from_file {
                kept.sending_from_file = kept.sending_from_file.saturating_sub(1);
                // Until its outcome is written the row still counts in the
                // file.
                kept.beside(self.model, 1);
            }
            // Not settled: it goes back to where it waited.
            if let Some(held) = self.held.and_then(|seq| kept.release(seq)) {
                let max_held = self.delivery.policy.max_queued;
                kept.hold(held.item, max_held);
            }
        }
        if let Some((id, generation)) = self.unsettled {
            outbox.store.settle(Settle {
                id,
                generation,
                outcome: Outcome::Release,
            });
        }
        self.delivery.in_flight_gauge(self.model, Step::Down);
        outbox.wake.notify_one();
    }
}

impl UsageReportDelivery {
    /// The task that decides who is sent when: it hears what the store did,
    /// starts attempts as places come free, claims reports from the file and
    /// writes to it what memory holds. It also publishes the outbox gauges.
    /// One per delivery, for the life of the process.
    pub(super) async fn dispatch(
        self: Arc<Self>,
        mut from_store: mpsc::UnboundedReceiver<Event<Item>>,
    ) {
        let Some(outbox) = &self.outbox else {
            return;
        };
        let mut recheck = tokio::time::interval(outbox.config().recheck_interval);
        recheck.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The moment as of which `advance` last looked at what is due.
        let mut looked_at = Instant::now();
        loop {
            // Only ever a time `advance` has not seen pass: whatever was due
            // when it looked is waiting for something else, a place or an
            // answer of the store, and that wakes this loop when it comes.
            let timer = self.next_timer(outbox, looked_at);
            let due = async {
                match timer {
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
                _ = due => {}
            }
            while let Ok(event) = from_store.try_recv() {
                self.on_store_event(outbox, event);
            }
            looked_at = self.advance(outbox, rechecking);
            self.publish(outbox);
            if outbox.is_idle() {
                self.idle.notify_waiters();
            }
        }
    }

    /// When the dispatcher has something to do that nothing will wake it
    /// for: a report due again, the next probe. `looked_at` is the moment
    /// `advance` last looked as of. A time before it was seen then, and
    /// asking for it again would only run the loop without end; a time after
    /// it was not, even when it has passed by now, and must not be left to
    /// the next look at the file.
    fn next_timer(&self, outbox: &Outbox, looked_at: Instant) -> Option<Instant> {
        let kept = outbox.kept();
        if kept.stopping && !kept.closed {
            return None;
        }
        [
            kept.retry_at,
            kept.memory_due.first().map(|(due_at, _)| *due_at),
            kept.breaker.probe_at.filter(|_| kept.breaker.engaged),
        ]
        .into_iter()
        .flatten()
        .filter(|at| *at > looked_at)
        .min()
    }

    /// Take in what the store did.
    fn on_store_event(self: &Arc<Self>, outbox: &Outbox, event: Event<Item>) {
        let path = outbox.config().path.display();
        match event {
            Event::Committed(committed) => {
                let Committed {
                    stored,
                    no_room,
                    overtaken,
                    evicted,
                    expired,
                    rejected_evicted,
                    claimed,
                    stats,
                    health,
                } = *committed;
                let now = Instant::now();
                let now_ms = outbox.config().clock.now_ms();
                let item = |row: Row| self.item_from(outbox, row, now, now_ms);
                let claimed = claimed.map(|claimed| {
                    (
                        claimed.fresh.into_iter().map(item).collect::<Vec<_>>(),
                        claimed.retries.into_iter().map(item).collect::<Vec<_>>(),
                        claimed.more_fresh,
                        claimed.more_retries,
                        claimed.next_retry_ms,
                    )
                });
                let pending = stats
                    .pending
                    .into_iter()
                    .map(|(label, rows)| (outbox.label(label), rows))
                    .collect();
                let (was, unstarted, limited) = {
                    let mut kept = outbox.kept();
                    let was = (kept.health, kept.stalled);
                    kept.health = health;
                    kept.stalled = false;
                    kept.synced = true;
                    kept.stored = pending;
                    kept.oldest_completed_at_ms = stats.oldest_completed_at_ms;
                    kept.rejected = stats.rejected;
                    kept.used_bytes = stats.used_bytes;
                    let limited = (stats.limited_by_volume != kept.limited_by_volume)
                        .then_some(stats.limited_by_volume);
                    kept.limited_by_volume = stats.limited_by_volume;
                    if stored > 0 || !overtaken.is_empty() {
                        kept.look_fresh = true;
                        kept.look_retries = true;
                    }
                    // Reports held in memory since the writer was given up
                    // on, which it wrote after all: the ones nobody has
                    // started on are the file's from here on.
                    if !overtaken.is_empty() {
                        let written: HashSet<*const Job> = overtaken
                            .iter()
                            .map(|item| Arc::as_ptr(&item.job))
                            .collect();
                        let waiting: Vec<u64> = kept
                            .memory
                            .iter()
                            .filter(|(_, held)| {
                                !held.sending && written.contains(&Arc::as_ptr(&held.item.job))
                            })
                            .map(|(seq, _)| *seq)
                            .collect();
                        for seq in waiting {
                            kept.release(seq);
                        }
                    }
                    let mut unstarted = Vec::new();
                    if let Some((fresh, retries, more_fresh, more_retries, next_retry_ms)) = claimed
                    {
                        kept.asked = false;
                        kept.look_fresh |= more_fresh;
                        kept.look_retries |= more_retries;
                        if let Some(due_ms) = next_retry_ms {
                            let wait = u64::try_from(due_ms.saturating_sub(now_ms)).unwrap_or(0);
                            // A time further ahead than any backoff was
                            // written under another clock.
                            let wait = Duration::from_millis(wait).min(outbox.config().max_backoff);
                            let due_at = now + wait;
                            kept.retry_at = Some(kept.retry_at.map_or(due_at, |at| at.min(due_at)));
                        }
                        if kept.stopping {
                            // Too late to start them; `close` gives every
                            // lease back.
                        } else if kept.breaker.engaged && !kept.asked_for_a_probe {
                            // Asked for before the billing API stopped
                            // answering: the reports that waited longest,
                            // which is not what a probe takes.
                            unstarted.extend(fresh.into_iter().chain(retries));
                        } else if kept.breaker.engaged {
                            // One report is enough to find out whether the
                            // billing API answers. The other goes back.
                            let mut both = fresh.into_iter().chain(retries);
                            if let Some(probe) = both.next() {
                                if probe.attempts == 0 {
                                    kept.fresh.push_back(probe);
                                } else {
                                    kept.retries.push_back(probe);
                                }
                            }
                            unstarted.extend(both);
                        } else {
                            kept.fresh.extend(fresh);
                            kept.retries.extend(retries);
                        }
                    }
                    (was, unstarted, limited)
                };
                // Said when it begins and when it ends: a volume like that
                // is a mistake in how the outbox was set up, or a volume
                // something else is filling.
                match limited {
                    Some(true) => warn!(
                        path = %path,
                        max_bytes = outbox.config().max_bytes,
                        limited_to = stats.max_bytes,
                        "The volume of the usage report outbox has no room for \
                         VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_BYTES and the journal beside it: the \
                         file is kept to what the volume has room for"
                    ),
                    Some(false) => info!(
                        path = %path,
                        max_bytes = outbox.config().max_bytes,
                        "The volume of the usage report outbox has room for \
                         VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_BYTES again"
                    ),
                    None => {}
                }
                // The file had no room for these: they are held in memory
                // and sent from there.
                for item in no_room {
                    let item = self.bypassed(item, "full");
                    outbox.kept().hold(item, self.policy.max_queued);
                    outbox.in_transit.fetch_sub(1, Ordering::SeqCst);
                }
                outbox.in_transit.fetch_sub(stored, Ordering::SeqCst);
                self.give_back(outbox, unstarted);
                self.say_health(outbox, was, (health, false), None);
                for row in evicted {
                    self.say_gave_up(outbox, &item(row), GaveUp::FileFull);
                }
                for (row, reason) in expired {
                    self.say_gave_up(outbox, &item(row), GaveUp::TooOld(reason));
                }
                if rejected_evicted > 0 {
                    metrics::counter!("inference_proxy_usage_report_outbox_rejected_evicted_total")
                        .increment(rejected_evicted as u64);
                }
            }
            Event::Failed {
                error,
                reports,
                claim,
                health,
            } => {
                let was = {
                    let mut kept = outbox.kept();
                    let was = (kept.health, kept.stalled);
                    kept.health = health;
                    kept.stalled = false;
                    kept.synced = true;
                    if claim {
                        kept.asked = false;
                    }
                    was
                };
                if let Some((op, _)) = &error {
                    metrics::counter!("inference_proxy_usage_report_outbox_errors_total", "op" => *op)
                        .increment(1);
                }
                outbox.in_transit.fetch_sub(reports.len(), Ordering::SeqCst);
                let unstored = reports.len();
                for item in reports {
                    let item = self.bypassed(item, "write_failed");
                    outbox.kept().hold(item, self.policy.max_queued);
                }
                self.say_health(
                    outbox,
                    was,
                    (health, false),
                    error.map(|error| (error, unstored)),
                );
            }
            Event::Replaced {
                why,
                kept_as,
                generation,
            } => {
                metrics::counter!("inference_proxy_usage_report_outbox_replaced_total", "why" => why)
                    .increment(1);
                // The rows leased from a database that is no longer at the
                // path are not in the one that is: they are held in memory,
                // and written to it with the rest.
                let strangers: Vec<Item> = {
                    let mut kept = outbox.kept();
                    let changed = kept.generation != generation;
                    kept.generation = generation;
                    kept.look_fresh = true;
                    kept.look_retries = true;
                    if changed {
                        let mut strangers: Vec<Item> = kept.fresh.drain(..).collect();
                        strangers.extend(kept.retries.drain(..));
                        strangers
                    } else {
                        Vec::new()
                    }
                };
                for item in strangers {
                    let item = item.out_of_the_file();
                    outbox.kept().hold(item, self.policy.max_queued);
                }
                match why {
                    "corrupt" => error!(
                        path = %path,
                        kept_as = %kept_as.as_deref().unwrap_or(outbox.config().path.as_path()).display(),
                        "Usage report outbox was not a readable database any more: it was moved \
                         aside for a person to look at, and a new file started. The reports it \
                         held are not sent unless they are put back"
                    ),
                    "deleted" => warn!(
                        path = %path,
                        "Usage report outbox was deleted under the running process: it has been \
                         written back as it was"
                    ),
                    "lost" => error!(
                        path = %path,
                        "Usage report outbox was deleted or emptied under the running process \
                         and could not be written back: a new one was started, and the reports \
                         it held are lost, except those this process was sending"
                    ),
                    _ => warn!(
                        path = %path,
                        "Usage report outbox is another file than the one that was open: it has \
                         been opened in its place"
                    ),
                }
            }
        }
    }

    /// Say what changed about the file, when something did: the outbox is
    /// spoken of in the log when its state changes, never while it stays
    /// what it was. `failed`: the step that failed, why, and how many
    /// reports it left unwritten.
    fn say_health(
        &self,
        outbox: &Outbox,
        was: (Health, bool),
        is: (Health, bool),
        failed: Option<((&'static str, String), usize)>,
    ) {
        let path = outbox.config().path.display();
        let usable = |(health, stalled): (Health, bool)| health.available && !stalled;
        match (usable(was), usable(is)) {
            (true, false) => {
                let ((op, reason), unstored) =
                    failed.unwrap_or((("stalled", "the writer does not come back".to_string()), 0));
                error!(
                    path = %path,
                    op,
                    error = %reason,
                    unstored,
                    "Usage report outbox unavailable: reports are held in memory, sent from \
                     there, and written to the file when it is back"
                );
            }
            (false, true) => info!(path = %path, "Usage report outbox available again"),
            (false, false) => {
                if let Some(((op, reason), _)) = failed {
                    // Said when it began; the counter has every later try.
                    debug!(path = %path, op, error = %reason, "Usage report outbox still unavailable");
                }
            }
            (true, true) => {}
        }
        let full = {
            let mut kept = outbox.kept();
            let held = !kept.memory.is_empty() || outbox.in_transit.load(Ordering::SeqCst) > 0;
            let full = is.0.full || (kept.full && held);
            let changed = full != kept.full;
            kept.full = full;
            changed.then_some(full)
        };
        match full {
            Some(true) => warn!(
                path = %path,
                max_bytes = outbox.config().max_bytes,
                "Usage report outbox is full: new reports are held in memory and sent from \
                 there until it has room. What it holds is still sent"
            ),
            Some(false) => info!(path = %path, "Usage report outbox has room again"),
            None => {}
        }
    }

    /// Start attempts in the places that are free, claim reports for the
    /// places to come, write to the file what memory holds, and look after
    /// what only time changes. `rechecking`: the file is looked at whether
    /// or not anything here says it changed. Returns the moment as of which
    /// it looked at what is due.
    fn advance(self: &Arc<Self>, outbox: &Outbox, rechecking: bool) -> Instant {
        let config = outbox.config();
        let cap = self.policy.max_in_flight.max(1);
        let max_held = self.policy.max_queued;
        let now = Instant::now();
        let (limit, limit_reason) = max_age(&self.policy, config);

        // A writer that does not come back from a transaction: what it was
        // handed is taken over and sent from memory, and the file counts as
        // unavailable until the writer is heard of again.
        if rechecking {
            let stalled = outbox.store.stalled_for() > config.stall_timeout;
            let was = {
                let kept = outbox.kept();
                (kept.health, kept.stalled)
            };
            if stalled && !was.1 {
                let taken = outbox.store.take_over();
                outbox.in_transit.fetch_sub(taken.len(), Ordering::SeqCst);
                metrics::counter!("inference_proxy_usage_report_outbox_errors_total", "op" => "stalled")
                    .increment(1);
                let unstored = taken.len();
                {
                    let mut kept = outbox.kept();
                    kept.stalled = true;
                    kept.asked = false;
                }
                for item in taken {
                    let item = self.bypassed(item, "stalled");
                    outbox.kept().hold(item, max_held);
                }
                let reason = format!(
                    "the writer has been in one transaction for more than {} s",
                    config.stall_timeout.as_secs()
                );
                self.say_health(
                    outbox,
                    was,
                    (was.0, true),
                    Some((("stalled", reason), unstored)),
                );
            }
        }

        let mut start = Vec::new();
        let mut gave_up = Vec::new();
        let mut give_back = Vec::new();
        let mut write = Vec::new();
        let mut ask = None;
        // A report was handed over since the billing API stopped answering,
        // and none that was has been tried.
        let newcomer =
            |kept: &Kept| !kept.breaker.newcomer_tried && outbox.newcomer.load(Ordering::Relaxed);
        {
            let mut kept = outbox.kept();
            let kept = &mut *kept;
            gave_up.append(&mut kept.dropped);
            let usable = kept.health.available && !kept.stalled && !kept.stopping;

            // Reports in memory too old to be sent end there, and wait for
            // the file to keep them in `rejected`.
            if rechecking && !kept.stopping {
                let too_old: Vec<u64> = kept
                    .memory
                    .iter()
                    .filter(|(_, held)| {
                        !held.sending
                            && matches!(held.item.home, Home::Memory { ended: None, .. })
                            && held.item.job.since_completion() > limit
                    })
                    .map(|(seq, _)| *seq)
                    .collect();
                for seq in too_old {
                    if let Some(held) = kept.release(seq) {
                        let ended = Item {
                            home: Home::Memory {
                                due_at: now,
                                ended: Some(Rejection {
                                    reason: limit_reason,
                                    status: None,
                                    outcome: held
                                        .item
                                        .last_failure
                                        .map_or(NEVER_SENT, |(reason, _)| reason),
                                }),
                            },
                            ..held.item
                        };
                        gave_up.push((ended.clone(), GaveUp::TooOld(limit_reason)));
                        kept.hold(ended, max_held);
                    }
                }
            }

            // What memory holds is written to the file as soon as the file
            // takes reports: those that wait, those that wait for their next
            // attempt, and those that ended.
            if usable && !kept.health.full && outbox.store.is_accepting() {
                let waiting: Vec<u64> = kept
                    .memory
                    .iter()
                    .filter(|(_, held)| !held.sending)
                    .map(|(seq, _)| *seq)
                    .collect();
                write.extend(waiting.into_iter().filter_map(|seq| kept.release(seq)));
            }

            // The retry of a report of the file is due.
            if kept.retry_at.is_some_and(|at| at <= now) {
                kept.retry_at = None;
                kept.look_retries = true;
            }
            if rechecking {
                kept.look_fresh = true;
                kept.look_retries = true;
            }

            // Attempts, in the places that are free.
            loop {
                let in_flight = kept.in_flight();
                if in_flight >= cap || (kept.stopping && !kept.closed) {
                    break;
                }
                let fresh = kept.ready(Class::Fresh, now);
                let retry = kept.ready(Class::Retry, now);
                let class = if kept.breaker.engaged {
                    // One report at a time, `pause` apart. One report that
                    // came in since does not wait for the pause.
                    let early = fresh && newcomer(kept);
                    let due = kept.breaker.probe_at.is_none_or(|at| at <= now);
                    if in_flight > 0 || !(early || due) || !(fresh || retry) {
                        break;
                    }
                    // A probe that comes due when nothing was handed over
                    // since the last one: the next report to come is news
                    // again, and may go ahead of the pause.
                    if !outbox.newcomer.load(Ordering::Relaxed) {
                        kept.breaker.newcomer_tried = false;
                    }
                    match (fresh, retry) {
                        (true, _) => {
                            // The newest report there is has been tried.
                            if outbox.newcomer.swap(false, Ordering::Relaxed) {
                                kept.breaker.newcomer_tried = true;
                            }
                            Class::Fresh
                        }
                        (false, true) => Class::Retry,
                        (false, false) => break,
                    }
                } else {
                    // Retries hold `retry_places` and no more, so that a
                    // first attempt always finds a place; and the last free
                    // place is left to a retry while none is in one, so that
                    // retries are not kept out either. With one place the
                    // two take turns.
                    let (can_fresh, can_retry) = if cap == 1 {
                        (
                            fresh && (!retry || kept.fresh_since_retry < RETRY_EVERY - 1),
                            retry && (!fresh || kept.fresh_since_retry >= RETRY_EVERY - 1),
                        )
                    } else {
                        (
                            fresh && (in_flight + 1 < cap || !retry || kept.sending_retries > 0),
                            retry && kept.sending_retries < outbox.retry_places,
                        )
                    };
                    match (can_fresh, can_retry) {
                        (true, _) => Class::Fresh,
                        (false, true) => Class::Retry,
                        (false, false) => break,
                    }
                };
                let Some((item, held)) = kept.next(class, now, kept.breaker.engaged) else {
                    break;
                };
                // A row of a database that is no longer at the path.
                if let Home::File { generation, .. } = item.home {
                    if generation != kept.generation {
                        kept.sending_from_file -= 1;
                        kept.beside(item.model(), 1);
                        let item = item.out_of_the_file();
                        kept.hold(item, max_held);
                        continue;
                    }
                }
                match class {
                    Class::Fresh => {
                        kept.sending_fresh += 1;
                        kept.fresh_since_retry = kept.fresh_since_retry.saturating_add(1);
                    }
                    Class::Retry => {
                        kept.sending_retries += 1;
                        kept.fresh_since_retry = 0;
                    }
                }
                let probe = kept.breaker.engaged;
                start.push((item, held, class, probe));
            }

            // Reports for the places to come. While the billing API is not
            // answering nothing is leased ahead: a leased report nobody
            // starts only keeps another process from sending it.
            if usable && !kept.asked {
                let (fresh, retries) = if kept.breaker.engaged {
                    // One report at hand is all a probe takes.
                    while kept.fresh.len() + kept.retries.len() > 1 {
                        give_back.extend(kept.retries.pop_back().or_else(|| kept.fresh.pop_back()));
                    }
                    let fresh_at_hand = kept.ready(Class::Fresh, now);
                    let at_hand = fresh_at_hand || kept.ready(Class::Retry, now);
                    let due = kept.breaker.probe_at.is_none_or(|at| at <= now);
                    if kept.in_flight() > 0 {
                        (0, 0)
                    } else if due {
                        // A report never tried when the file may hold one,
                        // else one that failed before.
                        match (at_hand, kept.look_fresh) {
                            (true, _) => (0, 0),
                            (false, true) => (1, 0),
                            (false, false) => (0, 1),
                        }
                    } else if !fresh_at_hand && newcomer(kept) {
                        // The report that may go ahead of the pause: it is
                        // in the file, or on its way into it with this very
                        // transaction.
                        kept.look_fresh = true;
                        (1, 0)
                    } else {
                        (0, 0)
                    }
                } else {
                    (
                        (cap * (1 + ROUNDS_AHEAD))
                            .saturating_sub(kept.sending_fresh + kept.fresh.len()),
                        (outbox.retry_places * 2)
                            .saturating_sub(kept.sending_retries + kept.retries.len()),
                    )
                };
                let fresh = if kept.look_fresh { fresh } else { 0 };
                let retries = if kept.look_retries { retries } else { 0 };
                if fresh + retries > 0 || rechecking {
                    kept.asked = true;
                    kept.asked_for_a_probe = kept.breaker.engaged;
                    kept.look_fresh &= fresh == 0;
                    kept.look_retries &= retries == 0;
                    ask = Some(Claim {
                        fresh,
                        newest_first: kept.breaker.engaged,
                        retries,
                        lease: outbox.lease,
                        expire: Some((limit, limit_reason)),
                    });
                }
            }
        }

        for (item, held, class, probe) in start {
            let model = item.model();
            let place = Place {
                delivery: Arc::clone(self),
                class,
                model,
                unsettled: match item.home {
                    Home::File { id, generation, .. } => Some((id, generation)),
                    Home::Memory { .. } => None,
                },
                from_file: matches!(item.home, Home::File { .. }),
                held,
            };
            self.in_flight_gauge(model, Step::Up);
            tokio::spawn(Arc::clone(self).send(item, place, probe));
        }
        self.give_back(outbox, give_back);
        for held in write {
            if let Some((item, _)) = outbox.keep(held.item, usize::MAX) {
                outbox.kept().hold(item, max_held);
            }
        }
        match ask {
            Some(claim) => outbox.store.claim(claim),
            // The way an unavailable file is tried again, and how the
            // numbers stay fresh while nothing is claimed.
            None if rechecking && !outbox.kept().stopping => outbox.store.sync(),
            None => {}
        }
        gave_up_lines(self, outbox, gave_up);
        now
    }

    /// Count `item` as a report the file did not take, for `why`, unless it
    /// was counted before.
    fn bypassed(&self, mut item: Item, why: &'static str) -> Item {
        if !item.bypassed {
            item.bypassed = true;
            self.count(
                "inference_proxy_usage_report_outbox_bypassed_total",
                &item.job.reporter,
                ("reason", why),
            );
        }
        item
    }

    /// Give back the leases of reports that will not be started for now.
    fn give_back(&self, outbox: &Outbox, unstarted: Vec<Item>) {
        if unstarted.is_empty() {
            return;
        }
        for item in &unstarted {
            if let Home::File { id, generation, .. } = item.home {
                outbox.store.settle(Settle {
                    id,
                    generation,
                    outcome: Outcome::Release,
                });
            }
        }
        let mut kept = outbox.kept();
        kept.look_fresh = true;
        kept.look_retries = true;
    }

    /// One attempt at `item`, in `place`, and what follows from it.
    async fn send(self: Arc<Self>, item: Item, mut place: Place, probe: bool) {
        let Some(outbox) = &self.outbox else {
            return;
        };
        let verdict = self.try_once(outbox, &item).await;
        self.settle(outbox, item, &mut place, probe, verdict);
    }

    /// Send `item` once, unless it should not be sent at all.
    async fn try_once(&self, outbox: &Outbox, item: &Item) -> Verdict {
        let job = &item.job;
        let timeout = self.policy.attempt_timeout;
        let (limit, reason) = max_age(&self.policy, outbox.config());
        if job.since_completion() > limit {
            return Verdict::TooOld(reason);
        }
        // The report waited for its place longer than its lease allows for:
        // another process may take it while this one is still sending. It is
        // given back and claimed again, with a new lease.
        if let Home::File { lease_until, .. } = item.home {
            let left = lease_until.saturating_duration_since(Instant::now());
            if left < timeout + outbox.lease_slack {
                return Verdict::NotSent;
            }
        }
        if let Some((reason, _)) = item.last_failure {
            self.count(
                "inference_proxy_usage_report_retries_total",
                &job.reporter,
                ("reason", reason),
            );
        }
        let attempt = self.attempt(job).await;
        let passing = retry_reason(attempt.outcome, &attempt.answer)
            .or_else(|| caller_refused(&attempt.answer))
            .and_then(passing_failure);
        match passing {
            Some(failure) => Verdict::Failed(attempt, failure),
            None if attempt.outcome == UsageReportOutcome::Accepted => Verdict::Accepted(attempt),
            None => Verdict::Refused(attempt),
        }
    }

    /// What becomes of `item` after `verdict`: its lines and series, its row
    /// or its place in memory, and what it says about the billing API.
    fn settle(
        self: &Arc<Self>,
        outbox: &Outbox,
        item: Item,
        place: &mut Place,
        probe: bool,
        verdict: Verdict,
    ) {
        let config = outbox.config();
        let now = Instant::now();
        let attempts = item.attempts.saturating_add(1);
        // What is written to the row, or what the report becomes in memory.
        let (outcome, next): (Option<Outcome>, Option<Item>) = match &verdict {
            Verdict::Accepted(attempt) => {
                self.end(&item.job, attempt, attempts, None);
                (Some(Outcome::Delete), None)
            }
            Verdict::Refused(attempt) => {
                self.end(&item.job, attempt, attempts, None);
                let why = Rejection {
                    reason: "rejected",
                    status: attempt.answer.as_ref().ok().map(|status| status.as_u16()),
                    outcome: attempt.outcome.as_label(),
                };
                (
                    Some(Outcome::Reject { why, attempts }),
                    Some(Item {
                        attempts,
                        home: Home::Memory {
                            due_at: now,
                            ended: Some(why),
                        },
                        ..item.clone()
                    }),
                )
            }
            Verdict::Failed(attempt, failure) => {
                let delay = outbox.backoff(&self.policy, attempts);
                self.say_retrying(&item.job, attempt, attempts, delay);
                (
                    Some(Outcome::Retry {
                        attempts,
                        next_attempt_at_ms: config
                            .clock
                            .now_ms()
                            .saturating_add(delay.as_millis() as i64),
                        last_outcome: failure.0,
                    }),
                    Some(Item {
                        attempts,
                        last_failure: Some(*failure),
                        home: Home::Memory {
                            due_at: now + delay,
                            ended: None,
                        },
                        ..item.clone()
                    }),
                )
            }
            Verdict::TooOld(reason) => {
                let why = Rejection {
                    reason,
                    status: None,
                    outcome: item.last_failure.map_or(NEVER_SENT, |(reason, _)| reason),
                };
                let ended = Item {
                    home: Home::Memory {
                        due_at: now,
                        ended: Some(why),
                    },
                    ..item.clone()
                };
                self.say_gave_up(outbox, &ended, GaveUp::TooOld(reason));
                (
                    Some(Outcome::Reject {
                        why,
                        attempts: item.attempts,
                    }),
                    Some(ended),
                )
            }
            Verdict::NotSent => (Some(Outcome::Release), Some(item.clone())),
        };

        let mut engaged = None;
        let mut give_back = Vec::new();
        let mut write = None;
        {
            let mut kept = outbox.kept();
            let kept = &mut *kept;
            // What this says about the billing API.
            match &verdict {
                Verdict::Accepted(_) | Verdict::Refused(_) => {
                    if kept.breaker.engaged {
                        engaged = Some(false);
                        outbox.paused.store(false, Ordering::Relaxed);
                        // Whatever the file holds is worth a look again.
                        kept.look_fresh = true;
                        kept.look_retries = true;
                    }
                    kept.breaker = Breaker::default();
                }
                Verdict::Failed(_, failure) => {
                    let breaker = &mut kept.breaker;
                    breaker.failures = breaker.failures.saturating_add(1);
                    if item.attempts == 0 {
                        breaker.fresh_failures = breaker.fresh_failures.saturating_add(1);
                    }
                    let since = *breaker.since.get_or_insert(now);
                    // A 401 is about this process, so about every report.
                    let at_once = failure.0 == "http_401";
                    let first_attempts = breaker.fresh_failures >= config.breaker_after;
                    let nothing_answered = breaker.failures >= config.breaker_after
                        && now.saturating_duration_since(since) >= config.breaker_window;
                    if breaker.engaged {
                        if probe {
                            breaker.probes_failed = breaker.probes_failed.saturating_add(1);
                            breaker.probe_at =
                                Some(now + outbox.pause(&self.policy, breaker.probes_failed));
                        }
                    } else if at_once || first_attempts || nothing_answered {
                        breaker.engaged = true;
                        breaker.probes_failed = 1;
                        breaker.probe_at = Some(now + outbox.pause(&self.policy, 1));
                        breaker.newcomer_tried = false;
                        outbox.newcomer.store(false, Ordering::Relaxed);
                        outbox.paused.store(true, Ordering::Relaxed);
                        engaged = Some(true);
                        // Nothing leased ahead stays leased.
                        give_back.extend(kept.fresh.drain(..));
                        give_back.extend(kept.retries.drain(..));
                    }
                }
                Verdict::TooOld(_) | Verdict::NotSent => {}
            }

            let in_file = match item.home {
                Home::File { id, generation, .. } => {
                    (generation == kept.generation).then_some((id, generation))
                }
                Home::Memory { .. } => None,
            };
            match (in_file, outcome) {
                // Its row is told. (Released without its lock: `Place`.)
                (Some((id, generation)), Some(outcome)) => {
                    if let Outcome::Retry { .. } = &outcome {
                        let due_at = match &next {
                            Some(Item {
                                home: Home::Memory { due_at, .. },
                                ..
                            }) => *due_at,
                            _ => now,
                        };
                        kept.retry_at = Some(kept.retry_at.map_or(due_at, |at| at.min(due_at)));
                    }
                    if matches!(outcome, Outcome::Release) {
                        kept.look_fresh = true;
                        kept.look_retries = true;
                    }
                    outbox.store.settle(Settle {
                        id,
                        generation,
                        outcome,
                    });
                    place.unsettled = None;
                }
                // It is not in the file (it never was, or the file it was in
                // is no longer the one at the path): what is left of it is
                // held here, and written to the file when the file takes it.
                _ => {
                    place.unsettled = None;
                    // (It is marked as being sent, so it waits nowhere.) A
                    // report that is no longer there was handed to the store
                    // when shutdown began (`close`): what is written there,
                    // or came back from there, is the report now, and this
                    // attempt has nothing more to say about it.
                    let handed_on = place
                        .held
                        .take()
                        .is_some_and(|seq| kept.release(seq).is_none());
                    let next = next.filter(|_| !handed_on).map(|next| match next.home {
                        Home::File { .. } => next.out_of_the_file(),
                        Home::Memory { .. } => next,
                    });
                    if let Some(next) = next {
                        if outbox.store.is_accepting() && kept.health.available && !kept.stalled {
                            write = Some(next);
                        } else {
                            kept.hold(next, self.policy.max_queued);
                        }
                    }
                }
            }
        }
        if let Some(item) = write {
            if let Some((item, _)) = outbox.keep(item, usize::MAX) {
                outbox.kept().hold(item, self.policy.max_queued);
            }
        }
        self.give_back(outbox, give_back);
        match engaged {
            Some(true) => warn!(
                after = config.breaker_after,
                "The billing API is not answering usage reports: one report at a time is sent \
                 until one gets an answer"
            ),
            Some(false) => info!("The billing API answers usage reports again"),
            None => {}
        }
    }

    /// A row of the outbox as a report to send. Who it is for is read from
    /// the report itself; where it goes and the bearer are the running
    /// process's.
    fn item_from(self: &Arc<Self>, outbox: &Outbox, row: Row, now: Instant, now_ms: i64) -> Item {
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
        let job = Job {
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
            completed_before: age,
        };
        Item {
            job: Arc::new(job),
            attempts: row.attempts,
            last_failure: row.last_outcome.as_deref().and_then(passing_failure),
            bypassed: false,
            home: Home::File {
                id: row.id,
                generation: row.generation,
                lease_until: now + outbox.lease,
            },
        }
    }

    /// The line and the series of a report given up on without an answer of
    /// its own, with its ids.
    fn say_gave_up(&self, outbox: &Outbox, item: &Item, why: GaveUp) {
        let reporter = &item.job.reporter;
        let since_completion_ms = item.job.since_completion().as_millis() as u64;
        let (outcome, reason) = match why {
            GaveUp::FileFull | GaveUp::MemoryFull => (UsageReportOutcome::QueueFull, "queue_full"),
            GaveUp::TooOld(reason) => (UsageReportOutcome::DeadlineExceeded, reason),
            // A report that ends with the process has no outcome of its own.
            GaveUp::Shutdown => (UsageReportOutcome::QueueFull, "shutdown"),
        };
        if !matches!(why, GaveUp::Shutdown) {
            record_usage_report_outcome(reporter, outcome, None);
        }
        self.count(
            "inference_proxy_usage_reports_dropped_total",
            reporter,
            ("reason", reason),
        );
        let last_attempt = item.last_failure.map(|(reason, _)| reason);
        match why {
            GaveUp::FileFull => warn!(
                request_id = %reporter.request_id.as_deref().unwrap_or(""),
                org_id = %reporter.org_id.as_deref().unwrap_or(""),
                workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                model = %reporter.model_name,
                auth_path = reporter.request_source.auth_path.as_label(),
                ingress_route = reporter.request_source.ingress_route.as_label(),
                outcome = outcome.as_label(),
                since_completion_ms,
                max_pending = outbox.config().max_pending,
                "Usage report dropped: the outbox is full and this report waited longest — \
                 usage NOT billed"
            ),
            GaveUp::MemoryFull => warn!(
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
                "Usage report dropped: the outbox could not take it, and what waits in memory \
                 is full — usage NOT billed"
            ),
            GaveUp::TooOld(reason) => warn!(
                request_id = %reporter.request_id.as_deref().unwrap_or(""),
                org_id = %reporter.org_id.as_deref().unwrap_or(""),
                workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                model = %reporter.model_name,
                auth_path = reporter.request_source.auth_path.as_label(),
                ingress_route = reporter.request_source.ingress_route.as_label(),
                outcome = outcome.as_label(),
                reason,
                attempts = item.attempts,
                last_attempt,
                since_completion_ms,
                "Usage report moved to rejected: older than a report may grow — usage NOT \
                 billed unless it is put back"
            ),
            GaveUp::Shutdown => warn!(
                request_id = %reporter.request_id.as_deref().unwrap_or(""),
                org_id = %reporter.org_id.as_deref().unwrap_or(""),
                workspace_id = %reporter.workspace_id.as_deref().unwrap_or(""),
                api_key_id = %reporter.api_key_id.as_deref().unwrap_or(""),
                model = %reporter.model_name,
                auth_path = reporter.request_source.auth_path.as_label(),
                ingress_route = reporter.request_source.ingress_route.as_label(),
                attempts = item.attempts,
                last_attempt,
                since_completion_ms,
                "Usage report left in memory at shutdown: the outbox did not take it — usage \
                 NOT billed"
            ),
        }
    }

    /// The outbox gauges, and `queue_depth`, which with an outbox is set
    /// from the rows of the file rather than counted up and down: the file
    /// is shared, and a row another process sent never passed through here.
    fn publish(&self, outbox: &Outbox) {
        let (health, paused, stored, mut waiting, oldest, rejected, used_bytes, in_memory) = {
            let kept = outbox.kept();
            (
                Health {
                    available: kept.health.available && !kept.stalled,
                    full: kept.full,
                },
                kept.breaker.engaged,
                kept.stored.clone(),
                kept.beside.clone(),
                kept.oldest_completed_at_ms,
                kept.rejected.clone(),
                kept.used_bytes,
                kept.memory.len(),
            )
        };
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        metrics::gauge!("inference_proxy_usage_report_outbox_available")
            .set(flag(health.available));
        metrics::gauge!("inference_proxy_usage_report_outbox_full").set(flag(health.full));
        metrics::gauge!("inference_proxy_usage_report_outbox_billing_paused").set(flag(paused));
        for (model, rows) in stored {
            model_gauge!(model, "inference_proxy_usage_report_outbox_pending").set(rows as f64);
            *waiting.entry(model).or_default() += i64::try_from(rows).unwrap_or(i64::MAX);
        }
        for (model, waiting) in waiting {
            model_gauge!(model, "inference_proxy_usage_report_queue_depth")
                .set(waiting.max(0) as f64);
        }
        let oldest_age_ms = oldest.map_or(0, |at| {
            outbox.config().clock.now_ms().saturating_sub(at).max(0)
        });
        metrics::gauge!("inference_proxy_usage_report_outbox_oldest_pending_age_seconds")
            .set(oldest_age_ms as f64 / 1000.0);
        // Every reason there is has its series from the start, so that an
        // alert on it never has to tell 0 from absent.
        for reason in REJECTED_REASONS {
            if !rejected.iter().any(|(known, _)| known == reason) {
                metrics::gauge!("inference_proxy_usage_report_outbox_rejected", "reason" => reason)
                    .set(0.0);
            }
        }
        for (reason, rows) in rejected {
            metrics::gauge!("inference_proxy_usage_report_outbox_rejected", "reason" => reason)
                .set(rows as f64);
        }
        metrics::gauge!("inference_proxy_usage_report_outbox_bytes").set(used_bytes as f64);
        metrics::gauge!("inference_proxy_usage_report_outbox_unwritten")
            .set(outbox.in_transit.load(Ordering::SeqCst) as f64);
        metrics::gauge!("inference_proxy_usage_report_outbox_in_memory").set(in_memory as f64);
    }

    /// Shutdown with an outbox. The attempts in flight get `shutdown_drain`
    /// to end; what memory holds is then handed to the store, which writes
    /// it, gives back every lease this process holds and closes the file.
    /// What the file did not take is said and counted. `Drained` counts what
    /// this process still held itself; `left_waiting` is what stayed in
    /// memory only, and is lost.
    pub(super) async fn close(&self, outbox: &Outbox) -> Drained {
        let config = outbox.config();
        let started_at = Instant::now();
        let max_held = self.policy.max_queued;
        let (pending, unstarted) = {
            let mut kept = outbox.kept();
            kept.stopping = true;
            // Leased and not started: given back with every other lease.
            let mut unstarted: Vec<Item> = kept.fresh.drain(..).collect();
            unstarted.extend(kept.retries.drain(..));
            // What this process holds itself: in memory, on its way to the
            // file, and the rows of the file it is sending.
            let pending = kept.memory.len()
                + outbox.in_transit.load(Ordering::SeqCst)
                + kept.sending_from_file;
            (pending, unstarted)
        };
        drop(unstarted);

        let give_up_at = tokio::time::Instant::now() + self.policy.shutdown_drain;
        while outbox.kept().in_flight() > 0 && tokio::time::Instant::now() < give_up_at {
            // The places wake the dispatcher, not this; attempts take
            // milliseconds to seconds, so this looks often enough.
            tokio::time::sleep_until(
                (tokio::time::Instant::now() + Duration::from_millis(5)).min(give_up_at),
            )
            .await;
        }
        let left_in_flight = outbox.kept().in_flight();

        // Everything memory holds, the reports still being sent included.
        // From here on what is handed to the store is the report: an attempt
        // that ends after this finds its report gone from memory and leaves
        // it at that (`settle`), so a report is in one place only and is
        // counted once, as written or as left. Should the attempt be
        // accepted after all, the next process sends the report once more,
        // and the billing API tells the two apart by their completion id.
        let in_memory: Vec<Item> = {
            let mut kept = outbox.kept();
            let all: Vec<u64> = kept.memory.keys().copied().collect();
            all.into_iter()
                .filter_map(|seq| kept.release(seq))
                .map(|held| held.item)
                .collect()
        };
        for item in in_memory {
            outbox.in_transit.fetch_add(1, Ordering::SeqCst);
            if let Err((item, _)) = outbox.store.offer_anyway(item) {
                outbox.in_transit.fetch_sub(1, Ordering::SeqCst);
                outbox.kept().hold(item, max_held);
            }
        }

        // The disk may be why the process is stopping: this is not waited
        // for without end either.
        let closed = tokio::time::timeout(config.close_timeout, outbox.store.close()).await;
        let left = match closed {
            Ok(Ok(left)) => left,
            // The writer did not come back. What it was handed is here.
            _ => {
                let taken = outbox.store.take_over();
                outbox.in_transit.fetch_sub(taken.len(), Ordering::SeqCst);
                for item in taken {
                    outbox.kept().hold(item, max_held);
                }
                None
            }
        };
        // What the last transaction could not write comes back through the
        // dispatcher, which is still running.
        let settled_by = Instant::now() + Duration::from_secs(1);
        while outbox.in_transit.load(Ordering::SeqCst) > 0 && Instant::now() < settled_by {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        let unwritten: Vec<Item> = {
            let mut kept = outbox.kept();
            let all: Vec<u64> = kept.memory.keys().copied().collect();
            all.into_iter()
                .filter_map(|seq| kept.release(seq))
                .map(|held| held.item)
                .collect()
        };
        outbox.kept().closed = true;
        let waited = started_at.elapsed();
        match &left {
            Some(left) => info!(
                pending,
                in_outbox = left.pending_total(),
                left_in_flight,
                left_unwritten = unwritten.len(),
                waited_ms = waited.as_millis() as u64,
                "Usage report outbox closed for shutdown: what it holds is sent by the next \
                 process"
            ),
            None => warn!(
                pending,
                left_in_flight,
                left_unwritten = unwritten.len(),
                waited_ms = waited.as_millis() as u64,
                "Usage report outbox could not be closed for shutdown: what was not written \
                 is lost, and the leases of this process are left to run out"
            ),
        }
        // Reports the file never took end with the process, as they do
        // without an outbox: each is said and counted.
        for item in &unwritten {
            self.say_gave_up(outbox, item, GaveUp::Shutdown);
        }
        if !unwritten.is_empty() {
            warn!(
                pending,
                left_waiting = unwritten.len(),
                left_in_flight,
                waited_ms = waited.as_millis() as u64,
                "Usage reports left undelivered at shutdown — usage NOT billed"
            );
        }
        Drained {
            pending,
            left_waiting: unwritten.len(),
            left_in_flight,
            waited,
        }
    }
}

/// Write the lines of the reports in `gave_up`.
fn gave_up_lines(delivery: &UsageReportDelivery, outbox: &Outbox, gave_up: Vec<(Item, GaveUp)>) {
    for (item, why) in gave_up {
        delivery.say_gave_up(outbox, &item, why);
    }
}

/// The entry of `PASSING_FAILURES` a reason, or a row's `last_outcome`, names.
fn passing_failure(label: &str) -> Option<Passing> {
    PASSING_FAILURES
        .into_iter()
        .find(|(reason, _)| *reason == label)
}

/// With an outbox, one more answer leaves a report where it is: a 401 says
/// that the billing API does not accept this process (a usage token that is
/// wrong, or was rotated under a running process), not that anything is
/// wrong with the report. Every report gets that answer until the token is
/// put right, so ending them would move the whole traffic of that time to
/// `rejected`. They wait instead, and are sent with the right token.
///
/// A 403 is not that: the billing API's handler never answers one for the
/// token, so it is about the report and final like any other 4xx. Without an
/// outbox a 401 is final too, as it always was (`retry_reason`): nothing in
/// memory outlives the restart that puts a token right.
fn caller_refused(answer: &Result<reqwest::StatusCode, reqwest::Error>) -> Option<&'static str> {
    match answer {
        Ok(reqwest::StatusCode::UNAUTHORIZED) => Some("http_401"),
        _ => None,
    }
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

#[cfg(test)]
#[path = "usage_report_outbox_tests.rs"]
mod tests;
