//! The durable outbox of usage reports: a SQLite file that keeps every report
//! from the completion of its request until the billing API has accepted it
//! or refused it for good (`VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH`).
//!
//! This module is the store and nothing else. `usage_report.rs` decides what
//! becomes of a report; here it is written down, leased out for an attempt
//! and removed.
//!
//! SQLite calls block, so all of them are made by one thread of this module's
//! own, the writer, which owns the connection. Everything else talks to it
//! through an inbox in memory: handing a report over is a lock and a push,
//! whatever the disk is doing, and nothing here is ever awaited or called on
//! a runtime worker. The writer gathers what is in the inbox into one
//! transaction (group commit): reports handed over within `commit_interval`
//! of each other are written together, and so is whatever outcome or claim
//! is waiting at that moment.
//!
//! The file is shared state. Two processes may have it open at once (a
//! blue/green switch overlaps the old process and the new one), so a report
//! being sent is marked with a lease: its holder and a time after which
//! anyone may take it. A process that dies leaves its leases to run out.
//!
//! What a row holds is what sending the report again takes and what its log
//! lines and metrics need: the serialized report, the request id, three
//! labels, when its request completed and how its attempts went. Never the
//! bearer token, which is read from the configuration when a report is sent,
//! and never anything of a request or a response.
//!
//! The store must never cost the process anything else. When the file cannot
//! be opened or written, the reports of that transaction are handed back to
//! be delivered from memory, as without an outbox, and opening is tried
//! again every `reopen_interval`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OpenFlags, TransactionBehavior};
use tokio::sync::{mpsc, oneshot};

/// `PRAGMA user_version` of the schema below.
const SCHEMA_VERSION: i64 = 1;

/// Reports written by one transaction at most. A longer backlog takes several,
/// so the write lock, which another process may be waiting for, is never held
/// for long.
const MAX_BATCH: usize = 2_000;

/// How often the write-ahead log is flushed to disk and folded into the file
/// while there is something to fold. With `synchronous=NORMAL` a commit is
/// not flushed by itself, so this is the most a host crash can take.
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(5);

/// How often the rows of `rejected` are counted again when nothing this
/// process did changed them: another process, or a person with the `sqlite3`
/// CLI, may have.
const RECOUNT_INTERVAL: Duration = if cfg!(test) {
    Duration::from_millis(200)
} else {
    Duration::from_secs(5)
};

/// `pending`: one row per report not yet accepted. `id` is never reused
/// (`AUTOINCREMENT`), so an outcome written late cannot land on another
/// report. A report is due when `next_attempt_at_ms` has passed and nobody
/// holds a lease on it (`lease_until_ms` has passed); `pending_due` serves
/// exactly that question, oldest first.
///
/// `rejected`: reports the billing API refused for good, or that used up an
/// explicit attempt cap. Nothing reads it but a person.
///
/// `pending_counts`: the rows of `pending` per `model_label`, kept by two
/// triggers so that reading the backlog never means counting a million rows,
/// and so that a row moved by hand is counted like any other.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS pending (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    body               TEXT NOT NULL,
    request_id         TEXT,
    model_label        TEXT,
    auth_path          TEXT NOT NULL,
    ingress_route      TEXT NOT NULL,
    completed_at_ms    INTEGER NOT NULL,
    attempts           INTEGER NOT NULL DEFAULT 0,
    next_attempt_at_ms INTEGER NOT NULL DEFAULT 0,
    lease_until_ms     INTEGER NOT NULL DEFAULT 0,
    lease_owner        TEXT,
    last_outcome       TEXT
);
CREATE INDEX IF NOT EXISTS pending_due ON pending (next_attempt_at_ms, id);
CREATE TABLE IF NOT EXISTS rejected (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rejected_at_ms  INTEGER NOT NULL,
    reason          TEXT NOT NULL,
    status          INTEGER,
    outcome         TEXT NOT NULL,
    attempts        INTEGER NOT NULL,
    body            TEXT NOT NULL,
    request_id      TEXT,
    model_label     TEXT,
    auth_path       TEXT NOT NULL,
    ingress_route   TEXT NOT NULL,
    completed_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_counts (
    model_label TEXT NOT NULL PRIMARY KEY,
    n           INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TRIGGER IF NOT EXISTS pending_counts_insert AFTER INSERT ON pending BEGIN
    INSERT OR IGNORE INTO pending_counts (model_label, n)
        VALUES (COALESCE(NEW.model_label, ''), 0);
    UPDATE pending_counts SET n = n + 1
        WHERE model_label = COALESCE(NEW.model_label, '');
END;
CREATE TRIGGER IF NOT EXISTS pending_counts_delete AFTER DELETE ON pending BEGIN
    UPDATE pending_counts SET n = n - 1
        WHERE model_label = COALESCE(OLD.model_label, '');
END;
PRAGMA user_version = 1;
";

/// The columns of a `pending` row that leave the store, in `Row::read` order.
const ROW_COLUMNS: &str = "id, body, request_id, model_label, auth_path, ingress_route, \
                           completed_at_ms, attempts, last_outcome, next_attempt_at_ms";

/// Where the outbox is and how much it may hold
/// (`VLLM_PROXY_USAGE_REPORT_OUTBOX_*`). The durations are not settings: they
/// are fields so that a test can shorten them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageOutboxConfig {
    /// The SQLite file (`VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH`). Created with
    /// mode 0600 when it does not exist; its directory must.
    pub path: PathBuf,
    /// Reports the file may hold while they wait to be accepted
    /// (`VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_PENDING`, default 1000000). Past
    /// it the report that has waited longest is dropped and counted.
    pub max_pending: usize,
    /// Rows kept in `rejected` (`VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_REJECTED`,
    /// default 10000). Past it the oldest is removed and counted.
    pub max_rejected: usize,
    /// How long the writer gathers reports before it commits them: a report
    /// handed over less than this before the process is killed may be lost.
    pub commit_interval: Duration,
    /// How long a transaction waits for another process's write lock.
    pub busy_timeout: Duration,
    /// How long the store is left alone after it failed, before opening it
    /// is tried again.
    pub reopen_interval: Duration,
    /// How often the file is looked at when nothing in this process says it
    /// changed: for reports another process left, leases that ran out and
    /// rows put back by hand.
    pub recheck_interval: Duration,
    /// What a lease lasts beyond two attempt timeouts (one for the attempt,
    /// one for the wait for a place before it).
    pub lease_margin: Duration,
    /// The longest a place stays taken after an attempt failed in a way that
    /// can pass.
    pub max_pause: Duration,
}

impl UsageOutboxConfig {
    pub const DEFAULT_MAX_PENDING: usize = 1_000_000;
    pub const DEFAULT_MAX_REJECTED: usize = 10_000;

    /// The outbox at `path` with every default.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            max_pending: Self::DEFAULT_MAX_PENDING,
            max_rejected: Self::DEFAULT_MAX_REJECTED,
            commit_interval: Duration::from_millis(50),
            busy_timeout: Duration::from_secs(5),
            reopen_interval: Duration::from_secs(5),
            recheck_interval: Duration::from_secs(2),
            lease_margin: Duration::from_secs(30),
            max_pause: Duration::from_secs(30),
        }
    }
}

/// Milliseconds since the Unix epoch. The file is shared by processes and
/// outlives them, so the times in it are wall-clock times.
pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis() as i64)
}

/// A report the store can keep.
pub(crate) trait Persist: Send + 'static {
    fn row(&self) -> NewRow<'_>;
}

/// What is written for a new report.
pub(crate) struct NewRow<'a> {
    /// The serialized report, as it is sent.
    pub body: &'a [u8],
    pub request_id: Option<&'a str>,
    pub model_label: Option<&'a str>,
    pub auth_path: &'a str,
    pub ingress_route: &'a str,
    /// How long ago its request completed.
    pub age: Duration,
}

/// A report as the store hands it out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub id: i64,
    pub body: String,
    pub request_id: Option<String>,
    pub model_label: Option<String>,
    pub auth_path: String,
    pub ingress_route: String,
    pub completed_at_ms: i64,
    /// Attempts made so far, by any process.
    pub attempts: u32,
    /// Why the last of them failed, when there was one.
    pub last_outcome: Option<String>,
    next_attempt_at_ms: i64,
}

impl Row {
    /// Read a row whatever is in it. A person may have written to the file
    /// by hand, and a row that could not be read would fail the transaction
    /// it is part of: every claim after it, for every other report.
    fn read(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        use rusqlite::types::ValueRef;
        let text = |column: usize| -> rusqlite::Result<Option<String>> {
            Ok(match row.get_ref(column)? {
                ValueRef::Null => None,
                ValueRef::Text(bytes) | ValueRef::Blob(bytes) => {
                    Some(String::from_utf8_lossy(bytes).into_owned())
                }
                ValueRef::Integer(number) => Some(number.to_string()),
                ValueRef::Real(number) => Some(number.to_string()),
            })
        };
        let integer = |column: usize| -> rusqlite::Result<i64> {
            Ok(match row.get_ref(column)? {
                ValueRef::Integer(number) => number,
                _ => 0,
            })
        };
        Ok(Self {
            id: row.get(0)?,
            body: text(1)?.unwrap_or_default(),
            request_id: text(2)?,
            model_label: text(3)?,
            auth_path: text(4)?.unwrap_or_default(),
            ingress_route: text(5)?.unwrap_or_default(),
            completed_at_ms: integer(6)?,
            attempts: integer(7)?.clamp(0, i64::from(u32::MAX)) as u32,
            last_outcome: text(8)?,
            next_attempt_at_ms: integer(9)?,
        })
    }
}

/// What became of a report this process held a lease on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Settle {
    /// It is done with: accepted, or dropped at its deadline.
    Delete { id: i64 },
    /// An attempt failed in a way that can pass. The lease is given back and
    /// the report is due again at `next_attempt_at_ms`.
    Retry {
        id: i64,
        attempts: u32,
        next_attempt_at_ms: i64,
        last_outcome: &'static str,
    },
    /// No attempt will be made again: the row moves to `rejected`.
    Reject {
        id: i64,
        reason: &'static str,
        status: Option<u16>,
        outcome: &'static str,
        attempts: u32,
    },
    /// It was not attempted after all: the lease is given back as it was.
    Release { id: i64 },
}

/// The backlog as of a transaction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Stats {
    /// Rows of `pending` per model label, waiting or being sent by anyone.
    pub pending: Vec<(Option<String>, u64)>,
    /// When the request of the oldest of them completed.
    pub oldest_completed_at_ms: Option<i64>,
    pub rejected: u64,
}

impl Stats {
    pub fn pending_total(&self) -> u64 {
        self.pending.iter().map(|(_, count)| count).sum()
    }
}

/// What a committed transaction did.
#[derive(Debug, Default)]
pub(crate) struct Committed {
    /// Reports written.
    pub stored: usize,
    /// Reports dropped to keep `pending` within its bound, oldest first.
    pub evicted: Vec<Row>,
    /// Rows removed from `rejected` to keep it within its bound.
    pub rejected_evicted: usize,
    /// The reports leased to this process, longest due first, when the
    /// transaction carried a claim.
    pub claimed: Option<Vec<Row>>,
    /// When the next report becomes due, if the claim found fewer than it
    /// asked for and one is waiting for its retry.
    pub next_due_ms: Option<i64>,
    pub stats: Stats,
}

/// What the writer has to say, in the order things happened.
pub(crate) enum Event<J> {
    Committed(Committed),
    /// Nothing was written. `error` says which step failed, when one was
    /// tried at all (the store is left alone for a while after a failure).
    /// The reports are handed back; outcomes are kept and written by the
    /// next transaction that succeeds.
    Failed {
        error: Option<(&'static str, String)>,
        reports: Vec<J>,
        /// The transaction carried a claim, which found nothing.
        claim: bool,
    },
    /// Something beside a transaction failed; the store goes on.
    Warning {
        op: &'static str,
        error: String,
    },
}

/// Why `offer` did not take a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The store is not usable at the moment.
    Unavailable,
    /// As many reports as allowed already wait to be written.
    BufferFull,
}

impl Refused {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Unavailable => "unavailable",
            Self::BufferFull => "buffer_full",
        }
    }
}

struct Claim {
    limit: usize,
    lease: Duration,
}

enum Stop {
    /// Write what is left, give this process's leases back, answer.
    Close(oneshot::Sender<Option<Stats>>),
    /// The handle is gone: write what is left and end.
    Abandon,
}

struct Inbox<J> {
    reports: Vec<J>,
    /// When the oldest of `reports` was handed over.
    first_report_at: Option<Instant>,
    /// Reports the writer took from here and has not put in a transaction
    /// yet (`Writer::backlog`). They still count as waiting to be written.
    held_by_writer: usize,
    settles: Vec<Settle>,
    claim: Option<Claim>,
    /// A transaction is wanted now, even an empty one: for fresh numbers,
    /// or to find out whether the store is back.
    sync: bool,
    /// How the writer is to end, until it has read it.
    stop: Option<Stop>,
    /// The writer was told to end: nothing more is taken.
    stopping: bool,
}

struct Shared<J> {
    config: UsageOutboxConfig,
    /// Who holds the leases this process takes: unique per handle.
    owner: String,
    inbox: Mutex<Inbox<J>>,
    work: Condvar,
    available: AtomicBool,
    events: mpsc::UnboundedSender<Event<J>>,
    /// Makes opening and every transaction fail, as a full disk would.
    #[cfg(test)]
    fault: AtomicBool,
}

impl<J> Shared<J> {
    fn inbox(&self) -> MutexGuard<'_, Inbox<J>> {
        // Nothing panics while holding the lock; if something ever does, the
        // inbox is still what there is.
        self.inbox.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fault(&self) -> bool {
        #[cfg(test)]
        return self.fault.load(Ordering::Relaxed);
        #[cfg(not(test))]
        false
    }
}

/// The handle on the store: every method is a lock and a push, and none
/// touches the file.
pub(crate) struct Store<J: Persist> {
    shared: Arc<Shared<J>>,
}

impl<J: Persist> Store<J> {
    /// Start the writer, which opens (or creates) the file at once. Whatever
    /// happens to the file is told through `events`; this never fails.
    pub fn start(config: UsageOutboxConfig, events: mpsc::UnboundedSender<Event<J>>) -> Self {
        let shared = Arc::new(Shared {
            config,
            owner: uuid::Uuid::new_v4().to_string(),
            inbox: Mutex::new(Inbox {
                reports: Vec::new(),
                first_report_at: None,
                held_by_writer: 0,
                settles: Vec::new(),
                claim: None,
                // The first transaction opens the file and reads the backlog
                // a previous process left.
                sync: true,
                stop: None,
                stopping: false,
            }),
            work: Condvar::new(),
            // Until the file says otherwise: a report handed over before it
            // is open waits for it rather than going around it.
            available: AtomicBool::new(true),
            events,
            #[cfg(test)]
            fault: AtomicBool::new(false),
        });
        let writer = Writer::new(shared.clone());
        let spawned = std::thread::Builder::new()
            .name("usage-outbox".to_string())
            .spawn(move || writer.run());
        if let Err(error) = spawned {
            // Without its thread the store never opens: reports are refused
            // here from the start and delivered from memory.
            shared.available.store(false, Ordering::Relaxed);
            let _ = shared.events.send(Event::Failed {
                error: Some(("open", format!("no thread for the writer: {error}"))),
                reports: Vec::new(),
                claim: false,
            });
        }
        Self { shared }
    }

    pub fn config(&self) -> &UsageOutboxConfig {
        &self.shared.config
    }

    /// Whether the last thing tried on the file worked.
    pub fn is_available(&self) -> bool {
        self.shared.available.load(Ordering::Relaxed)
    }

    /// Hand a report over to be written. Never blocks on the file: the report
    /// is taken, or given back when the store is unavailable or
    /// `max_buffered` reports already wait to be written.
    pub fn offer(&self, report: J, max_buffered: usize) -> Result<(), (J, Refused)> {
        if !self.is_available() {
            return Err((report, Refused::Unavailable));
        }
        let wake = {
            let mut inbox = self.shared.inbox();
            if inbox.stopping {
                return Err((report, Refused::Unavailable));
            }
            if inbox.reports.len() + inbox.held_by_writer >= max_buffered {
                return Err((report, Refused::BufferFull));
            }
            inbox.reports.push(report);
            // The writer sleeps through the reports that join a batch: it is
            // woken by the first, which starts the interval, and when the
            // batch is as large as a transaction gets.
            let first = inbox.reports.len() == 1;
            if first {
                inbox.first_report_at = Some(Instant::now());
            }
            first || inbox.reports.len() == MAX_BATCH
        };
        if wake {
            self.shared.work.notify_one();
        }
        Ok(())
    }

    /// Record what became of a leased report. Written at once.
    pub fn settle(&self, settle: Settle) {
        self.shared.inbox().settles.push(settle);
        self.shared.work.notify_one();
    }

    /// Ask for up to `limit` due reports, leased to this process for `lease`.
    /// The answer is the `claimed` of a `Committed`, or a `Failed` with
    /// `claim` set.
    pub fn claim(&self, limit: usize, lease: Duration) {
        self.shared.inbox().claim = Some(Claim { limit, lease });
        self.shared.work.notify_one();
    }

    /// Ask for a transaction now: its `Committed` carries fresh numbers, and
    /// when the store was unavailable this is what tries it again.
    pub fn sync(&self) {
        self.shared.inbox().sync = true;
        self.shared.work.notify_one();
    }

    /// Write what is left, give back every lease this process holds and stop
    /// the writer. The answer is the backlog left in the file, `None` when
    /// the last transaction failed.
    pub fn close(&self) -> oneshot::Receiver<Option<Stats>> {
        let (done, closed) = oneshot::channel();
        {
            let mut inbox = self.shared.inbox();
            // A second call finds the writer gone, and its answer with it.
            if !inbox.stopping {
                inbox.stopping = true;
                inbox.stop = Some(Stop::Close(done));
            }
        }
        self.shared.work.notify_one();
        closed
    }

    #[cfg(test)]
    pub fn inject_fault(&self, on: bool) {
        self.shared.fault.store(on, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub fn owner(&self) -> &str {
        &self.shared.owner
    }
}

impl<J: Persist> Drop for Store<J> {
    /// The writer ends with its handle. It is not waited for (this may well
    /// run on the writer's own thread, when the last report it wrote held
    /// the last reference), and leases are left to run out: a process that
    /// stops in good order calls `close` first.
    fn drop(&mut self) {
        let mut inbox = self.shared.inbox();
        if !inbox.stopping {
            inbox.stopping = true;
            inbox.stop = Some(Stop::Abandon);
        }
        drop(inbox);
        self.shared.work.notify_one();
    }
}

/// What the writer took from the inbox for one transaction.
struct Work<J> {
    reports: Vec<J>,
    settles: Vec<Settle>,
    claim: Option<Claim>,
    stop: Option<Stop>,
}

/// A failed step, by name.
struct Failure {
    op: &'static str,
    error: String,
}

fn failed<E: std::fmt::Display>(op: &'static str) -> impl Fn(E) -> Failure {
    move |error| Failure {
        op,
        error: error.to_string(),
    }
}

/// The thread that owns the connection.
struct Writer<J> {
    shared: Arc<Shared<J>>,
    conn: Option<Connection>,
    /// Reports taken from the inbox that did not fit the transaction of the
    /// moment (`MAX_BATCH`), oldest first. Taking everything and dividing it
    /// here keeps the inbox lock, which a request may be waiting for, to a
    /// swap of two pointers.
    backlog: VecDeque<J>,
    /// Outcomes a failed transaction did not write. They are written by the
    /// next one that succeeds; until then their reports stay leased, so at
    /// worst they are sent once more when the lease runs out. Never more
    /// than the reports this process holds a lease on.
    unwritten: Vec<Settle>,
    /// Not before this is the file opened again after a failure.
    retry_at: Instant,
    /// The rows of `rejected`, and when they were last counted.
    rejected: Option<(u64, Instant)>,
    /// Something was committed since the last checkpoint.
    dirty: bool,
    checkpointed_at: Instant,
}

impl<J: Persist> Writer<J> {
    fn new(shared: Arc<Shared<J>>) -> Self {
        Self {
            shared,
            conn: None,
            backlog: VecDeque::new(),
            unwritten: Vec::new(),
            retry_at: Instant::now(),
            rejected: None,
            dirty: false,
            checkpointed_at: Instant::now(),
        }
    }

    fn run(mut self) {
        loop {
            let mut work = self.wait_for_work();
            let stop = work.stop.take();
            let stats = self.serve(work, matches!(stop, Some(Stop::Close(_))));
            match stop {
                Some(Stop::Close(done)) => {
                    // Closing the last connection folds the log into the
                    // file; the answer is sent once that is done.
                    self.conn = None;
                    let _ = done.send(stats);
                    return;
                }
                Some(Stop::Abandon) => return,
                None => {}
            }
        }
    }

    /// Sleep until there is something to do, and take it. Reports alone wait
    /// until the oldest has been there for `commit_interval`, or until there
    /// are enough for a transaction; anything else is served at once, with
    /// whatever reports are there. Nothing is waited for while reports of an
    /// earlier round are left.
    fn wait_for_work(&mut self) -> Work<J> {
        let interval = self.shared.config.commit_interval;
        let mut inbox = self.shared.inbox();
        if self.backlog.is_empty() {
            loop {
                if inbox.stop.is_some()
                    || inbox.claim.is_some()
                    || inbox.sync
                    || !inbox.settles.is_empty()
                    || inbox.reports.len() >= MAX_BATCH
                {
                    break;
                }
                match inbox.first_report_at {
                    Some(first) => {
                        let left = interval.saturating_sub(first.elapsed());
                        if left.is_zero() {
                            break;
                        }
                        inbox = self
                            .shared
                            .work
                            .wait_timeout(inbox, left)
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                    }
                    None => {
                        inbox = self
                            .shared
                            .work
                            .wait(inbox)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                }
            }
            self.backlog = std::mem::take(&mut inbox.reports).into();
            inbox.first_report_at = None;
        } else if inbox.stop.is_some() {
            self.backlog.extend(std::mem::take(&mut inbox.reports));
            inbox.first_report_at = None;
        }
        // A writer that is ending writes all it has.
        let take = if inbox.stop.is_some() {
            self.backlog.len()
        } else {
            self.backlog.len().min(MAX_BATCH)
        };
        inbox.held_by_writer = self.backlog.len() - take;
        inbox.sync = false;
        let (settles, claim, stop) = (
            std::mem::take(&mut inbox.settles),
            inbox.claim.take(),
            inbox.stop.take(),
        );
        drop(inbox);
        Work {
            reports: self.backlog.drain(..take).collect(),
            settles,
            claim,
            stop,
        }
    }

    /// One transaction for `work`, or the report that there was none. Returns
    /// the backlog when it committed.
    fn serve(&mut self, mut work: Work<J>, release_leases: bool) -> Option<Stats> {
        let claim = work.claim.is_some();
        self.unwritten.append(&mut work.settles);
        if self.conn.is_none() {
            if Instant::now() < self.retry_at {
                self.hand_back(None, work.reports, claim);
                return None;
            }
            match self.open() {
                Ok(conn) => self.conn = Some(conn),
                Err(failure) => {
                    self.hand_back(Some(failure), work.reports, claim);
                    return None;
                }
            }
        }
        match self.transact(&work, release_leases) {
            Ok(committed) => {
                // A transaction that only looked wrote nothing to flush.
                self.dirty |= release_leases
                    || committed.stored > 0
                    || !self.unwritten.is_empty()
                    || committed
                        .claimed
                        .as_ref()
                        .is_some_and(|rows| !rows.is_empty());
                self.unwritten.clear();
                self.shared.available.store(true, Ordering::Relaxed);
                let stats = committed.stats.clone();
                let _ = self.shared.events.send(Event::Committed(committed));
                self.checkpoint_when_due();
                Some(stats)
            }
            Err(failure) => {
                // Whatever state the connection is in, the next try starts
                // from a new one.
                self.conn = None;
                self.hand_back(Some(failure), work.reports, claim);
                None
            }
        }
    }

    fn hand_back(&mut self, failure: Option<Failure>, reports: Vec<J>, claim: bool) {
        self.shared.available.store(false, Ordering::Relaxed);
        if failure.is_some() {
            self.retry_at = Instant::now() + self.shared.config.reopen_interval;
        }
        let _ = self.shared.events.send(Event::Failed {
            error: failure.map(|failure| (failure.op, failure.error)),
            reports,
            claim,
        });
    }

    fn open(&self) -> Result<Connection, Failure> {
        if self.shared.fault() {
            return Err(failed("open")("injected fault"));
        }
        open(&self.shared.config)
    }

    /// Everything `work` asks for, in one transaction: the outcomes first
    /// (they free rows), then the new reports, the bound, the claim (which
    /// may take a report written a line above) and the numbers.
    fn transact(&mut self, work: &Work<J>, release_leases: bool) -> Result<Committed, Failure> {
        if self.shared.fault() {
            return Err(failed("begin")("injected fault"));
        }
        let config = &self.shared.config;
        let owner = self.shared.owner.as_str();
        let conn = self.conn.as_mut().expect("opened by the caller");
        let now = now_ms();
        // The write lock is taken at once: a transaction that started by
        // reading could not wait for it later (SQLite fails such an upgrade
        // without consulting the busy timeout).
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(failed("begin"))?;
        let mut committed = Committed::default();

        let mut rejected_changed = false;
        for settle in &self.unwritten {
            match settle {
                Settle::Delete { id } => {
                    tx.prepare_cached("DELETE FROM pending WHERE id = ?1")
                        .and_then(|mut statement| statement.execute([id]))
                        .map_err(failed("settle"))?;
                }
                // Only while this process still holds the lease: once it ran
                // out the report may be another process's, and its lease is
                // not ours to give back.
                Settle::Retry {
                    id,
                    attempts,
                    next_attempt_at_ms,
                    last_outcome,
                } => {
                    tx.prepare_cached(
                        "UPDATE pending SET attempts = ?2, next_attempt_at_ms = ?3, \
                         last_outcome = ?4, lease_until_ms = 0, lease_owner = NULL \
                         WHERE id = ?1 AND lease_owner = ?5",
                    )
                    .and_then(|mut statement| {
                        statement.execute(params![
                            id,
                            attempts,
                            next_attempt_at_ms,
                            last_outcome,
                            owner
                        ])
                    })
                    .map_err(failed("settle"))?;
                }
                Settle::Reject {
                    id,
                    reason,
                    status,
                    outcome,
                    attempts,
                } => {
                    let moved = tx
                        .prepare_cached(
                            "INSERT INTO rejected (rejected_at_ms, reason, status, outcome, \
                             attempts, body, request_id, model_label, auth_path, ingress_route, \
                             completed_at_ms) \
                             SELECT ?2, ?3, ?4, ?5, ?6, body, request_id, model_label, auth_path, \
                             ingress_route, completed_at_ms FROM pending WHERE id = ?1",
                        )
                        .and_then(|mut statement| {
                            statement.execute(params![id, now, reason, status, outcome, attempts])
                        })
                        .map_err(failed("settle"))?;
                    tx.prepare_cached("DELETE FROM pending WHERE id = ?1")
                        .and_then(|mut statement| statement.execute([id]))
                        .map_err(failed("settle"))?;
                    rejected_changed |= moved > 0;
                }
                Settle::Release { id } => {
                    tx.prepare_cached(
                        "UPDATE pending SET lease_until_ms = 0, lease_owner = NULL \
                         WHERE id = ?1 AND lease_owner = ?2",
                    )
                    .and_then(|mut statement| statement.execute(params![id, owner]))
                    .map_err(failed("settle"))?;
                }
            }
        }
        if rejected_changed {
            // All but the newest `max_rejected`.
            committed.rejected_evicted = tx
                .prepare_cached(
                    "DELETE FROM rejected WHERE id <= \
                     (SELECT id FROM rejected ORDER BY id DESC LIMIT 1 OFFSET ?1)",
                )
                .and_then(|mut statement| statement.execute([config.max_rejected as i64]))
                .map_err(failed("settle"))?;
        }

        if !work.reports.is_empty() {
            // Due from the moment its request completed, so that reports
            // come up in the order they completed.
            let mut insert = tx
                .prepare_cached(
                    "INSERT INTO pending (body, request_id, model_label, auth_path, \
                     ingress_route, completed_at_ms, next_attempt_at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
                )
                .map_err(failed("insert"))?;
            for report in &work.reports {
                let row = report.row();
                let completed_at_ms = now - row.age.as_millis().min(i64::MAX as u128) as i64;
                insert
                    .execute(params![
                        String::from_utf8_lossy(row.body),
                        row.request_id,
                        row.model_label,
                        row.auth_path,
                        row.ingress_route,
                        completed_at_ms
                    ])
                    .map_err(failed("insert"))?;
            }
            committed.stored = work.reports.len();

            let pending: i64 = tx
                .prepare_cached("SELECT COALESCE(SUM(n), 0) FROM pending_counts")
                .and_then(|mut statement| statement.query_row([], |row| row.get(0)))
                .map_err(failed("evict"))?;
            let excess = pending - config.max_pending.min(i64::MAX as usize) as i64;
            if excess > 0 {
                // The oldest by id, which is the order reports were written
                // in. A report somebody is sending right now is left alone:
                // its answer is on its way.
                let mut evict = tx
                    .prepare_cached(&format!(
                        "DELETE FROM pending WHERE id IN (SELECT id FROM pending \
                         WHERE lease_until_ms <= ?1 ORDER BY id LIMIT ?2) \
                         RETURNING {ROW_COLUMNS}"
                    ))
                    .map_err(failed("evict"))?;
                let evicted = evict
                    .query_map(params![now, excess], Row::read)
                    .and_then(Iterator::collect::<rusqlite::Result<Vec<_>>>)
                    .map_err(failed("evict"))?;
                committed.evicted = evicted;
                committed.evicted.sort_by_key(|row| row.id);
            }
        }

        if let Some(claim) = &work.claim {
            let lease_until = now + claim.lease.as_millis().min(i64::MAX as u128) as i64;
            let mut take = tx
                .prepare_cached(&format!(
                    "UPDATE pending SET lease_until_ms = ?1, lease_owner = ?2 \
                     WHERE id IN (SELECT id FROM pending \
                     WHERE next_attempt_at_ms <= ?3 AND lease_until_ms <= ?3 \
                     ORDER BY next_attempt_at_ms, id LIMIT ?4) \
                     RETURNING {ROW_COLUMNS}"
                ))
                .map_err(failed("claim"))?;
            let mut claimed = take
                .query_map(
                    params![lease_until, owner, now, claim.limit as i64],
                    Row::read,
                )
                .and_then(Iterator::collect::<rusqlite::Result<Vec<_>>>)
                .map_err(failed("claim"))?;
            claimed.sort_by_key(|row| (row.next_attempt_at_ms, row.id));
            if claimed.len() < claim.limit {
                committed.next_due_ms = tx
                    .prepare_cached(
                        "SELECT MIN(next_attempt_at_ms) FROM pending WHERE next_attempt_at_ms > ?1",
                    )
                    .and_then(|mut statement| statement.query_row([now], |row| row.get(0)))
                    .map_err(failed("claim"))?;
            }
            committed.claimed = Some(claimed);
        }

        if release_leases {
            tx.prepare_cached(
                "UPDATE pending SET lease_until_ms = 0, lease_owner = NULL WHERE lease_owner = ?1",
            )
            .and_then(|mut statement| statement.execute([owner]))
            .map_err(failed("release"))?;
        }

        committed.stats.pending = tx
            // A model whose rows are all gone is still listed, with 0, so
            // that whoever reads this sees its count come down.
            .prepare_cached("SELECT model_label, n FROM pending_counts")
            .and_then(|mut statement| {
                statement
                    .query_map([], |row| {
                        let label: String = row.get(0)?;
                        let count: i64 = row.get(1)?;
                        Ok(((!label.is_empty()).then_some(label), count.max(0) as u64))
                    })
                    .and_then(Iterator::collect::<rusqlite::Result<Vec<_>>>)
            })
            .map_err(failed("stats"))?;
        committed.stats.oldest_completed_at_ms = tx
            .prepare_cached("SELECT completed_at_ms FROM pending ORDER BY id LIMIT 1")
            .and_then(|mut statement| {
                let mut rows = statement.query([])?;
                rows.next()?.map(|row| row.get(0)).transpose()
            })
            .map_err(failed("stats"))?;
        let counted = self
            .rejected
            .filter(|(_, at)| !rejected_changed && at.elapsed() < RECOUNT_INTERVAL);
        let rejected = match counted {
            Some((rejected, _)) => rejected,
            None => {
                let rejected: i64 = tx
                    .prepare_cached("SELECT COUNT(*) FROM rejected")
                    .and_then(|mut statement| statement.query_row([], |row| row.get(0)))
                    .map_err(failed("stats"))?;
                rejected.max(0) as u64
            }
        };
        committed.stats.rejected = rejected;

        tx.commit().map_err(failed("commit"))?;
        if counted.is_none() {
            self.rejected = Some((rejected, Instant::now()));
        }
        Ok(committed)
    }

    /// Flush the write-ahead log and fold it into the file, every
    /// `CHECKPOINT_INTERVAL` while something was written. Passive: it does
    /// what it can without waiting for anyone.
    fn checkpoint_when_due(&mut self) {
        if !self.dirty || self.checkpointed_at.elapsed() < CHECKPOINT_INTERVAL {
            return;
        }
        let Some(conn) = &self.conn else { return };
        self.checkpointed_at = Instant::now();
        match conn.query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |_| Ok(())) {
            Ok(()) => self.dirty = false,
            Err(error) => {
                let _ = self.shared.events.send(Event::Warning {
                    op: "checkpoint",
                    error: error.to_string(),
                });
            }
        }
    }
}

/// Open the file, creating it and its schema when they are not there.
fn open(config: &UsageOutboxConfig) -> Result<Connection, Failure> {
    create_private(&config.path).map_err(failed("open"))?;
    let conn = Connection::open_with_flags(
        &config.path,
        // One thread ever uses the connection, so SQLite need not lock it.
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(failed("open"))?;
    conn.busy_timeout(config.busy_timeout)
        .map_err(failed("open"))?;
    // WAL: the other process on the file can read while this one writes, and
    // a commit is one append. It is a property of the file and stays set.
    let mode: String = conn
        .pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))
        .map_err(failed("open"))?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(failed("open")(format!(
            "journal mode is {mode}, not WAL (a file system without shared memory?)"
        )));
    }
    // NORMAL: a commit is written, not flushed. It survives the process (a
    // crash, a kill), and the log is flushed at every checkpoint, so a host
    // crash can only take the last seconds; the file itself cannot be
    // corrupted by either. FULL would flush on every commit.
    conn.pragma_update(None, "synchronous", "NORMAL")
        .map_err(failed("open"))?;
    migrate(&conn).map_err(failed("open"))?;
    Ok(conn)
}

/// Bring the schema to `SCHEMA_VERSION`. A file written by a newer version
/// of the proxy is refused rather than guessed at.
fn migrate(conn: &Connection) -> Result<(), String> {
    let version = |conn: &Connection| -> rusqlite::Result<i64> {
        conn.pragma_query_value(None, "user_version", |row| row.get(0))
    };
    let found = version(conn).map_err(|error| error.to_string())?;
    if found > SCHEMA_VERSION {
        return Err(format!(
            "the file has schema version {found}, this proxy knows {SCHEMA_VERSION}"
        ));
    }
    if found == SCHEMA_VERSION {
        return Ok(());
    }
    // Two processes may get here together; the second finds everything made.
    conn.execute_batch(&format!("BEGIN IMMEDIATE;{SCHEMA}COMMIT;"))
        .map_err(|error| {
            let _ = conn.execute_batch("ROLLBACK");
            error.to_string()
        })
}

/// Make sure the file exists and nobody but its owner can read it. SQLite
/// gives the write-ahead log and the shared-memory file beside it the mode
/// of the file itself.
fn create_private(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)?;
        if file.metadata()?.permissions().mode() & 0o077 != 0 {
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
#[path = "usage_outbox_tests.rs"]
mod tests;
