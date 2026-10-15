//! The durable outbox of usage reports: a SQLite file that keeps every report
//! from the completion of its request until the billing API has accepted it
//! or refused it for good (`VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH`).
//!
//! This module is the store and nothing else. `usage_report_outbox.rs`
//! decides what becomes of a report; here it is written down, leased out for
//! an attempt and removed.
//!
//! SQLite calls block, so all of them are made by one thread of this module's
//! own, the writer, which owns the connection. Everything else talks to it
//! through an inbox in memory: handing a report over is a lock and a push,
//! whatever the disk is doing, and nothing here is ever awaited or called on
//! a runtime worker. The writer gathers what is in the inbox into one
//! transaction (group commit): reports and outcomes handed over within
//! `commit_interval` of each other are written together, and so is the claim
//! that is waiting at that moment.
//!
//! The file is shared state. Two processes may have it open at once (a
//! blue/green switch overlaps the old process and the new one), so a report
//! being sent is marked with a lease: its holder and a time after which
//! anyone else may take it. A process that dies leaves its leases to run out.
//!
//! What a row holds is what sending the report again takes and what its log
//! lines and metrics need: the serialized report, the request id, three
//! labels, when its request completed and how its attempts went. Never the
//! bearer token, which is read from the configuration when a report is sent,
//! and never anything of a request or a response.
//!
//! The store must never cost the process anything else:
//!
//! - No part of the file is ever mapped into memory. The journal is a
//!   rollback journal (`journal_mode=TRUNCATE`, `mmap_size=0`), not a
//!   write-ahead log, whose index SQLite maps: a mapped page that cannot be
//!   read (the file truncated under the process, a disk error) is a signal
//!   that ends the process, where a failed `read` is an error code.
//! - When the file cannot be opened or written, the reports of that
//!   transaction are handed back to be delivered from memory, and the file
//!   is tried again every `reopen_interval`. A file that is corrupt is moved
//!   aside for a person and a new one started.
//! - A file deleted or replaced under the process is noticed before every
//!   transaction, and by SQLite when one is under way (it does not write to
//!   a database whose file was moved). The connection is let go of at once
//!   and never used again, not even to read what it still holds: SQLite
//!   finds a rollback journal by the path of its database, so a connection
//!   whose file is no longer at the path takes the journal of whatever is
//!   there now for its own, and plays it into its file, which ruins both.
//!   What the file held is said and counted as lost (`Writer::let_go`), the
//!   reports this process has in hand are written to the file that is
//!   started or found at the path, and nothing is ever copied back to the
//!   path as a file.
//! - The file has a size of its own (`max_bytes`) below that of its volume,
//!   so that it is full while the volume still has room for the journal:
//!   leasing and removing reports then go on, and only new reports are
//!   refused. On a volume too small for `max_bytes` the size is what the
//!   volume has room for (`fitting`), looked at again as the volume changes.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{ffi, params, Connection, OpenFlags, TransactionBehavior};
use tokio::sync::{mpsc, oneshot};

/// `PRAGMA user_version` of the schema below.
const SCHEMA_VERSION: i64 = 1;

/// Reports written by one transaction at most. A longer backlog takes several,
/// so the write lock, which another process may be waiting for, is never held
/// for long.
const MAX_BATCH: usize = 2_000;

/// Reports moved to `rejected` for their age, or dropped at the bound, by one
/// transaction at most.
const SWEEP_BATCH: usize = 500;

/// Rows removed from `rejected` at its bound by one transaction at most:
/// more than one transaction can add to it (new reports that had ended in
/// memory, reports too old), so the table is back at its bound however fast
/// it is filled, and what a transaction hands out to be said row by row
/// stays a megabyte or two.
const EVICT_BATCH: usize = MAX_BATCH + SWEEP_BATCH + 500;

/// New pages one report can take at the very worst, besides those its body
/// overflows into: a page of the table and one of each index it is in, when
/// they all split at once. A report takes a fraction of one page as a rule;
/// this is what room is kept for between two looks at the file's size.
const WORST_PAGES_PER_REPORT: i64 = 4;

/// With fewer pages than this left for new reports, the file is full, and
/// it stays full until it has twice as many: what is done to the reports in
/// it takes and frees a page now and then, and a file at the line must not
/// be full and not full by turns.
const ROOM_PAGES: i64 = 32;

/// By how many pages a file on a volume that was full is made to grow, to
/// find out whether it can again.
const ROOM_PROBE_PAGES: i64 = 16;

/// `pending`: one row per report not yet accepted. `id` is never reused
/// within a database (`AUTOINCREMENT`), and an outcome is written only to a
/// row its sender still holds the lease of, handed out by the database that
/// is open now (`Settle::generation`): so an outcome written late cannot
/// land on another report, in this database or in the one that took its
/// place. A report never tried (`attempts = 0`) is due from the moment it
/// is written; one that failed is due again at `next_attempt_at_ms`. The two
/// are asked for separately (`pending_fresh`, `pending_retry`), because a
/// retry must never stand in front of a first attempt.
///
/// `rejected`: reports the billing API refused for good, or that grew older
/// than a report may. Nothing reads it but a person.
///
/// `pending_counts`, `rejected_counts`: the rows of the two tables, per model
/// label and per reason, kept by triggers so that reading the backlog never
/// means counting a million rows, and so that a row moved by hand is counted
/// like any other.
///
/// `meta`: `db_id` names this database, whichever file holds it (one started
/// after a file was deleted or moved aside is another), and `probed_at_ms`
/// is the row written to find out whether the file takes writes.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS pending (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    body               TEXT NOT NULL,
    request_id         TEXT,
    model_label        TEXT,
    auth_path          TEXT NOT NULL,
    ingress_route      TEXT NOT NULL,
    completed_at_ms    INTEGER NOT NULL,
    attempts           INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at_ms INTEGER NOT NULL DEFAULT 0,
    lease_until_ms     INTEGER NOT NULL DEFAULT 0,
    lease_owner        TEXT,
    last_outcome       TEXT
);
CREATE INDEX IF NOT EXISTS pending_fresh ON pending (next_attempt_at_ms, id) WHERE attempts = 0;
CREATE INDEX IF NOT EXISTS pending_retry ON pending (next_attempt_at_ms, id) WHERE attempts > 0;
CREATE INDEX IF NOT EXISTS pending_completed ON pending (completed_at_ms, id);
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
CREATE TABLE IF NOT EXISTS rejected_counts (
    reason TEXT NOT NULL PRIMARY KEY,
    n      INTEGER NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT NOT NULL PRIMARY KEY,
    value
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
CREATE TRIGGER IF NOT EXISTS rejected_counts_insert AFTER INSERT ON rejected BEGIN
    INSERT OR IGNORE INTO rejected_counts (reason, n) VALUES (NEW.reason, 0);
    UPDATE rejected_counts SET n = n + 1 WHERE reason = NEW.reason;
END;
CREATE TRIGGER IF NOT EXISTS rejected_counts_delete AFTER DELETE ON rejected BEGIN
    UPDATE rejected_counts SET n = n - 1 WHERE reason = OLD.reason;
END;
PRAGMA user_version = 1;
";

/// The columns of a `pending` row that leave the store, in `Row::read` order.
const ROW_COLUMNS: &str = "id, body, request_id, model_label, auth_path, ingress_route, \
                           completed_at_ms, attempts, last_outcome, next_attempt_at_ms";

/// The time the file's rows are stamped with: the system's wall clock, which
/// is what two processes and two runs of one process have in common. A test
/// steps it to see what a clock that jumps does.
#[derive(Clone, Default)]
pub struct Clock {
    step_ms: Arc<AtomicI64>,
}

impl Clock {
    /// Milliseconds since the Unix epoch.
    pub fn now_ms(&self) -> i64 {
        let system = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_millis() as i64);
        system.saturating_add(self.step_ms.load(Ordering::Relaxed))
    }

    /// Move this clock, and every clone of it, forward (or back) from here
    /// on, as an operator or a time service moving the system clock would.
    pub fn step(&self, by_ms: i64) {
        self.step_ms.fetch_add(by_ms, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for Clock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Clock(+{} ms)", self.step_ms.load(Ordering::Relaxed))
    }
}

impl PartialEq for Clock {
    /// Two configurations do not differ by their clock.
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Clock {}

/// Where the outbox is and how much it may hold
/// (`VLLM_PROXY_USAGE_REPORT_OUTBOX_*`). What is not a setting is a field all
/// the same, so that a test can shorten it.
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
    /// default 100000). Past it the oldest is removed and counted.
    pub max_rejected: usize,
    /// The size the file may grow to
    /// (`VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_BYTES`, default 1 GiB). The
    /// volume must be larger: see `VOLUME_MARGIN_BYTES` and
    /// docs/gateway-mode.md.
    pub max_bytes: u64,
    /// How old a report may grow before it is moved to `rejected`
    /// (`VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_AGE_SECS`, default 7 days), so
    /// that nothing is tried for ever.
    pub max_age: Duration,
    /// How long the writer gathers reports and outcomes before it commits
    /// them: a report handed over less than this before the process is
    /// killed may be lost.
    pub commit_interval: Duration,
    /// How long a transaction waits for another process's lock.
    pub busy_timeout: Duration,
    /// How long the file is left alone after it failed, before it is tried
    /// again.
    pub reopen_interval: Duration,
    /// How often the file is looked at when nothing in this process says it
    /// changed: for reports another process left, leases that ran out and
    /// rows put back by hand.
    pub recheck_interval: Duration,
    /// What a lease lasts beyond two attempt timeouts (one for the attempt,
    /// one for the wait for a place before it).
    pub lease_margin: Duration,
    /// Longer than any lease any process takes. A lease that ends further
    /// ahead than this was written under another clock and counts as over.
    pub max_lease: Duration,
    /// The longest a report waits between two attempts.
    pub max_backoff: Duration,
    /// The longest the process waits between two probes of a billing API
    /// that is not answering: what an outage costs it in requests, and how
    /// long after its end the backlog starts to go out.
    pub max_pause: Duration,
    /// First attempts that failed in a row, with no report accepted or
    /// refused in between, after which the billing API counts as not
    /// answering.
    pub breaker_after: u32,
    /// When the failures in a row are not all first attempts, how long
    /// nothing must have been answered as well: reports that failed before
    /// fail again without that saying anything about the billing API.
    pub breaker_window: Duration,
    /// How long the writer may take over a transaction while work waits for
    /// it before the store counts as unavailable. Longer than `busy_timeout`.
    pub stall_timeout: Duration,
    /// How long shutdown waits for the writer to write what is left and
    /// give its leases back. The disk may be why the process is stopping.
    pub close_timeout: Duration,
    pub clock: Clock,
}

impl UsageOutboxConfig {
    pub const DEFAULT_MAX_PENDING: usize = 1_000_000;
    pub const DEFAULT_MAX_REJECTED: usize = 100_000;
    pub const DEFAULT_MAX_BYTES: u64 = 1 << 30;
    pub const DEFAULT_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 3600);
    /// The least `max_bytes` can be: room for the schema, the headroom and a
    /// few thousand reports.
    pub const MIN_MAX_BYTES: u64 = 4 << 20;
    /// What the volume must have beyond `max_bytes`: the rollback journal of
    /// the largest transaction, with a margin.
    pub const VOLUME_MARGIN_BYTES: u64 = 64 << 20;

    /// The outbox at `path` with every default.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            max_pending: Self::DEFAULT_MAX_PENDING,
            max_rejected: Self::DEFAULT_MAX_REJECTED,
            max_bytes: Self::DEFAULT_MAX_BYTES,
            max_age: Self::DEFAULT_MAX_AGE,
            commit_interval: Duration::from_millis(50),
            busy_timeout: Duration::from_secs(5),
            reopen_interval: Duration::from_secs(5),
            recheck_interval: Duration::from_secs(2),
            lease_margin: Duration::from_secs(30),
            // The attempt timeout is at most 300 s (`config.rs`).
            max_lease: Duration::from_secs(2 * 300 + 60),
            max_backoff: Duration::from_secs(600),
            max_pause: Duration::from_secs(5),
            breaker_after: 20,
            breaker_window: Duration::from_secs(5),
            stall_timeout: Duration::from_secs(15),
            close_timeout: Duration::from_secs(10),
            clock: Clock::default(),
        }
    }
}

/// A report the store can keep. Cheap to clone: the writer keeps a copy of
/// what it is writing where `take_over` finds it.
pub(crate) trait Persist: Clone + Send + 'static {
    fn row(&self) -> NewRow<'_>;
}

/// What is written for a report handed over.
pub(crate) struct NewRow<'a> {
    /// The serialized report, as it is sent.
    pub body: &'a [u8],
    pub request_id: Option<&'a str>,
    pub model_label: Option<&'a str>,
    pub auth_path: &'a str,
    pub ingress_route: &'a str,
    /// How long ago its request completed.
    pub age: Duration,
    /// Attempts already made, when it was sent from memory before the file
    /// could take it, and why the last of them failed.
    pub attempts: u32,
    pub last_outcome: Option<&'a str>,
    /// When it is due again, from now. Zero for a report never tried.
    pub due_in: Duration,
    /// It has ended already, without the file to keep it in: written to
    /// `rejected` and not to `pending`.
    pub rejected: Option<Rejection>,
}

/// Why a report is in `rejected`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rejection {
    /// `rejected` (the billing API refused it), `max_age`, `deadline`.
    pub reason: &'static str,
    /// The HTTP status of its last answer, when it had one.
    pub status: Option<u16>,
    /// How its last attempt ended.
    pub outcome: &'static str,
}

/// A report as the store hands it out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Row {
    pub id: i64,
    /// Which database the `id` is of (`Store::generation`). An outcome for a
    /// row of another one is not written.
    pub generation: u64,
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
    fn read(row: &rusqlite::Row<'_>, generation: u64) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            generation,
            body: text_of(row, 1)?.unwrap_or_default(),
            request_id: text_of(row, 2)?,
            model_label: text_of(row, 3)?,
            auth_path: text_of(row, 4)?.unwrap_or_default(),
            ingress_route: text_of(row, 5)?.unwrap_or_default(),
            completed_at_ms: number_of(row, 6)?,
            attempts: number_of(row, 7)?.clamp(0, i64::from(u32::MAX)) as u32,
            last_outcome: text_of(row, 8)?,
            next_attempt_at_ms: number_of(row, 9)?,
        })
    }
}

/// A column as text, whatever is in it.
fn text_of(row: &rusqlite::Row<'_>, column: usize) -> rusqlite::Result<Option<String>> {
    use rusqlite::types::ValueRef;
    Ok(match row.get_ref(column)? {
        ValueRef::Null => None,
        ValueRef::Text(bytes) | ValueRef::Blob(bytes) => {
            Some(String::from_utf8_lossy(bytes).into_owned())
        }
        ValueRef::Integer(number) => Some(number.to_string()),
        ValueRef::Real(number) => Some(number.to_string()),
    })
}

/// A column as a number, 0 when it holds something else.
fn number_of(row: &rusqlite::Row<'_>, column: usize) -> rusqlite::Result<i64> {
    use rusqlite::types::ValueRef;
    Ok(match row.get_ref(column)? {
        ValueRef::Integer(number) => number,
        _ => 0,
    })
}

/// What became of a report this process held a lease on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Settle {
    pub id: i64,
    /// The database the `id` is of.
    pub generation: u64,
    pub outcome: Outcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Accepted: the row is done with.
    Delete,
    /// An attempt failed in a way that can pass. The lease is given back and
    /// the report is due again at `next_attempt_at_ms`.
    Retry {
        attempts: u32,
        next_attempt_at_ms: i64,
        last_outcome: &'static str,
    },
    /// No attempt will be made again: the row moves to `rejected`.
    Reject { why: Rejection, attempts: u32 },
    /// It was not attempted after all: the lease is given back as it was.
    Release,
}

/// What a claim asks for.
#[derive(Clone, Debug)]
pub(crate) struct Claim {
    /// Reports never tried, at most.
    pub fresh: usize,
    /// Of those, the ones whose request completed last rather than first:
    /// what a probe of a billing API that is not answering takes, because
    /// the newest report is the one least like those that failed.
    pub newest_first: bool,
    /// Reports that failed before and are due again, at most.
    pub retries: usize,
    /// How long they are this process's.
    pub lease: Duration,
    /// Reports whose request completed longer ago than this are moved to
    /// `rejected` with this reason instead of being sent. An age, not a
    /// time: how old a report is is judged with the clock as the
    /// transaction reads it, like everything else in it. A time worked out
    /// when the claim was made would be of a clock that may have been put
    /// right since.
    pub expire: Option<(Duration, &'static str)>,
}

/// What a claim found.
#[derive(Debug, Default)]
pub(crate) struct Claimed {
    /// Leased to this process, longest waiting first.
    pub fresh: Vec<Row>,
    pub retries: Vec<Row>,
    /// As many as were asked for: there may be more.
    pub more_fresh: bool,
    pub more_retries: bool,
    /// When the next report that failed is due again, if it is not yet.
    pub next_retry_ms: Option<i64>,
}

/// The backlog as of a transaction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Stats {
    /// Rows of `pending` per model label, waiting or being sent by anyone.
    pub pending: Vec<(Option<String>, u64)>,
    /// When the request of the oldest of them completed.
    pub oldest_completed_at_ms: Option<i64>,
    /// Rows of `rejected` per reason.
    pub rejected: Vec<(String, u64)>,
    /// What the file holds, in bytes, and what it may grow to.
    pub used_bytes: u64,
    pub max_bytes: u64,
    /// It may grow to less than `UsageOutboxConfig::max_bytes`, because its
    /// volume has no room for that.
    pub limited_by_volume: bool,
}

impl Stats {
    pub fn pending_total(&self) -> u64 {
        self.pending.iter().map(|(_, count)| count).sum()
    }

    pub fn rejected_total(&self) -> u64 {
        self.rejected.iter().map(|(_, count)| count).sum()
    }
}

/// What the writer knows about the file after a transaction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Health {
    /// The last time something was really written to the file, it worked.
    /// A transaction that only looked says nothing either way.
    pub available: bool,
    /// The file has no room for new reports: it is at its size, or the
    /// volume is. Leasing and removing still work at the file's own size.
    pub full: bool,
}

/// What a committed transaction did.
#[derive(Debug)]
pub(crate) struct Committed<J> {
    /// Reports written.
    pub stored: usize,
    /// Reports the file had no room for, handed back.
    pub no_room: Vec<J>,
    /// Reports that were written although `take_over` had taken them while
    /// the transaction was committing. They are in the file now, and are not
    /// counted in `stored`: whoever holds them in memory need not send them
    /// from there. (One that was sent from memory meanwhile is sent again
    /// from the file, which the billing API tells apart by its id.)
    pub overtaken: Vec<J>,
    /// Reports dropped to keep `pending` within its bound, oldest first.
    pub evicted: Vec<Row>,
    /// Reports moved to `rejected` for their age, with the reason.
    pub expired: Vec<(Row, &'static str)>,
    /// Rows removed from `rejected` to keep it within its bound, oldest
    /// first: the last there was of these reports.
    pub rejected_evicted: Vec<Evicted>,
    /// What the claim found, when the transaction carried one.
    pub claimed: Option<Claimed>,
    pub stats: Stats,
    pub health: Health,
}

/// What the writer has to say, in the order things happened.
pub(crate) enum Event<J> {
    /// A transaction committed. (Boxed: it is by far the largest.)
    Committed(Box<Committed<J>>),
    /// Nothing was written. `error` says which step failed, when one was
    /// tried at all (the file is left alone for a while after a failure).
    /// The reports are handed back; outcomes are kept and written by the
    /// next transaction that succeeds.
    Failed {
        error: Option<(&'static str, String)>,
        reports: Vec<J>,
        /// The transaction carried a claim, which found nothing.
        claim: bool,
        health: Health,
    },
    /// The file at the path is not the one the writer had open, or cannot be
    /// read as a database.
    Replaced {
        /// `deleted`: it is gone from the path. `replaced`: another file is
        /// there. In both the connection to the old file was let go of.
        /// `corrupt`: what is at the path cannot be read as a database, and
        /// was moved aside (`kept_as`). `lost`: the file was emptied, or
        /// exchanged while the writer did not have it open.
        why: &'static str,
        /// Where the file is now, when it was moved aside.
        kept_as: Option<PathBuf>,
        /// The ids of rows handed out before mean nothing from here on.
        generation: u64,
        /// What was wrong with the file, when something was.
        error: Option<String>,
        /// Reports the file held when this process last wrote to it. What
        /// another process wrote to it since is not among them.
        held: Option<u64>,
    },
}

/// A row removed from `rejected` to make room: what there was of the report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Evicted {
    /// Its id in `rejected`.
    pub id: i64,
    pub reason: String,
    pub status: Option<u16>,
    pub attempts: u32,
    pub body: String,
    pub request_id: Option<String>,
    pub model_label: Option<String>,
    pub completed_at_ms: i64,
}

/// Why `offer` did not take a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The file cannot be used at the moment.
    Unavailable,
    /// The file works and has no room for new reports.
    Full,
    /// As many reports as allowed already wait to be written.
    BufferFull,
}

impl Refused {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Unavailable => "unavailable",
            Self::Full => "full",
            Self::BufferFull => "buffer_full",
        }
    }
}

enum Stop {
    /// Write what is left, give this process's leases back, answer.
    Close(oneshot::Sender<Option<Stats>>),
    /// The handle is gone: write what is left and end.
    Abandon,
}

struct Inbox<J> {
    reports: VecDeque<J>,
    settles: Vec<Settle>,
    /// When the oldest of `reports` and `settles` was handed over.
    waiting_since: Option<Instant>,
    claim: Option<Claim>,
    /// A transaction is wanted now, even an empty one: for fresh numbers,
    /// or to find out whether the file is back.
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
    /// Reports are taken: the file is available and has room.
    accepting: AtomicBool,
    /// It is for lack of room that they are not.
    full: AtomicBool,
    /// Milliseconds since `started` at which the writer began the work it is
    /// doing, 0 while it waits for some.
    busy_since_ms: AtomicU64,
    started: Instant,
    /// Copies of the reports the writer is writing. Should it never come
    /// back from that, this is where they are found.
    in_hand: Mutex<Vec<J>>,
    /// How often `take_over` took what the writer had in hand.
    takeovers: AtomicU64,
    /// The database the rows handed out belong to.
    generation: AtomicU64,
    events: mpsc::UnboundedSender<Event<J>>,
    /// Makes opening and every transaction fail, as a disk that fails would.
    #[cfg(test)]
    fault: AtomicBool,
    /// Makes every insert fail as on a full volume.
    #[cfg(test)]
    volume_full: AtomicBool,
    /// The size of the volume and what everything but the outbox takes of
    /// it, in bytes, in place of what the system says.
    #[cfg(test)]
    volume: Mutex<Option<(u64, u64)>>,
    /// Deletes the file once, between the look at it and the transaction.
    #[cfg(test)]
    unlink: AtomicBool,
    /// Makes every commit take this many milliseconds longer, as a disk that
    /// is slow to flush would.
    #[cfg(test)]
    commit_delay_ms: AtomicU64,
}

impl<J> Shared<J> {
    fn inbox(&self) -> MutexGuard<'_, Inbox<J>> {
        // Nothing panics while holding the lock; if something ever does, the
        // inbox is still what there is.
        self.inbox.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn in_hand(&self) -> MutexGuard<'_, Vec<J>> {
        self.in_hand.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fault(&self) -> bool {
        #[cfg(test)]
        return self.fault.load(Ordering::Relaxed);
        #[cfg(not(test))]
        false
    }

    fn volume_full(&self) -> bool {
        #[cfg(test)]
        return self.volume_full.load(Ordering::Relaxed);
        #[cfg(not(test))]
        false
    }

    /// The size of the volume the file is on and what is free on it, in
    /// bytes. `None` where the system does not say.
    fn volume(&self) -> Option<(u64, u64)> {
        #[cfg(test)]
        if let Some((total, others)) = *self.volume.lock().unwrap_or_else(|e| e.into_inner()) {
            let file = std::fs::metadata(&self.config.path).map_or(0, |file| file.len());
            return Some((total, total.saturating_sub(others).saturating_sub(file)));
        }
        volume(&self.config.path)
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
                // Room for two transactions' worth, so that handing a report
                // over does not have to grow the buffer.
                reports: VecDeque::with_capacity(2 * MAX_BATCH),
                settles: Vec::new(),
                waiting_since: None,
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
            accepting: AtomicBool::new(true),
            full: AtomicBool::new(false),
            busy_since_ms: AtomicU64::new(0),
            started: Instant::now(),
            in_hand: Mutex::new(Vec::new()),
            takeovers: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            events,
            #[cfg(test)]
            fault: AtomicBool::new(false),
            #[cfg(test)]
            volume_full: AtomicBool::new(false),
            #[cfg(test)]
            volume: Mutex::new(None),
            #[cfg(test)]
            unlink: AtomicBool::new(false),
            #[cfg(test)]
            commit_delay_ms: AtomicU64::new(0),
        });
        let writer = Writer::new(shared.clone());
        let spawned = std::thread::Builder::new()
            .name("usage-outbox".to_string())
            .spawn(move || writer.run());
        if let Err(error) = spawned {
            // Without its thread the store never opens: reports are refused
            // here from the start and delivered from memory.
            shared.accepting.store(false, Ordering::Relaxed);
            let _ = shared.events.send(Event::Failed {
                error: Some(("open", format!("no thread for the writer: {error}"))),
                reports: Vec::new(),
                claim: false,
                health: Health::default(),
            });
        }
        Self { shared }
    }

    pub fn config(&self) -> &UsageOutboxConfig {
        &self.shared.config
    }

    /// Whether a report handed over now is taken.
    pub fn is_accepting(&self) -> bool {
        self.shared.accepting.load(Ordering::Relaxed)
    }

    /// Hand a report over to be written. Never blocks on the file: the report
    /// is taken, or given back when the file is not taking any or
    /// `max_buffered` reports already wait to be written.
    pub fn offer(&self, report: J, max_buffered: usize) -> Result<(), (J, Refused)> {
        if !self.is_accepting() {
            let full = self.shared.full.load(Ordering::Relaxed);
            let why = if full {
                Refused::Full
            } else {
                Refused::Unavailable
            };
            return Err((report, why));
        }
        self.push(report, max_buffered)
    }

    /// `offer` although the file is not taking reports: for what memory
    /// holds when the process stops, which has this one chance of being
    /// written.
    pub fn offer_anyway(&self, report: J) -> Result<(), (J, Refused)> {
        self.push(report, usize::MAX)
    }

    fn push(&self, report: J, max_buffered: usize) -> Result<(), (J, Refused)> {
        let wake = {
            let mut inbox = self.shared.inbox();
            if inbox.stopping {
                return Err((report, Refused::Unavailable));
            }
            if inbox.reports.len() >= max_buffered {
                return Err((report, Refused::BufferFull));
            }
            inbox.reports.push_back(report);
            // The writer sleeps through what joins a batch: it is woken by
            // the first, which starts the interval, and when the batch is as
            // large as a transaction gets.
            let first = inbox.waiting_since.is_none();
            if first {
                inbox.waiting_since = Some(Instant::now());
            }
            first || inbox.reports.len() == MAX_BATCH
        };
        if wake {
            self.shared.work.notify_one();
        }
        Ok(())
    }

    /// Record what became of a leased report. Written with the next
    /// transaction, within `commit_interval`: until then the report stays
    /// leased, so nothing else is done with it.
    pub fn settle(&self, settle: Settle) {
        let first = {
            let mut inbox = self.shared.inbox();
            inbox.settles.push(settle);
            let first = inbox.waiting_since.is_none();
            if first {
                inbox.waiting_since = Some(Instant::now());
            }
            first
        };
        if first {
            self.shared.work.notify_one();
        }
    }

    /// Ask for due reports, leased to this process. The answer is the
    /// `claimed` of a `Committed`, or a `Failed` with `claim` set.
    pub fn claim(&self, claim: Claim) {
        self.shared.inbox().claim = Some(claim);
        self.shared.work.notify_one();
    }

    /// Ask for a transaction now: its `Committed` carries fresh numbers, and
    /// when the file was unavailable this is what tries it again.
    pub fn sync(&self) {
        self.shared.inbox().sync = true;
        self.shared.work.notify_one();
    }

    /// How long the writer has been at the work it is doing, or has left
    /// work waiting. A writer that waits for nothing is not stalled.
    pub fn stalled_for(&self) -> Duration {
        let busy_since = self.shared.busy_since_ms.load(Ordering::Relaxed);
        let busy = if busy_since == 0 {
            Duration::ZERO
        } else {
            self.shared
                .started
                .elapsed()
                .saturating_sub(Duration::from_millis(busy_since))
        };
        // Work that waits longer than the interval it is gathered for is
        // waiting for a writer that does not come.
        let waiting = self
            .shared
            .inbox()
            .waiting_since
            .map_or(Duration::ZERO, |since| {
                since
                    .elapsed()
                    .saturating_sub(self.shared.config.commit_interval)
            });
        busy.max(waiting)
    }

    /// The writer does not come back: take what it was handed and has not
    /// written, to be delivered from memory. From here on no report is
    /// taken, until the writer has written something again. Should it come
    /// back with these reports in a transaction, it leaves them out; if it
    /// was committing them already, they are written after all, and its
    /// `Committed` names them (`overtaken`).
    pub fn take_over(&self) -> Vec<J> {
        self.shared.accepting.store(false, Ordering::Relaxed);
        self.shared.full.store(false, Ordering::Relaxed);
        self.shared.takeovers.fetch_add(1, Ordering::SeqCst);
        let mut taken = std::mem::take(&mut *self.shared.in_hand());
        let mut inbox = self.shared.inbox();
        taken.extend(inbox.reports.drain(..));
        if inbox.settles.is_empty() {
            inbox.waiting_since = None;
        }
        taken
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
    pub fn inject_volume_full(&self, on: bool) {
        self.shared.volume_full.store(on, Ordering::Relaxed);
    }

    /// The volume is `total` bytes, of which everything but the outbox takes
    /// `others`, whatever the system says; `None` ends that.
    #[cfg(test)]
    pub fn inject_volume(&self, volume: Option<(u64, u64)>) {
        *self.shared.volume.lock().unwrap() = volume;
    }

    /// The file is deleted under the writer's next transaction, after the
    /// writer has looked at it.
    #[cfg(test)]
    pub fn inject_unlink(&self) {
        self.shared.unlink.store(true, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub fn inject_commit_delay(&self, delay: Duration) {
        self.shared
            .commit_delay_ms
            .store(delay.as_millis() as u64, Ordering::Relaxed);
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

/// A failed step, by name, and what kind of failure it was.
struct Failure {
    op: &'static str,
    error: String,
    /// The file or its volume has no room.
    full: bool,
    /// The file is not a database any more.
    corrupt: bool,
    /// The reports of this transaction were taken over while it ran.
    taken_over: bool,
}

impl Failure {
    fn new(op: &'static str, error: impl std::fmt::Display) -> Self {
        Self {
            op,
            error: error.to_string(),
            full: false,
            corrupt: false,
            taken_over: false,
        }
    }
}

/// `error` as the failure of step `op`.
fn failed(op: &'static str) -> impl Fn(rusqlite::Error) -> Failure {
    move |error| {
        let code = error.sqlite_error_code();
        Failure {
            full: code == Some(ffi::ErrorCode::DiskFull),
            corrupt: matches!(
                code,
                Some(ffi::ErrorCode::DatabaseCorrupt | ffi::ErrorCode::NotADatabase)
            ),
            ..Failure::new(op, error)
        }
    }
}

/// The open file.
struct Db {
    conn: Connection,
    /// The device and inode of the file that was opened, to notice a file
    /// that was deleted or replaced under the connection.
    identity: Option<(u64, u64)>,
    /// `meta.db_id` of the database in it.
    id: String,
    /// The most pages the file may have, and of those the ones new reports
    /// may not take: what leasing, retrying and rejecting need to go on in a
    /// file that is full.
    max_pages: i64,
    headroom_pages: i64,
    page_size: i64,
    /// `max_pages` is what the volume has room for, less than `max_bytes`.
    limited_by_volume: bool,
}

/// The thread that owns the connection.
struct Writer<J> {
    shared: Arc<Shared<J>>,
    db: Option<Db>,
    /// `db_id` of the database the rows handed out belong to.
    db_id: Option<String>,
    /// Outcomes a failed transaction did not write. They are written by the
    /// next one that succeeds; until then their reports stay leased, so at
    /// worst they are sent once more when the lease runs out. Never more
    /// than the reports this process holds a lease on.
    unwritten: Vec<Settle>,
    /// Not before this is the file opened again after a failure.
    retry_at: Instant,
    health: Health,
    /// An insert failed for lack of room although the file was below its own
    /// size: the volume is full. What ends that is a report that is written,
    /// or the file growing when that is tried (`find_room`).
    volume_full: bool,
    /// Not before this is that tried again.
    room_probe_at: Instant,
    /// Not before this is the volume looked at again, for the size the file
    /// may have on it.
    volume_check_at: Instant,
    /// Rows of `pending` and of `rejected` as of the last transaction: what
    /// a file held, when all that can be said of it is that it is gone.
    last_held: Option<u64>,
}

impl<J: Persist> Writer<J> {
    fn new(shared: Arc<Shared<J>>) -> Self {
        Self {
            shared,
            db: None,
            db_id: None,
            unwritten: Vec::new(),
            retry_at: Instant::now(),
            health: Health::default(),
            volume_full: false,
            room_probe_at: Instant::now(),
            volume_check_at: Instant::now(),
            last_held: None,
        }
    }

    fn run(mut self) {
        loop {
            let mut work = self.wait_for_work();
            let stop = work.stop.take();
            let busy_since = self.shared.started.elapsed().as_millis().max(1) as u64;
            self.shared
                .busy_since_ms
                .store(busy_since, Ordering::Relaxed);
            let stats = self.serve(work, matches!(stop, Some(Stop::Close(_))), stop.is_some());
            self.shared.busy_since_ms.store(0, Ordering::Relaxed);
            match stop {
                Some(Stop::Close(done)) => {
                    self.db = None;
                    let _ = done.send(stats);
                    return;
                }
                Some(Stop::Abandon) => return,
                None => {}
            }
        }
    }

    /// Sleep until there is something to do, and take it. Reports and
    /// outcomes alone wait until the oldest has been there for
    /// `commit_interval`, or until there are enough reports for a
    /// transaction; a claim, a sync and the end are served at once, with
    /// whatever else is there.
    fn wait_for_work(&mut self) -> Work<J> {
        let interval = self.shared.config.commit_interval;
        let mut inbox = self.shared.inbox();
        loop {
            if inbox.stop.is_some()
                || inbox.claim.is_some()
                || inbox.sync
                || inbox.reports.len() >= MAX_BATCH
            {
                break;
            }
            match inbox.waiting_since {
                Some(since) => {
                    let left = interval.saturating_sub(since.elapsed());
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
        // A writer that is ending writes all it has.
        let take = if inbox.stop.is_some() {
            inbox.reports.len()
        } else {
            inbox.reports.len().min(MAX_BATCH)
        };
        let reports: Vec<J> = inbox.reports.drain(..take).collect();
        // What is left has waited its interval: the next round starts at
        // once. Nothing left: the next report starts a new one.
        if inbox.reports.is_empty() {
            inbox.waiting_since = None;
        }
        inbox.sync = false;
        let work = Work {
            reports,
            settles: std::mem::take(&mut inbox.settles),
            claim: inbox.claim.take(),
            stop: inbox.stop.take(),
        };
        drop(inbox);
        work
    }

    /// One transaction for `work`, or the report that there was none. Returns
    /// the backlog when it committed. `last`: the writer ends after this one,
    /// so a file that failed a moment ago is tried all the same.
    fn serve(&mut self, mut work: Work<J>, release_leases: bool, last: bool) -> Option<Stats> {
        let claim = work.claim.is_some();
        self.unwritten.append(&mut work.settles);
        // Where `take_over` finds them, should this never return.
        let takeovers = self.shared.takeovers.load(Ordering::SeqCst);
        *self.shared.in_hand() = work.reports.clone();

        if last {
            self.retry_at = Instant::now();
        }
        let outcome = self.transact_on_the_file(&mut work, release_leases, takeovers);
        // Whatever follows, nobody takes these over any more: they are in
        // the file, or on their way back in an event.
        self.shared.in_hand().clear();
        let taken_over = self.shared.takeovers.load(Ordering::SeqCst) != takeovers;
        match outcome {
            Ok(mut committed) => {
                if taken_over {
                    // They are being delivered from memory already. Those
                    // that were written all the same (the writer was given
                    // up on while it committed) are told apart.
                    let written = committed.stored.min(work.reports.len());
                    committed.overtaken = work.reports.drain(..written).collect();
                    committed.stored = 0;
                    work.reports.clear();
                }
                committed.no_room = work
                    .reports
                    .split_off(committed.stored.min(work.reports.len()));
                self.unwritten.clear();
                self.set_health(committed.health);
                let stats = committed.stats.clone();
                let _ = self
                    .shared
                    .events
                    .send(Event::Committed(Box::new(committed)));
                Some(stats)
            }
            Err(failure) => {
                if taken_over {
                    // They are being delivered from memory already.
                    work.reports.clear();
                }
                if let Some(failure) = &failure {
                    self.health.available = false;
                    self.health.full |= failure.full;
                    self.retry_at = Instant::now() + self.shared.config.reopen_interval;
                }
                self.set_health(self.health);
                let _ = self.shared.events.send(Event::Failed {
                    error: failure.map(|failure| (failure.op, failure.error)),
                    reports: work.reports,
                    claim,
                    health: self.health,
                });
                None
            }
        }
    }

    fn set_health(&mut self, health: Health) {
        self.health = health;
        self.shared
            .full
            .store(health.available && health.full, Ordering::Relaxed);
        self.shared
            .accepting
            .store(health.available && !health.full, Ordering::Relaxed);
    }

    /// Open the file if it is not, make sure it is still the file at the
    /// path, and run the transaction: again when a try shows that the
    /// reports have no room (with fewer of them, and in the end with none:
    /// the rest of it must not wait for that), or that the file has to be
    /// replaced. `Err(None)`: nothing was tried.
    fn transact_on_the_file(
        &mut self,
        work: &mut Work<J>,
        release_leases: bool,
        takeovers: u64,
    ) -> Result<Committed<J>, Option<Failure>> {
        if self.db.is_none() && Instant::now() < self.retry_at {
            return Err(None);
        }
        // How many of the reports this try writes, at the most.
        let mut reports = work.reports.len();
        let (mut moved_aside, mut let_go) = (false, false);
        loop {
            let tried = self.ready().and_then(|()| {
                self.fit_the_volume();
                self.find_room();
                #[cfg(test)]
                if self.shared.unlink.swap(false, Ordering::Relaxed) {
                    let _ = std::fs::remove_file(&self.shared.config.path);
                }
                self.transact(work, reports, release_leases, takeovers)
            });
            match tried {
                Ok(committed) => return Ok(committed),
                Err(failure) if failure.taken_over && reports > 0 => {
                    reports = 0;
                }
                // The volume has no room for that many reports: half as
                // many are tried, so that what room there is gets used, and
                // in the end none. What frees room (an accepted report
                // removed) is in the same transaction, and goes ahead; the
                // reports left out are handed back.
                Err(failure) if failure.full && reports > 0 => {
                    self.volume_full = true;
                    self.room_probe_at = Instant::now() + self.shared.config.reopen_interval;
                    reports /= 2;
                }
                Err(failure) => {
                    // The file was taken from under this very transaction
                    // (after it was last looked at): SQLite does not write
                    // to a database whose file was moved, and says so. That
                    // comes before anything else the error may seem to say:
                    // a connection whose file is gone can find its database
                    // damaged, and what is at the path is not that database.
                    // It is let go of, and the transaction is made on the
                    // file at the path. (Once: a file that goes away again
                    // meanwhile is left to the next transaction.)
                    if self.left_the_path() {
                        if let_go {
                            return Err(Some(failure));
                        }
                        let_go = true;
                        continue;
                    }
                    // What this process had open and can no longer read, or
                    // what it found at the path and cannot read at all.
                    if failure.corrupt && !moved_aside {
                        moved_aside = true;
                        self.db = None;
                        if let Err(error) = self.move_aside(failure.error) {
                            return Err(Some(Failure::new("replace", error)));
                        }
                        continue;
                    }
                    // Whatever state the connection is in, the next try
                    // starts from a new one. The file is at the path, and
                    // what it holds with it.
                    self.db = None;
                    return Err(Some(failure));
                }
            }
        }
    }

    /// Have the file open, and the right one: a file deleted or replaced
    /// under the connection is noticed before a transaction is begun on what
    /// is no longer at the path.
    fn ready(&mut self) -> Result<(), Failure> {
        if self.shared.fault() {
            self.db = None;
            return Err(Failure::new("open", "injected fault"));
        }
        let path = self.shared.config.path.clone();
        if let Some(db) = &self.db {
            match whereabouts(&path) {
                Ok(there) if there == db.identity => return Ok(()),
                Ok(there) => self.let_go(if there.is_none() {
                    "deleted"
                } else {
                    "replaced"
                }),
                // The path cannot be looked at: nothing is known of the
                // file. This try fails as any other does, which closes the
                // connection, and the file is opened anew at the next one.
                Err(error) => {
                    return Err(Failure::new(
                        "open",
                        format!("cannot look at {}: {error}", path.display()),
                    ))
                }
            }
        }
        let db = open(&self.shared.config, self.shared.volume())?;
        self.adopt(db);
        Ok(())
    }

    /// The file of the open connection is no longer the one at the path:
    /// it is let go of, and `true`.
    fn left_the_path(&mut self) -> bool {
        let Some(db) = &self.db else {
            return false;
        };
        match whereabouts(&self.shared.config.path) {
            Ok(there) if there != db.identity => {
                self.let_go(if there.is_none() {
                    "deleted"
                } else {
                    "replaced"
                });
                true
            }
            _ => false,
        }
    }

    /// The file of the open connection is no longer at the path. The
    /// connection is closed, and nothing is done with it first: SQLite
    /// finds the journal of a database by the path the database was opened
    /// under, which for this one is now the journal of whatever is at the
    /// path, or will be. A connection that so much as begins to read then
    /// takes a transaction another process has under way on that file for
    /// one of its own that was interrupted, plays it into its own file and
    /// clears the journal. So what this database still holds is not read:
    /// it is said, with how many reports that was when it was last written
    /// to, and they are lost with the file. The rows handed out of it mean
    /// nothing in whatever is opened at the path next; their reports are
    /// still in this process, and are written there.
    fn let_go(&mut self, why: &'static str) {
        if self.db.take().is_none() {
            return;
        }
        let generation = self.shared.generation.fetch_add(1, Ordering::Relaxed) + 1;
        // Outcomes not yet written cannot be written there any more.
        self.unwritten.clear();
        self.db_id = None;
        let _ = self.shared.events.send(Event::Replaced {
            why,
            kept_as: None,
            generation,
            error: None,
            held: self.last_held.take(),
        });
    }

    /// Take `db`, just opened, as the file at the path. When this process
    /// had another database open before and no longer has it (the file was
    /// emptied under it, or exchanged while it could not be used), the rows
    /// handed out of that one mean nothing in this one, and what it held is
    /// in nobody's hands.
    fn adopt(&mut self, db: Db) {
        if self.db_id.as_deref().is_some_and(|known| known != db.id) {
            let generation = self.shared.generation.fetch_add(1, Ordering::Relaxed) + 1;
            self.unwritten.clear();
            let _ = self.shared.events.send(Event::Replaced {
                why: "lost",
                kept_as: None,
                generation,
                error: None,
                held: self.last_held.take(),
            });
        }
        self.db_id = Some(db.id.clone());
        self.db = Some(db);
    }

    /// The size the file may have follows its volume: looked at when the
    /// file is opened, and again every `reopen_interval`. A volume that has
    /// no room for `max_bytes` (too small, or filling up with something
    /// else) lowers it, so that the file is full, by its own size, while the
    /// volume still has room for the journal; room that returns raises it
    /// again. (Never below what the file already has: SQLite does not set
    /// that.)
    fn fit_the_volume(&mut self) {
        if Instant::now() < self.volume_check_at {
            return;
        }
        self.volume_check_at = Instant::now() + self.shared.config.reopen_interval;
        let volume = self.shared.volume();
        let Some(db) = self.db.as_mut() else {
            return;
        };
        if let Ok((max_pages, limited_by_volume)) =
            set_size(&db.conn, &self.shared.config, db.page_size, volume)
        {
            db.max_pages = max_pages;
            db.headroom_pages = headroom(max_pages);
            db.limited_by_volume = limited_by_volume;
        }
    }

    /// A volume that was full may have room again, and with no report waiting
    /// to be written nothing would find out: every `reopen_interval` the
    /// file is made to grow by a few pages, which stay with it for the
    /// reports to come. That it grew is the evidence that it can.
    fn find_room(&mut self) {
        if !self.volume_full || self.shared.volume_full() || Instant::now() < self.room_probe_at {
            return;
        }
        self.room_probe_at = Instant::now() + self.shared.config.reopen_interval;
        let Some(db) = &self.db else {
            return;
        };
        // Written as a row, so that the pages are really taken from the
        // volume, then removed, which leaves them to the file.
        let bytes = ROOM_PROBE_PAGES * db.page_size;
        let grown = db.conn.execute_batch(&format!(
            "BEGIN IMMEDIATE; \
             INSERT OR REPLACE INTO meta (key, value) VALUES ('room', zeroblob({bytes})); \
             COMMIT;"
        ));
        if grown.is_err() {
            let _ = db.conn.execute_batch("ROLLBACK");
            return;
        }
        if db
            .conn
            .execute_batch("DELETE FROM meta WHERE key = 'room'")
            .is_ok()
        {
            self.volume_full = false;
        }
    }

    /// What is at the path cannot be read as a database. It is kept for a
    /// person under another name, with its journal, and a new file is
    /// started at the path by the next `ready`. When it is the file this
    /// process last wrote to, the reports that held then go with it, and
    /// are said to.
    fn move_aside(&mut self, error: String) -> std::io::Result<()> {
        let path = &self.shared.config.path;
        let stamp = self.shared.config.clock.now_ms() / 1000;
        let name = |suffix: u32, journal: bool| {
            let mut name = path.clone().into_os_string();
            name.push(format!(".corrupt-{stamp}"));
            if suffix > 0 {
                name.push(format!("-{suffix}"));
            }
            if journal {
                name.push("-journal");
            }
            PathBuf::from(name)
        };
        let suffix = (0..)
            .find(|suffix| !name(*suffix, false).exists())
            .unwrap_or(0);
        std::fs::rename(path, name(suffix, false))?;
        let mut journal = path.clone().into_os_string();
        journal.push("-journal");
        // A journal that stayed behind would be played into the new file.
        if Path::new(&journal).exists() {
            std::fs::rename(&journal, name(suffix, true))?;
        }
        // The database that follows is another one.
        let generation = self.shared.generation.fetch_add(1, Ordering::Relaxed) + 1;
        self.unwritten.clear();
        self.db_id = None;
        let _ = self.shared.events.send(Event::Replaced {
            why: "corrupt",
            kept_as: Some(name(suffix, false)),
            generation,
            error: Some(error),
            held: self.last_held.take(),
        });
        Ok(())
    }

    /// Everything `work` asks for, in one transaction: the outcomes first
    /// (they free rows and room), then the reports too old to send, the new
    /// reports, the bound, the claim (which may take a report written a line
    /// above) and the numbers.
    fn transact(
        &mut self,
        work: &Work<J>,
        reports: usize,
        release_leases: bool,
        takeovers: u64,
    ) -> Result<Committed<J>, Failure> {
        let config = &self.shared.config;
        let owner = self.shared.owner.as_str();
        let generation = self.shared.generation.load(Ordering::Relaxed);
        let db = self.db.as_mut().expect("opened by `ready`");
        let changes_before = db.conn.total_changes();
        // The write lock is taken at once: a transaction that started by
        // reading could not wait for it later (SQLite fails such an upgrade
        // without consulting the busy timeout).
        let tx = db
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(failed("begin"))?;
        // The time of this transaction, read once it has the file to itself:
        // the wait for the lock may have been long, and every time written
        // or compared below is of the same moment as what is read.
        let now = config.clock.now_ms();
        let mut committed = Committed {
            stored: 0,
            no_room: Vec::new(),
            overtaken: Vec::new(),
            evicted: Vec::new(),
            expired: Vec::new(),
            rejected_evicted: Vec::new(),
            claimed: None,
            stats: Stats::default(),
            health: self.health,
        };
        let integer = |sql: &str, op: &'static str| -> Result<i64, Failure> {
            tx.prepare_cached(sql)
                .and_then(|mut statement| statement.query_row([], |row| row.get(0)))
                .map_err(failed(op))
        };

        // Evidence that the file takes writes, when that is in doubt: a
        // transaction that changes nothing proves nothing.
        if !self.health.available {
            tx.prepare_cached(
                "INSERT INTO meta (key, value) VALUES ('probed_at_ms', ?1) \
                 ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            )
            .and_then(|mut statement| statement.execute([now]))
            .map_err(failed("probe"))?;
        }

        let reject = "INSERT INTO rejected (rejected_at_ms, reason, status, outcome, attempts, \
                      body, request_id, model_label, auth_path, ingress_route, completed_at_ms) \
                      VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)";
        for settle in &self.unwritten {
            // The row is of a database that is no longer the one at the
            // path: its id would name another report here.
            if settle.generation != generation {
                continue;
            }
            let id = settle.id;
            // An outcome is written to the row its sender leased, and to no
            // other: only while this process is still the one named on it.
            // Once its lease ran out and somebody else took the report, the
            // row is theirs, and what they find out is what is written.
            match &settle.outcome {
                Outcome::Delete => {
                    tx.prepare_cached("DELETE FROM pending WHERE id = ?1 AND lease_owner = ?2")
                        .and_then(|mut statement| statement.execute(params![id, owner]))
                        .map_err(failed("settle"))?;
                }
                Outcome::Retry {
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
                Outcome::Reject { why, attempts } => {
                    tx.prepare_cached(
                        "INSERT INTO rejected (rejected_at_ms, reason, status, outcome, \
                         attempts, body, request_id, model_label, auth_path, ingress_route, \
                         completed_at_ms) \
                         SELECT ?2, ?3, ?4, ?5, ?6, body, request_id, model_label, auth_path, \
                         ingress_route, completed_at_ms FROM pending \
                         WHERE id = ?1 AND lease_owner = ?7",
                    )
                    .and_then(|mut statement| {
                        statement.execute(params![
                            id,
                            now,
                            why.reason,
                            why.status,
                            why.outcome,
                            attempts,
                            owner
                        ])
                    })
                    .map_err(failed("settle"))?;
                    tx.prepare_cached("DELETE FROM pending WHERE id = ?1 AND lease_owner = ?2")
                        .and_then(|mut statement| statement.execute(params![id, owner]))
                        .map_err(failed("settle"))?;
                }
                Outcome::Release => {
                    tx.prepare_cached(
                        "UPDATE pending SET lease_until_ms = 0, lease_owner = NULL \
                         WHERE id = ?1 AND lease_owner = ?2",
                    )
                    .and_then(|mut statement| statement.execute(params![id, owner]))
                    .map_err(failed("settle"))?;
                }
            }
        }

        // A lease is over when its time has passed, or when it ends further
        // ahead than any lease lasts: it was then written under a clock that
        // has since been set back. A report this process itself holds is
        // never taken again from here, whatever the clock says: it is being
        // sent, and its outcome will say what becomes of it.
        let lease_far = now.saturating_add(millis(config.max_lease));
        let free = "(lease_until_ms <= ?1 OR lease_until_ms > ?2) \
                    AND (lease_owner IS NULL OR lease_owner <> ?3)";

        // Reports too old to be sent go to `rejected`, where a person can
        // see them and put them back: nothing is tried for ever, and nothing
        // is destroyed for its age (a clock set forward makes everything
        // look old).
        if let Some((max_age, reason)) = work.claim.as_ref().and_then(|claim| claim.expire) {
            let before_ms = now.saturating_sub(millis(max_age));
            let expired = tx
                .prepare_cached(&format!(
                    "DELETE FROM pending WHERE id IN (SELECT id FROM pending \
                     WHERE completed_at_ms <= ?4 AND {free} \
                     ORDER BY completed_at_ms, id LIMIT ?5) RETURNING {ROW_COLUMNS}"
                ))
                .and_then(|mut statement| {
                    statement
                        .query_map(
                            params![now, lease_far, owner, before_ms, SWEEP_BATCH as i64],
                            |row| Row::read(row, generation),
                        )
                        .and_then(Iterator::collect::<rusqlite::Result<Vec<_>>>)
                })
                .map_err(failed("expire"))?;
            let mut insert = tx.prepare_cached(reject).map_err(failed("expire"))?;
            for row in &expired {
                insert
                    .execute(params![
                        now,
                        reason,
                        None::<u16>,
                        row.last_outcome.as_deref().unwrap_or("never_sent"),
                        row.attempts,
                        row.body,
                        row.request_id,
                        row.model_label,
                        row.auth_path,
                        row.ingress_route,
                        row.completed_at_ms
                    ])
                    .map_err(failed("expire"))?;
            }
            committed.expired = expired.into_iter().map(|row| (row, reason)).collect();
            committed
                .expired
                .sort_by_key(|(row, _)| (row.completed_at_ms, row.id));
        }

        // New reports stay out of the last pages the file may have: what is
        // done to the reports already in it must never want for room. The
        // room is measured, and measured again before the reports written
        // since could have used up half of it at the very worst: so the
        // file fills up to that line and not beyond, whatever the size of a
        // report, without a look at its size per report.
        let room = || -> Result<i64, Failure> {
            let used = integer("SELECT page_count FROM pragma_page_count", "insert")?
                - integer("SELECT freelist_count FROM pragma_freelist_count", "insert")?;
            Ok((db.max_pages - db.headroom_pages - used).max(0))
        };
        let mut may_take: Option<i64> = None;
        let mut fits = |body: usize| -> Result<bool, Failure> {
            let worst = WORST_PAGES_PER_REPORT
                + i64::try_from(body).unwrap_or(i64::MAX) / db.page_size.max(1);
            let mut left = match may_take {
                Some(left) => left,
                None => room()? / 2,
            };
            if left < worst {
                left = room()? / 2;
            }
            may_take = Some(if left < worst { left } else { left - worst });
            Ok(left >= worst)
        };
        let insert_pending = "INSERT INTO pending (body, request_id, model_label, auth_path, \
                              ingress_route, completed_at_ms, attempts, next_attempt_at_ms, \
                              last_outcome) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)";
        if reports > 0 {
            if self.shared.volume_full() {
                return Err(Failure {
                    full: true,
                    ..Failure::new("insert", "injected: database or disk is full")
                });
            }
            // Due from the moment its request completed, so that reports
            // come up in the order they completed.
            let mut insert = tx
                .prepare_cached(insert_pending)
                .map_err(failed("insert"))?;
            let mut insert_rejected = tx.prepare_cached(reject).map_err(failed("insert"))?;
            for report in work.reports.iter().take(reports) {
                let row = report.row();
                if !fits(row.body.len())? {
                    break;
                }
                let completed_at_ms = now.saturating_sub(millis(row.age));
                let body = String::from_utf8_lossy(row.body);
                match row.rejected {
                    Some(why) => insert_rejected.execute(params![
                        now,
                        why.reason,
                        why.status,
                        why.outcome,
                        row.attempts,
                        body,
                        row.request_id,
                        row.model_label,
                        row.auth_path,
                        row.ingress_route,
                        completed_at_ms
                    ]),
                    None => {
                        let due_at_ms = if row.attempts == 0 {
                            completed_at_ms
                        } else {
                            now.saturating_add(millis(row.due_in))
                        };
                        insert.execute(params![
                            body,
                            row.request_id,
                            row.model_label,
                            row.auth_path,
                            row.ingress_route,
                            completed_at_ms,
                            row.attempts,
                            due_at_ms,
                            row.last_outcome
                        ])
                    }
                }
                .map_err(failed("insert"))?;
                committed.stored += 1;
            }
        }

        // The bound on `pending`: the reports that have waited longest go.
        // One that somebody is sending right now is left alone: its answer
        // is on its way.
        let pending = integer("SELECT COALESCE(SUM(n), 0) FROM pending_counts", "evict")?;
        let excess = pending - i64::try_from(config.max_pending).unwrap_or(i64::MAX);
        if excess > 0 {
            committed.evicted = tx
                .prepare_cached(&format!(
                    "DELETE FROM pending WHERE id IN (SELECT id FROM pending \
                     WHERE {free} ORDER BY completed_at_ms, id LIMIT ?4) \
                     RETURNING {ROW_COLUMNS}"
                ))
                .and_then(|mut statement| {
                    statement
                        .query_map(
                            params![now, lease_far, owner, excess.min(SWEEP_BATCH as i64)],
                            |row| Row::read(row, generation),
                        )
                        .and_then(Iterator::collect::<rusqlite::Result<Vec<_>>>)
                })
                .map_err(failed("evict"))?;
            committed
                .evicted
                .sort_by_key(|row| (row.completed_at_ms, row.id));
        }
        // And the bound on `rejected`: all but the newest.
        let rejected = integer("SELECT COALESCE(SUM(n), 0) FROM rejected_counts", "evict")?;
        let excess = rejected - i64::try_from(config.max_rejected).unwrap_or(i64::MAX);
        if excess > 0 {
            // What is removed here is the last there was of these reports:
            // it is handed out, to be said row by row.
            committed.rejected_evicted = tx
                .prepare_cached(
                    "DELETE FROM rejected WHERE id IN \
                     (SELECT id FROM rejected ORDER BY id LIMIT ?1) \
                     RETURNING id, reason, status, attempts, body, request_id, model_label, \
                     completed_at_ms",
                )
                .and_then(|mut statement| {
                    statement
                        .query_map([excess.min(EVICT_BATCH as i64)], |row| {
                            Ok(Evicted {
                                id: row.get(0)?,
                                reason: text_of(row, 1)?.unwrap_or_default(),
                                status: u16::try_from(number_of(row, 2)?).ok().filter(|s| *s > 0),
                                attempts: number_of(row, 3)?.clamp(0, i64::from(u32::MAX)) as u32,
                                body: text_of(row, 4)?.unwrap_or_default(),
                                request_id: text_of(row, 5)?,
                                model_label: text_of(row, 6)?,
                                completed_at_ms: number_of(row, 7)?,
                            })
                        })
                        .and_then(Iterator::collect::<rusqlite::Result<Vec<_>>>)
                })
                .map_err(failed("evict"))?;
            committed.rejected_evicted.sort_by_key(|row| row.id);
        }

        if let Some(claim) = &work.claim {
            let lease_until = now.saturating_add(millis(claim.lease));
            let due_far = now.saturating_add(millis(config.max_backoff).saturating_mul(2));
            let take = |class: &str,
                        due: &str,
                        order: &str,
                        limit: usize|
             -> Result<Vec<Row>, Failure> {
                if limit == 0 {
                    return Ok(Vec::new());
                }
                let mut rows = tx
                    .prepare_cached(&format!(
                        "UPDATE pending SET lease_until_ms = ?4, lease_owner = ?3 \
                         WHERE id IN (SELECT id FROM pending WHERE {class} AND {due} AND {free} \
                         ORDER BY {order} LIMIT ?5) RETURNING {ROW_COLUMNS}"
                    ))
                    .and_then(|mut statement| {
                        statement
                            .query_map(
                                params![now, lease_far, owner, lease_until, limit as i64, due_far],
                                |row| Row::read(row, generation),
                            )
                            .and_then(Iterator::collect::<rusqlite::Result<Vec<_>>>)
                    })
                    .map_err(failed("claim"))?;
                rows.sort_by_key(|row| (row.next_attempt_at_ms, row.id));
                Ok(rows)
            };
            // A report never tried is due whatever its time says. One that
            // failed is due when its time has passed, or when that time is
            // further ahead than any backoff: the clock was set back.
            let oldest = "next_attempt_at_ms, id";
            let newest = "next_attempt_at_ms DESC, id DESC";
            let fresh = take(
                "attempts = 0",
                "?6 IS NOT NULL",
                if claim.newest_first { newest } else { oldest },
                claim.fresh,
            )?;
            let retries = take(
                "attempts > 0",
                "(next_attempt_at_ms <= ?1 OR next_attempt_at_ms > ?6)",
                oldest,
                claim.retries,
            )?;
            let mut claimed = Claimed {
                more_fresh: claim.fresh > 0 && fresh.len() >= claim.fresh,
                more_retries: claim.retries > 0 && retries.len() >= claim.retries,
                fresh,
                retries,
                next_retry_ms: None,
            };
            if !claimed.more_retries {
                claimed.next_retry_ms = tx
                    .prepare_cached(
                        "SELECT MIN(next_attempt_at_ms) FROM pending \
                         WHERE attempts > 0 AND next_attempt_at_ms > ?1",
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

        // A model, or a reason, whose rows are all gone is still listed,
        // with 0, so that whoever reads this sees its count come down.
        committed.stats.pending = tx
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
        committed.stats.rejected = tx
            .prepare_cached("SELECT reason, n FROM rejected_counts")
            .and_then(|mut statement| {
                statement
                    .query_map([], |row| {
                        let count: i64 = row.get(1)?;
                        Ok((row.get(0)?, count.max(0) as u64))
                    })
                    .and_then(Iterator::collect::<rusqlite::Result<Vec<_>>>)
            })
            .map_err(failed("stats"))?;
        committed.stats.oldest_completed_at_ms = tx
            .prepare_cached("SELECT MIN(completed_at_ms) FROM pending")
            .and_then(|mut statement| statement.query_row([], |row| row.get(0)))
            .map_err(failed("stats"))?;
        let used = integer("SELECT page_count FROM pragma_page_count", "stats")?
            - integer("SELECT freelist_count FROM pragma_freelist_count", "stats")?;
        committed.stats.used_bytes = (used.max(0) * db.page_size) as u64;
        committed.stats.max_bytes = (db.max_pages * db.page_size) as u64;
        committed.stats.limited_by_volume = db.limited_by_volume;

        // The reports of this transaction are being delivered from memory:
        // the writer was given up on while it waited. They must not be in
        // the file as well.
        if committed.stored > 0 && self.shared.takeovers.load(Ordering::SeqCst) != takeovers {
            return Err(Failure {
                taken_over: true,
                ..Failure::new("commit", "the reports were taken over")
            });
        }
        #[cfg(test)]
        {
            let delay = self.shared.commit_delay_ms.load(Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(delay));
        }
        tx.commit().map_err(failed("commit"))?;

        // What this says about the file. Something was written, so it takes
        // writes. It is full while new reports have no room below the
        // headroom, and, once an insert failed on a full volume, until an
        // insert works again.
        let wrote = db.conn.total_changes() != changes_before;
        if committed.stored > 0 {
            self.volume_full = false;
        }
        self.last_held = Some(committed.stats.pending_total() + committed.stats.rejected_total());
        let left_out = reports > 0 && committed.stored < work.reports.len();
        let room = db.max_pages - db.headroom_pages - used;
        let at_its_size =
            left_out || room < ROOM_PAGES || (self.health.full && room < 2 * ROOM_PAGES);
        committed.health = Health {
            available: self.health.available || wrote,
            full: at_its_size || self.volume_full,
        };
        Ok(committed)
    }
}

/// `duration` in milliseconds, for the file.
fn millis(duration: Duration) -> i64 {
    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

/// The device and inode of the file at `path`, `None` when there is none (or
/// on a system without them, where a file replaced under the process is not
/// noticed).
fn identity(path: &Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let metadata = std::fs::metadata(path).ok()?;
        Some((metadata.dev(), metadata.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// What is at `path` now: the device and inode of the file there, `None`
/// when there is none. An error when that cannot be found out, which says
/// nothing about the file.
fn whereabouts(path: &Path) -> std::io::Result<Option<(u64, u64)>> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match std::fs::metadata(path) {
            Ok(metadata) => Ok(Some((metadata.dev(), metadata.ino()))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(None)
    }
}

/// Open the file, creating it and its schema when they are not there.
/// `volume`: the size of its volume and what is free on it, when known.
fn open(config: &UsageOutboxConfig, volume: Option<(u64, u64)>) -> Result<Db, Failure> {
    create_private(&config.path).map_err(|error| Failure::new("open", error))?;
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
    // A rollback journal, never a write-ahead log: SQLite maps the index of
    // a write-ahead log into memory, and a mapped page that cannot be read
    // ends the process with a signal. With a rollback journal and no mapped
    // I/O every access to the file is a call that can fail. TRUNCATE keeps
    // the journal's file between transactions, so committing does not have
    // to create and delete one.
    let mode: String = conn
        .pragma_update_and_check(None, "journal_mode", "TRUNCATE", |row| row.get(0))
        .map_err(failed("open"))?;
    if !mode.eq_ignore_ascii_case("truncate") {
        return Err(Failure::new(
            "open",
            format!(
                "journal mode is {mode}, not TRUNCATE: is the file open elsewhere in WAL mode?"
            ),
        ));
    }
    conn.pragma_update(None, "mmap_size", 0)
        .map_err(failed("open"))?;
    // FULL: the journal is flushed before the file is changed and the file
    // before the journal is cleared, so a commit that returned is on the
    // disk and neither a crash of the process nor of the host can leave the
    // file half written. With a rollback journal NORMAL does not promise
    // that on power loss. The price is two flushes per transaction, paid by
    // the writer alone, and a transaction carries everything handed over
    // within `commit_interval`.
    conn.pragma_update(None, "synchronous", "FULL")
        .map_err(failed("open"))?;
    // Nothing of a statement is ever spilled to a file of its own.
    conn.pragma_update(None, "temp_store", "MEMORY")
        .map_err(failed("open"))?;
    migrate(&conn)?;

    // The file's own size: it is full while its volume still has room for
    // the journal, so what is done to the reports in it goes on.
    let page_size: i64 = conn
        .pragma_query_value(None, "page_size", |row| row.get(0))
        .map_err(failed("open"))?;
    let (max_pages, limited_by_volume) = set_size(&conn, config, page_size, volume)?;
    let id: String = conn
        .query_row("SELECT value FROM meta WHERE key = 'db_id'", [], |row| {
            row.get(0)
        })
        .map_err(failed("open"))?;
    Ok(Db {
        identity: identity(&config.path),
        id,
        max_pages,
        headroom_pages: headroom(max_pages),
        page_size,
        limited_by_volume,
        conn,
    })
}

/// Of `max_pages`, the pages new reports may not take.
fn headroom(max_pages: i64) -> i64 {
    (max_pages / 8).clamp(16, 4_096)
}

/// Tell SQLite the most pages the file may have: `max_bytes`, or what its
/// volume has room for when that is less. Returns what it set, and whether
/// the volume is why.
fn set_size(
    conn: &Connection,
    config: &UsageOutboxConfig,
    page_size: i64,
    volume: Option<(u64, u64)>,
) -> Result<(i64, bool), Failure> {
    let file_bytes = std::fs::metadata(&config.path).map_or(0, |file| file.len());
    let fits = volume.map(|(total, available)| fitting(file_bytes, total, available));
    let limited = fits.is_some_and(|fits| fits < config.max_bytes);
    let max_bytes = fits.map_or(config.max_bytes, |fits| fits.min(config.max_bytes));
    let wanted = i64::try_from(max_bytes).unwrap_or(i64::MAX) / page_size.max(1);
    // SQLite answers with what it set: never less than the file has.
    let max_pages: i64 = conn
        .pragma_update_and_check(None, "max_page_count", wanted.max(1), |row| row.get(0))
        .map_err(failed("open"))?;
    Ok((max_pages, limited))
}

/// The size a file of `file_bytes` may grow to on a volume of `total` bytes
/// of which `available` are free: all of it but a margin for the rollback
/// journal, `VOLUME_MARGIN_BYTES`, or a quarter of a volume smaller than
/// four times that.
fn fitting(file_bytes: u64, total: u64, available: u64) -> u64 {
    let margin = UsageOutboxConfig::VOLUME_MARGIN_BYTES.min(total / 4);
    file_bytes.saturating_add(available).saturating_sub(margin)
}

/// The size of the volume `path` is on and what is free on it for this
/// process, in bytes. `None` when the system does not say (or says the
/// volume has no size, as some do that have no limit of their own).
fn volume(path: &Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let directory = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let directory = std::ffi::CString::new(directory.as_os_str().as_bytes()).ok()?;
        let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: `directory` is a NUL-terminated string that outlives the
        // call, and `stat` is room for one `statvfs`, which the call fills
        // when it returns 0.
        let stat = unsafe {
            if libc::statvfs(directory.as_ptr(), stat.as_mut_ptr()) != 0 {
                return None;
            }
            stat.assume_init()
        };
        // (The integer types of these fields differ from one system to
        // another.)
        #[allow(clippy::unnecessary_cast)]
        let (block, blocks, available) = (
            stat.f_frsize as u64,
            stat.f_blocks as u64,
            stat.f_bavail as u64,
        );
        let total = blocks.saturating_mul(block);
        (total > 0).then(|| (total, available.saturating_mul(block)))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// Bring the schema to `SCHEMA_VERSION`, and give a new database its name.
/// A file written by a newer version of the proxy is refused rather than
/// guessed at.
fn migrate(conn: &Connection) -> Result<(), Failure> {
    let found: i64 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(failed("open"))?;
    if found > SCHEMA_VERSION {
        return Err(Failure::new(
            "open",
            format!("the file has schema version {found}, this proxy knows {SCHEMA_VERSION}"),
        ));
    }
    if found == SCHEMA_VERSION {
        return Ok(());
    }
    // Two processes may get here together; the second finds everything
    // made, and the name the first gave it.
    let db_id = uuid::Uuid::new_v4().to_string();
    conn.execute_batch(&format!(
        "BEGIN IMMEDIATE;{SCHEMA}\
         INSERT OR IGNORE INTO meta (key, value) VALUES ('db_id', '{db_id}');COMMIT;"
    ))
    .map_err(|error| {
        let _ = conn.execute_batch("ROLLBACK");
        failed("open")(error)
    })?;
    Ok(())
}

/// Make sure the file exists and nobody but its owner can read it. SQLite
/// gives the journal beside it the mode of the file itself.
///
/// A file that is there is not opened for this, only looked at by its path:
/// a process that closes any descriptor of a file gives up every lock it
/// holds on that file (the rule of POSIX locks), and SQLite may have this one
/// open and locked in this very process.
fn create_private(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let private = std::fs::Permissions::from_mode(0o600);
        match std::fs::metadata(path) {
            Ok(file) if file.permissions().mode() & 0o077 != 0 => {
                return std::fs::set_permissions(path, private);
            }
            Ok(_) => return Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        // A file made here is new: nobody holds a lock on it yet.
        let made = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path);
        match made {
            Ok(_) => {}
            // The other process on the outbox made it at this very moment.
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
#[path = "usage_outbox_tests.rs"]
mod tests;
