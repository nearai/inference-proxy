//! The store by itself: one or two handles on a file in a temporary
//! directory, and a plain SQLite connection beside them to look at what was
//! written, the way a person with the `sqlite3` CLI would.

use super::*;

/// A report as far as the store is concerned.
#[derive(Clone, Debug, PartialEq)]
struct Report {
    id: &'static str,
    model: Option<&'static str>,
    age: Duration,
    /// Sent from memory before the file took it: attempts made, why the last
    /// failed, when it is due again.
    tried: Option<(u32, &'static str, Duration)>,
    ended: Option<Rejection>,
    /// The serialized report: made once, borrowed by `row`.
    body: String,
}

fn report(id: &'static str) -> Report {
    Report {
        id,
        model: None,
        age: Duration::ZERO,
        tried: None,
        ended: None,
        body: format!(r#"{{"type":"chat_completion","id":"{id}"}}"#),
    }
}

impl Persist for Report {
    fn row(&self) -> NewRow<'_> {
        NewRow {
            body: self.body.as_bytes(),
            request_id: Some(self.id),
            model_label: self.model,
            auth_path: "cloud_api_key",
            ingress_route: "canonical",
            age: self.age,
            attempts: self.tried.map_or(0, |(attempts, _, _)| attempts),
            last_outcome: self.tried.map(|(_, outcome, _)| outcome),
            due_in: self.tried.map_or(Duration::ZERO, |(_, _, due_in)| due_in),
            rejected: self.ended,
        }
    }
}

/// A handle and what its writer says.
struct Handle {
    store: Store<Report>,
    events: mpsc::UnboundedReceiver<Event<Report>>,
}

/// Timings a test does not have to wait for.
fn config(path: &Path) -> UsageOutboxConfig {
    UsageOutboxConfig {
        commit_interval: Duration::from_millis(5),
        reopen_interval: Duration::from_millis(50),
        ..UsageOutboxConfig::at(path)
    }
}

/// A directory for one test and the outbox file in it.
fn file() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    (dir, path)
}

const LEASE: Duration = Duration::from_secs(60);

fn all(fresh: usize, retries: usize, lease: Duration) -> Claim {
    Claim {
        fresh,
        newest_first: false,
        retries,
        lease,
        expire: None,
    }
}

impl Handle {
    fn open(config: UsageOutboxConfig) -> Self {
        let (to_test, events) = mpsc::unbounded_channel();
        Self {
            store: Store::start(config, to_test),
            events,
        }
    }

    /// The handle once the file is open, the usual start of a test.
    fn opened(config: UsageOutboxConfig) -> Self {
        let mut handle = Self::open(config);
        handle.committed();
        handle
    }

    fn next(&mut self) -> Event<Report> {
        let give_up_at = Instant::now() + Duration::from_secs(30);
        loop {
            match self.events.try_recv() {
                Ok(event) => return event,
                Err(_) if Instant::now() < give_up_at => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("the store said nothing: {error}"),
            }
        }
    }

    /// The next transaction, which must have committed.
    fn committed(&mut self) -> Committed<Report> {
        match self.next() {
            Event::Committed(committed) => *committed,
            Event::Failed { error, .. } => panic!("a transaction failed: {error:?}"),
            Event::Replaced { why, .. } => panic!("the file was replaced: {why}"),
        }
    }

    /// The next transaction, which must have failed: why, what it handed
    /// back, whether it carried a claim, and what it left of the file.
    fn failed(&mut self) -> (Option<(&'static str, String)>, Vec<Report>, bool, Health) {
        match self.next() {
            Event::Failed {
                error,
                reports,
                claim,
                health,
            } => (error, reports, claim, health),
            Event::Committed(committed) => panic!("a transaction committed: {committed:?}"),
            Event::Replaced { why, .. } => panic!("the file was replaced: {why}"),
        }
    }

    /// The next thing said, which must be that the file was replaced.
    fn replaced(&mut self) -> (&'static str, Option<PathBuf>, u64) {
        match self.next() {
            Event::Replaced {
                why,
                kept_as,
                generation,
            } => (why, kept_as, generation),
            Event::Committed(committed) => panic!("a transaction committed: {committed:?}"),
            Event::Failed { error, .. } => panic!("a transaction failed: {error:?}"),
        }
    }

    /// Hand `reports` over and have them written now, with whatever else
    /// waits.
    fn keep(&mut self, reports: impl IntoIterator<Item = Report>) -> Committed<Report> {
        for report in reports {
            self.store.offer(report, usize::MAX).unwrap();
        }
        self.sync()
    }

    /// A transaction now, with whatever waits.
    fn sync(&mut self) -> Committed<Report> {
        self.store.sync();
        self.committed()
    }

    fn claim(&mut self, fresh: usize, retries: usize, lease: Duration) -> Claimed {
        self.store.claim(all(fresh, retries, lease));
        self.committed().claimed.expect("the answer to a claim")
    }

    /// Record an outcome and have it written now.
    fn settle(&mut self, row: &Row, outcome: Outcome) -> Committed<Report> {
        self.store.settle(Settle {
            id: row.id,
            generation: row.generation,
            outcome,
        });
        self.sync()
    }

    fn stats(&mut self) -> Stats {
        self.sync().stats
    }
}

/// The completion ids of `rows`.
fn ids(rows: &[Row]) -> Vec<String> {
    rows.iter()
        .map(|row| {
            let report: serde_json::Value = serde_json::from_str(&row.body).unwrap();
            report["id"].as_str().unwrap().to_string()
        })
        .collect()
}

/// One value read from the file by a connection of the test's own.
fn read<T: rusqlite::types::FromSql>(path: &Path, sql: &str) -> T {
    let conn = Connection::open(path).unwrap();
    conn.busy_timeout(Duration::from_secs(5)).unwrap();
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

fn failing(attempts: u32, in_ms: i64, clock: &Clock) -> Outcome {
    Outcome::Retry {
        attempts,
        next_attempt_at_ms: clock.now_ms() + in_ms,
        last_outcome: "http_5xx",
    }
}

const REFUSED: Rejection = Rejection {
    reason: "rejected",
    status: Some(422),
    outcome: "http_4xx",
};

// ---------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------

#[test]
fn the_file_is_private_has_a_rollback_journal_and_is_never_mapped() {
    let (dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep((0..200).map(|_| report("one")));
    let rows = handle.claim(50, 0, LEASE);
    for row in &rows.fresh {
        handle.settle(row, Outcome::Delete);
    }

    // A rollback journal: the file's header says so (1; a write-ahead log
    // would be 2), and the journal is kept, empty, between transactions.
    let header = std::fs::read(&path).unwrap();
    assert_eq!((header[18], header[19]), (1, 1));
    assert_eq!(
        std::fs::metadata(dir.path().join("outbox.db-journal"))
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        read::<i64>(&path, "PRAGMA user_version"),
        SCHEMA_VERSION,
        "the schema version the migration left"
    );
    for table in [
        "pending",
        "rejected",
        "pending_counts",
        "rejected_counts",
        "meta",
    ] {
        let found: i64 = read(
            &path,
            &format!(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = '{table}'"
            ),
        );
        assert_eq!(found, 1, "{table}");
    }
    // The file and its journal, and nothing else: no write-ahead log, and no
    // shared-memory file, which is what SQLite maps.
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["outbox.db", "outbox.db-journal"]);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for name in &names {
            let mode = std::fs::metadata(dir.path().join(name))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "{name}");
        }
    }
    // No part of it is in this process's address space.
    #[cfg(target_os = "linux")]
    {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let name = dir.path().to_string_lossy().into_owned();
        assert!(
            !maps.contains(&name),
            "{}",
            maps.lines()
                .filter(|line| line.contains(&name))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

#[cfg(unix)]
#[test]
fn a_file_that_others_could_read_is_made_private() {
    use std::os::unix::fs::PermissionsExt;
    let (_dir, path) = file();
    std::fs::write(&path, b"").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    let _handle = Handle::opened(config(&path));
    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600);
}

#[test]
fn what_is_handed_over_together_is_written_by_one_transaction() {
    let (_dir, path) = file();
    let interval = Duration::from_millis(600);
    let mut handle = Handle::opened(UsageOutboxConfig {
        commit_interval: interval,
        ..config(&path)
    });
    handle.keep(["earlier"].map(report));
    let earlier = handle.claim(1, 0, LEASE).fresh.remove(0);

    let started_at = Instant::now();
    for id in ["one", "two", "three"] {
        handle.store.offer(report(id), usize::MAX).unwrap();
    }
    // An outcome waits for the same transaction: its report stays leased
    // until then, so nothing else is done with it.
    handle.store.settle(Settle {
        id: earlier.id,
        generation: earlier.generation,
        outcome: Outcome::Delete,
    });
    // Nothing is written before the interval is over: the writer is waiting
    // for more to join these.
    std::thread::sleep(Duration::from_millis(50));
    let rows: i64 = read(&path, "SELECT COUNT(*) FROM pending");
    if started_at.elapsed() < interval {
        assert_eq!(rows, 1);
    }
    let committed = handle.committed();
    assert!(started_at.elapsed() >= interval);
    assert_eq!(committed.stored, 3);
    assert_eq!(committed.stats.pending, [(None, 3)]);
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 3);

    // A claim does not wait for the interval, and takes along a report that
    // was handed over just before it.
    let started_at = Instant::now();
    handle.store.offer(report("four"), usize::MAX).unwrap();
    handle.store.claim(all(10, 10, LEASE));
    let committed = handle.committed();
    assert!(started_at.elapsed() < interval);
    assert_eq!(committed.stored, 1);
    assert_eq!(
        ids(&committed.claimed.unwrap().fresh),
        ["one", "two", "three", "four"]
    );
}

#[test]
fn a_row_holds_the_report_its_labels_and_when_its_request_completed() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    let before = Clock::default().now_ms();
    handle.keep([Report {
        model: Some("example/alpha"),
        age: Duration::from_secs(90),
        ..report("one")
    }]);

    let rows = handle.claim(1, 0, LEASE).fresh;
    let row = &rows[0];
    assert_eq!(row.body, r#"{"type":"chat_completion","id":"one"}"#);
    assert_eq!(row.request_id.as_deref(), Some("one"));
    assert_eq!(row.model_label.as_deref(), Some("example/alpha"));
    assert_eq!(row.auth_path, "cloud_api_key");
    assert_eq!(row.ingress_route, "canonical");
    assert_eq!((row.attempts, row.last_outcome.as_deref()), (0, None));
    // 90 seconds before it was written, not when it was written.
    let completed_ago = before - row.completed_at_ms;
    assert!(
        (89_000..=91_000).contains(&completed_ago),
        "{completed_ago}"
    );

    // The columns are the ones documented, and no other.
    let conn = Connection::open(&path).unwrap();
    let columns: Vec<String> = conn
        .prepare("SELECT name FROM pragma_table_info('pending') ORDER BY cid")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        columns,
        [
            "id",
            "body",
            "request_id",
            "model_label",
            "auth_path",
            "ingress_route",
            "completed_at_ms",
            "attempts",
            "next_attempt_at_ms",
            "lease_until_ms",
            "lease_owner",
            "last_outcome",
        ]
    );
}

// ---------------------------------------------------------------------------
// Claims and outcomes
// ---------------------------------------------------------------------------

#[test]
fn a_claim_leases_the_reports_that_waited_longest_and_nobody_else_gets_them_until_the_lease_runs_out(
) {
    let (_dir, path) = file();
    let mut first = Handle::opened(config(&path));
    // Written in the order they completed.
    first.keep([
        Report {
            age: Duration::from_secs(3),
            ..report("oldest")
        },
        Report {
            age: Duration::from_secs(2),
            ..report("older")
        },
        Report {
            age: Duration::from_secs(1),
            ..report("old")
        },
    ]);

    let mut second = Handle::opened(config(&path));
    let lease = Duration::from_millis(1_500);
    let leased_at = Instant::now();
    assert_eq!(ids(&first.claim(2, 2, lease).fresh), ["oldest", "older"]);
    // The other handle gets what is left, and then nothing.
    assert_eq!(ids(&second.claim(5, 5, LEASE).fresh), ["old"]);
    assert_eq!(ids(&first.claim(5, 5, LEASE).fresh), [] as [&str; 0]);
    assert_eq!(ids(&second.claim(5, 5, LEASE).fresh), [] as [&str; 0]);
    assert!(leased_at.elapsed() < lease, "the test was too slow to tell");
    assert_eq!(
        read::<String>(
            &path,
            "SELECT group_concat(DISTINCT lease_owner) FROM pending WHERE id <= 2"
        ),
        first.store.owner()
    );

    // The holder of the first two never says what became of them. Once
    // their lease has run out they are anybody's but its own: it is sending
    // them, as far as it knows.
    std::thread::sleep(lease.saturating_sub(leased_at.elapsed()) + Duration::from_millis(20));
    assert_eq!(ids(&first.claim(5, 5, LEASE).fresh), [] as [&str; 0]);
    assert_eq!(ids(&second.claim(5, 5, LEASE).fresh), ["oldest", "older"]);
    // Leased rows still count as pending: they are not accepted yet.
    assert_eq!(second.stats().pending, [(None, 3)]);
}

#[test]
fn reports_never_tried_and_reports_that_failed_are_asked_for_apart() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    let clock = Clock::default();
    handle.keep(["a", "b", "c", "d", "e", "f"].map(report));
    let rows = handle.claim(3, 0, LEASE).fresh;
    assert_eq!(ids(&rows), ["a", "b", "c"]);
    // The three fail, and are due again at once.
    for row in &rows {
        handle.settle(row, failing(1, -1, &clock));
    }

    // However many of them are due, a claim for first attempts gets none of
    // them, and the other way round.
    let claimed = handle.claim(2, 0, LEASE);
    assert_eq!(ids(&claimed.fresh), ["d", "e"]);
    assert!(claimed.retries.is_empty() && claimed.more_fresh);
    let claimed = handle.claim(0, 2, LEASE);
    assert!(claimed.fresh.is_empty());
    assert_eq!(ids(&claimed.retries), ["a", "b"]);
    assert!(claimed.more_retries);
    let claimed = handle.claim(5, 5, LEASE);
    assert_eq!(ids(&claimed.fresh), ["f"]);
    assert_eq!(ids(&claimed.retries), ["c"]);
    assert!(!claimed.more_fresh && !claimed.more_retries);

    // Each is found through an index of its own, however many of the other
    // kind stand before it in time.
    let conn = Connection::open(&path).unwrap();
    for (class, index) in [
        ("attempts = 0", "pending_fresh"),
        ("attempts > 0", "pending_retry"),
    ] {
        let plan: String = conn
            .query_row(
                &format!(
                    "EXPLAIN QUERY PLAN SELECT id FROM pending WHERE {class} \
                     AND next_attempt_at_ms <= 5 AND lease_until_ms <= 5 \
                     ORDER BY next_attempt_at_ms, id LIMIT 8"
                ),
                [],
                |row| row.get(3),
            )
            .unwrap();
        assert!(plan.contains(index), "{class}: {plan}");
    }
    let plan: String = conn
        .query_row(
            "EXPLAIN QUERY PLAN SELECT MIN(completed_at_ms) FROM pending",
            [],
            |row| row.get(3),
        )
        .unwrap();
    assert!(plan.contains("pending_completed"), "{plan}");
}

#[test]
fn a_probe_takes_the_report_that_came_in_last() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep(["a", "b", "c"].map(report));
    std::thread::sleep(Duration::from_millis(3));
    handle.keep(["d", "e"].map(report));

    // The newest of those never tried, when that is what is asked for.
    handle.store.claim(Claim {
        newest_first: true,
        ..all(1, 0, LEASE)
    });
    let claimed = handle.committed().claimed.expect("the answer to a claim");
    assert_eq!(ids(&claimed.fresh), ["e"]);
    assert!(claimed.more_fresh);
    // Everybody else gets the one that waited longest, as ever.
    assert_eq!(ids(&handle.claim(2, 0, LEASE).fresh), ["a", "b"]);
    handle.store.claim(Claim {
        newest_first: true,
        ..all(5, 0, LEASE)
    });
    let claimed = handle.committed().claimed.expect("the answer to a claim");
    assert_eq!(ids(&claimed.fresh), ["c", "d"]);

    // Found from the other end of the same index.
    let plan: String = Connection::open(&path)
        .unwrap()
        .query_row(
            "EXPLAIN QUERY PLAN SELECT id FROM pending WHERE attempts = 0 \
             AND lease_until_ms <= 5 ORDER BY next_attempt_at_ms DESC, id DESC LIMIT 1",
            [],
            |row| row.get(3),
        )
        .unwrap();
    assert!(plan.contains("pending_fresh"), "{plan}");
}

#[test]
fn what_became_of_a_report_is_written_to_its_row() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    let clock = Clock::default();
    handle.keep(["accepted", "failing", "refused", "unstarted"].map(report));
    let rows = handle.claim(4, 0, LEASE).fresh;
    let row = |name: &str| &rows[ids(&rows).iter().position(|id| id == name).unwrap()];

    // Accepted: the row is gone.
    handle.settle(row("accepted"), Outcome::Delete);
    // Failed in a way that can pass: due again later, the lease given back.
    let retry_at = clock.now_ms() + 1_000;
    handle.settle(row("failing"), failing(1, 1_000, &clock));
    // Refused for good: moved to `rejected` with what the answer was.
    let committed = handle.settle(
        row("refused"),
        Outcome::Reject {
            why: REFUSED,
            attempts: 1,
        },
    );
    assert_eq!(committed.stats.rejected, [("rejected".to_string(), 1)]);
    // Not started after all: as if it had never been leased.
    let committed = handle.settle(row("unstarted"), Outcome::Release);
    assert_eq!(committed.stats.pending, [(None, 2)]);

    let (reason, status, outcome, body): (String, i64, String, String) = {
        let conn = Connection::open(&path).unwrap();
        conn.query_row(
            "SELECT reason, status, outcome, body FROM rejected",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap()
    };
    assert_eq!(
        (reason.as_str(), status, outcome.as_str()),
        ("rejected", 422, "http_4xx")
    );
    assert_eq!(body, report("refused").body);

    // Only the released one is due now. The failing one tells when it is.
    let claimed = handle.claim(4, 4, LEASE);
    assert_eq!(ids(&claimed.fresh), ["unstarted"]);
    assert!(claimed.retries.is_empty());
    let due_ms = claimed.next_retry_ms.expect("a retry is waiting");
    assert!(
        (retry_at - 50..=retry_at + 50).contains(&due_ms),
        "{due_ms} vs {retry_at}"
    );
    assert!(clock.now_ms() < retry_at, "the test was too slow to tell");
    std::thread::sleep(Duration::from_millis(
        (retry_at - clock.now_ms()).max(0) as u64 + 30,
    ));
    let again = handle.claim(4, 4, LEASE).retries;
    assert_eq!(ids(&again), ["failing"]);
    assert_eq!(
        (again[0].attempts, again[0].last_outcome.as_deref()),
        (1, Some("http_5xx"))
    );
}

#[test]
fn an_outcome_written_after_the_lease_ran_out_does_not_undo_the_next_holders_lease() {
    let (_dir, path) = file();
    let mut slow = Handle::opened(config(&path));
    let mut next = Handle::opened(config(&path));
    let clock = Clock::default();
    slow.keep([report("one")]);
    let row = slow.claim(1, 0, Duration::from_millis(50)).fresh.remove(0);
    std::thread::sleep(Duration::from_millis(70));
    assert_eq!(ids(&next.claim(1, 0, LEASE).fresh), ["one"]);

    // The first holder's attempt ends late, with a failure.
    slow.settle(&row, failing(1, -1, &clock));
    slow.settle(&row, Outcome::Release);
    // The row is still the second holder's, untouched.
    let claimed = slow.claim(1, 1, LEASE);
    assert!(claimed.fresh.is_empty() && claimed.retries.is_empty());
    assert_eq!(
        read::<String>(&path, "SELECT lease_owner FROM pending"),
        next.store.owner()
    );
    assert_eq!(read::<i64>(&path, "SELECT attempts FROM pending"), 0);
    // An acceptance, though, is an acceptance whoever got it.
    slow.settle(&row, Outcome::Delete);
    assert_eq!(next.stats().pending, [(None, 0)]);
}

#[test]
fn a_report_sent_from_memory_first_is_written_with_what_happened_to_it() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep([
        report("never-tried"),
        // Three attempts from memory, due again in a moment.
        Report {
            tried: Some((3, "timeout", Duration::from_millis(300))),
            ..report("tried")
        },
        // Refused while the file could not take it: straight to `rejected`.
        Report {
            tried: Some((1, "http_4xx", Duration::ZERO)),
            ended: Some(REFUSED),
            ..report("ended")
        },
    ]);
    let stats = handle.stats();
    assert_eq!(stats.pending, [(None, 2)]);
    assert_eq!(stats.rejected, [("rejected".to_string(), 1)]);
    assert_eq!(
        read::<String>(
            &path,
            "SELECT json_extract(body, '$.id') || ' ' || status || ' ' || attempts FROM rejected"
        ),
        "ended 422 1"
    );

    let claimed = handle.claim(5, 5, LEASE);
    assert_eq!(ids(&claimed.fresh), ["never-tried"]);
    assert!(claimed.retries.is_empty(), "not due yet");
    std::thread::sleep(Duration::from_millis(330));
    let retries = handle.claim(5, 5, LEASE).retries;
    assert_eq!(ids(&retries), ["tried"]);
    assert_eq!(
        (retries[0].attempts, retries[0].last_outcome.as_deref()),
        (3, Some("timeout"))
    );
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

#[test]
fn rejected_keeps_the_newest_rows_and_counts_them_by_reason_without_reading_them() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(UsageOutboxConfig {
        max_rejected: 2,
        ..config(&path)
    });
    handle.keep(["one", "two", "three", "four"].map(report));
    let rows = handle.claim(4, 0, LEASE).fresh;

    let mut evicted = 0;
    for (n, row) in rows.iter().enumerate() {
        let why = Rejection {
            reason: if n % 2 == 0 { "rejected" } else { "max_age" },
            status: None,
            outcome: "timeout",
        };
        evicted += handle
            .settle(row, Outcome::Reject { why, attempts: 3 })
            .rejected_evicted;
    }
    assert_eq!(evicted, 2);
    let stats = handle.stats();
    assert_eq!(stats.rejected_total(), 2);
    assert_eq!(stats.pending_total(), 0);
    let mut by_reason = stats.rejected.clone();
    by_reason.sort();
    assert_eq!(
        by_reason,
        [("max_age".to_string(), 1), ("rejected".to_string(), 1)]
    );
    let kept: String = read(
        &path,
        "SELECT group_concat(json_extract(body, '$.id')) FROM (SELECT body FROM rejected ORDER BY id)",
    );
    assert_eq!(kept, "three,four");
    // No answer at all: the status is empty, not zero.
    assert_eq!(
        read::<i64>(&path, "SELECT COUNT(*) FROM rejected WHERE status IS NULL"),
        2
    );
    // The count comes from a table of its own, kept by triggers: a row moved
    // by hand is counted, and nothing is counted by reading `rejected`.
    Connection::open(&path)
        .unwrap()
        .execute("DELETE FROM rejected WHERE reason = 'max_age'", [])
        .unwrap();
    let mut by_reason = handle.stats().rejected;
    by_reason.sort();
    assert_eq!(
        by_reason,
        [("max_age".to_string(), 0), ("rejected".to_string(), 1)]
    );
}

#[test]
fn the_bound_drops_the_reports_that_waited_longest_but_never_one_being_sent() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(UsageOutboxConfig {
        max_pending: 3,
        ..config(&path)
    });
    let committed = handle.keep(["one", "two", "three"].map(report));
    assert!(committed.evicted.is_empty());
    // "one" is being sent.
    assert_eq!(ids(&handle.claim(1, 0, LEASE).fresh), ["one"]);

    let committed = handle.keep(["four", "five"].map(report));
    assert_eq!(committed.stored, 2);
    assert_eq!(ids(&committed.evicted), ["two", "three"]);
    assert_eq!(committed.stats.pending, [(None, 3)]);
    assert_eq!(
        read::<String>(
            &path,
            "SELECT group_concat(json_extract(body, '$.id')) FROM (SELECT body FROM pending ORDER BY id)"
        ),
        "one,four,five"
    );
}

#[test]
fn the_backlog_is_counted_per_model_and_its_oldest_report_is_the_oldest_whoever_wrote_it() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    let stats = handle.stats();
    assert_eq!(
        (stats.pending_total(), stats.oldest_completed_at_ms),
        (0, None)
    );

    let before = Clock::default().now_ms();
    let committed = handle.keep([
        Report {
            age: Duration::from_secs(30),
            model: Some("example/alpha"),
            ..report("one")
        },
        Report {
            model: Some("example/beta"),
            ..report("two")
        },
        Report {
            model: Some("example/alpha"),
            ..report("three")
        },
    ]);
    let mut pending = committed.stats.pending.clone();
    pending.sort();
    assert_eq!(
        pending,
        [
            (Some("example/alpha".to_string()), 2),
            (Some("example/beta".to_string()), 1)
        ]
    );
    let oldest_ago = before - committed.stats.oldest_completed_at_ms.unwrap();
    assert!((29_000..=31_000).contains(&oldest_ago), "{oldest_ago}");
    assert!(committed.stats.used_bytes > 0);
    assert_eq!(
        committed.stats.max_bytes,
        UsageOutboxConfig::DEFAULT_MAX_BYTES
    );

    // A report put in by hand, older than all the others and with the
    // highest id of them: it is the oldest all the same.
    Connection::open(&path)
        .unwrap()
        .execute(
            "INSERT INTO pending (body, auth_path, ingress_route, completed_at_ms) \
             VALUES ('{}', 'cloud_api_key', 'other', ?1)",
            [before - 3_600_000],
        )
        .unwrap();
    assert_eq!(
        handle.stats().oldest_completed_at_ms,
        Some(before - 3_600_000)
    );

    // A model whose reports are all gone stays in the count, at zero, so
    // that its gauge comes down with it.
    let rows = handle.claim(5, 0, LEASE).fresh;
    for row in rows
        .iter()
        .filter(|row| row.model_label.as_deref() == Some("example/alpha"))
    {
        handle.settle(row, Outcome::Delete);
    }
    let mut pending = handle.stats().pending;
    pending.sort();
    assert_eq!(
        pending,
        [
            (None, 1),
            (Some("example/alpha".to_string()), 0),
            (Some("example/beta".to_string()), 1)
        ]
    );
}

#[test]
fn no_more_reports_wait_to_be_written_than_allowed() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(UsageOutboxConfig {
        commit_interval: Duration::from_secs(5),
        ..config(&path)
    });
    handle.store.offer(report("one"), 2).unwrap();
    handle.store.offer(report("two"), 2).unwrap();
    let (back, why) = handle.store.offer(report("three"), 2).unwrap_err();
    assert_eq!((back, why), (report("three"), Refused::BufferFull));
    assert_eq!(why.as_label(), "buffer_full");
    // What was taken is written when the next transaction comes.
    assert_eq!(handle.sync().stored, 2);
}

#[test]
fn many_reports_are_written_in_transactions_of_bounded_size() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(UsageOutboxConfig {
        commit_interval: Duration::from_millis(100),
        ..config(&path)
    });
    let reports = 2 * MAX_BATCH + 10;
    for _ in 0..reports {
        handle.store.offer(report("many"), usize::MAX).unwrap();
    }
    let mut transactions = Vec::new();
    while transactions.iter().sum::<usize>() < reports {
        transactions.push(handle.committed().stored);
    }
    assert!(transactions.iter().all(|stored| *stored <= MAX_BATCH));
    assert!(transactions.len() >= 3, "{transactions:?}");
    assert_eq!(
        read::<i64>(&path, "SELECT COUNT(*) FROM pending"),
        reports as i64
    );
}

// ---------------------------------------------------------------------------
// The size of the file
// ---------------------------------------------------------------------------

/// A report with a body of about a kilobyte, so that a small file fills up.
fn bulky(id: &'static str) -> Report {
    Report {
        body: format!(r#"{{"id":"{id}","padding":"{}"}}"#, "x".repeat(1_000)),
        ..report(id)
    }
}

#[test]
fn a_file_at_its_size_takes_no_new_report_and_goes_on_with_the_ones_it_holds() {
    let (_dir, path) = file();
    let max_bytes = UsageOutboxConfig::MIN_MAX_BYTES;
    let mut handle = Handle::opened(UsageOutboxConfig {
        max_bytes,
        ..config(&path)
    });
    let clock = Clock::default();

    // It fills up, to the line new reports stay behind and not beyond. What
    // has no room is handed back, by the transaction that wrote the rest.
    let (mut offered, mut stored, mut no_room) = (0, 0, 0);
    let mut health = Health::default();
    while handle.store.is_accepting() {
        let committed = handle.keep((0..300).map(|_| bulky("filling")));
        offered += 300;
        stored += committed.stored;
        no_room += committed.no_room.len();
        health = committed.health;
        assert!(stored < 100_000, "the file never filled up");
    }
    assert!(stored > 1_000, "{stored}");
    assert_eq!(stored + no_room, offered);
    assert_eq!(
        health,
        Health {
            available: true,
            full: true
        }
    );
    let (_, why) = handle
        .store
        .offer(bulky("refused"), usize::MAX)
        .unwrap_err();
    assert_eq!(why, Refused::Full);
    // Below the size it was given, and by the headroom at least.
    let size = std::fs::metadata(&path).unwrap().len();
    assert!(size <= max_bytes, "{size}");
    let stats = handle.stats();
    assert!(stats.used_bytes <= max_bytes - max_bytes / 8, "{stats:?}");
    assert_eq!(stats.max_bytes, max_bytes);

    // Everything that is done to a report it holds still works: leasing it,
    // a failure with its reason, a refusal that moves it to `rejected`, and
    // accepting it, which frees the room.
    let rows = handle.claim(90, 0, LEASE).fresh;
    assert_eq!(rows.len(), 90);
    for (n, row) in rows.iter().enumerate() {
        let outcome = match n % 3 {
            0 => failing(1, -1, &clock),
            1 => Outcome::Reject {
                why: REFUSED,
                attempts: 1,
            },
            _ => Outcome::Delete,
        };
        handle.store.settle(Settle {
            id: row.id,
            generation: row.generation,
            outcome,
        });
    }
    let committed = handle.sync();
    assert_eq!(committed.stats.rejected_total(), 30);
    assert_eq!(committed.stats.pending_total(), stored as u64 - 60);
    assert_eq!(handle.claim(0, 600, LEASE).retries.len(), 30);
    assert!(std::fs::metadata(&path).unwrap().len() <= max_bytes);
    assert!(committed.health.full);

    // Enough of them accepted, a few at a time, and it takes reports again
    // by itself: once, when it has room for a good many, and not at the
    // first page that comes free (pages come and go at the line).
    let line = max_bytes - max_bytes / 8;
    let mut accepting = handle.store.is_accepting();
    let mut least_room = u64::MAX;
    while !accepting {
        let rows = handle.claim(20, 0, LEASE).fresh;
        assert!(!rows.is_empty(), "the file never had room again");
        for row in &rows {
            handle.store.settle(Settle {
                id: row.id,
                generation: row.generation,
                outcome: Outcome::Delete,
            });
        }
        let committed = handle.sync();
        accepting = !committed.health.full;
        assert_eq!(accepting, handle.store.is_accepting());
        if accepting {
            least_room = line - committed.stats.used_bytes;
        }
    }
    assert!(least_room >= 64 * 4096, "{least_room} bytes of room");
    assert!(least_room < 80 * 4096, "{least_room} bytes of room");
    assert_eq!(handle.keep([bulky("after")]).stored, 1);
}

#[test]
fn a_full_volume_stops_new_reports_and_nothing_else_of_the_transaction() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep(["one", "two"].map(report));
    let rows = handle.claim(2, 0, LEASE).fresh;

    // The volume has no room for a new row; what is in the file can still
    // be changed (it was not growing).
    handle.store.inject_volume_full(true);
    handle.store.offer(report("three"), usize::MAX).unwrap();
    handle.store.settle(Settle {
        id: rows[0].id,
        generation: rows[0].generation,
        outcome: Outcome::Delete,
    });
    handle.store.claim(all(5, 5, LEASE));
    let committed = handle.committed();
    // The report is handed back; the outcome and the claim went through.
    assert_eq!(committed.stored, 0);
    assert_eq!(committed.no_room, [report("three")]);
    assert_eq!(committed.stats.pending, [(None, 1)]);
    assert!(committed.claimed.is_some());
    assert_eq!(
        committed.health,
        Health {
            available: true,
            full: true
        }
    );
    assert!(!handle.store.is_accepting());

    // It stays full for as long as a report offered all the same (at
    // shutdown, when there is no other chance) is handed back.
    handle.store.offer_anyway(report("last")).unwrap();
    let committed = handle.sync();
    assert_eq!((committed.stored, committed.no_room.len()), (0, 1));
    assert!(committed.health.full);
    // A transaction that writes no report says nothing about room.
    assert!(handle.sync().health.full);

    // Room again, and no report waiting to find out. The file is made to
    // grow a little every now and then; that it could is what says so.
    handle.store.inject_volume_full(false);
    std::thread::sleep(Duration::from_millis(60));
    let committed = handle.sync();
    assert_eq!(
        committed.health,
        Health {
            available: true,
            full: false
        }
    );
    assert!(handle.store.is_accepting());
    // The pages it grew by stay with the file, for the reports to come, and
    // nothing else is left of the try.
    assert_eq!(
        read::<i64>(&path, "SELECT COUNT(*) FROM meta WHERE key = 'room'"),
        0
    );
    assert!(read::<i64>(&path, "SELECT freelist_count FROM pragma_freelist_count") >= 16);
    assert_eq!(handle.keep([report("three")]).stored, 1);
    assert_eq!(handle.stats().pending, [(None, 2)]);
}

// ---------------------------------------------------------------------------
// A file that fails
// ---------------------------------------------------------------------------

#[test]
fn handing_a_report_over_never_waits_for_the_file() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));

    // Somebody else holds the write lock for a while.
    let locked = Duration::from_millis(1_500);
    let other = Connection::open(&path).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    let locked_at = Instant::now();

    let mut took = Vec::new();
    for round in 0..3_000 {
        let started_at = Instant::now();
        handle.store.offer(report("blocked"), usize::MAX).unwrap();
        took.push(started_at.elapsed());
        if round % 1_000 == 0 {
            // The writer is in its transaction by now, waiting for the lock.
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    assert!(
        locked_at.elapsed() < locked,
        "the test was too slow to tell"
    );
    assert!(
        handle.events.try_recv().is_err(),
        "nothing can have been written"
    );
    // A call that waited for the file would have taken the rest of the
    // second and a half. One in a hundred may be whatever else the machine
    // was doing; the others take microseconds.
    took.sort();
    let (p99, all) = (took[took.len() * 99 / 100], took.iter().sum::<Duration>());
    assert!(p99 < Duration::from_millis(1), "p99 {p99:?}");
    assert!(all < Duration::from_millis(300), "3000 calls took {all:?}");

    std::thread::sleep(locked.saturating_sub(locked_at.elapsed()));
    other.execute_batch("COMMIT").unwrap();
    let mut stored = 0;
    while stored < 3_000 {
        stored += handle.committed().stored;
    }
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 3_000);
}

#[test]
fn a_file_that_cannot_be_opened_hands_reports_back_and_is_tried_again() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("not-there-yet");
    let path = missing.join("outbox.db");
    let mut handle = Handle::open(UsageOutboxConfig {
        reopen_interval: Duration::from_millis(1_000),
        ..config(&path)
    });
    let opened_at = Instant::now();

    // The first thing the writer does is open the file.
    let (error, reports, claim, health) = handle.failed();
    let (op, _) = error.expect("opening was tried");
    assert_eq!((op, reports.len(), claim), ("open", 0, false));
    assert_eq!(health, Health::default());
    assert!(!handle.store.is_accepting());

    // From then on nothing is taken, so nothing waits for a file that is
    // not there.
    let (back, why) = handle.store.offer(report("one"), usize::MAX).unwrap_err();
    assert_eq!((back, why.as_label()), (report("one"), "unavailable"));
    // A claim is answered, with nothing, and the file is left alone for a
    // while: no second error yet.
    handle.store.claim(all(5, 5, LEASE));
    let (error, _, claim, _) = handle.failed();
    assert!(
        opened_at.elapsed() < Duration::from_millis(900),
        "the test was too slow to tell"
    );
    assert!(error.is_none() && claim, "{error:?}");

    // The directory appears. The next look after the interval opens it, and
    // writing to it is what says it is available.
    std::fs::create_dir(&missing).unwrap();
    std::thread::sleep(Duration::from_millis(1_050));
    let committed = handle.sync();
    assert_eq!(
        committed.health,
        Health {
            available: true,
            full: false
        }
    );
    assert!(handle.store.is_accepting());
    assert_eq!(handle.keep([report("one")]).stored, 1);
}

#[test]
fn a_transaction_that_fails_writes_nothing_hands_its_reports_back_and_keeps_the_outcomes() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep(["one", "two"].map(report));
    let rows = handle.claim(2, 0, LEASE).fresh;

    handle.store.inject_fault(true);
    handle.store.offer(report("three"), usize::MAX).unwrap();
    // "one" was accepted while the file could not be written.
    handle.store.settle(Settle {
        id: rows[0].id,
        generation: rows[0].generation,
        outcome: Outcome::Delete,
    });
    handle.store.sync();
    let (error, reports, _, health) = handle.failed();
    assert_eq!(error.unwrap().0, "open");
    assert_eq!(reports, [report("three")]);
    assert!(!health.available && !handle.store.is_accepting());
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 2);

    // Tried again after the interval, and failing again: a second error.
    std::thread::sleep(Duration::from_millis(60));
    handle.store.sync();
    let tried_again = loop {
        if let (Some((op, _)), _, _, _) = handle.failed() {
            break op;
        }
    };
    assert_eq!(tried_again, "open");

    // Back: the outcome that could not be written is written now, so "one"
    // is not sent a second time.
    handle.store.inject_fault(false);
    std::thread::sleep(Duration::from_millis(60));
    let committed = handle.sync();
    assert!(committed.health.available);
    assert_eq!(committed.stats.pending, [(None, 1)]);
    assert_eq!(
        read::<String>(&path, "SELECT json_extract(body, '$.id') FROM pending"),
        "two"
    );
}

#[test]
fn only_something_written_says_the_file_is_available_again() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(UsageOutboxConfig {
        reopen_interval: Duration::from_millis(20),
        ..config(&path)
    });
    handle.store.inject_fault(true);
    handle.store.sync();
    let (_, _, _, health) = handle.failed();
    assert!(!health.available);

    // The file works again, and nothing waits to be written. The next
    // transaction would change nothing by itself: it writes a row to find
    // out, and that row is the evidence.
    handle.store.inject_fault(false);
    std::thread::sleep(Duration::from_millis(40));
    let probed_before: Option<i64> = Connection::open(&path)
        .unwrap()
        .query_row(
            "SELECT value FROM meta WHERE key = 'probed_at_ms'",
            [],
            |row| row.get(0),
        )
        .ok();
    let committed = handle.sync();
    assert!(committed.health.available);
    let probed: i64 = read(&path, "SELECT value FROM meta WHERE key = 'probed_at_ms'");
    assert!(probed_before.is_none_or(|before| probed > before));
    // While it is available nothing is written to find out anything.
    handle.sync();
    assert_eq!(
        read::<i64>(&path, "SELECT value FROM meta WHERE key = 'probed_at_ms'"),
        probed
    );
}

#[test]
fn a_file_from_a_newer_proxy_is_left_alone() {
    let (_dir, path) = file();
    {
        let mut handle = Handle::opened(config(&path));
        handle.keep([report("one")]);
        let closed = handle.store.close();
        assert!(closed.blocking_recv().unwrap().is_some());
    }
    Connection::open(&path)
        .unwrap()
        .pragma_update(None, "user_version", SCHEMA_VERSION + 1)
        .unwrap();

    let mut handle = Handle::open(config(&path));
    let (op, error) = handle.failed().0.unwrap();
    assert_eq!(op, "open");
    assert!(error.contains("schema version"), "{error}");
    // Nothing in it was touched, and it was not moved.
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 1);
}

/// The files in `dir`, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn a_file_that_is_not_a_database_is_moved_aside_and_a_new_one_started() {
    let (dir, path) = file();
    let junk = b"this is not an SQLite file, whatever its name says";
    std::fs::write(&path, junk).unwrap();
    let mut handle = Handle::open(config(&path));

    let (why, kept_as, generation) = handle.replaced();
    let kept_as = kept_as.expect("where the old file is now");
    assert_eq!((why, generation), ("corrupt", 1));
    // Kept as it was, under a name that says when, for a person to look at.
    assert_eq!(std::fs::read(&kept_as).unwrap(), junk);
    let name = kept_as.file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.starts_with("outbox.db.corrupt-"), "{name}");
    // And the outbox works, in a new file.
    let committed = handle.committed();
    assert!(committed.health.available);
    assert_eq!(handle.keep([report("one")]).stored, 1);
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 1);
    assert_eq!(names(dir.path()).len(), 3, "{:?}", names(dir.path()));
}

#[test]
fn a_file_damaged_under_the_running_store_is_moved_aside_and_what_was_in_hand_is_kept() {
    let (dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep((0..400).map(|_| bulky("before")));
    let leased = handle.claim(1, 0, LEASE).fresh.remove(0);

    // Everything after the header is overwritten, and the header says the
    // file changed, so that what the connection remembers of it is not
    // trusted any more.
    {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let len = file.metadata().unwrap().len();
        let mut counter = [0u8; 4];
        file.seek(SeekFrom::Start(24)).unwrap();
        file.read_exact(&mut counter).unwrap();
        let changed = (u32::from_be_bytes(counter) + 1).to_be_bytes();
        for offset in [24, 92] {
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&changed).unwrap();
        }
        file.seek(SeekFrom::Start(100)).unwrap();
        file.write_all(&vec![0xA5u8; (len - 100) as usize]).unwrap();
    }
    handle.store.offer(report("after"), usize::MAX).unwrap();
    handle.store.sync();
    let (why, kept_as, generation) = handle.replaced();
    assert_eq!((why, generation), ("corrupt", 1));
    assert!(kept_as.unwrap().exists());
    // The report that was in hand is written to the new file.
    let committed = handle.committed();
    assert_eq!(committed.stored, 1);
    assert_eq!(committed.stats.pending, [(None, 1)]);
    assert!(committed.health.available);

    // An outcome for a row of the file that is gone is not written to the
    // new one, where its id is another report's.
    assert_eq!(leased.id, 1);
    handle.settle(&leased, Outcome::Delete);
    assert_eq!(
        read::<String>(
            &path,
            "SELECT json_extract(body, '$.id') FROM pending WHERE id = 1"
        ),
        "after"
    );
    assert_eq!(handle.claim(1, 0, LEASE).fresh[0].generation, 1);
    // The old file and its journal are there for a person, side by side.
    let kept: Vec<String> = names(dir.path())
        .into_iter()
        .filter(|name| name.contains(".corrupt-"))
        .collect();
    assert_eq!(kept.len(), 2, "{kept:?}");
    assert_eq!(format!("{}-journal", kept[0]), kept[1]);
}

#[test]
fn a_file_deleted_under_the_running_store_is_put_back_as_it_was() {
    let (dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    let clock = Clock::default();
    handle.keep(["one", "two", "three"].map(report));
    let rows = handle.claim(2, 0, LEASE).fresh;
    handle.settle(&rows[0], failing(1, 60_000, &clock));

    for name in names(dir.path()) {
        std::fs::remove_file(dir.path().join(name)).unwrap();
    }
    // Noticed by the next transaction, before anything is written to a file
    // nobody can open any more.
    handle.store.offer(report("four"), usize::MAX).unwrap();
    handle.store.sync();
    let (why, kept_as, generation) = handle.replaced();
    assert_eq!((why, kept_as, generation), ("deleted", None, 0));
    let committed = handle.committed();
    assert_eq!(committed.stored, 1);
    assert_eq!(committed.stats.pending, [(None, 4)]);

    // The same database: rows, attempts, leases and ids. So the outcome of a
    // report that was being sent finds its row.
    assert!(path.exists());
    assert_eq!(
        read::<String>(
            &path,
            "SELECT group_concat(json_extract(body, '$.id') || ':' || attempts || ':' || \
             (lease_owner IS NOT NULL)) FROM (SELECT * FROM pending ORDER BY id)"
        ),
        "one:1:0,two:0:1,three:0:0,four:0:0"
    );
    handle.settle(&rows[1], Outcome::Delete);
    assert_eq!(handle.stats().pending, [(None, 3)]);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

#[test]
fn a_file_emptied_under_the_running_store_is_started_anew_and_the_old_rows_mean_nothing_in_it() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep(["old-one", "old-two"].map(report));
    let leased = handle.claim(1, 0, LEASE).fresh.remove(0);

    // Truncated to nothing, which to SQLite is a database with no tables.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(0)
        .unwrap();
    // The transaction that finds out fails, and hands its report back.
    handle.store.offer(report("in-hand"), usize::MAX).unwrap();
    handle.store.sync();
    let (error, reports, _, health) = handle.failed();
    assert!(error.is_some());
    assert_eq!(reports.len(), 1);
    assert!(!health.available);

    // The next try makes the database anew, and says that it is another
    // one: whoever holds rows of the old one must not look for them here.
    std::thread::sleep(Duration::from_millis(60));
    handle
        .store
        .offer(reports.into_iter().next().unwrap(), usize::MAX)
        .unwrap_err();
    handle.store.sync();
    let (why, kept_as, generation) = handle.replaced();
    assert_eq!((why, kept_as, generation), ("lost", None, 1));
    let committed = handle.committed();
    assert!(committed.health.available);
    assert_eq!(committed.stats.pending, []);
    assert_eq!(handle.keep([report("new-one")]).stored, 1);
    // The outcome of the report leased from the old database names a row of
    // the new one by its id. It is not written.
    assert_eq!(leased.id, 1);
    handle.settle(&leased, Outcome::Delete);
    assert_eq!(handle.stats().pending, [(None, 1)]);
    assert_eq!(handle.claim(1, 0, LEASE).fresh[0].generation, 1);
}

#[test]
fn another_file_put_in_its_place_is_opened_and_the_old_rows_mean_nothing_in_it() {
    let (dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep(["old-one", "old-two"].map(report));
    let leased = handle.claim(1, 0, LEASE).fresh.remove(0);

    // A person moves the file away and another database is there instead
    // (here: one a second store made elsewhere).
    let elsewhere = dir.path().join("elsewhere.db");
    {
        let mut other = Handle::opened(config(&elsewhere));
        other.keep(["new-one", "new-two", "new-three"].map(report));
        assert!(other.store.close().blocking_recv().unwrap().is_some());
    }
    std::fs::rename(&path, dir.path().join("moved-away.db")).unwrap();
    std::fs::rename(&elsewhere, &path).unwrap();

    handle.store.sync();
    let (why, _, generation) = handle.replaced();
    assert_eq!((why, generation), ("replaced", 1));
    assert_eq!(handle.committed().stats.pending, [(None, 3)]);
    // The outcome of the report leased from the old database names a row of
    // the new one by its id. It is not written.
    assert_eq!(leased.id, 1);
    handle.settle(&leased, Outcome::Delete);
    assert_eq!(handle.stats().pending, [(None, 3)]);
    assert_eq!(ids(&handle.claim(1, 0, LEASE).fresh), ["new-one"]);
}

// ---------------------------------------------------------------------------
// A writer that does not come back
// ---------------------------------------------------------------------------

#[test]
fn what_a_writer_that_hangs_was_handed_can_be_taken_over_and_is_not_written_twice() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(UsageOutboxConfig {
        busy_timeout: Duration::from_secs(4),
        ..config(&path)
    });
    assert_eq!(handle.store.stalled_for(), Duration::ZERO);

    // The writer is stuck in a transaction (here: behind a lock).
    let other = Connection::open(&path).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    handle.store.offer(report("in-hand"), usize::MAX).unwrap();
    std::thread::sleep(Duration::from_millis(150));
    handle.store.offer(report("waiting"), usize::MAX).unwrap();
    std::thread::sleep(Duration::from_millis(150));
    assert!(handle.store.stalled_for() >= Duration::from_millis(250));

    // What it has in hand and what waits for it, oldest first. From then on
    // nothing is taken.
    assert_eq!(
        handle.store.take_over(),
        [report("in-hand"), report("waiting")]
    );
    assert!(!handle.store.is_accepting());
    assert!(handle.store.offer(report("later"), usize::MAX).is_err());

    // It comes back. The reports that were taken over are not written: they
    // are being delivered from memory.
    other.execute_batch("COMMIT").unwrap();
    let committed = handle.committed();
    assert_eq!((committed.stored, committed.no_room.len()), (0, 0));
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 0);
    // And having come back it takes reports again.
    let give_up_at = Instant::now() + Duration::from_secs(10);
    while handle.store.stalled_for() > Duration::ZERO || !handle.store.is_accepting() {
        assert!(Instant::now() < give_up_at, "the writer never came back");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(handle.keep([report("later")]).stored, 1);
}

#[test]
fn reports_taken_over_while_they_were_being_committed_are_named_by_the_transaction_that_wrote_them()
{
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    // A disk that takes its time over the flush: the writer has decided to
    // commit, and is not back yet.
    handle.store.inject_commit_delay(Duration::from_millis(400));
    handle.store.offer(report("one"), usize::MAX).unwrap();
    handle.store.offer(report("two"), usize::MAX).unwrap();
    std::thread::sleep(Duration::from_millis(150));
    assert!(handle.store.stalled_for() >= Duration::from_millis(100));
    assert_eq!(handle.store.take_over(), [report("one"), report("two")]);

    // It does come back, with the two written. They are not counted as
    // stored (whoever took them over has them), and they are named, so that
    // they need not be sent from memory as well.
    handle.store.inject_commit_delay(Duration::ZERO);
    let committed = handle.committed();
    assert_eq!((committed.stored, committed.no_room.len()), (0, 0));
    assert_eq!(committed.overtaken, [report("one"), report("two")]);
    assert_eq!(committed.stats.pending, [(None, 2)]);
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 2);
    assert!(committed.health.available);
}

// ---------------------------------------------------------------------------
// Clocks
// ---------------------------------------------------------------------------

#[test]
fn a_clock_set_back_strands_neither_a_lease_nor_a_retry() {
    let (_dir, path) = file();
    let clock = Clock::default();
    let timings = || UsageOutboxConfig {
        max_lease: Duration::from_secs(600),
        max_backoff: Duration::from_secs(600),
        clock: clock.clone(),
        ..config(&path)
    };
    let mut dead = Handle::opened(timings());
    let mut next = Handle::opened(timings());
    dead.keep(["leased", "failed"].map(report));
    let rows = dead.claim(2, 0, Duration::from_secs(60)).fresh;
    // One stays leased by a process that then dies; the other failed and is
    // due again in ten minutes.
    dead.settle(&rows[1], failing(1, 600_000, &clock));
    drop(dead);
    let claimed = next.claim(5, 5, LEASE);
    assert!(claimed.fresh.is_empty() && claimed.retries.is_empty());

    // The clock is set back by an hour. By the times in the rows, the lease
    // now runs for an hour and a minute and the retry is 70 minutes away.
    // No lease is that long and no retry that far: both count as over.
    clock.step(-3_600_000);
    let claimed = next.claim(5, 5, LEASE);
    assert_eq!(ids(&claimed.fresh), ["leased"]);
    assert_eq!(ids(&claimed.retries), ["failed"]);

    // Set back by less than the longest lease, the wait is at most that
    // long: here the rows are left alone, as they would be without a step.
    let rows = [claimed.fresh, claimed.retries].concat();
    next.settle(&rows[0], Outcome::Release);
    next.settle(&rows[1], failing(2, 60_000, &clock));
    drop(next);
    let mut later = Handle::opened(timings());
    clock.step(-30_000);
    let claimed = later.claim(5, 5, LEASE);
    assert_eq!(ids(&claimed.fresh), ["leased"]);
    assert!(claimed.retries.is_empty());
}

#[test]
fn a_clock_set_forward_never_hands_a_process_a_report_it_is_sending() {
    let (_dir, path) = file();
    let clock = Clock::default();
    let timings = || UsageOutboxConfig {
        clock: clock.clone(),
        ..config(&path)
    };
    let mut sending = Handle::opened(timings());
    let mut other = Handle::opened(timings());
    sending.keep(["one", "two"].map(report));
    let rows = sending.claim(1, 0, Duration::from_secs(60)).fresh;
    assert_eq!(ids(&rows), ["one"]);

    // An hour forward: every lease looks long over.
    clock.step(3_600_000);
    // The process that is sending "one" is not handed it again...
    assert_eq!(ids(&sending.claim(5, 5, LEASE).fresh), ["two"]);
    // ...and when it says what became of it, the row is told.
    sending.settle(&rows[0], failing(1, -1, &clock));
    assert_eq!(ids(&sending.claim(5, 5, LEASE).retries), ["one"]);
    // Another process may take what looks abandoned: that is what a lease
    // running out means, and the report is then sent twice at worst.
    sending.keep([report("three")]);
    assert_eq!(
        ids(&sending.claim(1, 0, Duration::from_secs(60)).fresh),
        ["three"]
    );
    clock.step(3_600_000);
    assert_eq!(ids(&other.claim(5, 0, LEASE).fresh), ["two", "three"]);
}

#[test]
fn a_report_too_old_to_send_is_moved_to_rejected_not_destroyed() {
    let (_dir, path) = file();
    let clock = Clock::default();
    let mut handle = Handle::opened(UsageOutboxConfig {
        clock: clock.clone(),
        ..config(&path)
    });
    handle.keep([
        Report {
            age: Duration::from_secs(7_200),
            ..report("old")
        },
        Report {
            age: Duration::from_secs(7_000),
            tried: Some((4, "http_5xx", Duration::ZERO)),
            ..report("old-and-tried")
        },
        report("young"),
    ]);
    // A report somebody is sending is not taken from under them.
    handle.keep([Report {
        age: Duration::from_secs(9_000),
        ..report("being-sent")
    }]);
    let being_sent = handle.claim(1, 0, LEASE).fresh;
    assert_eq!(ids(&being_sent), ["being-sent"]);
    handle.settle(&being_sent[0], Outcome::Release);
    let mut other = Handle::opened(UsageOutboxConfig {
        clock: clock.clone(),
        ..config(&path)
    });
    assert_eq!(ids(&other.claim(1, 0, LEASE).fresh), ["being-sent"]);

    handle.store.claim(Claim {
        expire: Some((clock.now_ms() - 3_600_000, "max_age")),
        ..all(5, 5, LEASE)
    });
    let committed = handle.committed();
    let expired: Vec<(String, &str)> = committed
        .expired
        .iter()
        .map(|(row, reason)| (ids(std::slice::from_ref(row)).remove(0), *reason))
        .collect();
    assert_eq!(
        expired,
        [
            ("old".to_string(), "max_age"),
            ("old-and-tried".to_string(), "max_age")
        ]
    );
    assert_eq!(ids(&committed.claimed.unwrap().fresh), ["young"]);
    assert_eq!(committed.stats.rejected, [("max_age".to_string(), 2)]);
    // Everything a person needs to put them back is there.
    assert_eq!(
        read::<String>(
            &path,
            "SELECT group_concat(json_extract(body, '$.id') || ':' || reason || ':' || outcome \
             || ':' || attempts || ':' || (status IS NULL)) FROM (SELECT * FROM rejected ORDER BY id)"
        ),
        "old:max_age:never_sent:0:1,old-and-tried:max_age:http_5xx:4:1"
    );

    // A clock set forward by a year makes everything look too old. Nothing
    // is lost by it: the reports are in `rejected`.
    clock.step(365 * 24 * 3_600_000);
    handle.settle(&committed_young(&path), Outcome::Release);
    handle.store.claim(Claim {
        expire: Some((clock.now_ms() - 7 * 24 * 3_600_000, "max_age")),
        ..all(5, 5, LEASE)
    });
    let committed = handle.committed();
    // "young", and the one whose holder's lease is a year over.
    assert_eq!(committed.expired.len(), 2);
    assert_eq!(committed.stats.rejected, [("max_age".to_string(), 4)]);
    assert_eq!(committed.stats.pending_total(), 0);
}

/// The row of the report "young", as leased.
fn committed_young(path: &Path) -> Row {
    let id: i64 = read(
        path,
        "SELECT id FROM pending WHERE json_extract(body, '$.id') = 'young'",
    );
    Row {
        id,
        generation: 0,
        body: String::new(),
        request_id: None,
        model_label: None,
        auth_path: String::new(),
        ingress_route: String::new(),
        completed_at_ms: 0,
        attempts: 0,
        last_outcome: None,
        next_attempt_at_ms: 0,
    }
}

// ---------------------------------------------------------------------------
// By hand, and the end
// ---------------------------------------------------------------------------

#[test]
fn a_row_written_by_hand_with_anything_in_it_does_not_stop_the_others() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(config(&path));
    handle.keep([report("before")]);
    // Not what the proxy writes: a binary body, a label that is a number, a
    // completion time that is text.
    let conn = Connection::open(&path).unwrap();
    conn.busy_timeout(Duration::from_secs(5)).unwrap();
    conn.execute(
        "INSERT INTO pending (body, request_id, model_label, auth_path, ingress_route, \
         completed_at_ms, last_outcome) \
         VALUES (x'ff00fe', 17, 2.5, x'00', '', 'yesterday', x'ff')",
        [],
    )
    .unwrap();
    // Attempts below zero are refused when they are written.
    assert!(conn
        .execute(
            "INSERT INTO pending (body, auth_path, ingress_route, completed_at_ms, attempts) \
             VALUES ('{}', 'a', 'b', 0, -3)",
            [],
        )
        .is_err());
    handle.keep([report("after")]);

    let rows = handle.claim(5, 5, LEASE).fresh;
    assert_eq!(rows.len(), 3);
    let odd = rows
        .iter()
        .find(|row| row.request_id.as_deref() == Some("17"));
    let odd = odd.expect("the row written by hand");
    assert_eq!(odd.model_label.as_deref(), Some("2.5"));
    assert_eq!(odd.completed_at_ms, 0);
    // The reports around it are what they were.
    let readable: Vec<&str> = rows
        .iter()
        .filter(|row| row.id != odd.id)
        .map(|row| row.request_id.as_deref().unwrap())
        .collect();
    assert_eq!(readable, ["before", "after"]);
    // And it can be settled like any other.
    let committed = handle.settle(
        odd,
        Outcome::Reject {
            why: REFUSED,
            attempts: 1,
        },
    );
    assert_eq!(committed.stats.rejected_total(), 1);
    assert_eq!(committed.stats.pending_total(), 2);
}

#[test]
fn closing_writes_what_is_left_and_gives_this_handles_leases_back() {
    let (_dir, path) = file();
    let mut leaving = Handle::opened(UsageOutboxConfig {
        commit_interval: Duration::from_secs(30),
        ..config(&path)
    });
    let mut staying = Handle::opened(config(&path));
    staying.keep(["theirs"].map(report));
    assert_eq!(ids(&staying.claim(1, 0, LEASE).fresh), ["theirs"]);
    leaving.keep([report("one")]);
    assert_eq!(ids(&leaving.claim(5, 0, LEASE).fresh), ["one"]);

    // Handed over and not written yet: the interval is half a minute.
    leaving.store.offer(report("two"), usize::MAX).unwrap();
    let left = leaving.store.close().blocking_recv().unwrap().unwrap();
    assert_eq!(left.pending, [(None, 3)]);
    // After that nothing is taken, and a second close has no answer.
    assert!(leaving.store.offer(report("three"), usize::MAX).is_err());
    assert!(leaving.store.close().blocking_recv().is_err());

    // Its lease is gone, so the report is the next handle's at once. The
    // lease of the handle that stays is still its own.
    assert_eq!(ids(&staying.claim(5, 0, LEASE).fresh), ["one", "two"]);
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL"
        ),
        3
    );
}

#[test]
fn a_handle_that_is_dropped_writes_what_it_was_handed_and_leaves_its_leases() {
    let (_dir, path) = file();
    let mut handle = Handle::opened(UsageOutboxConfig {
        commit_interval: Duration::from_secs(30),
        ..config(&path)
    });
    handle.keep([report("one")]);
    assert_eq!(ids(&handle.claim(1, 0, LEASE).fresh), ["one"]);
    handle.store.offer(report("two"), usize::MAX).unwrap();
    let Handle { store, mut events } = handle;
    drop(store);

    // The writer ends by itself: its side of the channel closes.
    let give_up_at = Instant::now() + Duration::from_secs(20);
    while !matches!(
        events.try_recv(),
        Err(mpsc::error::TryRecvError::Disconnected)
    ) {
        assert!(Instant::now() < give_up_at, "the writer is still there");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 2);
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE lease_until_ms > 0"
        ),
        1
    );
}
