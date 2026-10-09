//! The store by itself: one or two handles on a file in a temporary
//! directory, and a plain SQLite connection beside them to look at what was
//! written, the way a person with the `sqlite3` CLI would.

use super::*;

/// A report as far as the store is concerned.
#[derive(Debug, PartialEq)]
struct Report {
    id: &'static str,
    model: Option<&'static str>,
    age: Duration,
}

fn report(id: &'static str) -> Report {
    Report {
        id,
        model: None,
        age: Duration::ZERO,
    }
}

impl Report {
    fn body(&self) -> String {
        format!(r#"{{"type":"chat_completion","id":"{}"}}"#, self.id)
    }
}

impl Persist for Report {
    fn row(&self) -> NewRow<'_> {
        // The body is built per call and the row borrows it: leaked, in a
        // test, for a handful of reports.
        NewRow {
            body: self.body().leak().as_bytes(),
            request_id: Some(self.id),
            model_label: self.model,
            auth_path: "cloud_api_key",
            ingress_route: "canonical",
            age: self.age,
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
        let give_up_at = Instant::now() + Duration::from_secs(20);
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
    fn committed(&mut self) -> Committed {
        match self.next() {
            Event::Committed(committed) => committed,
            Event::Failed { error, .. } => panic!("a transaction failed: {error:?}"),
            Event::Warning { op, error } => panic!("{op} failed: {error}"),
        }
    }

    /// The next transaction, which must have failed: why, what it handed
    /// back, and whether it carried a claim.
    fn failed(&mut self) -> (Option<(&'static str, String)>, Vec<Report>, bool) {
        match self.next() {
            Event::Failed {
                error,
                reports,
                claim,
            } => (error, reports, claim),
            Event::Committed(committed) => panic!("a transaction committed: {committed:?}"),
            Event::Warning { op, error } => panic!("{op} failed: {error}"),
        }
    }

    /// Hand `reports` over and wait for the transaction that writes them,
    /// which comes when the commit interval is over.
    fn keep(&mut self, reports: impl IntoIterator<Item = Report>) -> Committed {
        for report in reports {
            self.store.offer(report, usize::MAX).unwrap();
        }
        self.committed()
    }

    /// `keep` for a handle whose commit interval is too long to wait for.
    fn keep_now(&mut self, reports: impl IntoIterator<Item = Report>) -> Committed {
        for report in reports {
            self.store.offer(report, usize::MAX).unwrap();
        }
        self.store.sync();
        self.committed()
    }

    fn claim(&mut self, limit: usize, lease: Duration) -> Vec<Row> {
        self.store.claim(limit, lease);
        self.committed().claimed.expect("the answer to a claim")
    }

    fn settle(&mut self, settle: Settle) -> Committed {
        self.store.settle(settle);
        self.committed()
    }

    fn stats(&mut self) -> Stats {
        self.store.sync();
        self.committed().stats
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

const LEASE: Duration = Duration::from_secs(60);

#[test]
fn the_file_is_created_private_in_wal_mode_with_the_schema() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(config(&path));
    handle.keep([report("one")]);

    assert_eq!(read::<String>(&path, "PRAGMA journal_mode"), "wal");
    assert_eq!(
        read::<i64>(&path, "PRAGMA user_version"),
        SCHEMA_VERSION,
        "the schema version the migration left"
    );
    for table in ["pending", "rejected", "pending_counts"] {
        let found: i64 = read(
            &path,
            &format!(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = '{table}'"
            ),
        );
        assert_eq!(found, 1, "{table}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // The file, and the log and shared-memory files SQLite puts beside
        // it, which hold the same rows.
        for suffix in ["", "-wal", "-shm"] {
            let file = dir.path().join(format!("outbox.db{suffix}"));
            let mode = std::fs::metadata(&file).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "{}", file.display());
        }
    }
}

#[cfg(unix)]
#[test]
fn a_file_that_others_could_read_is_made_private() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    std::fs::write(&path, b"").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    let _handle = Handle::opened(config(&path));
    let mode = std::fs::metadata(&path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600);
}

#[test]
fn reports_handed_over_together_are_written_by_one_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let interval = Duration::from_millis(600);
    let mut handle = Handle::opened(UsageOutboxConfig {
        commit_interval: interval,
        ..config(&path)
    });

    let started_at = Instant::now();
    for id in ["one", "two", "three"] {
        handle.store.offer(report(id), usize::MAX).unwrap();
    }
    // Nothing is written before the interval is over: the writer is waiting
    // for more reports to join these.
    std::thread::sleep(Duration::from_millis(50));
    let rows: i64 = read(&path, "SELECT COUNT(*) FROM pending");
    if started_at.elapsed() < interval {
        assert_eq!(rows, 0);
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
    handle.store.claim(10, LEASE);
    let committed = handle.committed();
    assert!(started_at.elapsed() < interval);
    assert_eq!(committed.stored, 1);
    assert_eq!(
        ids(&committed.claimed.unwrap()),
        ["one", "two", "three", "four"]
    );
}

#[test]
fn a_row_holds_the_report_its_labels_and_when_its_request_completed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(config(&path));
    let before = now_ms();
    handle.keep([Report {
        id: "one",
        model: Some("example/alpha"),
        age: Duration::from_secs(90),
    }]);

    let rows = handle.claim(1, LEASE);
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

#[test]
fn a_claim_leases_the_reports_longest_due_and_nobody_else_gets_them_until_the_lease_runs_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
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
    assert_eq!(ids(&first.claim(2, lease)), ["oldest", "older"]);
    // The other handle gets what is left, and then nothing.
    assert_eq!(ids(&second.claim(5, LEASE)), ["old"]);
    assert_eq!(ids(&first.claim(5, LEASE)), [] as [&str; 0]);
    assert_eq!(ids(&second.claim(5, LEASE)), [] as [&str; 0]);
    assert!(leased_at.elapsed() < lease, "the test was too slow to tell");
    assert_eq!(
        read::<String>(
            &path,
            "SELECT group_concat(DISTINCT lease_owner) FROM pending WHERE id <= 2"
        ),
        first.store.owner()
    );

    // The holder of the first two never says what became of them. Once
    // their lease has run out they are anybody's.
    std::thread::sleep(lease.saturating_sub(leased_at.elapsed()) + Duration::from_millis(20));
    assert_eq!(ids(&second.claim(5, LEASE)), ["oldest", "older"]);
    // Leased rows still count as pending: they are not accepted yet.
    assert_eq!(second.stats().pending, [(None, 3)]);
}

#[test]
fn what_became_of_a_report_is_written_to_its_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(config(&path));
    handle.keep(["accepted", "failing", "refused", "unstarted"].map(report));
    let rows = handle.claim(4, LEASE);
    let id = |name: &str| rows[ids(&rows).iter().position(|id| id == name).unwrap()].id;

    // Accepted: the row is gone.
    handle.settle(Settle::Delete { id: id("accepted") });
    // Failed in a way that can pass: due again later, the lease given back.
    let retry_at = now_ms() + 1_000;
    handle.settle(Settle::Retry {
        id: id("failing"),
        attempts: 1,
        next_attempt_at_ms: retry_at,
        last_outcome: "http_5xx",
    });
    // Refused for good: moved to `rejected` with what the answer was.
    let committed = handle.settle(Settle::Reject {
        id: id("refused"),
        reason: "rejected",
        status: Some(422),
        outcome: "http_4xx",
        attempts: 1,
    });
    assert_eq!(committed.stats.rejected, 1);
    // Not started after all: as if it had never been leased.
    let committed = handle.settle(Settle::Release {
        id: id("unstarted"),
    });
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
    assert_eq!(body, report("refused").body());

    // Only the released one is due now. The failing one tells when it is.
    handle.store.claim(4, LEASE);
    let committed = handle.committed();
    assert_eq!(ids(committed.claimed.as_ref().unwrap()), ["unstarted"]);
    assert_eq!(committed.next_due_ms, Some(retry_at));
    assert!(now_ms() < retry_at, "the test was too slow to tell");
    std::thread::sleep(Duration::from_millis(
        (retry_at - now_ms()).max(0) as u64 + 20,
    ));
    let again = handle.claim(4, LEASE);
    assert_eq!(ids(&again), ["failing"]);
    assert_eq!(
        (again[0].attempts, again[0].last_outcome.as_deref()),
        (1, Some("http_5xx"))
    );
}

#[test]
fn an_outcome_written_after_the_lease_ran_out_does_not_undo_the_next_holders_lease() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut slow = Handle::opened(config(&path));
    let mut next = Handle::opened(config(&path));
    slow.keep([report("one")]);
    let row = slow.claim(1, Duration::from_millis(50)).remove(0);
    std::thread::sleep(Duration::from_millis(70));
    assert_eq!(ids(&next.claim(1, LEASE)), ["one"]);

    // The first holder's attempt ends late, with a failure.
    slow.settle(Settle::Retry {
        id: row.id,
        attempts: 1,
        next_attempt_at_ms: 0,
        last_outcome: "timeout",
    });
    slow.settle(Settle::Release { id: row.id });
    // The row is still the second holder's, untouched.
    assert_eq!(ids(&slow.claim(1, LEASE)), [] as [&str; 0]);
    assert_eq!(
        read::<String>(&path, "SELECT lease_owner FROM pending"),
        next.store.owner()
    );
    assert_eq!(read::<i64>(&path, "SELECT attempts FROM pending"), 0);
    // An acceptance, though, is an acceptance whoever got it.
    slow.settle(Settle::Delete { id: row.id });
    assert_eq!(next.stats().pending, [(None, 0)]);
}

#[test]
fn rejected_keeps_the_newest_rows_and_counts_the_ones_it_removes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(UsageOutboxConfig {
        max_rejected: 2,
        ..config(&path)
    });
    handle.keep(["one", "two", "three", "four"].map(report));
    let rows = handle.claim(4, LEASE);

    let mut evicted = 0;
    for row in &rows {
        evicted += handle
            .settle(Settle::Reject {
                id: row.id,
                reason: "attempts_exhausted",
                status: None,
                outcome: "timeout",
                attempts: 3,
            })
            .rejected_evicted;
    }
    assert_eq!(evicted, 2);
    let stats = handle.stats();
    assert_eq!((stats.rejected, stats.pending_total()), (2, 0));
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
}

#[test]
fn the_bound_drops_the_reports_that_waited_longest_but_never_one_being_sent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(UsageOutboxConfig {
        max_pending: 3,
        ..config(&path)
    });
    let committed = handle.keep(["one", "two", "three"].map(report));
    assert!(committed.evicted.is_empty());
    // "one" is being sent.
    assert_eq!(ids(&handle.claim(1, LEASE)), ["one"]);

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
fn the_backlog_is_counted_per_model_and_its_oldest_report_is_known() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(config(&path));
    assert_eq!(handle.stats(), Stats::default());

    let before = now_ms();
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

    // A model whose reports are all gone stays in the count, at zero, so
    // that its gauge comes down with it.
    let rows = handle.claim(3, LEASE);
    for row in rows
        .iter()
        .filter(|row| row.model_label.as_deref() == Some("example/alpha"))
    {
        handle.settle(Settle::Delete { id: row.id });
    }
    let mut pending = handle.stats().pending;
    pending.sort();
    assert_eq!(
        pending,
        [
            (Some("example/alpha".to_string()), 0),
            (Some("example/beta".to_string()), 1)
        ]
    );
}

#[test]
fn a_rejected_report_put_back_by_hand_is_counted_and_due_like_any_other() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(config(&path));
    handle.keep([report("one")]);
    let row = handle.claim(1, LEASE).remove(0);
    handle.settle(Settle::Reject {
        id: row.id,
        reason: "rejected",
        status: Some(400),
        outcome: "http_4xx",
        attempts: 1,
    });
    assert_eq!(handle.stats().pending_total(), 0);

    // The statements docs/gateway-mode.md gives for sending a rejected
    // report again, run while the proxy has the file open.
    let conn = Connection::open(&path).unwrap();
    conn.busy_timeout(Duration::from_secs(5)).unwrap();
    conn.execute_batch(
        "BEGIN IMMEDIATE;
         INSERT INTO pending (body, request_id, model_label, auth_path, ingress_route, completed_at_ms)
           SELECT body, request_id, model_label, auth_path, ingress_route, completed_at_ms
           FROM rejected WHERE id = 1;
         DELETE FROM rejected WHERE id = 1;
         COMMIT;",
    )
    .unwrap();

    // The triggers counted the row as it was put back. The rows of
    // `rejected` are counted again on a schedule of their own.
    assert_eq!(handle.stats().pending_total(), 1);
    let give_up_at = Instant::now() + Duration::from_secs(20);
    while handle.stats().rejected != 0 {
        assert!(
            Instant::now() < give_up_at,
            "`rejected` was never recounted"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let again = handle.claim(1, LEASE);
    assert_eq!(ids(&again), ["one"]);
    assert_eq!(again[0].attempts, 0);
}

#[test]
fn a_row_written_by_hand_with_anything_in_it_does_not_stop_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(config(&path));
    handle.keep([report("before")]);
    // Not what the proxy writes: a binary body, a label that is a number,
    // attempts below zero, a completion time that is text.
    let conn = Connection::open(&path).unwrap();
    conn.busy_timeout(Duration::from_secs(5)).unwrap();
    conn.execute(
        "INSERT INTO pending (body, request_id, model_label, auth_path, ingress_route, \
         completed_at_ms, attempts, last_outcome) \
         VALUES (x'ff00fe', 17, 2.5, x'00', '', 'yesterday', -3, x'ff')",
        [],
    )
    .unwrap();
    handle.keep([report("after")]);

    let rows = handle.claim(5, LEASE);
    assert_eq!(rows.len(), 3);
    let odd = rows
        .iter()
        .find(|row| row.attempts == 0 && row.request_id.as_deref() == Some("17"));
    let odd = odd.expect("the row written by hand");
    assert_eq!(odd.model_label.as_deref(), Some("2.5"));
    assert_eq!(odd.completed_at_ms, 0);
    // The reports around it are what they were.
    let readable: Vec<&Row> = rows.iter().filter(|row| row.id != odd.id).collect();
    assert_eq!(
        readable
            .iter()
            .map(|row| row.request_id.as_deref().unwrap())
            .collect::<Vec<_>>(),
        ["before", "after"]
    );
    // And it can be settled like any other.
    let committed = handle.settle(Settle::Reject {
        id: odd.id,
        reason: "rejected",
        status: Some(400),
        outcome: "http_4xx",
        attempts: 1,
    });
    assert_eq!(committed.stats.rejected, 1);
    assert_eq!(committed.stats.pending_total(), 2);
}

#[test]
fn handing_a_report_over_never_waits_for_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(config(&path));

    // Somebody else holds the write lock for a while.
    let locked = Duration::from_millis(1_500);
    let other = Connection::open(&path).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    let locked_at = Instant::now();

    let mut slowest = Duration::ZERO;
    for round in 0..300 {
        let started_at = Instant::now();
        handle.store.offer(report("blocked"), usize::MAX).unwrap();
        slowest = slowest.max(started_at.elapsed());
        if round % 100 == 0 {
            // The writer is in its transaction by now, waiting for the lock.
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    // Far below the time the lock is held, whatever else the machine does.
    assert!(slowest < locked / 5, "{slowest:?}");
    assert!(
        locked_at.elapsed() < locked,
        "the test was too slow to tell"
    );
    assert!(
        handle.events.try_recv().is_err(),
        "nothing can have been written"
    );

    std::thread::sleep(locked.saturating_sub(locked_at.elapsed()));
    other.execute_batch("COMMIT").unwrap();
    let mut stored = 0;
    while stored < 300 {
        stored += handle.committed().stored;
    }
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 300);
}

#[test]
fn no_more_reports_wait_to_be_written_than_allowed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
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
    handle.store.sync();
    assert_eq!(handle.committed().stored, 2);
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
    let (error, reports, claim) = handle.failed();
    let (op, _) = error.expect("opening was tried");
    assert_eq!((op, reports.len(), claim), ("open", 0, false));
    assert!(!handle.store.is_available());

    // From then on nothing is taken, so nothing waits for a file that is
    // not there.
    let (back, why) = handle.store.offer(report("one"), usize::MAX).unwrap_err();
    assert_eq!((back, why.as_label()), (report("one"), "unavailable"));
    // A claim is answered, with nothing, and the file is left alone for a
    // while: no second error yet.
    handle.store.claim(5, LEASE);
    let (error, _, claim) = handle.failed();
    assert!(
        opened_at.elapsed() < Duration::from_millis(900),
        "the test was too slow to tell"
    );
    assert!(error.is_none() && claim, "{error:?}");

    // The directory appears. The next look after the interval opens it.
    std::fs::create_dir(&missing).unwrap();
    std::thread::sleep(Duration::from_millis(1_050));
    assert_eq!(handle.stats(), Stats::default());
    assert!(handle.store.is_available());
    assert_eq!(handle.keep([report("one")]).stored, 1);
}

#[test]
fn a_transaction_that_fails_writes_nothing_hands_its_reports_back_and_keeps_the_outcomes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(UsageOutboxConfig {
        commit_interval: Duration::from_millis(200),
        ..config(&path)
    });
    handle.keep(["one", "two"].map(report));
    let rows = handle.claim(2, LEASE);

    handle.store.inject_fault(true);
    handle.store.offer(report("three"), usize::MAX).unwrap();
    // "one" was accepted while the file could not be written.
    handle.store.settle(Settle::Delete { id: rows[0].id });
    let (error, reports, _) = handle.failed();
    assert_eq!(error.unwrap().0, "begin");
    assert_eq!(reports, [report("three")]);
    assert!(!handle.store.is_available());
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 2);

    // Tried again after the interval, and failing again: a second error.
    std::thread::sleep(Duration::from_millis(60));
    handle.store.sync();
    let tried_again = loop {
        if let (Some((op, _)), _, _) = handle.failed() {
            break op;
        }
    };
    assert_eq!(tried_again, "open");

    // Back: the outcome that could not be written is written now, so "one"
    // is not sent a second time.
    handle.store.inject_fault(false);
    std::thread::sleep(Duration::from_millis(60));
    let stats = handle.stats();
    assert_eq!(stats.pending, [(None, 1)]);
    assert_eq!(
        read::<String>(&path, "SELECT json_extract(body, '$.id') FROM pending"),
        "two"
    );
}

#[test]
fn a_file_from_a_newer_proxy_is_left_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
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
    // Nothing in it was touched.
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM pending"), 1);
}

#[test]
fn a_file_that_is_not_a_database_makes_the_store_unavailable_and_nothing_worse() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    std::fs::write(&path, b"this is not an SQLite file, whatever its name says").unwrap();
    let mut handle = Handle::open(config(&path));
    assert_eq!(handle.failed().0.unwrap().0, "open");
    assert!(!handle.store.is_available());
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"this is not an SQLite file, whatever its name says"
    );
}

#[test]
fn closing_writes_what_is_left_and_gives_this_handles_leases_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut leaving = Handle::opened(UsageOutboxConfig {
        commit_interval: Duration::from_secs(30),
        ..config(&path)
    });
    let mut staying = Handle::opened(config(&path));
    staying.keep(["theirs"].map(report));
    assert_eq!(ids(&staying.claim(1, LEASE)), ["theirs"]);
    leaving.keep_now([report("one")]);
    assert_eq!(ids(&leaving.claim(5, LEASE)), ["one"]);

    // Handed over and not written yet: the interval is half a minute.
    leaving.store.offer(report("two"), usize::MAX).unwrap();
    let left = leaving.store.close().blocking_recv().unwrap().unwrap();
    assert_eq!(left.pending, [(None, 3)]);
    // After that nothing is taken, and a second close has no answer.
    assert!(leaving.store.offer(report("three"), usize::MAX).is_err());
    assert!(leaving.store.close().blocking_recv().is_err());

    // Its lease is gone, so the report is the next handle's at once. The
    // lease of the handle that stays is still its own.
    assert_eq!(ids(&staying.claim(5, LEASE)), ["one", "two"]);
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    let mut handle = Handle::opened(UsageOutboxConfig {
        commit_interval: Duration::from_secs(30),
        ..config(&path)
    });
    handle.keep_now([report("one")]);
    assert_eq!(ids(&handle.claim(1, LEASE)), ["one"]);
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

#[test]
fn many_reports_are_written_in_transactions_of_bounded_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
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
