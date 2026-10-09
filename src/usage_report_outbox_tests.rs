//! Delivery through the outbox (`VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH`):
//! a stand-in billing API on a real socket, whose answers a test changes as
//! it goes, and the outbox as a file in a temporary directory, which a test
//! reads with a connection of its own.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU16, AtomicU64, AtomicUsize, Ordering};

use axum::extract::State as Shared;
use axum::http::{HeaderMap, StatusCode};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusRecorder};

use super::*;

const TOKEN: &str = "usage-secret";

/// What the stand-in has seen and how it answers.
struct Intake {
    /// The status every request gets, unless its report is in `refused`.
    status: AtomicU16,
    /// How long an answer takes, read when the request arrives.
    delay_ms: AtomicU64,
    /// Reports that get a status of their own, by completion id.
    refused: Mutex<BTreeMap<String, u16>>,
    received: AtomicUsize,
    /// The `id` of every report answered with a success, in order.
    written: Mutex<Vec<String>>,
    authorizations: Mutex<BTreeSet<String>>,
    request_ids: Mutex<BTreeSet<String>>,
    bodies: Mutex<Vec<bytes::Bytes>>,
}

async fn usage(
    Shared(intake): Shared<Arc<Intake>>,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> StatusCode {
    intake.received.fetch_add(1, Ordering::SeqCst);
    let header = |name: &str| {
        headers
            .get(name)
            .map(|value| value.to_str().unwrap().to_string())
            .unwrap_or_default()
    };
    intake
        .authorizations
        .lock()
        .unwrap()
        .insert(header("authorization"));
    intake
        .request_ids
        .lock()
        .unwrap()
        .insert(header("x-request-id"));
    intake.bodies.lock().unwrap().push(body.clone());
    let report: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let id = report["id"].as_str().unwrap().to_string();
    let status = intake
        .refused
        .lock()
        .unwrap()
        .get(&id)
        .copied()
        .unwrap_or_else(|| intake.status.load(Ordering::SeqCst));
    let delay = intake.delay_ms.load(Ordering::SeqCst);
    if delay > 0 {
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
    let status = StatusCode::from_u16(status).unwrap();
    if status.is_success() {
        intake.written.lock().unwrap().push(id);
    }
    status
}

/// The stand-in billing API.
struct Billing {
    url: String,
    intake: Arc<Intake>,
    client: reqwest::Client,
}

impl Billing {
    /// Answering `status` at once. Must be called on the runtime that is to
    /// serve it.
    async fn start(status: u16) -> Self {
        let intake = Arc::new(Intake {
            status: AtomicU16::new(status),
            delay_ms: AtomicU64::new(0),
            refused: Mutex::default(),
            received: AtomicUsize::new(0),
            written: Mutex::default(),
            authorizations: Mutex::default(),
            request_ids: Mutex::default(),
            bodies: Mutex::default(),
        });
        let app = axum::Router::new()
            .route("/v1/internal/usage", axum::routing::post(usage))
            .with_state(intake.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url,
            intake,
            client: reqwest::Client::new(),
        }
    }

    /// From now on every request is answered `status`.
    fn answer(&self, status: u16) {
        self.intake.status.store(status, Ordering::SeqCst);
    }

    /// From now on every answer takes this long.
    fn take(&self, delay: Duration) {
        self.intake
            .delay_ms
            .store(delay.as_millis() as u64, Ordering::SeqCst);
    }

    /// The report for completion `id` is answered `status`, whatever the
    /// others get; `None` ends that.
    fn refuse(&self, id: &str, status: Option<u16>) {
        let mut refused = self.intake.refused.lock().unwrap();
        match status {
            Some(status) => refused.insert(id.to_string(), status),
            None => refused.remove(id),
        };
    }

    fn received(&self) -> usize {
        self.intake.received.load(Ordering::SeqCst)
    }

    fn written(&self) -> Vec<String> {
        self.intake.written.lock().unwrap().clone()
    }

    /// The reports written, sorted, to compare with what was handed over:
    /// equal means none was lost and none was written twice.
    fn written_sorted(&self) -> Vec<String> {
        let mut written = self.written();
        written.sort();
        written
    }

    /// Hand a report for completion `id` to `delivery`, the way a finished
    /// request does.
    fn report(&self, delivery: &Arc<UsageReportDelivery>, id: &str, model: ModelLabel) {
        let reporter = UsageReporter {
            http_client: self.client.clone(),
            cloud_api_url: self.url.clone(),
            model_name: model.unwrap_or("test-model").to_string(),
            cloud_api_usage_token: Some(TOKEN.to_string()),
            org_id: Some("org-1".to_string()),
            workspace_id: Some("ws-1".to_string()),
            api_key_id: Some("key-1".to_string()),
            discount_to_user: Some(0.2),
            model_label: model,
            request_id: Some(format!("request-{id}")),
            request_source: RequestSource {
                auth_path: AuthPath::CloudApiKey,
                ingress_route: IngressRouteKind::Long,
            },
            delivery: delivery.clone(),
        };
        crate::proxy::spawn_usage_report(
            &reporter,
            serde_json::json!({
                "type": "chat_completion",
                "model": reporter.model_name,
                "input_tokens": 1200,
                "output_tokens": 340,
                "cache_read_tokens": 800,
                "id": id,
            }),
        );
    }

    /// A delivery with its outbox at `path`, sending here.
    fn delivery(
        &self,
        policy: UsageReportPolicy,
        outbox: UsageOutboxConfig,
    ) -> Arc<UsageReportDelivery> {
        UsageReportDelivery::with_outbox(policy, outbox, self.client.clone(), &self.url, TOKEN)
    }
}

/// The ids "{prefix}-0" to "{prefix}-{count - 1}", sorted as `written_sorted`
/// sorts them.
fn ids(prefix: &str, count: usize) -> Vec<String> {
    let mut ids: Vec<String> = (0..count).map(|n| format!("{prefix}-{n}")).collect();
    ids.sort();
    ids
}

/// What a process with an outbox and no other setting gets, with a backoff a
/// test does not have to wait for.
fn durable(change: impl FnOnce(&mut UsageReportPolicy)) -> UsageReportPolicy {
    let mut policy = UsageReportPolicy {
        initial_backoff: Duration::from_millis(10),
        ..UsageReportPolicy::durable()
    };
    change(&mut policy);
    policy
}

/// The outbox at `path` with timings a test does not have to wait for.
fn outbox(path: &Path) -> UsageOutboxConfig {
    UsageOutboxConfig {
        commit_interval: Duration::from_millis(5),
        reopen_interval: Duration::from_millis(50),
        recheck_interval: Duration::from_millis(20),
        lease_margin: Duration::from_millis(400),
        max_pause: Duration::from_millis(40),
        ..UsageOutboxConfig::at(path)
    }
}

/// A directory for one test and the outbox file in it.
fn file() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outbox.db");
    (dir, path)
}

/// One value read from the outbox by a connection of the test's own. 0 or
/// empty while the file or its tables are not there yet.
fn read<T: rusqlite::types::FromSql + Default>(path: &Path, sql: &str) -> T {
    let Ok(conn) = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return T::default();
    };
    let _ = conn.busy_timeout(Duration::from_secs(5));
    conn.query_row(sql, [], |row| row.get(0))
        .unwrap_or_default()
}

fn pending_rows(path: &Path) -> i64 {
    read(path, "SELECT COUNT(*) FROM pending")
}

/// Wait until `done`, for at most 30 seconds.
async fn eventually(what: &str, mut done: impl FnMut() -> bool) {
    let give_up_at = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < give_up_at, "never happened: {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Wait until nothing is pending, in memory or in the file.
async fn delivered(delivery: &UsageReportDelivery) {
    let drained = delivery.drain(Duration::from_secs(60)).await;
    assert_eq!(drained.left(), 0, "{drained:?}");
}

/// The value of every series of `family` whose labels include `labels`,
/// summed; 0 when there is none.
fn value(recorder: &PrometheusRecorder, family: &str, labels: &[&str]) -> f64 {
    recorder
        .handle()
        .render()
        .lines()
        .filter(|line| {
            let name = line.split(['{', ' ']).next().unwrap_or_default();
            name == family && labels.iter().all(|label| line.contains(label))
        })
        .map(|line| line.rsplit(' ').next().unwrap().parse::<f64>().unwrap())
        .sum()
}

/// The names of the series about usage reports that exist.
fn families(recorder: &PrometheusRecorder) -> BTreeSet<String> {
    recorder
        .handle()
        .render()
        .lines()
        .filter(|line| !line.starts_with('#') && line.contains("usage_report"))
        .map(|line| line.split(['{', ' ']).next().unwrap().to_string())
        .collect()
}

const REPORTS: &str = "inference_proxy_usage_reports_total";
const ATTEMPTS: &str = "inference_proxy_usage_report_attempts_total";
const RETRIES: &str = "inference_proxy_usage_report_retries_total";
const DROPPED: &str = "inference_proxy_usage_reports_dropped_total";
const WAITING: &str = "inference_proxy_usage_report_queue_depth";
const IN_FLIGHT: &str = "inference_proxy_usage_reports_in_flight";
const TIME_TO_ACCEPTED_COUNT: &str = "inference_proxy_usage_report_time_to_accepted_seconds_count";
const TIME_TO_ACCEPTED_SUM: &str = "inference_proxy_usage_report_time_to_accepted_seconds_sum";
const AVAILABLE: &str = "inference_proxy_usage_report_outbox_available";
const PENDING: &str = "inference_proxy_usage_report_outbox_pending";
const OLDEST_AGE: &str = "inference_proxy_usage_report_outbox_oldest_pending_age_seconds";
const REJECTED: &str = "inference_proxy_usage_report_outbox_rejected";
const REJECTED_EVICTED: &str = "inference_proxy_usage_report_outbox_rejected_evicted_total";
const ERRORS: &str = "inference_proxy_usage_report_outbox_errors_total";
const BYPASSED: &str = "inference_proxy_usage_report_outbox_bypassed_total";

/// The log lines written on this thread, as JSON, the format production uses.
#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Logs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Keeps a `Logs` capture going until dropped.
struct LogCapture {
    _subscriber: tracing::subscriber::DefaultGuard,
    _registered: tracing::Dispatch,
}

impl Logs {
    fn capture(&self) -> LogCapture {
        let logs = self.clone();
        let subscriber = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .json()
                .with_max_level(tracing::Level::INFO)
                .with_writer(move || logs.clone())
                .finish(),
        );
        // See `usage_report_tests.rs`: a second registered subscriber keeps
        // a test on another thread from switching a shared callsite off.
        let registered = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        LogCapture {
            _subscriber: subscriber,
            _registered: registered,
        }
    }

    fn contents(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }

    /// The fields of every line with this message, oldest first.
    fn lines(&self, message: &str) -> Vec<serde_json::Value> {
        self.contents()
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|line| line["fields"]["message"] == message)
            .map(|mut line| line["fields"].take())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Written first, sent from the file
// ---------------------------------------------------------------------------

#[test]
fn a_process_with_an_outbox_sends_until_accepted_eight_at_a_time_unless_told_otherwise() {
    assert_eq!(
        UsageReportPolicy::durable(),
        UsageReportPolicy {
            max_attempts: UsageReportPolicy::UNTIL_ACCEPTED,
            max_in_flight: 8,
            ..UsageReportPolicy::default()
        }
    );
    assert_eq!(UsageReportPolicy::durable().attempt_cap(), None);
    assert_eq!(UsageReportPolicy::default().attempt_cap(), Some(1));
}

#[tokio::test]
async fn a_report_is_written_to_the_file_sent_from_it_and_removed_once_accepted() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));

    billing.report(&delivery, "chatcmpl-1", None);
    // It is in the file, and it stays there while the billing API fails.
    eventually("a second attempt", || billing.received() >= 2).await;
    assert_eq!(pending_rows(&path), 1);
    assert_eq!(billing.written(), Vec::<String>::new());
    let stored: String = read(&path, "SELECT body FROM pending");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&stored).unwrap(),
        serde_json::json!({
            "type": "chat_completion",
            "model": "test-model",
            "input_tokens": 1200,
            "output_tokens": 340,
            "cache_read_tokens": 800,
            "id": "chatcmpl-1",
            "organization_id": "org-1",
            "workspace_id": "ws-1",
            "api_key_id": "key-1",
            "discount_to_user": 0.2,
        })
    );
    assert_eq!(
        read::<String>(
            &path,
            "SELECT request_id || ' ' || auth_path || ' ' || ingress_route || ' ' || last_outcome FROM pending"
        ),
        "request-chatcmpl-1 cloud_api_key long http_5xx"
    );
    assert!(read::<i64>(&path, "SELECT attempts FROM pending") >= 1);

    billing.answer(200);
    delivered(&delivery).await;
    assert_eq!(billing.written(), ["chatcmpl-1"]);
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(delivery.pending(), (0, 0));

    // Every attempt was the request a report has always been: the bearer
    // from the configuration, the request id of its request, the same bytes.
    assert_eq!(
        *billing.intake.authorizations.lock().unwrap(),
        BTreeSet::from(["Bearer usage-secret".to_string()])
    );
    assert_eq!(
        *billing.intake.request_ids.lock().unwrap(),
        BTreeSet::from(["request-chatcmpl-1".to_string()])
    );
    let bodies = billing.intake.bodies.lock().unwrap();
    assert!(bodies.iter().all(|body| body == &bodies[0]));
    assert_eq!(bodies[0], stored.as_bytes());

    // One report, counted once, at its final outcome; its attempts and its
    // retries as without an outbox, under the labels it was handed over with.
    let source = "auth_path=\"cloud_api_key\",ingress_route=\"long\"}";
    let attempts = billing.received() as f64;
    assert_eq!(value(&recorder, REPORTS, &[]), 1.0);
    assert_eq!(
        value(&recorder, REPORTS, &["{outcome=\"accepted\",", source]),
        1.0
    );
    assert_eq!(value(&recorder, ATTEMPTS, &[]), attempts);
    assert_eq!(
        value(&recorder, ATTEMPTS, &["{outcome=\"http_5xx\",", source]),
        attempts - 1.0
    );
    assert_eq!(
        value(&recorder, RETRIES, &["{reason=\"http_5xx\",", source]),
        attempts - 1.0
    );
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    assert_eq!(value(&recorder, TIME_TO_ACCEPTED_COUNT, &[source]), 1.0);
}

#[tokio::test]
async fn every_report_is_accepted_once_the_billing_api_is_back_however_often_it_failed() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(durable(|policy| policy.max_in_flight = 4), outbox(&path));

    for id in ids("down", 40) {
        billing.report(&delivery, &id, None);
    }
    // Far more attempts than a report ever got from memory.
    eventually("many failed attempts", || billing.received() >= 60).await;
    assert_eq!(billing.written(), Vec::<String>::new());
    assert_eq!(pending_rows(&path), 40);
    // Rate limiting is an answer that can pass too.
    billing.answer(429);
    let so_far = billing.received();
    eventually("attempts that were rate limited", || {
        billing.received() >= so_far + 10
    })
    .await;
    assert_eq!(pending_rows(&path), 40);

    billing.answer(200);
    delivered(&delivery).await;
    // None lost, none written twice.
    assert_eq!(billing.written_sorted(), ids("down", 40));
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 0);
    assert_eq!(value(&recorder, REPORTS, &[]), 40.0);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 40.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    assert!(value(&recorder, RETRIES, &["reason=\"http_429\""]) >= 10.0);
    assert_eq!(
        value(&recorder, ATTEMPTS, &[]),
        billing.received() as f64,
        "every attempt is counted"
    );
}

#[tokio::test]
async fn a_billing_api_that_is_down_gets_fewer_requests_not_the_backlog_in_a_loop() {
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    // The places pause for up to 300 ms after a failure, so each of the two
    // makes about three attempts a second however long the backlog is.
    let delivery = billing.delivery(
        durable(|policy| {
            policy.max_in_flight = 2;
            policy.initial_backoff = Duration::from_millis(200);
        }),
        UsageOutboxConfig {
            max_pause: Duration::from_millis(300),
            ..outbox(&path)
        },
    );
    for id in ids("backlog", 300) {
        billing.report(&delivery, &id, None);
    }
    eventually("the backlog is written", || pending_rows(&path) == 300).await;
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let attempts = billing.received();
    assert!(
        (2..=40).contains(&attempts),
        "{attempts} attempts in 1.5 s against a billing API that is down"
    );
    // And nothing is leased while the places wait: another process could
    // take any of it.
    eventually("no lease is held", || {
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL",
        ) == 0
    })
    .await;

    billing.answer(200);
    delivered(&delivery).await;
    assert_eq!(billing.written_sorted(), ids("backlog", 300));
}

// ---------------------------------------------------------------------------
// Restarts and two processes on one file
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_new_process_delivers_what_the_last_one_left_and_nothing_twice() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;

    let first = billing.delivery(durable(|_| {}), outbox(&path));
    for id in ids("early", 10) {
        billing.report(&first, &id, None);
    }
    delivered(&first).await;
    // The billing API goes away; what completes now can only be kept.
    billing.answer(503);
    for id in ids("late", 15) {
        billing.report(&first, &id, None);
    }
    eventually("the reports are in the file", || pending_rows(&path) == 15).await;
    eventually("each was tried", || {
        read::<i64>(&path, "SELECT COUNT(*) FROM pending WHERE attempts = 0") == 0
    })
    .await;

    // The process stops in good order: nothing is lost by it, nothing is
    // waited for, and no lease is left behind.
    let drained = first.drain_at_shutdown().await.unwrap();
    assert_eq!(drained.left_waiting, 0);
    assert_eq!(pending_rows(&path), 15);
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE lease_until_ms > 0"
        ),
        0
    );
    let closed = logs.lines(
        "Usage report outbox closed for shutdown: what it holds is sent by the next process",
    );
    assert_eq!(closed.len(), 1, "{}", logs.contents());
    assert_eq!(closed[0]["in_outbox"], 15);
    assert!(logs
        .lines("Usage reports left undelivered at shutdown — usage NOT billed")
        .is_empty());
    drop(first);

    // Some time passes before the next process is up, with a billing API
    // that answers again.
    tokio::time::sleep(Duration::from_millis(400)).await;
    billing.answer(200);
    let accepted_before = value(&recorder, TIME_TO_ACCEPTED_SUM, &[]);
    let second = billing.delivery(durable(|_| {}), outbox(&path));
    delivered(&second).await;

    let mut expected = ids("early", 10);
    expected.extend(ids("late", 15));
    expected.sort();
    assert_eq!(billing.written_sorted(), expected);
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 25.0);
    // The time to acceptance of a report counts from the completion of its
    // request, in whichever process that was: 15 reports, each at least the
    // 400 ms between the two.
    assert_eq!(value(&recorder, TIME_TO_ACCEPTED_COUNT, &[]), 25.0);
    let waited = value(&recorder, TIME_TO_ACCEPTED_SUM, &[]) - accepted_before;
    assert!(waited >= 15.0 * 0.4, "{waited}");
    // And so do its attempts, and why the last of them failed.
    let accepted = logs.lines("Direct-key usage report accepted by Cloud API");
    let late: Vec<_> = accepted
        .iter()
        .filter(|line| line["request_id"].as_str().unwrap().contains("late"))
        .collect();
    assert_eq!(late.len(), 15);
    for line in late {
        assert!(line["attempts"].as_u64().unwrap() >= 2, "{line}");
        assert!(
            line["since_completion_ms"].as_u64().unwrap() >= 400,
            "{line}"
        );
        assert_eq!(line["org_id"], "org-1");
        assert_eq!(line["workspace_id"], "ws-1");
        assert_eq!(line["api_key_id"], "key-1");
        assert_eq!(line["model"], "test-model");
        assert_eq!(line["ingress_route"], "long");
    }
    assert!(value(&recorder, RETRIES, &["reason=\"http_5xx\""]) >= 15.0);
    assert!(!logs.contents().contains(TOKEN));
}

/// A process that is killed, with reports leased and their attempts in
/// flight: its runtime goes away without a word, as with `kill -9`.
#[test]
fn a_process_that_dies_leaves_its_reports_to_the_next_once_its_leases_run_out() {
    let (_dir, path) = file();
    let surviving = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let billing = surviving.block_on(Billing::start(200));
    // Whatever the first process sends is never answered in its lifetime.
    billing.take(Duration::from_secs(120));

    // Two places, so four reports are leased: two being sent, two next.
    let policy = || {
        durable(|policy| {
            policy.attempt_timeout = Duration::from_millis(600);
            policy.max_in_flight = 2;
        })
    };
    let dying = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let first = dying.block_on(async { billing.delivery(policy(), outbox(&path)) });
    dying.block_on(async {
        for id in ids("orphan", 10) {
            billing.report(&first, &id, None);
        }
        eventually("its attempts are in flight", || billing.received() == 2).await;
        eventually("all of them are written", || pending_rows(&path) == 10).await;
    });
    const LEASED: &str = "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL";
    dying.block_on(eventually("four reports are leased", || {
        read::<i64>(&path, LEASED) == 4
    }));
    // The first of them is anybody's at this time, and not before.
    let free_at_ms: i64 = read(
        &path,
        "SELECT MIN(lease_until_ms) FROM pending WHERE lease_owner IS NOT NULL",
    );
    dying.shutdown_background();
    drop(first);

    // The next process sends at once what nobody holds, and the rest when
    // the dead holder's leases have run out: 2 × 600 ms and the margin after
    // it took them.
    billing.take(Duration::ZERO);
    let second = surviving.block_on(async { billing.delivery(policy(), outbox(&path)) });
    let first_orphan_at_ms = surviving.block_on(async {
        eventually("what was not leased is delivered", || {
            billing.written().len() >= 6
        })
        .await;
        if usage_outbox::now_ms() < free_at_ms {
            assert_eq!(billing.written().len(), 6);
            assert_eq!(read::<i64>(&path, LEASED), 4);
        }
        eventually("a leased report is delivered", || {
            billing.written().len() > 6
        })
        .await;
        let at_ms = usage_outbox::now_ms();
        delivered(&second).await;
        at_ms
    });
    assert!(
        first_orphan_at_ms >= free_at_ms,
        "a leased report was sent {} ms before its lease ran out",
        free_at_ms - first_orphan_at_ms
    );
    assert_eq!(billing.written_sorted(), ids("orphan", 10));
    assert_eq!(pending_rows(&path), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_processes_on_one_file_lose_nothing_and_send_nothing_twice() {
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(2));
    // The old process and the new one of a blue/green switch, both serving.
    let old = billing.delivery(durable(|policy| policy.max_in_flight = 4), outbox(&path));
    let new = billing.delivery(durable(|policy| policy.max_in_flight = 4), outbox(&path));

    let mut expected = Vec::new();
    for round in 0..150 {
        for (name, delivery) in [("old", &old), ("new", &new)] {
            let id = format!("{name}-{round}");
            billing.report(delivery, &id, None);
            expected.push(id);
        }
        if round % 10 == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    // The billing API fails for a while in the middle of it.
    billing.answer(503);
    tokio::time::sleep(Duration::from_millis(150)).await;
    billing.answer(200);
    delivered(&old).await;
    delivered(&new).await;

    expected.sort();
    assert_eq!(billing.written_sorted(), expected);
    assert_eq!(pending_rows(&path), 0);

    // The old one is stopped. What it is handed until then is not lost, and
    // what it leaves is the new one's.
    billing.answer(503);
    for round in 0..20 {
        billing.report(&old, &format!("last-{round}"), None);
    }
    old.drain_at_shutdown().await.unwrap();
    billing.answer(200);
    expected.extend((0..20).map(|round| format!("last-{round}")));
    expected.sort();
    // The new one finds them the next time it looks at the file.
    eventually("what the old process left is delivered", || {
        billing.written().len() >= expected.len()
    })
    .await;
    delivered(&new).await;
    assert_eq!(billing.written_sorted(), expected);
    assert_eq!(pending_rows(&path), 0);
}

// ---------------------------------------------------------------------------
// Reports the billing API will not take
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_report_refused_for_good_moves_to_rejected_is_counted_once_and_holds_nothing_up() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.refuse("refused", Some(422));
    // One place: a report that blocked it would block everything.
    let delivery = billing.delivery(durable(|policy| policy.max_in_flight = 1), outbox(&path));

    let started_at = Instant::now();
    billing.report(&delivery, "refused", None);
    for id in ids("after", 20) {
        billing.report(&delivery, &id, None);
    }
    delivered(&delivery).await;
    assert!(started_at.elapsed() < Duration::from_secs(5));
    assert_eq!(billing.written_sorted(), ids("after", 20));
    // Sent once, never again.
    assert_eq!(billing.received(), 21);

    assert_eq!(pending_rows(&path), 0);
    let kept: String = read(
        &path,
        "SELECT reason || ' ' || status || ' ' || outcome || ' ' || attempts || ' ' || \
         json_extract(body, '$.id') || ' ' || request_id FROM rejected",
    );
    assert_eq!(kept, "rejected 422 http_4xx 1 refused request-refused");
    assert_eq!(value(&recorder, REPORTS, &[]), 21.0);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"http_4xx\""]), 1.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 1.0);
    assert_eq!(value(&recorder, DROPPED, &["reason=\"rejected\""]), 1.0);
    assert_eq!(value(&recorder, RETRIES, &[]), 0.0);
    eventually("the gauge follows", || {
        value(&recorder, REJECTED, &[]) == 1.0
    })
    .await;
    let refusals = logs.lines("Usage reporting returned non-success");
    assert_eq!(refusals.len(), 1, "{}", logs.contents());
    assert_eq!(refusals[0]["request_id"], "request-refused");
    assert_eq!(refusals[0]["status"], "422 Unprocessable Entity");

    // A person fixes what was wrong and puts it back, with the statements of
    // docs/gateway-mode.md. The running process finds it by itself.
    billing.refuse("refused", None);
    let by_hand = rusqlite::Connection::open(&path).unwrap();
    by_hand.busy_timeout(Duration::from_secs(5)).unwrap();
    by_hand
        .execute_batch(
            "BEGIN IMMEDIATE;
             INSERT INTO pending (body, request_id, model_label, auth_path, ingress_route, completed_at_ms)
               SELECT body, request_id, model_label, auth_path, ingress_route, completed_at_ms
               FROM rejected WHERE id = 1;
             DELETE FROM rejected WHERE id = 1;
             COMMIT;",
        )
        .unwrap();
    eventually("the report put back is accepted", || {
        billing.written().contains(&"refused".to_string())
    })
    .await;
    delivered(&delivery).await;
    assert_eq!(pending_rows(&path), 0);
    eventually("the gauge follows", || {
        value(&recorder, REJECTED, &[]) == 0.0
    })
    .await;
}

#[tokio::test]
async fn a_usage_token_the_billing_api_does_not_accept_keeps_every_report_until_it_does() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    // The token was rotated on the other side: every report gets a 401,
    // which says nothing about the report.
    let billing = Billing::start(401).await;
    let delivery = billing.delivery(durable(|policy| policy.max_in_flight = 2), outbox(&path));
    for id in ids("report", 12) {
        billing.report(&delivery, &id, None);
    }
    eventually("every report was refused at least once", || {
        read::<i64>(&path, "SELECT COUNT(*) FROM pending WHERE attempts > 0") == 12
    })
    .await;
    // None of them is given up on: they wait.
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 0);
    assert_eq!(pending_rows(&path), 12);
    assert_eq!(value(&recorder, REPORTS, &[]), 0.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    assert!(value(&recorder, ATTEMPTS, &["outcome=\"http_4xx\""]) >= 12.0);
    assert_eq!(
        read::<String>(
            &path,
            "SELECT group_concat(DISTINCT last_outcome) FROM pending"
        ),
        "http_401"
    );
    // A gateway in front that forbids the caller is the same case.
    billing.answer(403);
    eventually("attempts that were forbidden", || {
        value(&recorder, RETRIES, &["reason=\"http_401\""]) >= 1.0
            && read::<i64>(
                &path,
                "SELECT COUNT(*) FROM pending WHERE last_outcome = 'http_403'",
            ) >= 1
    })
    .await;
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 0);

    // The token is put right. Nothing has to be put back by hand.
    billing.answer(200);
    delivered(&delivery).await;
    assert_eq!(billing.written_sorted(), ids("report", 12));
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 12.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);

    // From memory, a 401 ends a report as it always has: nothing there
    // outlives the restart that puts a token right.
    billing.answer(401);
    let before = billing.received();
    let in_memory = UsageReportDelivery::new(UsageReportPolicy {
        max_attempts: 5,
        initial_backoff: Duration::from_millis(5),
        max_in_flight: 2,
        ..UsageReportPolicy::default()
    });
    billing.report(&in_memory, "in-memory", None);
    delivered(&in_memory).await;
    assert_eq!(billing.received(), before + 1);
    assert_eq!(value(&recorder, DROPPED, &["reason=\"rejected\""]), 1.0);
}

#[tokio::test]
async fn an_explicit_attempt_cap_ends_in_rejected_and_the_table_keeps_the_newest() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|policy| {
            policy.max_attempts = 3;
            policy.max_in_flight = 1;
        }),
        UsageOutboxConfig {
            max_rejected: 2,
            ..outbox(&path)
        },
    );
    for id in ["one", "two", "three"] {
        billing.report(&delivery, id, None);
    }
    delivered(&delivery).await;

    // Three attempts each, then kept where a person can see them: not sent
    // for ever, and not gone either.
    assert_eq!(billing.received(), 9);
    assert_eq!(billing.written(), Vec::<String>::new());
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"http_5xx\""]), 3.0);
    assert_eq!(
        value(&recorder, DROPPED, &["reason=\"attempts_exhausted\""]),
        3.0
    );
    // The table holds two; the oldest made room and was counted.
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 2);
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM rejected WHERE reason = 'attempts_exhausted' \
             AND status = 503 AND outcome = 'http_5xx' AND attempts = 3"
        ),
        2
    );
    eventually("the eviction is counted", || {
        value(&recorder, REJECTED_EVICTED, &[]) == 1.0
    })
    .await;
    assert_eq!(value(&recorder, REJECTED, &[]), 2.0);
}

#[tokio::test]
async fn with_a_deadline_a_report_has_a_maximum_age_whichever_process_kept_it() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let policy = || {
        durable(|policy| {
            policy.attempt_timeout = Duration::from_millis(200);
            policy.deadline = Some(Duration::from_millis(2_000));
        })
    };
    let first = billing.delivery(policy(), outbox(&path));
    billing.report(&first, "too-old", None);
    eventually("it was tried", || billing.received() >= 1).await;
    first.drain_at_shutdown().await.unwrap();
    assert_eq!(pending_rows(&path), 1);

    // Past its deadline when the next process finds it: even a billing API
    // that would take it is not asked.
    tokio::time::sleep(Duration::from_millis(2_000)).await;
    billing.answer(200);
    let attempts = billing.received();
    let second = billing.delivery(policy(), outbox(&path));
    delivered(&second).await;
    assert_eq!(billing.received(), attempts);
    assert_eq!(billing.written(), Vec::<String>::new());
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 0);
    assert_eq!(
        value(&recorder, REPORTS, &["outcome=\"deadline_exceeded\""]),
        1.0
    );
    assert_eq!(value(&recorder, DROPPED, &["reason=\"deadline\""]), 1.0);
    let lines =
        logs.lines("Usage report dropped: not accepted before its deadline — usage NOT billed");
    assert_eq!(lines.len(), 1, "{}", logs.contents());
    assert_eq!(lines[0]["request_id"], "request-too-old");
    assert_eq!(lines[0]["last_attempt"], "http_5xx");
    assert!(lines[0]["attempts"].as_u64().unwrap() >= 1);
    assert!(lines[0]["since_completion_ms"].as_u64().unwrap() >= 1_800);
}

#[tokio::test]
async fn a_full_outbox_drops_the_reports_that_waited_longest_and_counts_them() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|policy| policy.max_in_flight = 1),
        UsageOutboxConfig {
            max_pending: 5,
            ..outbox(&path)
        },
    );
    for id in ids("report", 5) {
        billing.report(&delivery, &id, None);
    }
    eventually("the first five are written", || pending_rows(&path) == 5).await;
    for id in ["report-5", "report-6", "report-7"] {
        billing.report(&delivery, id, None);
    }
    eventually("the bound is applied", || {
        value(&recorder, REPORTS, &["outcome=\"queue_full\""]) == 3.0
    })
    .await;
    assert_eq!(pending_rows(&path), 5);
    assert_eq!(value(&recorder, DROPPED, &["reason=\"queue_full\""]), 3.0);

    const FULL: &str =
        "Usage report dropped: the outbox is full and this report waited longest — usage NOT billed";
    let lines = logs.lines(FULL);
    assert_eq!(lines.len(), 3, "{}", logs.contents());
    let mut dropped = BTreeSet::new();
    for line in &lines {
        let request_id = line["request_id"].as_str().unwrap();
        dropped.insert(request_id.trim_start_matches("request-").to_string());
        assert_eq!(line["outcome"], "queue_full");
        assert_eq!(line["max_pending"], 5);
        assert_eq!(line["org_id"], "org-1");
        assert_eq!(line["model"], "test-model");
    }
    // The oldest go: three of the first five. Which three depends on the
    // moment, because a report somebody is sending, or about to, is spared:
    // its answer may be on its way.
    assert!(
        dropped.iter().all(|id| ids("report", 5).contains(id)),
        "{dropped:?}"
    );

    billing.answer(200);
    delivered(&delivery).await;
    let written: BTreeSet<String> = billing.written().into_iter().collect();
    assert_eq!(written.len(), 5);
    assert!(written.is_disjoint(&dropped));
    for id in ["report-5", "report-6", "report-7"] {
        assert!(written.contains(id), "{id} in {written:?}");
    }
    assert_eq!(value(&recorder, REPORTS, &[]), 8.0);
}

// ---------------------------------------------------------------------------
// The file never costs a request anything
// ---------------------------------------------------------------------------

#[tokio::test]
async fn handing_a_report_over_never_waits_for_the_file() {
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    billing.report(&delivery, "before", None);
    delivered(&delivery).await;

    // Another connection holds the write lock: nothing can be written.
    let other = rusqlite::Connection::open(&path).unwrap();
    other.busy_timeout(Duration::from_secs(5)).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    let locked_at = Instant::now();

    let mut slowest = Duration::ZERO;
    for id in ids("locked", 500) {
        let started_at = Instant::now();
        billing.report(&delivery, &id, None);
        slowest = slowest.max(started_at.elapsed());
    }
    assert!(slowest < Duration::from_millis(400), "{slowest:?}");
    // For two seconds the reports wait in memory. None is sent: a report is
    // written first.
    while locked_at.elapsed() < Duration::from_secs(2) {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let started_at = Instant::now();
        billing.report(
            &delivery,
            &format!("during-{:?}", locked_at.elapsed()),
            None,
        );
        slowest = slowest.max(started_at.elapsed());
    }
    assert!(slowest < Duration::from_millis(400), "{slowest:?}");
    assert_eq!(billing.written(), ["before"]);
    let handed_over = delivery.pending().0;
    assert!(handed_over >= 500, "{handed_over}");

    other.execute_batch("COMMIT").unwrap();
    delivered(&delivery).await;
    let written = billing.written();
    assert_eq!(written.len(), handed_over + 1);
    assert!(ids("locked", 500).iter().all(|id| written.contains(id)));
}

#[tokio::test]
async fn a_file_that_cannot_be_opened_costs_the_durability_of_the_reports_and_nothing_else() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("not-there-yet");
    let path = missing.join("outbox.db");
    let billing = Billing::start(200).await;
    // Starting never fails, whatever the path.
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));

    for id in ids("memory", 10) {
        billing.report(&delivery, &id, None);
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    delivered(&delivery).await;
    // Delivered all the same, from memory, as without an outbox.
    assert_eq!(billing.written_sorted(), ids("memory", 10));
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 10.0);
    assert_eq!(value(&recorder, AVAILABLE, &[]), 0.0);
    assert!(value(&recorder, ERRORS, &["op=\"open\""]) >= 1.0);
    assert_eq!(value(&recorder, BYPASSED, &[]), 10.0);
    assert!(!path.exists());
    const UNAVAILABLE: &str = "Usage report outbox unavailable: reports are delivered from memory \
                               and not kept across a restart until it is back";
    let said = logs.lines(UNAVAILABLE);
    // Said once, however often opening is tried again.
    assert_eq!(said.len(), 1, "{}", logs.contents());
    assert_eq!(said[0]["op"], "open");
    let tries = value(&recorder, ERRORS, &["op=\"open\""]);
    eventually("opening is tried again", || {
        value(&recorder, ERRORS, &["op=\"open\""]) > tries
    })
    .await;
    assert_eq!(logs.lines(UNAVAILABLE).len(), 1);

    // The directory appears. From then on reports are kept.
    std::fs::create_dir(&missing).unwrap();
    eventually("the outbox is back", || {
        value(&recorder, AVAILABLE, &[]) == 1.0
    })
    .await;
    assert_eq!(logs.lines("Usage report outbox available again").len(), 1);
    billing.answer(503);
    billing.report(&delivery, "kept", None);
    eventually("it is written", || pending_rows(&path) == 1).await;
    billing.answer(200);
    delivered(&delivery).await;
    assert!(billing.written().contains(&"kept".to_string()));
    assert_eq!(value(&recorder, BYPASSED, &[]), 10.0);
}

#[tokio::test]
async fn a_file_that_stops_taking_writes_costs_the_durability_of_the_reports_and_nothing_else() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    let delivery = billing.delivery(durable(|policy| policy.max_in_flight = 1), outbox(&path));
    billing.report(&delivery, "kept-1", None);
    delivered(&delivery).await;
    assert_eq!(value(&recorder, AVAILABLE, &[]), 1.0);
    assert_eq!(value(&recorder, BYPASSED, &[]), 0.0);

    // The disk is full, say: every write fails from here on.
    let store = &delivery.outbox.as_ref().unwrap().store;
    store.inject_fault(true);
    for id in ids("memory", 10) {
        billing.report(&delivery, &id, None);
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    delivered(&delivery).await;
    let mut expected = ids("memory", 10);
    expected.push("kept-1".to_string());
    expected.sort();
    assert_eq!(billing.written_sorted(), expected);
    assert_eq!(value(&recorder, AVAILABLE, &[]), 0.0);
    assert!(value(&recorder, ERRORS, &["op=\"begin\""]) >= 1.0);
    // The one whose write failed, and the ones that did not even try.
    assert_eq!(value(&recorder, BYPASSED, &[]), 10.0);
    assert!(value(&recorder, BYPASSED, &["reason=\"write_failed\""]) >= 1.0);

    // While it lasts the billing API fails too. Reports wait in memory, for
    // a place, as they do without an outbox.
    billing.answer(503);
    for id in ["waiting-0", "waiting-1", "waiting-2"] {
        billing.report(&delivery, id, None);
    }
    eventually("one is being sent", || delivery.pending() == (2, 1)).await;
    assert_eq!(pending_rows(&path), 0);

    // The disk is back: what waited in memory is written to the file after
    // all, and new reports are kept again.
    store.inject_fault(false);
    eventually("the outbox is back", || {
        value(&recorder, AVAILABLE, &[]) == 1.0
    })
    .await;
    eventually("what waited is in the file", || pending_rows(&path) == 2).await;
    billing.report(&delivery, "kept-2", None);
    eventually("and so is a new report", || pending_rows(&path) == 3).await;
    billing.answer(200);
    delivered(&delivery).await;
    expected.extend(["waiting-0", "waiting-1", "waiting-2", "kept-2"].map(String::from));
    expected.sort();
    assert_eq!(billing.written_sorted(), expected);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 15.0);
}

#[tokio::test]
async fn the_file_never_holds_the_bearer_or_where_reports_are_sent() {
    let (dir, path) = file();
    let billing = Billing::start(503).await;
    billing.refuse("refused", Some(400));
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    billing.report(&delivery, "refused", Some("example/alpha"));
    for id in ids("pending", 20) {
        billing.report(&delivery, &id, None);
    }
    eventually("a row in each table, and attempts made", || {
        read::<i64>(&path, "SELECT COUNT(*) FROM rejected") == 1
            && read::<i64>(&path, "SELECT COUNT(*) FROM pending WHERE attempts > 0") == 20
    })
    .await;

    let host = billing.url.trim_start_matches("http://");
    let holds_a_secret = |when: &str| {
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let file = entry.unwrap().path();
            let bytes = std::fs::read(&file).unwrap();
            let text = String::from_utf8_lossy(&bytes);
            for secret in [TOKEN, "Bearer", host, "authorization"] {
                assert!(
                    !text.contains(secret),
                    "{secret} in {} {when}",
                    file.display()
                );
            }
            // What is there is the reports.
            if file == path {
                assert!(text.contains("org-1") || bytes.len() <= 4096 * 8, "{when}");
            }
        }
    };
    holds_a_secret("while the process runs");
    delivery.drain_at_shutdown().await.unwrap();
    holds_a_secret("after it stopped");
    assert_eq!(pending_rows(&path), 20);
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

#[tokio::test]
async fn shutdown_gives_attempts_in_flight_their_time_and_leaves_the_rest_to_the_next_process() {
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(1_200));
    let policy = || {
        durable(|policy| {
            policy.max_in_flight = 2;
            policy.shutdown_drain = Duration::from_secs(20);
        })
    };
    let first = billing.delivery(policy(), outbox(&path));
    for id in ids("report", 8) {
        billing.report(&first, &id, None);
    }
    eventually("two attempts are in flight", || billing.received() == 2).await;

    let drained = first.drain_at_shutdown().await.unwrap();
    // The two attempts were waited for and nothing was started after them.
    assert!(drained.waited >= Duration::from_millis(400), "{drained:?}");
    assert!(drained.waited < Duration::from_secs(10), "{drained:?}");
    assert_eq!((drained.left_waiting, drained.left_in_flight), (0, 0));
    assert_eq!(billing.written().len(), 2);
    assert_eq!(billing.received(), 2);
    assert_eq!(pending_rows(&path), 6);
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE lease_until_ms > 0"
        ),
        0
    );
    // A report handed over after that is not swallowed: it is delivered from
    // memory, as without an outbox.
    billing.take(Duration::ZERO);
    billing.report(&first, "afterwards", None);
    eventually("it is sent", || {
        billing.written().contains(&"afterwards".to_string())
    })
    .await;

    // The next process has the rest at once: it waits for no lease.
    let started_at = Instant::now();
    let second = billing.delivery(policy(), outbox(&path));
    delivered(&second).await;
    assert!(started_at.elapsed() < Duration::from_secs(5));
    let mut expected = ids("report", 8);
    expected.push("afterwards".to_string());
    expected.sort();
    assert_eq!(billing.written_sorted(), expected);
}

#[tokio::test]
async fn shutdown_without_a_drain_waits_for_nothing_and_still_loses_nothing() {
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_secs(30));
    let first = billing.delivery(durable(|policy| policy.max_in_flight = 2), outbox(&path));
    for id in ids("report", 6) {
        billing.report(&first, &id, None);
    }
    eventually("two attempts are in flight", || billing.received() == 2).await;
    // One more, handed over as the process is told to stop.
    billing.report(&first, "last", None);

    let drained = first.drain_at_shutdown().await.unwrap();
    assert!(drained.waited < Duration::from_secs(5), "{drained:?}");
    assert_eq!((drained.left_waiting, drained.left_in_flight), (0, 2));
    // Everything is in the file, the last one included, and free to take:
    // the two attempts in flight are sent again by the next process, which
    // the billing API deduplicates.
    assert_eq!(pending_rows(&path), 7);
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE lease_until_ms > 0"
        ),
        0
    );
}

// ---------------------------------------------------------------------------
// Metrics and log lines
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_outbox_has_its_gauges_and_queue_depth_keeps_meaning_waiting() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    // One place, and an answer that takes a while: one report is being sent
    // and the others wait.
    billing.take(Duration::from_millis(2_500));
    let policy = || {
        durable(|policy| {
            policy.max_in_flight = 1;
            policy.attempt_timeout = Duration::from_secs(30);
            policy.shutdown_drain = Duration::from_secs(30);
        })
    };
    let delivery = billing.delivery(policy(), outbox(&path));
    billing.report(&delivery, "alpha-0", Some("example/alpha"));
    eventually("it is being sent", || billing.received() == 1).await;
    for (id, model) in [
        ("alpha-1", Some("example/alpha")),
        ("alpha-2", Some("example/alpha")),
        ("beta-0", Some("example/beta")),
        ("plain-0", None),
    ] {
        billing.report(&delivery, id, model);
    }
    let alpha = "model=\"example/alpha\"";
    let beta = "model=\"example/beta\"";
    eventually("the gauges follow", || {
        value(&recorder, PENDING, &[]) == 5.0 && value(&recorder, WAITING, &[]) == 4.0
    })
    .await;
    // The rows of the file per model, and of those the ones nobody sends.
    assert_eq!(value(&recorder, PENDING, &[alpha]), 3.0);
    assert_eq!(value(&recorder, PENDING, &[beta]), 1.0);
    assert_eq!(value(&recorder, WAITING, &[alpha]), 2.0);
    assert_eq!(value(&recorder, WAITING, &[beta]), 1.0);
    assert_eq!(value(&recorder, IN_FLIGHT, &[alpha]), 1.0);
    assert_eq!(value(&recorder, IN_FLIGHT, &[]), 1.0);
    assert_eq!(delivery.pending(), (4, 1));
    // The report of the single model has its series without the label.
    let unlabelled = |family: &str| {
        recorder
            .handle()
            .render()
            .lines()
            .filter(|line| line.starts_with(&format!("{family} ")))
            .map(|line| line.rsplit(' ').next().unwrap().parse::<f64>().unwrap())
            .sum::<f64>()
    };
    assert_eq!(unlabelled(PENDING), 1.0);
    assert_eq!(unlabelled(WAITING), 1.0);
    assert_eq!(value(&recorder, AVAILABLE, &[]), 1.0);
    assert_eq!(value(&recorder, REJECTED, &[]), 0.0);
    // The age of the oldest report grows while it waits.
    let age = value(&recorder, OLDEST_AGE, &[]);
    eventually("the oldest report gets older", || {
        value(&recorder, OLDEST_AGE, &[]) >= age + 0.1
    })
    .await;

    // No attempt has ended yet: these are the gauges, and nothing else.
    assert_eq!(
        families(&recorder),
        [WAITING, AVAILABLE, OLDEST_AGE, PENDING, REJECTED, IN_FLIGHT]
            .map(String::from)
            .into()
    );

    // The process stops once the report it is sending is through, and a new
    // one reads the labels of the other four back from the file.
    let drained = delivery.drain_at_shutdown().await.unwrap();
    assert_eq!(drained.left_in_flight, 0);
    assert_eq!(billing.written(), ["alpha-0"]);
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    billing.answer(503);
    billing.take(Duration::ZERO);
    let delivery = billing.delivery(policy(), outbox(&path));
    eventually("the backlog is read", || {
        value(&recorder, PENDING, &[]) == 4.0
    })
    .await;
    assert_eq!(value(&recorder, PENDING, &[alpha]), 2.0);
    assert_eq!(value(&recorder, PENDING, &[beta]), 1.0);
    eventually("an attempt of each model failed", || {
        value(&recorder, ATTEMPTS, &[alpha]) >= 1.0 && value(&recorder, ATTEMPTS, &[beta]) >= 1.0
    })
    .await;
    billing.answer(200);
    delivered(&delivery).await;
    assert_eq!(
        value(&recorder, REPORTS, &[alpha, "outcome=\"accepted\""]),
        2.0
    );
    assert_eq!(
        value(&recorder, REPORTS, &[beta, "outcome=\"accepted\""]),
        1.0
    );
    // Everything is back to nothing, and stays there to be read.
    eventually("the gauges are back to 0", || {
        value(&recorder, PENDING, &[]) == 0.0
            && value(&recorder, WAITING, &[]) == 0.0
            && value(&recorder, IN_FLIGHT, &[]) == 0.0
            && value(&recorder, OLDEST_AGE, &[]) == 0.0
    })
    .await;
    assert!(recorder
        .handle()
        .render()
        .contains(&format!("{PENDING}{{{alpha}}} 0")));
}

#[tokio::test]
async fn a_report_sent_until_accepted_says_so_in_its_lines() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    billing.report(&delivery, "one", None);
    eventually("an attempt failed", || billing.received() >= 1).await;
    billing.answer(200);
    delivered(&delivery).await;

    let enabled = logs.lines(
        "Usage report outbox enabled: a report is kept on disk until the billing API accepts \
         it or refuses it for good",
    );
    assert_eq!(enabled.len(), 1, "{}", logs.contents());
    assert_eq!(enabled[0]["path"], path.display().to_string());
    assert_eq!(enabled[0]["until_accepted"], true);
    assert_eq!(enabled[0]["max_pending"], 1_000_000);
    // No attempt cap is named where there is none.
    let configured = logs.lines("Usage report delivery configured");
    assert_eq!(configured.len(), 1);
    assert!(
        configured[0].get("max_attempts").is_none(),
        "{}",
        configured[0]
    );
    assert_eq!(configured[0]["max_in_flight"], 8);
    let retrying = logs.lines("Usage report attempt failed, retrying");
    assert!(!retrying.is_empty());
    assert!(retrying[0].get("max_attempts").is_none(), "{}", retrying[0]);
    assert_eq!(retrying[0]["attempt"], 1);
    assert_eq!(retrying[0]["request_id"], "request-one");
    assert!(!logs.contents().contains(TOKEN));
}

#[tokio::test]
async fn an_outbox_is_always_delivered_under_a_cap() {
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(100));
    // "No cap" is what the policy says, and not what an outbox does.
    let delivery = billing.delivery(
        UsageReportPolicy {
            max_attempts: UsageReportPolicy::UNTIL_ACCEPTED,
            ..UsageReportPolicy::default()
        },
        outbox(&path),
    );
    assert_eq!(delivery.policy().max_in_flight, 8);
    for id in ids("report", 40) {
        billing.report(&delivery, &id, None);
    }
    let mut most = 0;
    while billing.written().len() < 40 {
        most = most.max(delivery.pending().1);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(most, 8);
    delivered(&delivery).await;
}

#[test]
fn the_labels_a_row_is_written_with_are_read_back_as_they_were() {
    for path in [AuthPath::TrustedConfigToken, AuthPath::CloudApiKey] {
        assert_eq!(auth_path_from(path.as_label()), path);
    }
    for route in [
        IngressRouteKind::Canonical,
        IngressRouteKind::Indexed,
        IngressRouteKind::Long,
        IngressRouteKind::LongIndexed,
        IngressRouteKind::Other,
        IngressRouteKind::Missing,
    ] {
        // A variant added to the enum has to be added to `ingress_route_from`.
        match route {
            IngressRouteKind::Canonical
            | IngressRouteKind::Indexed
            | IngressRouteKind::Long
            | IngressRouteKind::LongIndexed
            | IngressRouteKind::Other
            | IngressRouteKind::Missing => {}
        }
        assert_eq!(ingress_route_from(route.as_label()), route);
    }
    assert_eq!(auth_path_from("something new"), AuthPath::CloudApiKey);
    assert_eq!(ingress_route_from("something new"), IngressRouteKind::Other);
    // Every reason an attempt is made again for can be kept in a row.
    for (reason, outcome) in PASSING_FAILURES {
        assert_eq!(passing_failure(reason), Some((reason, outcome)));
    }
    assert_eq!(passing_failure("http_4xx"), None);
    // The two an outbox adds, and nothing else of the 4xx.
    let answer = |status: u16| Ok(reqwest::StatusCode::from_u16(status).unwrap());
    assert_eq!(caller_refused(&answer(401)), Some("http_401"));
    assert_eq!(caller_refused(&answer(403)), Some("http_403"));
    for status in [200, 400, 402, 404, 409, 422, 429, 500, 503] {
        assert_eq!(caller_refused(&answer(status)), None, "{status}");
    }
}

// ---------------------------------------------------------------------------
// How fast
// ---------------------------------------------------------------------------

/// Not a test of anything but a measurement, run by hand:
///
/// ```text
/// cargo test --release --lib throughput_of_the_outbox -- --ignored --nocapture
/// ```
///
/// 20,000 reports handed over as fast as one thread can, with the production
/// timings, against a billing API that accepts at once: how long handing one
/// over takes, and how many a second get through the file and out.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement, run by hand"]
async fn throughput_of_the_outbox() {
    const REPORTS: usize = 20_000;
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    // Room for the whole burst to wait for the writer: this measures the
    // file, not the bound on what waits for it.
    let policy = durable(|policy| policy.max_queued = 100_000);
    let delivery = billing.delivery(policy, UsageOutboxConfig::at(&path));
    eventually("the file is open", || {
        pending_rows(&path) == 0 && path.exists()
    })
    .await;

    let ids: Vec<String> = (0..REPORTS).map(|n| format!("chatcmpl-{n}")).collect();
    let mut took = Vec::with_capacity(REPORTS);
    let started_at = Instant::now();
    for id in &ids {
        let handed_over_at = Instant::now();
        billing.report(&delivery, id, None);
        took.push(handed_over_at.elapsed());
    }
    let handing_over = started_at.elapsed();
    delivered(&delivery).await;
    let through = started_at.elapsed();

    took.sort();
    let at = |quantile: f64| took[((REPORTS - 1) as f64 * quantile) as usize];
    println!(
        "handing over {REPORTS} reports: {handing_over:?} in all; per report p50 {:?}, p99 {:?}, \
         p99.9 {:?}, max {:?}",
        at(0.5),
        at(0.99),
        at(0.999),
        at(1.0)
    );
    println!(
        "through the outbox and accepted: {through:?} in all, {:.0} reports a second",
        REPORTS as f64 / through.as_secs_f64()
    );
    let written: BTreeSet<String> = billing.written().into_iter().collect();
    assert_eq!(written.len(), REPORTS);
    assert_eq!(billing.written().len(), REPORTS, "none was sent twice");
    assert_eq!(pending_rows(&path), 0);
}
