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
use crate::usage_outbox::Clock;

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
    /// The requests being answered right now, and the most there ever were,
    /// for the reports in `refused` alone and for all of them.
    answering: AtomicUsize,
    most_answering: AtomicUsize,
    answering_refused: AtomicUsize,
    most_answering_refused: AtomicUsize,
    /// The `id` of every report answered with a success, in order, and when.
    written: Mutex<Vec<String>>,
    written_at: Mutex<BTreeMap<String, Instant>>,
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
    let refused = intake.refused.lock().unwrap().get(&id).copied();
    let status = refused.unwrap_or_else(|| intake.status.load(Ordering::SeqCst));
    let _all = Answering::start(&intake.answering, &intake.most_answering);
    let _refused = refused
        .map(|_| Answering::start(&intake.answering_refused, &intake.most_answering_refused));
    let delay = intake.delay_ms.load(Ordering::SeqCst);
    if delay > 0 {
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
    let status = StatusCode::from_u16(status).unwrap();
    if status.is_success() {
        intake
            .written_at
            .lock()
            .unwrap()
            .entry(id.clone())
            .or_insert_with(Instant::now);
        intake.written.lock().unwrap().push(id);
    }
    status
}

/// One request being answered; it stops counting when the answer is out, or
/// when the caller hung up.
struct Answering<'a>(&'a AtomicUsize);

impl<'a> Answering<'a> {
    fn start(now: &'a AtomicUsize, most: &AtomicUsize) -> Self {
        most.fetch_max(now.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
        Self(now)
    }
}

impl Drop for Answering<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
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
            answering: AtomicUsize::new(0),
            most_answering: AtomicUsize::new(0),
            answering_refused: AtomicUsize::new(0),
            most_answering_refused: AtomicUsize::new(0),
            written: Mutex::default(),
            written_at: Mutex::default(),
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

    /// How many of the reports written have an id that starts with `prefix`.
    fn written_of(&self, prefix: &str) -> usize {
        let written = self.intake.written.lock().unwrap();
        written.iter().filter(|id| id.starts_with(prefix)).count()
    }

    /// When the report `id` was first written.
    fn written_at(&self, id: &str) -> Option<Instant> {
        self.intake.written_at.lock().unwrap().get(id).copied()
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
        max_backoff: Duration::from_millis(200),
        max_pause: Duration::from_millis(40),
        ..UsageOutboxConfig::at(path)
    }
}

/// The store of `delivery`, to make its file fail.
fn store(delivery: &UsageReportDelivery) -> &Store<Item> {
    &delivery.outbox.as_ref().unwrap().store
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

/// Wait until the file at `path` is a database with its tables.
async fn opened(path: &Path) {
    eventually("the file is open", || {
        read::<i64>(path, "SELECT COUNT(*) FROM meta WHERE key = 'db_id'") == 1
    })
    .await;
}

/// Put rows into the file as an earlier process would have left them:
/// reports for the completions `ids`, whose requests completed a minute ago,
/// that failed `attempts` times, the last time with a 500, and are due now.
fn left_behind(path: &Path, ids: &[String], attempts: u32) {
    let mut conn = rusqlite::Connection::open(path).unwrap();
    conn.busy_timeout(Duration::from_secs(5)).unwrap();
    let now_ms = Clock::default().now_ms();
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    for id in ids {
        let body = serde_json::json!({
            "type": "chat_completion",
            "model": "test-model",
            "input_tokens": 1200,
            "output_tokens": 340,
            "id": id,
            "organization_id": "org-1",
            "workspace_id": "ws-1",
            "api_key_id": "key-1",
        });
        tx.execute(
            "INSERT INTO pending (body, request_id, auth_path, ingress_route, completed_at_ms, \
             attempts, next_attempt_at_ms, last_outcome) \
             VALUES (?1, ?2, 'cloud_api_key', 'long', ?3, ?4, ?3, ?5)",
            rusqlite::params![
                body.to_string(),
                format!("request-{id}"),
                now_ms - 60_000,
                attempts,
                (attempts > 0).then_some("http_5xx")
            ],
        )
        .unwrap();
    }
    tx.commit().unwrap();
}

/// How long after it was handed over each report of `handed_over` was
/// written, slowest last. A report not written at all fails the test.
fn lags(billing: &Billing, handed_over: &BTreeMap<String, Instant>) -> Vec<Duration> {
    let mut lags: Vec<Duration> = handed_over
        .iter()
        .map(|(id, at)| {
            let written = billing
                .written_at(id)
                .unwrap_or_else(|| panic!("{id} was not written"));
            written.saturating_duration_since(*at)
        })
        .collect();
    lags.sort();
    lags
}

/// The scheduler state of `delivery`'s outbox.
fn kept(delivery: &UsageReportDelivery) -> MutexGuard<'_, Kept> {
    delivery.outbox.as_ref().unwrap().kept()
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
const FULL: &str = "inference_proxy_usage_report_outbox_full";
const PAUSED: &str = "inference_proxy_usage_report_outbox_billing_paused";
const PENDING: &str = "inference_proxy_usage_report_outbox_pending";
const OLDEST_AGE: &str = "inference_proxy_usage_report_outbox_oldest_pending_age_seconds";
const REJECTED: &str = "inference_proxy_usage_report_outbox_rejected";
const REJECTED_EVICTED: &str = "inference_proxy_usage_report_outbox_rejected_evicted_total";
const ERRORS: &str = "inference_proxy_usage_report_outbox_errors_total";
const BYPASSED: &str = "inference_proxy_usage_report_outbox_bypassed_total";
const BYTES: &str = "inference_proxy_usage_report_outbox_bytes";
const UNWRITTEN: &str = "inference_proxy_usage_report_outbox_unwritten";
const IN_MEMORY: &str = "inference_proxy_usage_report_outbox_in_memory";
const REPLACED: &str = "inference_proxy_usage_report_outbox_replaced_total";

const UNAVAILABLE_LINE: &str = "Usage report outbox unavailable: reports are held in memory, \
                                sent from there, and written to the file when it is back";
const AVAILABLE_LINE: &str = "Usage report outbox available again";
const FULL_LINE: &str = "Usage report outbox is full: new reports are held in memory and sent \
                         from there until it has room. What it holds is still sent";
const ROOM_LINE: &str = "Usage report outbox has room again";
const NOT_ANSWERING_LINE: &str = "The billing API is not answering usage reports: one report at a \
                                  time is sent until one gets an answer";
const ANSWERING_LINE: &str = "The billing API answers usage reports again";
const TOO_OLD_LINE: &str = "Usage report moved to rejected: older than a report may grow — \
                            usage NOT billed unless it is put back";
const CLOSED_LINE: &str =
    "Usage report outbox closed for shutdown: what it holds is sent by the next process";
const NOT_CLOSED_LINE: &str = "Usage report outbox could not be closed for shutdown: what was \
                               not written is lost, and the leases of this process are left to \
                               run out";
const LEFT_LINE: &str =
    "Usage report left in memory at shutdown: the outbox did not take it — usage NOT billed";
const UNDELIVERED_LINE: &str = "Usage reports left undelivered at shutdown — usage NOT billed";
const FILE_FULL_LINE: &str = "Usage report dropped: the outbox is full and this report waited \
                              longest — usage NOT billed";
const ENABLED_LINE: &str = "Usage report outbox enabled: a report is kept on disk until the \
                            billing API accepts it or refuses it for good";

/// CPU time this thread has used so far, as the kernel accounts for it:
/// to the nanosecond where it keeps scheduler statistics, to the clock tick
/// (10 ms as a rule) otherwise. `None` where neither can be read.
#[cfg(target_os = "linux")]
fn thread_cpu() -> Option<Duration> {
    let precise = std::fs::read_to_string("/proc/thread-self/schedstat")
        .ok()
        .and_then(|stat| stat.split_whitespace().next()?.parse().ok())
        .map(Duration::from_nanos);
    precise.or_else(|| {
        // After the name, which may hold anything: state, then numbers, of
        // which the 12th and 13th are the user and system time in ticks.
        let stat = std::fs::read_to_string("/proc/thread-self/stat").ok()?;
        let after_name = stat.rsplit_once(") ")?.1;
        let mut fields = after_name.split_whitespace().skip(11);
        let ticks: u64 =
            fields.next()?.parse::<u64>().ok()? + fields.next()?.parse::<u64>().ok()?;
        Some(Duration::from_millis(ticks * 10))
    })
}

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
    eventually("its row says how the attempts went", || {
        read::<String>(
            &path,
            "SELECT request_id || ' ' || auth_path || ' ' || ingress_route || ' ' || last_outcome \
             FROM pending WHERE attempts > 0",
        ) == "request-chatcmpl-1 cloud_api_key long http_5xx"
    })
    .await;

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
    eventually("many failed attempts", || billing.received() >= 40).await;
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
    assert_eq!(
        value(&recorder, ATTEMPTS, &[]),
        billing.received() as f64,
        "every attempt is counted"
    );
}

// ---------------------------------------------------------------------------
// Reports that keep failing, and a billing API that is down
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reports_that_keep_failing_never_stand_in_front_of_new_ones() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(10));
    // Eight places, of which the reports that failed before may hold two.
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    opened(&path).await;

    // 400 reports an earlier process left, which the billing API answers
    // with a 500 every time. Their backoff is at its ceiling, 200 ms here:
    // 2000 of them are due every second, more than every place could send.
    let poison = ids("poison", 400);
    for id in &poison {
        billing.refuse(id, Some(500));
    }
    left_behind(&path, &poison, 8);

    // New reports, 100 a second for three seconds.
    let mut handed_over = BTreeMap::new();
    let mut tick = tokio::time::interval(Duration::from_millis(10));
    for n in 0..300 {
        tick.tick().await;
        let id = format!("healthy-{n}");
        handed_over.insert(id.clone(), Instant::now());
        billing.report(&delivery, &id, None);
    }
    eventually("the healthy ones are written", || {
        billing.written_of("healthy") == 300
    })
    .await;

    // They went out as they came in: no lag builds up behind the retries.
    // (Tens of milliseconds on an idle machine. The bounds leave room for a
    // busy one; behind retries that took the places, the new reports would
    // not go out at all while the 400 keep coming due.)
    let lags = lags(&billing, &handed_over);
    let (median, worst) = (lags[lags.len() / 2], lags[lags.len() - 1]);
    println!("healthy reports beside 400 failing ones: median {median:?}, worst {worst:?}");
    assert!(median < Duration::from_secs(1), "median {median:?}");
    assert!(worst < Duration::from_secs(5), "worst {worst:?}");
    // The retries were made all the while, in the places they may have and
    // in no more.
    let most = billing.intake.most_answering_refused.load(Ordering::SeqCst);
    assert!((1..=2).contains(&most), "{most} retries in flight at once");
    assert!(billing.intake.most_answering.load(Ordering::SeqCst) <= 8);
    let retries = billing.received() - 300;
    assert!(retries >= 100, "{retries} retries in three seconds");
    // And all this is not an outage: nothing was ever paused for it.
    eventually("only the 400 are left", || pending_rows(&path) == 400).await;
    assert!(
        logs.lines(NOT_ANSWERING_LINE).is_empty(),
        "{}",
        logs.contents()
    );
}

#[tokio::test]
async fn reports_that_fail_from_their_first_attempt_hold_the_next_ones_up_for_a_moment_only() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(5));
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));

    // 400 reports, none of which the billing API will take, on a lane where
    // nothing else completes. Twenty first attempts fail in a row: that
    // looks like the billing API being down, and is treated as that.
    for id in ids("poison", 400) {
        billing.refuse(&id, Some(500));
        billing.report(&delivery, &id, None);
    }
    eventually("it looks like an outage", || {
        !logs.lines(NOT_ANSWERING_LINE).is_empty()
    })
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let tried: i64 = read(&path, "SELECT COUNT(*) FROM pending WHERE attempts > 0");
    assert!((20..200).contains(&tried), "{tried} of 400 were tried");

    // Requests complete again, 100 a second. The first report does not wait
    // for the pause, is accepted, and that ends the outage that never was.
    let mut handed_over = BTreeMap::new();
    let mut tick = tokio::time::interval(Duration::from_millis(10));
    for n in 0..200 {
        tick.tick().await;
        let id = format!("healthy-{n}");
        handed_over.insert(id.clone(), Instant::now());
        billing.report(&delivery, &id, None);
    }
    eventually("the healthy ones are written", || {
        billing.written_of("healthy") == 200
    })
    .await;
    assert!(!logs.lines(ANSWERING_LINE).is_empty());

    // The reports nobody had tried yet are older, so they are tried first,
    // twenty at a time, each time ended by the next report that comes in:
    // for a moment the new ones wait behind them. Then every one of the 400
    // has failed once, is a retry, and stands in nobody's way any more.
    let all = lags(&billing, &handed_over);
    let late: BTreeMap<String, Instant> = (150..200)
        .map(|n| format!("healthy-{n}"))
        .map(|id| (id.clone(), handed_over[&id]))
        .collect();
    let late = lags(&billing, &late);
    println!(
        "healthy reports after 400 that failed from the start: worst {:?}, and of the last 50 {:?}",
        all[all.len() - 1],
        late[late.len() - 1]
    );
    assert!(
        all[all.len() - 1] < Duration::from_secs(5),
        "{:?}",
        all[all.len() - 1]
    );
    assert!(
        late[late.len() - 1] < Duration::from_secs(3),
        "{:?}",
        late[late.len() - 1]
    );
    eventually("only the 400 are left, each tried", || {
        pending_rows(&path) == 400
            && read::<i64>(&path, "SELECT COUNT(*) FROM pending WHERE attempts > 0") == 400
    })
    .await;
}

#[tokio::test]
async fn each_time_such_reports_look_like_an_outage_the_next_report_that_comes_in_ends_it() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(5));
    let delivery = billing.delivery(
        // Ten seconds or more between two probes, from the first one: a
        // report that had to wait for one would be seen to, on any machine.
        durable(|policy| policy.initial_backoff = Duration::from_secs(20)),
        UsageOutboxConfig {
            max_pause: Duration::from_secs(30),
            max_backoff: Duration::from_millis(50),
            ..outbox(&path)
        },
    );
    // A hundred reports the billing API will not take, and nothing else for
    // a while: twenty of them failed, the others have not been tried.
    for id in ids("poison", 100) {
        billing.refuse(&id, Some(500));
        billing.report(&delivery, &id, None);
    }
    eventually("it looks like an outage", || {
        kept(&delivery).breaker.engaged
    })
    .await;
    eventually("nothing is being sent", || delivery.pending().1 == 0).await;

    // Requests complete again. The untried ones are older and go first,
    // twenty at a time, and each time that looks like an outage again. Each
    // time the next report that comes in is tried at once, is accepted, and
    // ends it: none of them waits for a probe.
    let mut handed_over = BTreeMap::new();
    let mut tick = tokio::time::interval(Duration::from_millis(10));
    for n in 0..150 {
        tick.tick().await;
        let id = format!("healthy-{n}");
        handed_over.insert(id.clone(), Instant::now());
        billing.report(&delivery, &id, None);
    }
    eventually("the healthy ones are written", || {
        billing.written_of("healthy") == 150
    })
    .await;
    let lags = lags(&billing, &handed_over);
    let worst = lags[lags.len() - 1];
    println!(
        "healthy reports behind 80 untried failing ones: worst {worst:?}; it looked like an \
         outage {} times",
        logs.lines(NOT_ANSWERING_LINE).len()
    );
    assert!(worst < Duration::from_secs(5), "{worst:?}");
    assert!(logs.lines(NOT_ANSWERING_LINE).len() >= 2);
    eventually("every one of the hundred has been tried", || {
        read::<i64>(&path, "SELECT COUNT(*) FROM pending WHERE attempts > 0") == 100
    })
    .await;
}

#[tokio::test]
async fn a_share_of_the_reports_failing_slows_the_others_down_by_nothing() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(5));
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));

    // For three seconds 30 % of the reports that come in are answered with
    // a 500 (one organization's, say), the others accepted.
    let mut handed_over = BTreeMap::new();
    let mut bad = Vec::new();
    let mut tick = tokio::time::interval(Duration::from_millis(5));
    for n in 0u32..600 {
        tick.tick().await;
        let failing = (n * 7919) % 100 < 30;
        let id = format!("{}-{n}", if failing { "bad" } else { "good" });
        if failing {
            billing.refuse(&id, Some(500));
            bad.push(id.clone());
        } else {
            handed_over.insert(id.clone(), Instant::now());
        }
        billing.report(&delivery, &id, None);
    }
    eventually("the good ones are written", || {
        billing.written_of("good") == handed_over.len()
    })
    .await;
    let lags = lags(&billing, &handed_over);
    let (median, worst) = (lags[lags.len() / 2], lags[lags.len() - 1]);
    println!("good reports while 30 % fail: median {median:?}, worst {worst:?}");
    // Tens of milliseconds on an idle machine; the bounds are for a busy one.
    assert!(median < Duration::from_secs(1), "median {median:?}");
    assert!(worst < Duration::from_secs(5), "worst {worst:?}");
    assert_eq!(billing.written_of("bad"), 0);
    // Some reports failing while others are accepted is not the billing API
    // being down: it was never treated as that.
    assert!(
        logs.lines(NOT_ANSWERING_LINE).is_empty(),
        "{}",
        logs.contents()
    );

    // The incident ends: the ones that failed are accepted too, without
    // anything put back by hand.
    for id in &bad {
        billing.refuse(id, None);
    }
    delivered(&delivery).await;
    assert_eq!(billing.written().len(), 600);
    assert_eq!(pending_rows(&path), 0);
}

#[tokio::test]
async fn reports_that_failed_before_failing_in_a_row_are_no_outage_while_others_are_accepted() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    let delivery = billing.delivery(
        durable(|_| {}),
        UsageOutboxConfig {
            // Four failures in a row would do, were they first attempts, and
            // so would four of any kind with nothing answered for a minute.
            breaker_after: 4,
            breaker_window: Duration::from_secs(60),
            max_backoff: Duration::from_millis(20),
            ..outbox(&path)
        },
    );
    opened(&path).await;

    // Forty reports an earlier process left, which fail every time and are
    // due again every 10 to 20 ms: between two reports that come in, a tenth
    // of a second apart, dozens of them fail in a row.
    let poison = ids("poison", 40);
    for id in &poison {
        billing.refuse(id, Some(500));
    }
    left_behind(&path, &poison, 8);
    for n in 0..15 {
        billing.report(&delivery, &format!("healthy-{n}"), None);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    eventually("the healthy ones are written", || {
        billing.written_of("healthy") == 15
    })
    .await;

    // Reports that failed before fail again whether or not anything is
    // wrong with the billing API. However many of them do so in a row, that
    // is not what an outage looks like while it accepts the others.
    let retries = billing.received() - 15;
    assert!(retries >= 15 * 8, "{retries} retries failed meanwhile");
    assert!(
        logs.lines(NOT_ANSWERING_LINE).is_empty(),
        "{}",
        logs.contents()
    );
    assert!(!kept(&delivery).breaker.engaged);
    assert_eq!(kept(&delivery).breaker.fresh_failures, 0);
}

#[tokio::test]
async fn the_backoff_of_a_report_grows_to_minutes_and_a_place_never_waits_it_out() {
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    // The production ceilings, ten minutes and five seconds, from half a second.
    let production = UsageOutboxConfig::at(&path);
    let delivery = billing.delivery(
        UsageReportPolicy::durable(),
        UsageOutboxConfig {
            max_backoff: production.max_backoff,
            max_pause: production.max_pause,
            ..outbox(&path)
        },
    );
    let scheduler = delivery.outbox.as_ref().unwrap();
    let range = |attempts: u32| -> (Duration, Duration) {
        let seen: Vec<Duration> = (0..200)
            .map(|_| scheduler.backoff(&delivery.policy, attempts))
            .collect();
        (*seen.iter().min().unwrap(), *seen.iter().max().unwrap())
    };
    let ms = Duration::from_millis;
    // Half of it fixed, half random: 250 to 500 ms after the first failure,
    // doubling from there.
    for (attempts, ceiling) in [(1, 500), (2, 1_000), (5, 8_000), (8, 64_000), (11, 512_000)] {
        let (least, most) = range(attempts);
        assert!(
            least >= ms(ceiling / 2) && most <= ms(ceiling),
            "{attempts}: {least:?}..{most:?}"
        );
        assert!(
            most > ms(ceiling * 3 / 4),
            "{attempts}: never above {most:?}"
        );
    }
    // Past 30 seconds, which is where a report that waits in a place stops,
    // and up to ten minutes and no further.
    for attempts in [12, 20, 1_000, u32::MAX] {
        let (least, most) = range(attempts);
        assert!(
            least >= ms(300_000) && most <= ms(600_000),
            "{attempts}: {least:?}..{most:?}"
        );
    }
    // The pause between two probes of a billing API that is down doubles
    // the same way, and stops at five seconds.
    let pauses = |probes: u32| -> (Duration, Duration) {
        let seen: Vec<Duration> = (0..200)
            .map(|_| scheduler.pause(&delivery.policy, probes))
            .collect();
        (*seen.iter().min().unwrap(), *seen.iter().max().unwrap())
    };
    for (probes, ceiling) in [(1, 500), (3, 2_000), (4, 4_000), (5, 5_000), (40, 5_000)] {
        let (least, most) = pauses(probes);
        assert!(
            least >= ms(ceiling / 2) && most <= ms(ceiling),
            "{probes}: {least:?}..{most:?}"
        );
    }

    // A report that waits for its next attempt holds no place: with one
    // place and a report backing off, the next report goes straight through.
    let (_dir, path) = file();
    let delivery = billing.delivery(
        durable(|policy| {
            policy.max_in_flight = 1;
            policy.initial_backoff = Duration::from_secs(20);
        }),
        UsageOutboxConfig {
            max_backoff: Duration::from_secs(600),
            ..outbox(&path)
        },
    );
    billing.refuse("failing", Some(503));
    billing.report(&delivery, "failing", None);
    eventually("it failed", || billing.received() == 1).await;
    eventually("no place is held", || delivery.pending() == (1, 0)).await;
    let started_at = Instant::now();
    for id in ids("next", 20) {
        billing.report(&delivery, &id, None);
    }
    eventually("the others are written", || billing.written().len() == 20).await;
    assert!(started_at.elapsed() < Duration::from_secs(5));
    assert_eq!(
        billing.received(),
        21,
        "the one that failed waits ten seconds or more"
    );
}

#[tokio::test]
async fn a_report_is_tried_again_when_it_is_due_not_when_the_file_is_next_looked_at() {
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|_| {}),
        UsageOutboxConfig {
            // Nothing but a report coming due makes the dispatcher look.
            recheck_interval: Duration::from_secs(60),
            reopen_interval: Duration::from_secs(60),
            max_backoff: Duration::from_millis(10),
            // Never an outage, however long this takes on a busy machine.
            breaker_after: u32::MAX,
            ..outbox(&path)
        },
    );
    opened(&path).await;
    let scheduler = delivery.outbox.as_ref().unwrap();

    // What came due between the dispatcher's last look and now has not been
    // seen by it, and nothing else would make it look: it is what the timer
    // is set for, although it has passed. What was due when it looked is
    // not, or the dispatcher would run in a loop while a report waits for a
    // place. (Nothing is awaited in between: the dispatcher does not run.)
    let looked_at = Instant::now();
    let came_due = looked_at + Duration::from_micros(20);
    kept(&delivery).retry_at = Some(came_due);
    std::thread::sleep(Duration::from_millis(2));
    assert_eq!(delivery.next_timer(scheduler, looked_at), Some(came_due));
    assert_eq!(delivery.next_timer(scheduler, came_due), None);
    kept(&delivery).retry_at = None;

    // The same from outside: eight reports held in memory that fail every
    // time, each due again a few milliseconds later, many of them coming due
    // just as the dispatcher is busy with another. They are all tried again
    // on time, thousands of times, without one look at the file.
    store(&delivery).inject_fault(true);
    for id in ids("failing", 8) {
        billing.report(&delivery, &id, None);
    }
    let started_at = Instant::now();
    eventually("thousands of attempts", || billing.received() >= 3_000).await;
    println!(
        "3000 attempts at eight reports that fail: {:?}",
        started_at.elapsed()
    );
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(delivery.pending().0 + delivery.pending().1, 8);
}

#[tokio::test]
async fn a_billing_api_that_is_down_is_probed_by_one_report_at_a_time() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    billing.take(Duration::from_millis(5));
    let delivery = billing.delivery(
        durable(|policy| policy.initial_backoff = Duration::from_millis(100)),
        UsageOutboxConfig {
            breaker_after: 6,
            max_pause: Duration::from_millis(300),
            ..outbox(&path)
        },
    );
    for id in ids("backlog", 300) {
        billing.report(&delivery, &id, None);
    }
    eventually("the backlog is written", || pending_rows(&path) == 300).await;
    eventually("it is seen that nothing gets through", || {
        value(&recorder, PAUSED, &[]) == 1.0
    })
    .await;
    // The attempts in flight when that was seen come back; then it is one
    // report at a time, a pause apart: 300 reports waiting make no
    // difference to what the billing API is sent.
    tokio::time::sleep(Duration::from_millis(200)).await;
    billing.intake.most_answering.store(0, Ordering::SeqCst);
    let before = billing.received();
    tokio::time::sleep(Duration::from_millis(2_000)).await;
    let attempts = billing.received() - before;
    assert!(
        (3..=16).contains(&attempts),
        "{attempts} attempts in 2 s of outage"
    );
    assert_eq!(billing.intake.most_answering.load(Ordering::SeqCst), 1);
    // Each probe is another report (the newest never tried), so an outage
    // uses up nobody's attempts, and nothing is leased beyond the one at hand.
    assert_eq!(
        read::<i64>(&path, "SELECT COALESCE(MAX(attempts), 0) FROM pending"),
        1
    );
    assert!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL"
        ) <= 1
    );
    assert_eq!(
        logs.lines(NOT_ANSWERING_LINE).len(),
        1,
        "{}",
        logs.contents()
    );
    assert_eq!(logs.lines(NOT_ANSWERING_LINE)[0]["after"], 6);

    // It is back. The next probe is accepted and that ends it at once: the
    // backlog goes out at the full rate, not a pause apart.
    billing.answer(200);
    let back_at = Instant::now();
    delivered(&delivery).await;
    // (A pause apart, 300 reports would take most of a minute.)
    assert!(
        back_at.elapsed() < Duration::from_secs(15),
        "{:?}",
        back_at.elapsed()
    );
    assert_eq!(billing.written_sorted(), ids("backlog", 300));
    assert_eq!(value(&recorder, PAUSED, &[]), 0.0);
    assert_eq!(logs.lines(ANSWERING_LINE).len(), 1);
    assert_eq!(logs.lines(NOT_ANSWERING_LINE).len(), 1);
    assert!(billing.intake.most_answering.load(Ordering::SeqCst) > 1);
}

#[tokio::test]
async fn when_only_old_reports_fail_a_new_one_is_sent_at_once() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    let delivery = billing.delivery(
        // Pauses of ten seconds or more from the first one, which no test
        // waits out.
        durable(|policy| policy.initial_backoff = Duration::from_secs(20)),
        UsageOutboxConfig {
            breaker_after: 6,
            breaker_window: Duration::from_millis(100),
            max_pause: Duration::from_secs(30),
            max_backoff: Duration::from_millis(50),
            ..outbox(&path)
        },
    );
    opened(&path).await;
    // A quiet lane: nothing completes, and the ten reports an earlier
    // process left fail every time. They are no first attempts, so they say
    // little; but nothing at all is answered for a while, and that looks
    // like the billing API being down.
    let poison = ids("poison", 10);
    for id in &poison {
        billing.refuse(id, Some(500));
    }
    left_behind(&path, &poison, 3);
    eventually("it looks like an outage", || {
        logs.lines(NOT_ANSWERING_LINE).len() == 1
    })
    .await;
    eventually("nothing is being sent", || delivery.pending().1 == 0).await;
    let probe_in = kept(&delivery)
        .breaker
        .probe_at
        .unwrap()
        .saturating_duration_since(Instant::now());
    assert!(probe_in > Duration::from_secs(8), "{probe_in:?}");

    // A new report. It does not wait for the pause: it is the probe, it is
    // accepted, and that ends the matter.
    let started_at = Instant::now();
    billing.report(&delivery, "new", None);
    eventually("it is written", || billing.written() == ["new"]).await;
    assert!(
        started_at.elapsed() < Duration::from_secs(5),
        "{:?}",
        started_at.elapsed()
    );
    eventually("the billing API counts as answering", || {
        logs.lines(ANSWERING_LINE).len() == 1
    })
    .await;
    eventually("only the ten are left", || pending_rows(&path) == 10).await;
}

#[tokio::test]
async fn in_an_outage_one_new_report_goes_ahead_of_the_pause_and_no_more() {
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        // Ten seconds or more between two probes: none comes due by itself
        // while this looks at what is sent between them.
        durable(|policy| policy.initial_backoff = Duration::from_secs(20)),
        UsageOutboxConfig {
            breaker_after: 6,
            max_pause: Duration::from_secs(30),
            max_backoff: Duration::from_millis(50),
            ..outbox(&path)
        },
    );
    for id in ids("first", 10) {
        billing.report(&delivery, &id, None);
    }
    eventually("the billing API counts as down", || {
        kept(&delivery).breaker.engaged
    })
    .await;
    eventually("nothing is being sent", || delivery.pending().1 == 0).await;
    let before = billing.received();

    // Requests keep completing while it is down. The first of their reports
    // is tried at once; the others wait for the probes like everything else,
    // so the billing API is not sent a request per request served.
    for id in ids("during", 50) {
        billing.report(&delivery, &id, None);
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(billing.received(), before + 1);
    assert!(kept(&delivery).breaker.newcomer_tried);
    assert_eq!(pending_rows(&path), 60);

    // The next probe comes due (here: at once, not seconds later). Reports
    // were handed over since the last one, so what comes next is no news.
    let probe_now = || {
        kept(&delivery).breaker.probe_at = Some(Instant::now());
        delivery.outbox.as_ref().unwrap().wake.notify_one();
    };
    probe_now();
    eventually("the probe is sent", || billing.received() == before + 2).await;
    eventually("and has failed", || delivery.pending().1 == 0).await;
    billing.report(&delivery, "no-news", None);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(billing.received(), before + 2);
    // Then a probe comes due after a pause in which nothing was handed over
    // (the lane was quiet, or what was failing has all been seen). The next
    // report to come is news again, and goes ahead of the pause.
    probe_now();
    eventually("the probe is sent", || billing.received() == before + 3).await;
    eventually("and has failed", || delivery.pending().1 == 0).await;
    probe_now();
    eventually("the probe is sent", || billing.received() == before + 4).await;
    eventually("and has failed", || delivery.pending().1 == 0).await;
    billing.report(&delivery, "news", None);
    eventually("it is tried at once", || billing.received() == before + 5).await;
}

#[tokio::test]
async fn a_usage_token_the_billing_api_does_not_accept_keeps_every_report_until_it_does() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    // The token was rotated on the other side: every report gets a 401,
    // which says nothing about the report.
    let billing = Billing::start(401).await;
    let delivery = billing.delivery(
        durable(|policy| {
            policy.max_in_flight = 2;
            policy.initial_backoff = Duration::from_millis(400);
        }),
        UsageOutboxConfig {
            max_pause: Duration::from_millis(200),
            ..outbox(&path)
        },
    );
    for id in ids("report", 12) {
        billing.report(&delivery, &id, None);
    }
    // It is about this process, so about every report: the first 401 is
    // enough to stop sending them one after the other.
    eventually("the first answers", || value(&recorder, PAUSED, &[]) == 1.0).await;
    assert!(billing.received() <= 2, "{}", billing.received());
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(billing.received() <= 9, "{}", billing.received());
    // None of them is given up on: they wait.
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 0);
    assert_eq!(pending_rows(&path), 12);
    assert_eq!(value(&recorder, REPORTS, &[]), 0.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    assert_eq!(
        read::<String>(
            &path,
            "SELECT group_concat(DISTINCT last_outcome) FROM pending WHERE attempts > 0"
        ),
        "http_401"
    );

    // The token is put right. Nothing has to be put back by hand.
    billing.answer(200);
    delivered(&delivery).await;
    assert_eq!(billing.written_sorted(), ids("report", 12));
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 12.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    assert!(value(&recorder, RETRIES, &["reason=\"http_401\""]) >= 1.0);

    // A 403 is not that. The billing API never answers one for the token:
    // it is about the report, and final.
    billing.refuse("forbidden", Some(403));
    billing.report(&delivery, "forbidden", None);
    billing.report(&delivery, "after", None);
    delivered(&delivery).await;
    assert_eq!(
        read::<String>(
            &path,
            "SELECT reason || ' ' || status || ' ' || json_extract(body, '$.id') FROM rejected"
        ),
        "rejected 403 forbidden"
    );
    assert!(billing.written().contains(&"after".to_string()));
    assert_eq!(logs.lines(NOT_ANSWERING_LINE).len(), 1);
}

#[tokio::test]
async fn a_report_that_is_never_accepted_ends_in_rejected_when_it_is_too_old() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.refuse("poison", Some(500));
    // Days in production; a second and a half here.
    let delivery = billing.delivery(
        // An attempt cap is not what ends a report with an outbox, whatever
        // the policy says: three attempts would be a moment of an outage.
        durable(|policy| policy.max_attempts = 3),
        UsageOutboxConfig {
            max_age: Duration::from_millis(1_500),
            ..outbox(&path)
        },
    );
    assert_eq!(delivery.policy().attempt_cap(), None);
    billing.report(&delivery, "poison", None);
    eventually("it is tried more often than the cap", || {
        billing.received() >= 5
    })
    .await;
    assert_eq!(pending_rows(&path), 1);

    eventually("it is too old", || pending_rows(&path) == 0).await;
    let attempts = billing.received() as i64;
    assert_eq!(
        read::<String>(
            &path,
            "SELECT reason || ' ' || outcome || ' ' || (status IS NULL) || ' ' || \
             json_extract(body, '$.id') FROM rejected"
        ),
        "max_age http_5xx 1 poison"
    );
    assert_eq!(
        read::<i64>(&path, "SELECT attempts FROM rejected"),
        attempts
    );
    eventually("it is counted, once", || {
        value(&recorder, REPORTS, &["outcome=\"deadline_exceeded\""]) == 1.0
    })
    .await;
    assert_eq!(value(&recorder, DROPPED, &["reason=\"max_age\""]), 1.0);
    eventually("the gauge follows", || {
        value(&recorder, REJECTED, &["reason=\"max_age\""]) == 1.0
    })
    .await;
    let lines = logs.lines(TOO_OLD_LINE);
    assert_eq!(lines.len(), 1, "{}", logs.contents());
    assert_eq!(lines[0]["request_id"], "request-poison");
    assert_eq!(lines[0]["reason"], "max_age");
    assert_eq!(lines[0]["last_attempt"], "http_5xx");
    assert_eq!(lines[0]["attempts"], attempts);
    assert!(lines[0]["since_completion_ms"].as_u64().unwrap() >= 1_500);
    // And nothing is sent again after that.
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(billing.received() as i64, attempts);
}

#[tokio::test]
async fn a_report_too_old_is_moved_while_it_waits_and_is_not_sent_when_its_place_comes() {
    // A report that waits for its next attempt, twenty seconds away or more.
    // It is seen to be too old when it is, not when that attempt comes.
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|policy| policy.initial_backoff = Duration::from_secs(40)),
        UsageOutboxConfig {
            max_age: Duration::from_millis(800),
            max_backoff: Duration::from_secs(600),
            ..outbox(&path)
        },
    );
    billing.report(&delivery, "waiting", None);
    eventually("it failed once", || {
        read::<i64>(&path, "SELECT COUNT(*) FROM pending WHERE attempts = 1") == 1
    })
    .await;
    let started_at = Instant::now();
    eventually("it is too old", || {
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM rejected WHERE reason = 'max_age'",
        ) == 1
    })
    .await;
    assert!(
        started_at.elapsed() < Duration::from_secs(8),
        "{:?}",
        started_at.elapsed()
    );
    assert_eq!(billing.received(), 1, "it was not tried again for that");
    assert_eq!(
        read::<String>(&path, "SELECT outcome || ' ' || attempts FROM rejected"),
        "http_5xx 1"
    );

    // Reports leased to this process and waiting for its one place, behind
    // attempts that take their time. Nobody else looks at them meanwhile, so
    // each is looked at when its turn comes: too old by then, it is not sent.
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    billing.take(Duration::from_millis(700));
    let delivery = billing.delivery(
        durable(|policy| {
            policy.max_in_flight = 1;
            policy.initial_backoff = Duration::from_secs(40);
        }),
        UsageOutboxConfig {
            max_age: Duration::from_millis(1_000),
            max_backoff: Duration::from_secs(600),
            ..outbox(&path)
        },
    );
    for id in ids("queued", 4) {
        billing.report(&delivery, &id, None);
    }
    eventually("the four are in rejected", || {
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM rejected WHERE reason = 'max_age'",
        ) == 4
    })
    .await;
    // The first was sent, and the second while it was young enough; the
    // others had waited more than a second when the place was theirs.
    assert!(
        (1..=2).contains(&billing.received()),
        "{} of the four were sent",
        billing.received()
    );
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM rejected WHERE outcome = 'never_sent' AND attempts = 0"
        ),
        4 - billing.received() as i64
    );
    assert_eq!(pending_rows(&path), 0);
}

#[tokio::test]
async fn with_a_deadline_a_report_too_old_goes_to_rejected_whichever_process_kept_it() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
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
    // that would take it is not asked. But it is not destroyed either.
    tokio::time::sleep(Duration::from_millis(2_000)).await;
    billing.answer(200);
    let attempts = billing.received();
    let second = billing.delivery(policy(), outbox(&path));
    delivered(&second).await;
    assert_eq!(billing.received(), attempts);
    assert_eq!(billing.written(), Vec::<String>::new());
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(
        read::<String>(
            &path,
            "SELECT reason || ' ' || json_extract(body, '$.id') FROM rejected"
        ),
        "deadline too-old"
    );
    eventually("it is counted", || {
        value(&recorder, REPORTS, &["outcome=\"deadline_exceeded\""]) == 1.0
    })
    .await;
    assert_eq!(value(&recorder, DROPPED, &["reason=\"deadline\""]), 1.0);
}

#[tokio::test]
async fn a_clock_set_forward_destroys_nothing() {
    let (_dir, path) = file();
    let clock = Clock::default();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|_| {}),
        UsageOutboxConfig {
            clock: clock.clone(),
            ..outbox(&path)
        },
    );
    for id in ids("report", 20) {
        billing.report(&delivery, &id, None);
    }
    eventually("they are in the file", || pending_rows(&path) == 20).await;

    // A year forward: every report is older than a report may grow. They
    // are all still there to be put back.
    clock.step(365 * 24 * 3_600_000);
    eventually("they are too old", || pending_rows(&path) == 0).await;
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM rejected WHERE reason = 'max_age'"
        ),
        20
    );
    // Put back with the documented statement, and the clock right again,
    // they are sent like any other.
    clock.step(-365 * 24 * 3_600_000);
    billing.answer(200);
    put_back(&path, "max_age");
    eventually("they are accepted", || billing.written().len() == 20).await;
    assert_eq!(billing.written_sorted(), ids("report", 20));
}

// ---------------------------------------------------------------------------
// Restarts, a process that dies, two processes
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
            "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL"
        ),
        0
    );
    let closed = logs.lines(CLOSED_LINE);
    assert_eq!(closed.len(), 1, "{}", logs.contents());
    assert_eq!(closed[0]["in_outbox"], 15);
    assert_eq!(closed[0]["left_unwritten"], 0);
    assert!(logs.lines(UNDELIVERED_LINE).is_empty());
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
/// flight: its runtime goes away without a word and its file with it, as
/// with `kill -9`. (`tests/usage_outbox.rs` kills a real one.)
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

    // Two places, so eight reports are leased: two being sent, and three
    // rounds of two for the places to come. A lease lasts two attempt
    // timeouts and the margin, 4.4 s here.
    let policy = || {
        durable(|policy| {
            policy.attempt_timeout = Duration::from_secs(2);
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
        for id in ids("orphan", 12) {
            billing.report(&first, &id, None);
        }
        eventually("its attempts are in flight", || billing.received() == 2).await;
        eventually("all of them are written", || pending_rows(&path) == 12).await;
    });
    // What the process that is about to die holds, by its name in the file.
    let dead = store(&first).owner().to_string();
    let its_leases = format!("SELECT COUNT(*) FROM pending WHERE lease_owner = '{dead}'");
    dying.block_on(eventually("eight reports are leased", || {
        read::<i64>(&path, &its_leases) == 8
    }));
    // The first of them is anybody's at this time, and not before.
    let free_at_ms: i64 = read(
        &path,
        &format!("SELECT MIN(lease_until_ms) FROM pending WHERE lease_owner = '{dead}'"),
    );
    assert_eq!(billing.received(), 2, "none of its attempts had ended");
    // It dies. A killed process writes nothing more, not even that it gives
    // its leases back, which is what a task that ends in a living one does.
    store(&first).inject_fault(true);
    dying.shutdown_background();
    drop(first);

    // The next process sends at once what nobody holds, and the rest when
    // the dead holder's leases have run out: two attempt timeouts and the
    // margin after it took them.
    billing.take(Duration::ZERO);
    let clock = Clock::default();
    let second = surviving.block_on(async { billing.delivery(policy(), outbox(&path)) });
    let first_orphan_at_ms = surviving.block_on(async {
        eventually("what was not leased is delivered", || {
            billing.written().len() >= 4
        })
        .await;
        // Read first, then the clock: if the leases had not run out after
        // these were read, they had not when they were read.
        let (written, still_leased) = (billing.written().len(), read::<i64>(&path, &its_leases));
        if clock.now_ms() < free_at_ms {
            assert_eq!((written, still_leased), (4, 8));
        }
        eventually("a leased report is delivered", || {
            billing.written().len() > 4
        })
        .await;
        let at_ms = clock.now_ms();
        delivered(&second).await;
        at_ms
    });
    assert!(
        first_orphan_at_ms >= free_at_ms,
        "a leased report was sent {} ms before its lease ran out",
        free_at_ms - first_orphan_at_ms
    );
    assert_eq!(billing.written_sorted(), ids("orphan", 12));
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
        value(&recorder, REJECTED, &["reason=\"rejected\""]) == 1.0
    })
    .await;
    // The other reasons a row can be there for have their series too, at 0.
    assert_eq!(value(&recorder, REJECTED, &[]), 1.0);
    let rendered = recorder.handle().render();
    assert!(
        rendered.contains(&format!("{REJECTED}{{reason=\"max_age\"}} 0")),
        "{rendered}"
    );
    let refusals = logs.lines("Usage reporting returned non-success");
    assert_eq!(refusals.len(), 1, "{}", logs.contents());
    assert_eq!(refusals[0]["request_id"], "request-refused");
    assert_eq!(refusals[0]["status"], "422 Unprocessable Entity");

    // A person fixes what was wrong and puts it back, with the statements of
    // docs/gateway-mode.md. The running process finds it by itself.
    billing.refuse("refused", None);
    put_back(&path, "rejected");
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
async fn rejected_keeps_the_newest_and_says_how_many_made_room() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    let billing = Billing::start(400).await;
    let delivery = billing.delivery(
        durable(|policy| policy.max_in_flight = 1),
        UsageOutboxConfig {
            max_rejected: 2,
            ..outbox(&path)
        },
    );
    for id in ["one", "two", "three"] {
        billing.report(&delivery, id, None);
        // Apart enough to be told apart by the millisecond they completed in.
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    delivered(&delivery).await;
    assert_eq!(billing.received(), 3);
    assert_eq!(pending_rows(&path), 0);
    // The table holds two; the oldest made room and was counted.
    assert_eq!(
        read::<String>(
            &path,
            "SELECT group_concat(json_extract(body, '$.id')) FROM (SELECT * FROM rejected ORDER BY id)"
        ),
        "two,three"
    );
    eventually("the eviction is counted", || {
        value(&recorder, REJECTED_EVICTED, &[]) == 1.0
    })
    .await;
    assert_eq!(value(&recorder, REJECTED, &[]), 2.0);
    assert_eq!(value(&recorder, DROPPED, &["reason=\"rejected\""]), 3.0);
}

#[tokio::test]
async fn an_outbox_at_its_bound_drops_the_reports_that_waited_longest_and_counts_them() {
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
    // Each has failed once: from here on at most two of them are leased at
    // any moment, to be tried again.
    eventually("the first five are written and were tried", || {
        read::<i64>(&path, "SELECT COUNT(*) FROM pending WHERE attempts > 0") == 5
    })
    .await;
    for id in ["report-5", "report-6", "report-7"] {
        billing.report(&delivery, id, None);
    }
    eventually("the bound is applied", || {
        value(&recorder, REPORTS, &["outcome=\"queue_full\""]) == 3.0
    })
    .await;
    assert_eq!(pending_rows(&path), 5);
    assert_eq!(value(&recorder, DROPPED, &["reason=\"queue_full\""]), 3.0);

    let lines = logs.lines(FILE_FULL_LINE);
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

/// The value below which `share` of `times` lie.
fn quantile(times: &mut [Duration], share: f64) -> Duration {
    times.sort();
    times[((times.len() - 1) as f64 * share) as usize]
}

#[tokio::test]
async fn handing_a_report_over_never_waits_for_the_file() {
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    let delivery = billing.delivery(
        // The whole burst waits for the writer; this is not about the bound
        // on that, nor about the writer being given up on.
        durable(|policy| policy.max_queued = 100_000),
        UsageOutboxConfig {
            stall_timeout: Duration::from_secs(60),
            ..outbox(&path)
        },
    );
    billing.report(&delivery, "before", None);
    delivered(&delivery).await;

    // Another connection holds the write lock: nothing can be written, and
    // the writer waits for it.
    let other = rusqlite::Connection::open(&path).unwrap();
    other.busy_timeout(Duration::from_secs(5)).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    let locked_at = Instant::now();

    // What a request pays for its report: the same with the file locked as
    // with it free. Microseconds; a call that waited for the file would take
    // the busy timeout, seconds.
    let mut took = Vec::with_capacity(5_000);
    for id in ids("locked", 5_000) {
        let started_at = Instant::now();
        billing.report(&delivery, &id, None);
        took.push(started_at.elapsed());
    }
    let handing_over = locked_at.elapsed();
    let (median, p99) = (quantile(&mut took, 0.5), quantile(&mut took, 0.99));
    println!(
        "5000 reports handed over in {handing_over:?} with the file locked: median {median:?}, \
         p99 {p99:?}, slowest {:?}",
        took[took.len() - 1]
    );
    assert!(median < Duration::from_micros(500), "median {median:?}");
    assert!(p99 < Duration::from_millis(5), "p99 {p99:?}");
    assert!(handing_over < Duration::from_secs(2), "{handing_over:?}");

    // While the lock is held the reports wait to be written. None is sent:
    // a report is written first.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(billing.written(), ["before"]);
    assert_eq!(delivery.pending(), (5_000, 0));

    other.execute_batch("COMMIT").unwrap();
    delivered(&delivery).await;
    let mut expected = ids("locked", 5_000);
    expected.push("before".to_string());
    expected.sort();
    assert_eq!(billing.written_sorted(), expected);
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
    // Delivered all the same, from memory.
    assert_eq!(billing.written_sorted(), ids("memory", 10));
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 10.0);
    assert_eq!(value(&recorder, AVAILABLE, &[]), 0.0);
    assert!(value(&recorder, ERRORS, &["op=\"open\""]) >= 1.0);
    assert_eq!(value(&recorder, BYPASSED, &[]), 10.0);
    assert!(!path.exists());
    let said = logs.lines(UNAVAILABLE_LINE);
    // Said when it began, and not again however often opening is tried.
    assert_eq!(said.len(), 1, "{}", logs.contents());
    assert_eq!(said[0]["op"], "open");
    let tries = value(&recorder, ERRORS, &["op=\"open\""]);
    eventually("opening is tried again", || {
        value(&recorder, ERRORS, &["op=\"open\""]) >= tries + 3.0
    })
    .await;
    assert_eq!(logs.lines(UNAVAILABLE_LINE).len(), 1);
    assert!(logs.lines(AVAILABLE_LINE).is_empty());

    // The directory appears. From then on reports are kept.
    std::fs::create_dir(&missing).unwrap();
    eventually("the outbox is back", || {
        value(&recorder, AVAILABLE, &[]) == 1.0
    })
    .await;
    assert_eq!(logs.lines(AVAILABLE_LINE).len(), 1);
    billing.answer(503);
    billing.report(&delivery, "kept", None);
    eventually("it is written", || pending_rows(&path) == 1).await;
    billing.answer(200);
    delivered(&delivery).await;
    assert!(billing.written().contains(&"kept".to_string()));
    assert_eq!(value(&recorder, BYPASSED, &[]), 10.0);
    assert_eq!(logs.lines(UNAVAILABLE_LINE).len(), 1);
}

#[tokio::test]
async fn a_file_that_stops_taking_writes_costs_the_durability_of_the_reports_and_nothing_else() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    let delivery = billing.delivery(durable(|policy| policy.max_in_flight = 1), outbox(&path));
    billing.report(&delivery, "kept-1", None);
    delivered(&delivery).await;
    assert_eq!(value(&recorder, AVAILABLE, &[]), 1.0);
    assert_eq!(value(&recorder, BYPASSED, &[]), 0.0);

    // The disk fails, say: every transaction fails from here on.
    store(&delivery).inject_fault(true);
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
    assert!(value(&recorder, ERRORS, &["op=\"open\""]) >= 1.0);
    // The ones whose write failed, and the ones that did not even try.
    assert_eq!(value(&recorder, BYPASSED, &[]), 10.0);
    assert!(value(&recorder, BYPASSED, &["reason=\"write_failed\""]) >= 1.0);
    assert!(value(&recorder, BYPASSED, &["reason=\"unavailable\""]) >= 1.0);
    // It flaps neither in the log nor in the gauge while it lasts.
    let errors = value(&recorder, ERRORS, &[]);
    eventually("the file is tried again", || {
        value(&recorder, ERRORS, &[]) >= errors + 3.0
    })
    .await;
    assert_eq!(value(&recorder, AVAILABLE, &[]), 0.0);
    assert_eq!(logs.lines(UNAVAILABLE_LINE).len(), 1, "{}", logs.contents());
    assert!(logs.lines(AVAILABLE_LINE).is_empty());

    // While it lasts the billing API fails too. Reports wait in memory.
    billing.answer(503);
    for id in ["waiting-0", "waiting-1", "waiting-2"] {
        billing.report(&delivery, id, None);
    }
    eventually("each was tried", || {
        value(&recorder, ATTEMPTS, &["outcome=\"http_5xx\""]) >= 6.0
    })
    .await;
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(value(&recorder, IN_MEMORY, &[]), 3.0);

    // The disk is back: what memory holds is written to the file after all,
    // with what happened to it so far, and new reports are kept again.
    store(&delivery).inject_fault(false);
    eventually("the outbox is back", || {
        value(&recorder, AVAILABLE, &[]) == 1.0
    })
    .await;
    eventually("what waited is in the file", || pending_rows(&path) == 3).await;
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE attempts > 0 AND last_outcome = 'http_5xx'"
        ),
        3
    );
    eventually("memory holds nothing", || {
        value(&recorder, IN_MEMORY, &[]) == 0.0
    })
    .await;
    billing.report(&delivery, "kept-2", None);
    eventually("and so is a new report", || pending_rows(&path) == 4).await;
    billing.answer(200);
    delivered(&delivery).await;
    expected.extend(["waiting-0", "waiting-1", "waiting-2", "kept-2"].map(String::from));
    expected.sort();
    assert_eq!(billing.written_sorted(), expected);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 15.0);
    assert_eq!(logs.lines(AVAILABLE_LINE).len(), 1);
    assert_eq!(logs.lines(UNAVAILABLE_LINE).len(), 1);
}

#[tokio::test]
async fn reports_held_in_memory_follow_the_rules_of_the_file_and_are_written_to_it_when_it_is_back()
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    // Eight places, as a process with an outbox and nothing else set has.
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    billing.report(&delivery, "warmup", None);
    delivered(&delivery).await;

    // A hiccup of the file: the reports of that moment are held in memory.
    // The billing API answers these eight with a 500, every time.
    store(&delivery).inject_fault(true);
    for id in ids("poison", 8) {
        billing.refuse(&id, Some(500));
        billing.report(&delivery, &id, None);
    }
    eventually("each was tried a few times", || billing.received() > 8 * 4).await;
    // A report that waits for its next attempt holds no place: there are
    // moments when none of the eight is being sent, although none is done.
    eventually("no place is held", || delivery.pending() == (8, 0)).await;
    assert_eq!(value(&recorder, IN_MEMORY, &[]), 8.0);
    assert_eq!(pending_rows(&path), 0);

    // The file is back. Every report memory holds is written to it, the
    // ones that wait for their next attempt included, with their attempts.
    store(&delivery).inject_fault(false);
    eventually("the eight are in the file", || {
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE attempts >= 4 AND last_outcome = 'http_5xx'",
        ) == 8
    })
    .await;
    eventually("memory holds nothing", || {
        value(&recorder, IN_MEMORY, &[]) == 0.0
    })
    .await;

    // And they stand in nobody's way: fifty reports the billing API takes
    // go through at once.
    let started_at = Instant::now();
    for id in ids("healthy", 50) {
        billing.report(&delivery, &id, None);
    }
    eventually("the healthy ones are written", || {
        billing.written_of("healthy") == 50
    })
    .await;
    assert!(
        started_at.elapsed() < Duration::from_secs(3),
        "{:?}",
        started_at.elapsed()
    );
    eventually("only the eight are left", || pending_rows(&path) == 8).await;

    // A usage token the billing API does not accept is no reason to end a
    // report held in memory either: it is kept, and written when it can be.
    store(&delivery).inject_fault(true);
    billing.answer(401);
    billing.report(&delivery, "unauthorized", None);
    eventually("it was refused", || {
        value(&recorder, ATTEMPTS, &["outcome=\"http_4xx\""]) >= 1.0
    })
    .await;
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"http_4xx\""]), 0.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    store(&delivery).inject_fault(false);
    eventually("it is in the file", || {
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE last_outcome = 'http_401'",
        ) == 1
    })
    .await;
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 0);
    billing.answer(200);
    eventually("it is accepted with the right token", || {
        billing.written().contains(&"unauthorized".to_string())
    })
    .await;
}

#[tokio::test]
async fn a_report_waiting_out_its_backoff_in_memory_is_written_as_soon_as_the_file_is_back() {
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        // The production backoff, from far up: after one failure a report
        // waits twenty seconds or more for its next attempt.
        durable(|policy| policy.initial_backoff = Duration::from_secs(40)),
        UsageOutboxConfig {
            max_backoff: Duration::from_secs(600),
            ..outbox(&path)
        },
    );
    opened(&path).await;
    store(&delivery).inject_fault(true);
    for id in ids("backing-off", 3) {
        billing.report(&delivery, &id, None);
    }
    eventually("each failed once and waits", || {
        billing.received() == 3 && delivery.pending() == (3, 0)
    })
    .await;
    assert_eq!(pending_rows(&path), 0);

    // The file is back. The three are written to it at once, not when their
    // next attempt comes: until then a kill would have lost them.
    store(&delivery).inject_fault(false);
    let started_at = Instant::now();
    eventually("they are in the file", || {
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE attempts = 1 AND last_outcome = 'http_5xx'",
        ) == 3
    })
    .await;
    assert!(
        started_at.elapsed() < Duration::from_secs(8),
        "{:?}",
        started_at.elapsed()
    );
    assert_eq!(billing.received(), 3, "none was tried again to get there");
    eventually("memory holds nothing", || kept(&delivery).memory.is_empty()).await;
    // And they are due when they were due, not at once.
    let in_ten_seconds = Clock::default().now_ms() + 10_000;
    assert_eq!(
        read::<i64>(
            &path,
            &format!("SELECT COUNT(*) FROM pending WHERE next_attempt_at_ms > {in_ten_seconds}")
        ),
        3
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(billing.received(), 3);
}

#[tokio::test]
async fn at_shutdown_what_memory_holds_is_written_to_a_file_that_works() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|_| {}),
        // The file is left alone for a good while after it failed.
        UsageOutboxConfig {
            reopen_interval: Duration::from_secs(60),
            recheck_interval: Duration::from_secs(60),
            ..outbox(&path)
        },
    );
    opened(&path).await;
    store(&delivery).inject_fault(true);
    for id in ids("memory", 6) {
        billing.report(&delivery, &id, None);
    }
    eventually("each was tried", || billing.received() >= 12).await;
    assert_eq!(pending_rows(&path), 0);

    // The disk works again, and before anything here has found out the
    // process is told to stop. The last transaction tries the file whatever
    // happened before: the reports are in it, waiting or not, with their
    // attempts, for the next process.
    store(&delivery).inject_fault(false);
    let drained = delivery.drain_at_shutdown().await.unwrap();
    assert_eq!(drained.left_waiting, 0, "{drained:?}");
    assert_eq!(
        read::<i64>(
            &path,
            "SELECT COUNT(*) FROM pending WHERE attempts > 0 AND last_outcome = 'http_5xx' \
             AND lease_owner IS NULL"
        ),
        6
    );
    let closed = logs.lines(CLOSED_LINE);
    assert_eq!(closed.len(), 1, "{}", logs.contents());
    assert_eq!(closed[0]["in_outbox"], 6);
    assert!(logs.lines(LEFT_LINE).is_empty());

    billing.answer(200);
    let next = billing.delivery(durable(|_| {}), outbox(&path));
    delivered(&next).await;
    assert_eq!(billing.written_sorted(), ids("memory", 6));
}

#[tokio::test]
async fn a_report_being_sent_from_memory_at_shutdown_is_written_and_counted_once() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|_| {}),
        UsageOutboxConfig {
            reopen_interval: Duration::from_secs(60),
            recheck_interval: Duration::from_secs(60),
            ..outbox(&path)
        },
    );
    opened(&path).await;
    store(&delivery).inject_fault(true);
    // An answer that takes longer than the shutdown does: the four reports
    // are held in memory and are all being sent when it begins.
    billing.take(Duration::from_millis(600));
    for id in ids("sending", 4) {
        billing.report(&delivery, &id, None);
    }
    eventually("the four are being sent", || {
        billing.intake.answering.load(Ordering::SeqCst) == 4
    })
    .await;

    // The disk works again and the process is told to stop, without a drain.
    // The four are written like anything else memory holds, and none of them
    // is said or counted to be lost for being in a place at that moment.
    store(&delivery).inject_fault(false);
    let drained = delivery.drain_at_shutdown().await.unwrap();
    assert_eq!(
        (drained.left_waiting, drained.left_in_flight),
        (0, 4),
        "{drained:?}"
    );
    assert_eq!(pending_rows(&path), 4);
    let closed = logs.lines(CLOSED_LINE);
    assert_eq!(closed.len(), 1, "{}", logs.contents());
    assert_eq!(closed[0]["left_unwritten"], 0);
    assert!(logs.lines(LEFT_LINE).is_empty(), "{}", logs.contents());
    assert!(logs.lines(UNDELIVERED_LINE).is_empty());
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);

    // Their attempts end, as failures. Each report is in the file and
    // nowhere else: what an attempt finds out after its report was written
    // is not kept beside it.
    eventually("the attempts have ended", || delivery.pending().1 == 0).await;
    assert_eq!(kept(&delivery).memory.len(), 0);
    assert_eq!(pending_rows(&path), 4);

    billing.answer(200);
    billing.take(Duration::ZERO);
    let next = billing.delivery(durable(|_| {}), outbox(&path));
    delivered(&next).await;
    assert_eq!(billing.written_sorted(), ids("sending", 4));
}

// ---------------------------------------------------------------------------
// A file that is full, a volume that is full
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_file_at_its_size_keeps_sending_what_it_holds_and_says_once_that_it_is_full() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let max_bytes = 512 << 10;
    let delivery = billing.delivery(
        durable(|_| {}),
        UsageOutboxConfig {
            max_bytes,
            ..outbox(&path)
        },
    );

    // The billing API is down and requests keep completing: the file fills
    // up to the size it was given.
    for id in ids("report", 3_000) {
        billing.report(&delivery, &id, None);
    }
    eventually("the file is full", || value(&recorder, FULL, &[]) == 1.0).await;
    eventually("every report is somewhere", || {
        value(&recorder, UNWRITTEN, &[]) == 0.0
    })
    .await;
    let in_file = pending_rows(&path);
    let in_memory = value(&recorder, IN_MEMORY, &[]) as i64;
    assert!(in_file > 300, "{in_file}");
    assert_eq!(in_file + in_memory, 3_000);
    assert!(std::fs::metadata(&path).unwrap().len() <= max_bytes);
    assert!(value(&recorder, BYTES, &[]) <= max_bytes as f64);
    // Full is not unavailable: the file works, and what it holds is sent.
    assert_eq!(value(&recorder, AVAILABLE, &[]), 1.0);
    // Each report the file did not take is counted once, however often it
    // is offered again. (One is, now and then, and the last pages fill up.)
    let bypassed = value(&recorder, BYPASSED, &["reason=\"full\""]);
    assert!(
        (in_memory as f64..=2_900.0).contains(&bypassed),
        "{bypassed} for {in_memory}"
    );
    assert_eq!(value(&recorder, BYPASSED, &[]), bypassed);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    // Attempts go on. (The billing API is down, so one at a time.)
    let attempts = billing.received();
    eventually("attempts go on", || billing.received() >= attempts + 5).await;
    // Said once, when it began; it does not flap while it lasts.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(logs.lines(FULL_LINE).len(), 1, "{}", logs.contents());
    assert_eq!(logs.lines(FULL_LINE)[0]["max_bytes"], max_bytes);
    assert!(logs.lines(ROOM_LINE).is_empty());
    assert!(logs.lines(UNAVAILABLE_LINE).is_empty());
    assert_eq!(value(&recorder, FULL, &[]), 1.0);

    // The billing API is back. The rows of the file are sent and removed
    // while it is still full...
    let in_file = pending_rows(&path);
    billing.take(Duration::from_millis(5));
    billing.answer(200);
    eventually("rows of the full file are accepted", || {
        value(&recorder, FULL, &[]) == 1.0 && pending_rows(&path) < in_file - 8
    })
    .await;
    // ... and in the end every report is accepted, from the file and from
    // memory, which makes room; none is lost and none is sent twice.
    billing.take(Duration::ZERO);
    delivered(&delivery).await;
    assert_eq!(billing.written_sorted(), ids("report", 3_000));
    assert_eq!(
        value(&recorder, REPORTS, &["outcome=\"accepted\""]),
        3_000.0
    );
    eventually("it has room again", || value(&recorder, FULL, &[]) == 0.0).await;
    assert_eq!(logs.lines(ROOM_LINE).len(), 1);
    assert_eq!(logs.lines(FULL_LINE).len(), 1);
    // And it is used again.
    billing.answer(503);
    billing.report(&delivery, "afterwards", None);
    eventually("a new report is written", || pending_rows(&path) == 1).await;
}

#[tokio::test]
async fn a_volume_that_is_full_stops_new_reports_being_kept_and_nothing_else() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    for id in ids("kept", 30) {
        billing.report(&delivery, &id, None);
    }
    eventually("they are in the file", || pending_rows(&path) == 30).await;

    // Something else fills the volume: the file cannot grow, although it is
    // far below its own size.
    store(&delivery).inject_volume_full(true);
    for id in ids("memory", 20) {
        billing.report(&delivery, &id, None);
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    eventually("the file counts as full", || {
        value(&recorder, FULL, &[]) == 1.0
    })
    .await;
    eventually("the reports are held in memory", || {
        value(&recorder, IN_MEMORY, &[]) == 20.0
    })
    .await;
    assert_eq!(value(&recorder, AVAILABLE, &[]), 1.0);
    assert_eq!(pending_rows(&path), 30);
    assert_eq!(value(&recorder, BYPASSED, &[]), 20.0);
    assert!(value(&recorder, BYPASSED, &["reason=\"full\""]) >= 1.0);
    // It is offered a report now and then, to find out; that says nothing
    // new in the log.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(logs.lines(FULL_LINE).len(), 1, "{}", logs.contents());
    assert!(logs.lines(ROOM_LINE).is_empty());

    // The rows it holds are sent and removed all the same.
    billing.answer(200);
    eventually("everything is accepted", || billing.written().len() == 50).await;
    eventually("the rows are gone", || pending_rows(&path) == 0).await;
    let mut expected = ids("kept", 30);
    expected.extend(ids("memory", 20));
    expected.sort();
    assert_eq!(billing.written_sorted(), expected);

    // Space returns. The next report is kept again, and it says so once.
    billing.answer(503);
    for id in ids("still-full", 5) {
        billing.report(&delivery, &id, None);
    }
    eventually("they are held in memory", || {
        value(&recorder, IN_MEMORY, &[]) == 5.0
    })
    .await;
    store(&delivery).inject_volume_full(false);
    eventually("they are written", || pending_rows(&path) == 5).await;
    eventually("it has room again", || value(&recorder, FULL, &[]) == 0.0).await;
    assert_eq!(logs.lines(ROOM_LINE).len(), 1);
    assert_eq!(logs.lines(FULL_LINE).len(), 1);
    billing.answer(200);
    delivered(&delivery).await;
    assert_eq!(billing.written().len(), 55);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
}

// ---------------------------------------------------------------------------
// A writer that does not come back, a file that is not the file any more
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_writer_that_hangs_is_seen_and_its_reports_are_sent_from_memory() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    // A writer in a call that does not return (a disk that hangs, not one
    // that fails): stood in for by a write lock it waits for without end.
    let delivery = billing.delivery(
        durable(|_| {}),
        UsageOutboxConfig {
            busy_timeout: Duration::from_secs(3_600),
            stall_timeout: Duration::from_millis(600),
            close_timeout: Duration::from_millis(300),
            ..outbox(&path)
        },
    );
    billing.report(&delivery, "before", None);
    delivered(&delivery).await;

    let other = rusqlite::Connection::open(&path).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    for id in ids("parked", 200) {
        billing.report(&delivery, &id, None);
    }
    // Until the writer is given up on, a series says how many reports were
    // handed over and are not written, and none is lost sight of.
    eventually("the series says so", || {
        value(&recorder, UNWRITTEN, &[]) == 200.0
    })
    .await;
    assert_eq!(value(&recorder, AVAILABLE, &[]), 1.0);
    assert_eq!(delivery.pending(), (200, 0));
    assert_eq!(billing.written(), ["before"]);

    // It does not come back in the time it is given: the store counts as
    // unavailable, and what it was handed is sent from memory.
    eventually("the reports are accepted", || {
        billing.written().len() == 201
    })
    .await;
    assert_eq!(value(&recorder, AVAILABLE, &[]), 0.0);
    assert_eq!(value(&recorder, ERRORS, &["op=\"stalled\""]), 1.0);
    assert_eq!(value(&recorder, BYPASSED, &["reason=\"stalled\""]), 200.0);
    assert_eq!(value(&recorder, UNWRITTEN, &[]), 0.0);
    let said = logs.lines(UNAVAILABLE_LINE);
    assert_eq!(said.len(), 1, "{}", logs.contents());
    assert_eq!(said[0]["op"], "stalled");
    assert_eq!(said[0]["unstored"], 200);
    // What comes in meanwhile goes the same way, at once.
    for id in ids("meanwhile", 20) {
        billing.report(&delivery, &id, None);
    }
    eventually("they are accepted too", || billing.written().len() == 221).await;
    assert_eq!(
        value(&recorder, BYPASSED, &["reason=\"unavailable\""]),
        20.0
    );
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);

    // Shutdown with the writer still hanging, and reports the billing API
    // did not take: it does not wait for the writer without end, and says
    // of each report it could not keep that it is lost, and how many.
    billing.answer(503);
    for id in ids("lost", 3) {
        billing.report(&delivery, &id, None);
    }
    eventually("they were tried", || billing.received() >= 221 + 3).await;
    let started_at = Instant::now();
    let drained = delivery.drain_at_shutdown().await.unwrap();
    assert!(
        started_at.elapsed() < Duration::from_secs(5),
        "{:?}",
        started_at.elapsed()
    );
    assert_eq!(drained.left_waiting, 3, "{drained:?}");
    assert_eq!(logs.lines(NOT_CLOSED_LINE).len(), 1, "{}", logs.contents());
    assert_eq!(logs.lines(NOT_CLOSED_LINE)[0]["left_unwritten"], 3);
    let left = logs.lines(LEFT_LINE);
    assert_eq!(left.len(), 3);
    assert!(left.iter().all(|line| line["request_id"]
        .as_str()
        .unwrap()
        .starts_with("request-lost")));
    assert_eq!(logs.lines(UNDELIVERED_LINE).len(), 1);
    assert_eq!(logs.lines(UNDELIVERED_LINE)[0]["left_waiting"], 3);
    assert_eq!(value(&recorder, DROPPED, &["reason=\"shutdown\""]), 3.0);

    // The lock goes away after all. Nothing of what was sent from memory is
    // in the file: it is not sent a second time by whoever opens it next.
    other.execute_batch("ROLLBACK").unwrap();
    drop(other);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(billing.written().len(), 221);
    let written: BTreeSet<String> = billing.written().into_iter().collect();
    assert_eq!(written.len(), 221, "none was sent twice");
}

#[tokio::test]
async fn a_writer_that_comes_back_after_all_is_used_again() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|_| {}),
        UsageOutboxConfig {
            busy_timeout: Duration::from_secs(3_600),
            stall_timeout: Duration::from_millis(300),
            ..outbox(&path)
        },
    );
    opened(&path).await;
    let other = rusqlite::Connection::open(&path).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    for id in ids("parked", 10) {
        billing.report(&delivery, &id, None);
    }
    eventually("the writer is given up on", || {
        value(&recorder, AVAILABLE, &[]) == 0.0
    })
    .await;
    eventually("the reports are tried from memory", || {
        billing.received() >= 10
    })
    .await;

    // The lock is released. The writer finishes what it was at, without the
    // reports that were taken from it, and once it has written something
    // the file is used again: what memory holds goes into it.
    other.execute_batch("ROLLBACK").unwrap();
    eventually("the store is back", || {
        value(&recorder, AVAILABLE, &[]) == 1.0
    })
    .await;
    eventually("the reports are in the file", || pending_rows(&path) == 10).await;
    assert_eq!(logs.lines(UNAVAILABLE_LINE).len(), 1, "{}", logs.contents());
    assert_eq!(logs.lines(AVAILABLE_LINE).len(), 1);
    billing.answer(200);
    delivered(&delivery).await;
    assert_eq!(billing.written_sorted(), ids("parked", 10));
}

#[tokio::test]
async fn reports_a_slow_writer_wrote_after_it_was_given_up_on_are_not_kept_twice() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    // The billing API takes nothing for now, so what is where can be seen.
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|policy| policy.initial_backoff = Duration::from_secs(2)),
        UsageOutboxConfig {
            stall_timeout: Duration::from_millis(200),
            max_pause: Duration::from_secs(30),
            breaker_after: 1,
            ..outbox(&path)
        },
    );
    opened(&path).await;
    // A disk that takes a second over a flush: longer than the writer is
    // given, and it does finish.
    store(&delivery).inject_commit_delay(Duration::from_millis(1_000));
    for id in ids("slow", 30) {
        billing.report(&delivery, &id, None);
    }
    eventually("the writer is given up on", || {
        value(&recorder, ERRORS, &["op=\"stalled\""]) == 1.0
    })
    .await;
    assert_eq!(value(&recorder, BYPASSED, &["reason=\"stalled\""]), 30.0);
    store(&delivery).inject_commit_delay(Duration::ZERO);

    // It comes back, with the thirty written. Each of them is kept once: in
    // the file. What memory held of them is let go, except the one that was
    // being sent, which is in both until its attempt has ended.
    eventually("the store is back", || {
        value(&recorder, AVAILABLE, &[]) == 1.0
    })
    .await;
    eventually("each report is kept once", || {
        pending_rows(&path) == 30
            && value(&recorder, IN_MEMORY, &[]) == 0.0
            && value(&recorder, UNWRITTEN, &[]) == 0.0
    })
    .await;
    let waiting = delivery.pending();
    assert_eq!(waiting.0 + waiting.1, 30, "{waiting:?}");

    billing.answer(200);
    delivered(&delivery).await;
    let written: BTreeSet<String> = billing.written().into_iter().collect();
    assert_eq!(written.len(), 30);
    // At most the one that was being sent from memory was sent from the
    // file as well; the billing API tells those apart by their id.
    assert!(billing.written().len() <= 31, "{:?}", billing.written());
    assert_eq!(pending_rows(&path), 0);
}

#[tokio::test]
async fn a_file_that_is_no_database_is_moved_aside_counted_and_said_once() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (dir, path) = file();
    std::fs::write(&path, b"this is not an SQLite file, whatever its name says").unwrap();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    for id in ids("report", 5) {
        billing.report(&delivery, &id, None);
    }
    // A new file is started, and the reports are kept in it.
    eventually("the reports are in a new file", || pending_rows(&path) == 5).await;
    eventually("it is counted", || {
        value(&recorder, REPLACED, &["why=\"corrupt\""]) == 1.0
    })
    .await;
    assert_eq!(value(&recorder, AVAILABLE, &[]), 1.0);
    const MOVED: &str = "Usage report outbox was not a readable database any more: it was moved \
                         aside for a person to look at, and a new file started. The reports it \
                         held are not sent unless they are put back";
    let said = logs.lines(MOVED);
    assert_eq!(said.len(), 1, "{}", logs.contents());
    // What was there is kept beside it, under a name that says when.
    let kept_as = said[0]["kept_as"].as_str().unwrap();
    assert!(kept_as.contains("outbox.db.corrupt-"), "{kept_as}");
    assert_eq!(
        std::fs::read(kept_as).unwrap(),
        b"this is not an SQLite file, whatever its name says"
    );
    assert!(dir.path().join(kept_as).exists());
    billing.answer(200);
    delivered(&delivery).await;
    assert_eq!(billing.written_sorted(), ids("report", 5));
    assert_eq!(value(&recorder, REPLACED, &[]), 1.0);
}

#[tokio::test]
async fn a_file_emptied_under_the_running_process_is_started_anew_and_its_gauges_follow() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(
        durable(|_| {}),
        UsageOutboxConfig {
            // A report may wait an hour for its next attempt here, so one
            // that does is not taken for a row of another clock.
            max_backoff: Duration::from_secs(3_600),
            ..outbox(&path)
        },
    );
    opened(&path).await;
    // Twelve reports an earlier process left, which failed and are due again
    // in an hour: in the file, and in nobody's hands.
    {
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.busy_timeout(Duration::from_secs(5)).unwrap();
        let now_ms = Clock::default().now_ms();
        for id in ids("held", 12) {
            conn.execute(
                "INSERT INTO pending (body, request_id, auth_path, ingress_route, \
                 completed_at_ms, attempts, next_attempt_at_ms, last_outcome) \
                 VALUES (?1, ?2, 'cloud_api_key', 'long', ?3, 4, ?4, 'http_5xx')",
                rusqlite::params![
                    serde_json::json!({"id": id, "model": "test-model"}).to_string(),
                    format!("request-{id}"),
                    now_ms - 60_000,
                    now_ms + 3_600_000
                ],
            )
            .unwrap();
        }
    }
    eventually("the gauges say what the file holds", || {
        value(&recorder, PENDING, &[]) == 12.0 && value(&recorder, OLDEST_AGE, &[]) >= 60.0
    })
    .await;

    // Somebody cuts the file to nothing. To SQLite that is a database
    // without tables: what it held is gone, and nothing can bring it back.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(0)
        .unwrap();
    eventually("a new database is started in it", || {
        value(&recorder, REPLACED, &["why=\"lost\""]) == 1.0
    })
    .await;
    const LOST: &str = "Usage report outbox was deleted or emptied under the running process \
                        and could not be written back: a new one was started, and the reports \
                        it held are lost, except those this process was sending";
    assert_eq!(logs.lines(LOST).len(), 1, "{}", logs.contents());
    eventually("the file is in use again", || {
        value(&recorder, AVAILABLE, &[]) == 1.0
    })
    .await;

    // The new database has never heard of the reports the old one held, nor
    // of whose they were. The gauges do not stay at what the old one said:
    // they say what the file holds now, which is nothing, without a report
    // having to come in first.
    eventually("the gauges say the file holds nothing", || {
        value(&recorder, PENDING, &[]) == 0.0 && value(&recorder, OLDEST_AGE, &[]) == 0.0
    })
    .await;
    assert_eq!(pending_rows(&path), 0);
    assert_eq!(delivery.pending(), (0, 0));
    assert_eq!(billing.received(), 0, "none of them was ever in a place");
    // And reports are kept in it like in any other.
    billing.report(&delivery, "after", None);
    eventually("a new report is kept and counted", || {
        pending_rows(&path) == 1 && value(&recorder, PENDING, &[]) == 1.0
    })
    .await;
}

#[tokio::test]
async fn a_file_deleted_under_the_running_process_is_put_back_with_what_it_held() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let logs = Logs::default();
    let _capture = logs.capture();
    let (dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    for id in ids("before", 10) {
        billing.report(&delivery, &id, None);
    }
    eventually("they are in the file", || pending_rows(&path) == 10).await;

    for entry in std::fs::read_dir(dir.path()).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    for id in ids("after", 10) {
        billing.report(&delivery, &id, None);
    }
    // The next transaction notices, before it writes to a file nobody can
    // open any more, and writes the database back to the path.
    eventually("the file is back with everything", || {
        pending_rows(&path) == 20
    })
    .await;
    eventually("it is counted", || {
        value(&recorder, REPLACED, &["why=\"deleted\""]) == 1.0
    })
    .await;
    assert_eq!(value(&recorder, AVAILABLE, &[]), 1.0);
    const PUT_BACK: &str = "Usage report outbox was deleted under the running process: it has \
                            been written back as it was";
    assert_eq!(logs.lines(PUT_BACK).len(), 1, "{}", logs.contents());

    billing.answer(200);
    delivered(&delivery).await;
    let mut expected = ids("before", 10);
    expected.extend(ids("after", 10));
    expected.sort();
    assert_eq!(billing.written_sorted(), expected);
    assert_eq!(value(&recorder, REPLACED, &[]), 1.0);
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
        let mut files = 0;
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
                assert!(text.contains("org-1"), "{when}");
            }
            files += 1;
        }
        // The file and its journal.
        assert_eq!(files, 2, "{when}");
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
            "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL"
        ),
        0
    );
    // A report handed over after that is not swallowed: while the process
    // lasts it is sent from memory.
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
            "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL"
        ),
        0
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn nothing_runs_in_a_loop_while_shutdown_waits_for_an_attempt() {
    // Once with the reports in the file, once with the file failing and the
    // reports held in memory: what waits is kept in different places, and in
    // neither may it keep the dispatcher going round.
    for in_memory in [false, true] {
        let (_dir, path) = file();
        let billing = Billing::start(503).await;
        let delivery = billing.delivery(
            durable(|policy| {
                policy.max_in_flight = 2;
                policy.attempt_timeout = Duration::from_secs(10);
                policy.shutdown_drain = Duration::from_millis(1_500);
            }),
            UsageOutboxConfig {
                // Not looked at again during the drain for another reason.
                reopen_interval: Duration::from_secs(60),
                close_timeout: Duration::from_millis(500),
                ..outbox(&path)
            },
        );
        opened(&path).await;
        store(&delivery).inject_fault(in_memory);
        // One attempt that stays in flight through the whole drain.
        billing.take(Duration::from_secs(8));
        billing.report(&delivery, "slow", None);
        eventually("the slow attempt is in flight", || billing.received() == 1).await;
        // And a report that fails at once and is waiting for its next
        // attempt, which has long been due, when shutdown begins.
        billing.take(Duration::ZERO);
        billing.report(&delivery, "failing", None);
        eventually("it failed twice", || billing.received() >= 3).await;
        assert_eq!(pending_rows(&path), if in_memory { 0 } else { 2 });

        // Everything here runs on this thread: the dispatcher, the drain,
        // the attempts. Waiting for the attempt costs next to nothing of it.
        let Some(cpu_before) = thread_cpu() else {
            eprintln!(
                "SKIPPED nothing_runs_in_a_loop_while_shutdown_waits_for_an_attempt: the CPU time \
                 of a thread cannot be read here"
            );
            return;
        };
        let started_at = Instant::now();
        let drained = delivery.drain_at_shutdown().await.unwrap();
        let (cpu, wall) = (thread_cpu().unwrap() - cpu_before, started_at.elapsed());
        println!(
            "a drain of {wall:?} used {cpu:?} of this thread (reports held in memory: \
             {in_memory}); {drained:?}"
        );
        assert!(wall >= Duration::from_millis(1_400), "{wall:?}");
        assert_eq!(drained.left_in_flight, 1);
        assert!(
            cpu < wall / 10,
            "the thread was busy for {cpu:?} of a {wall:?} drain (in memory: {in_memory})"
        );
    }
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
    assert_eq!(value(&recorder, FULL, &[]), 0.0);
    assert_eq!(value(&recorder, PAUSED, &[]), 0.0);
    assert_eq!(value(&recorder, REJECTED, &[]), 0.0);
    assert_eq!(value(&recorder, UNWRITTEN, &[]), 0.0);
    assert_eq!(value(&recorder, IN_MEMORY, &[]), 0.0);
    let bytes = value(&recorder, BYTES, &[]);
    assert!(
        bytes >= 4_096.0 && bytes <= std::fs::metadata(&path).unwrap().len() as f64,
        "{bytes}"
    );
    // The age of the oldest report grows while it waits.
    let age = value(&recorder, OLDEST_AGE, &[]);
    eventually("the oldest report gets older", || {
        value(&recorder, OLDEST_AGE, &[]) >= age + 0.1
    })
    .await;

    // No attempt has ended yet: these are the gauges, and nothing else.
    assert_eq!(
        families(&recorder),
        [
            WAITING, AVAILABLE, FULL, PAUSED, OLDEST_AGE, PENDING, REJECTED, BYTES, UNWRITTEN,
            IN_MEMORY, IN_FLIGHT
        ]
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
async fn the_age_of_the_oldest_report_is_that_of_the_oldest_whoever_holds_it() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    opened(&path).await;
    // A report whose request completed a minute ago, which failed since and
    // waits for its next attempt in an hour: the oldest there is, although
    // it is the last that will be sent.
    left_behind(&path, &["old".to_string()], 4);
    rusqlite::Connection::open(&path)
        .unwrap()
        .execute(
            "UPDATE pending SET next_attempt_at_ms = next_attempt_at_ms + 3600000",
            [],
        )
        .unwrap();
    billing.report(&delivery, "new", None);
    // Both are in the file, and the age is that of the old one.
    eventually("the age is that of the old one", || {
        value(&recorder, PENDING, &[]) == 2.0 && value(&recorder, OLDEST_AGE, &[]) >= 60.0
    })
    .await;
    assert!(value(&recorder, OLDEST_AGE, &[]) < 120.0);
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

    let enabled = logs.lines(ENABLED_LINE);
    assert_eq!(enabled.len(), 1, "{}", logs.contents());
    assert_eq!(enabled[0]["path"], path.display().to_string());
    assert_eq!(enabled[0]["max_pending"], 1_000_000);
    assert_eq!(enabled[0]["max_rejected"], 100_000);
    assert_eq!(enabled[0]["max_bytes"], 1u64 << 30);
    assert_eq!(enabled[0]["max_age_secs"], 7 * 24 * 3_600);
    assert_eq!(enabled[0]["max_in_flight"], 8);
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

#[cfg(unix)]
#[tokio::test]
async fn a_directory_others_can_write_to_is_said() {
    use std::os::unix::fs::PermissionsExt;
    let logs = Logs::default();
    let _capture = logs.capture();
    const OPEN: &str = "The directory of the usage report outbox can be written to by other \
                        users: whoever can put a row into the file has it sent with the usage \
                        token. Make it 0700";
    let billing = Billing::start(200).await;
    let (dir, path) = file();
    // As it should be, and as it is where others may look but not write.
    for mode in [0o700, 0o755] {
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(mode)).unwrap();
        let private = billing.delivery(durable(|_| {}), outbox(&path));
        opened(&path).await;
        assert!(logs.lines(OPEN).is_empty(), "{mode:o}: {}", logs.contents());
        private.drain_at_shutdown().await.unwrap();
    }

    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o775)).unwrap();
    let _open = billing.delivery(durable(|_| {}), outbox(&path));
    let said = logs.lines(OPEN);
    assert_eq!(said.len(), 1, "{}", logs.contents());
    assert_eq!(said[0]["mode"], "775");
    assert_eq!(said[0]["directory"], dir.path().display().to_string());
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
    // The one answer an outbox adds to those, and nothing else of the 4xx:
    // a 403 is about the report, and final.
    let answer = |status: u16| Ok(reqwest::StatusCode::from_u16(status).unwrap());
    assert_eq!(caller_refused(&answer(401)), Some("http_401"));
    for status in [200, 400, 402, 403, 404, 409, 422, 429, 500, 503] {
        assert_eq!(caller_refused(&answer(status)), None, "{status}");
    }
    // Every reason a row is in `rejected` for has its series.
    for reason in ["rejected", "max_age", "deadline"] {
        assert!(REJECTED_REASONS.contains(&reason));
    }
}

// ---------------------------------------------------------------------------
// Putting rejected reports back, as documented
// ---------------------------------------------------------------------------

/// The statements docs/gateway-mode.md gives for putting rejected reports
/// back, as they are printed there.
fn documented_replay() -> &'static str {
    let docs = include_str!("../docs/gateway-mode.md");
    let block = docs
        .split("```")
        .find(|block| block.starts_with("sql\n") && block.contains(".bail on"))
        .expect("docs/gateway-mode.md has the statements, in a fenced block");
    &block["sql\n".len()..]
}

/// The same statements for a connection of the test's own, which has no use
/// for the lines that speak to the shell. `reason`: which rows are put back.
fn put_back(path: &Path, reason: &str) {
    let statements: String = documented_replay()
        .lines()
        .filter(|line| !line.starts_with('.'))
        .map(|line| format!("{line}\n"))
        .collect();
    assert_eq!(
        statements.matches("reason = 'rejected'").count(),
        2,
        "{statements}"
    );
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.busy_timeout(Duration::from_secs(5)).unwrap();
    conn.execute_batch(&statements.replace("'rejected'", &format!("'{reason}'")))
        .unwrap();
}

/// Run `script` on the file at `path` with the `sqlite3` shell, the way the
/// documentation says to.
fn sqlite3(path: &Path, script: &str) -> std::process::Output {
    use std::io::Write;
    let mut shell = std::process::Command::new("sqlite3")
        .arg(path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    shell
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    shell.wait_with_output().unwrap()
}

#[tokio::test]
async fn the_documented_way_to_put_reports_back_works_as_printed() {
    let script = documented_replay();
    // Whatever else it says: it stops at the first error, so that a row is
    // never deleted from `rejected` without having been put back, and it
    // waits for a file the gateway is writing to.
    assert!(
        script.starts_with(".bail on\n.timeout 5000\nBEGIN IMMEDIATE;\n"),
        "{script}"
    );
    assert!(script.trim_end().ends_with("COMMIT;"), "{script}");
    if std::process::Command::new("sqlite3")
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!(
            "SKIPPED the_documented_way_to_put_reports_back_works_as_printed: there is no \
             `sqlite3` command here, so the statements of docs/gateway-mode.md were not run"
        );
        return;
    }

    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    for id in ["refused-0", "refused-1"] {
        billing.refuse(id, Some(422));
    }
    let delivery = billing.delivery(durable(|_| {}), outbox(&path));
    for id in ["refused-0", "accepted", "refused-1"] {
        billing.report(&delivery, id, None);
    }
    delivered(&delivery).await;
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 2);
    assert_eq!(billing.written(), ["accepted"]);

    // What was wrong is put right, and the statements are run as printed,
    // while the gateway runs. It finds the rows by itself and sends them.
    for id in ["refused-0", "refused-1"] {
        billing.refuse(id, None);
    }
    let output = sqlite3(&path, script);
    assert!(output.status.success(), "{output:?}");
    eventually("the reports put back are accepted", || {
        billing.written().len() == 3
    })
    .await;
    delivered(&delivery).await;
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 0);
    assert_eq!(pending_rows(&path), 0);
    delivery.drain_at_shutdown().await.unwrap();

    // With the file busy for a moment (the gateway writes to it all the
    // time) the shell waits, and then does it all.
    let reject = |conn: &rusqlite::Connection| {
        conn.execute(
            "INSERT INTO rejected (rejected_at_ms, reason, status, outcome, attempts, body, \
             request_id, auth_path, ingress_route, completed_at_ms) \
             VALUES (0, 'rejected', 422, 'http_4xx', 1, '{\"id\":\"by-hand\"}', 'request-1', \
             'cloud_api_key', 'long', 1000)",
            [],
        )
        .unwrap();
    };
    let busy = rusqlite::Connection::open(&path).unwrap();
    reject(&busy);
    busy.execute_batch("BEGIN IMMEDIATE").unwrap();
    let waiting = {
        let path = path.clone();
        std::thread::spawn(move || sqlite3(&path, script))
    };
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert!(!waiting.is_finished());
    busy.execute_batch("COMMIT").unwrap();
    let output = waiting.join().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 0);
    // A report put back is as old as the moment it was put back: it is not
    // sent to `rejected` again for the age it had.
    let now_ms = Clock::default().now_ms();
    let completed_at_ms: i64 = read(&path, "SELECT completed_at_ms FROM pending");
    assert!(
        (now_ms - completed_at_ms).abs() < 60_000,
        "{completed_at_ms} at {now_ms}"
    );
    assert_eq!(
        read::<String>(
            &path,
            "SELECT json_extract(body, '$.id') || ' ' || request_id || ' ' || auth_path || ' ' \
             || ingress_route || ' ' || attempts || ' ' || (lease_owner IS NULL) FROM pending"
        ),
        "by-hand request-1 cloud_api_key long 0 1"
    );

    // With the file busy for longer than the shell waits, it gives up, says
    // so with its exit status, and has done nothing by halves: the row is
    // still in `rejected`, and there only.
    reject(&busy);
    busy.execute_batch("BEGIN IMMEDIATE").unwrap();
    let waiting = {
        let path = path.clone();
        std::thread::spawn(move || sqlite3(&path, script))
    };
    while !waiting.is_finished() {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let output = waiting.join().unwrap();
    busy.execute_batch("COMMIT").unwrap();
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("locked"),
        "{output:?}"
    );
    assert_eq!(read::<i64>(&path, "SELECT COUNT(*) FROM rejected"), 1);
    assert_eq!(pending_rows(&path), 1);
}

// ---------------------------------------------------------------------------
// How fast, and how it behaves over minutes: measurements, run by hand
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
    opened(&path).await;

    let ids: Vec<String> = (0..REPORTS).map(|n| format!("chatcmpl-{n}")).collect();
    let mut took = Vec::with_capacity(REPORTS);
    let started_at = Instant::now();
    for id in &ids {
        let handed_over_at = Instant::now();
        billing.report(&delivery, id, None);
        took.push(handed_over_at.elapsed());
    }
    let handing_over = started_at.elapsed();
    eventually("they are all in the file or out", || {
        delivery
            .outbox
            .as_ref()
            .unwrap()
            .in_transit
            .load(Ordering::SeqCst)
            == 0
    })
    .await;
    let written_in = started_at.elapsed();
    delivered(&delivery).await;
    let through = started_at.elapsed();

    println!(
        "handing over {REPORTS} reports: {handing_over:?} in all; per report median {:?}, p99 \
         {:?}, p99.9 {:?}, slowest {:?}",
        quantile(&mut took, 0.5),
        quantile(&mut took, 0.99),
        quantile(&mut took, 0.999),
        quantile(&mut took, 1.0)
    );
    println!(
        "written to the file: {written_in:?} in all, {:.0} reports a second",
        REPORTS as f64 / written_in.as_secs_f64()
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

/// What handing a report over costs while another connection holds the
/// file's write lock, with the production timings: 30,000 reports.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement, run by hand"]
async fn handing_over_while_the_file_is_locked() {
    const REPORTS: usize = 30_000;
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    let policy = durable(|policy| policy.max_queued = 100_000);
    let delivery = billing.delivery(policy, UsageOutboxConfig::at(&path));
    billing.report(&delivery, "before", None);
    delivered(&delivery).await;

    let other = rusqlite::Connection::open(&path).unwrap();
    other.execute_batch("BEGIN IMMEDIATE").unwrap();
    let locked_at = Instant::now();
    let mut took = Vec::with_capacity(REPORTS);
    for n in 0..REPORTS {
        let at = Instant::now();
        billing.report(&delivery, &format!("locked-{n}"), None);
        took.push(at.elapsed());
    }
    println!(
        "{REPORTS} reports handed over in {:?} with the write lock held elsewhere: median {:?}, \
         p99 {:?}, p99.9 {:?}, slowest {:?}",
        locked_at.elapsed(),
        quantile(&mut took, 0.5),
        quantile(&mut took, 0.99),
        quantile(&mut took, 0.999),
        quantile(&mut took, 1.0)
    );
    other.execute_batch("COMMIT").unwrap();
    delivered(&delivery).await;
    let written: BTreeSet<String> = billing.written().into_iter().collect();
    assert_eq!(written.len(), REPORTS + 1);
    assert_eq!(billing.written().len(), REPORTS + 1, "none was sent twice");
}

/// What a variable of the environment says, or `default`.
fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Reports that always get a 500, then reports the billing API takes, 17 a
/// second, with the production timings: what gets through, and how late.
///
/// `POISON` (400) reports are handed over first and left alone for `WARMUP`
/// (70) seconds, then the others come for `SECS` (120).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement of minutes, run by hand"]
async fn reports_that_always_fail_beside_reports_that_do_not() {
    let (poison, warmup, secs) = (
        setting("POISON", 400),
        setting("WARMUP", 70),
        setting("SECS", 120),
    );
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(20));
    let delivery = billing.delivery(UsageReportPolicy::durable(), UsageOutboxConfig::at(&path));
    for n in 0..poison {
        let id = format!("poison-{n}");
        billing.refuse(&id, Some(500));
        billing.report(&delivery, &id, None);
    }
    tokio::time::sleep(Duration::from_secs(warmup)).await;
    println!(
        "after {warmup} s with {poison} failing reports alone: {} attempts, {} of them tried, \
         billing API counted as down: {}",
        billing.received(),
        read::<i64>(&path, "SELECT COUNT(*) FROM pending WHERE attempts > 0"),
        kept(&delivery).breaker.engaged
    );

    let attempts_before = billing.received();
    let started_at = Instant::now();
    let mut handed_over = BTreeMap::new();
    let mut tick = tokio::time::interval(Duration::from_millis(1000 / 17));
    let mut last = (Instant::now(), 0usize, billing.received());
    while started_at.elapsed() < Duration::from_secs(secs) {
        tick.tick().await;
        let id = format!("healthy-{}", handed_over.len());
        handed_over.insert(id.clone(), Instant::now());
        billing.report(&delivery, &id, None);
        if last.0.elapsed() >= Duration::from_secs(30) {
            let healthy = billing.written_of("healthy");
            println!(
                "t={:>4}s: healthy handed over {}, accepted {healthy} (+{} in 30 s), attempts in \
                 30 s {}, rows {}, in flight {}",
                started_at.elapsed().as_secs(),
                handed_over.len(),
                healthy - last.1,
                billing.received() - last.2,
                pending_rows(&path),
                delivery.pending().1
            );
            last = (Instant::now(), healthy, billing.received());
        }
    }
    eventually("the healthy ones are written", || {
        billing.written_of("healthy") == handed_over.len()
    })
    .await;
    let mut lags = lags(&billing, &handed_over);
    println!(
        "{poison} failing reports, 17 healthy a second for {secs} s: healthy handed over {}, \
         accepted {}; from handed over to accepted: median {:?}, p99 {:?}, slowest {:?}; attempts \
         on the billing API {} ({:.1} a second); rows left {}",
        handed_over.len(),
        billing.written_of("healthy"),
        quantile(&mut lags, 0.5),
        quantile(&mut lags, 0.99),
        quantile(&mut lags, 1.0),
        billing.received() - attempts_before,
        (billing.received() - attempts_before) as f64 / secs as f64,
        pending_rows(&path)
    );
}

/// For `SECS` (300) seconds the billing API answers a 500 to `SHARE` (30)
/// percent of the reports that come in, 17 a second, then takes everything:
/// how late the others are meanwhile, and how long the rest takes afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement of minutes, run by hand"]
async fn a_share_of_the_reports_failing_for_minutes() {
    let (share, secs) = (setting("SHARE", 30), setting("SECS", 300));
    let (_dir, path) = file();
    let billing = Billing::start(200).await;
    billing.take(Duration::from_millis(20));
    let delivery = billing.delivery(UsageReportPolicy::durable(), UsageOutboxConfig::at(&path));
    let started_at = Instant::now();
    let mut good = BTreeMap::new();
    let mut bad = Vec::new();
    let mut engaged = 0;
    let mut tick = tokio::time::interval(Duration::from_millis(1000 / 17));
    let mut n = 0u64;
    while started_at.elapsed() < Duration::from_secs(secs) {
        tick.tick().await;
        n += 1;
        let failing = (n * 7919) % 100 < share;
        let id = format!("{}-{n}", if failing { "bad" } else { "good" });
        if failing {
            billing.refuse(&id, Some(500));
            bad.push(id.clone());
        } else {
            good.insert(id.clone(), Instant::now());
        }
        billing.report(&delivery, &id, None);
        engaged += usize::from(kept(&delivery).breaker.engaged);
    }
    eventually("the good ones are written", || {
        billing.written_of("good") == good.len()
    })
    .await;
    let mut lags = lags(&billing, &good);
    println!(
        "{share} % failing for {secs} s: good handed over {}, accepted {}; from handed over to \
         accepted: median {:?}, p99 {:?}, slowest {:?}; bad {}; attempts {}; rows {}; moments the \
         billing API counted as down: {engaged} of {n}",
        good.len(),
        billing.written_of("good"),
        quantile(&mut lags, 0.5),
        quantile(&mut lags, 0.99),
        quantile(&mut lags, 1.0),
        bad.len(),
        billing.received(),
        pending_rows(&path),
    );
    // The incident ends.
    for id in &bad {
        billing.refuse(id, None);
    }
    let ended_at = Instant::now();
    let mut reported = 0;
    while billing.written_of("bad") < bad.len() {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if ended_at.elapsed().as_secs() / 60 > reported {
            reported = ended_at.elapsed().as_secs() / 60;
            println!(
                "{reported} min after the incident: {} of {} that failed are accepted",
                billing.written_of("bad"),
                bad.len()
            );
        }
        if ended_at.elapsed() > Duration::from_secs(900) {
            break;
        }
    }
    println!(
        "everything that failed was accepted {:?} after the incident ended ({} of {})",
        ended_at.elapsed(),
        billing.written_of("bad"),
        bad.len()
    );
}

/// A backlog of `ROWS` (102,000) reports in the file and a billing API that
/// is down for `SECS` (60), then back: how many requests it is sent while it
/// is down, and how fast the backlog goes out once it is back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement of minutes, run by hand"]
async fn a_backlog_through_an_outage() {
    let (rows, secs) = (setting("ROWS", 102_000) as usize, setting("SECS", 60));
    let (_dir, path) = file();
    let billing = Billing::start(503).await;
    billing.take(Duration::from_millis(20));
    // Written by an earlier process, never tried.
    {
        let first = billing.delivery(UsageReportPolicy::durable(), UsageOutboxConfig::at(&path));
        opened(&path).await;
        first.drain_at_shutdown().await.unwrap();
    }
    left_behind(&path, &ids("backlog", rows), 0);
    let delivery = billing.delivery(UsageReportPolicy::durable(), UsageOutboxConfig::at(&path));
    eventually("the billing API counts as down", || {
        kept(&delivery).breaker.engaged
    })
    .await;
    let engaged_after = billing.received();
    let started_at = Instant::now();
    tokio::time::sleep(Duration::from_secs(secs)).await;
    let during = billing.received() - engaged_after;
    println!(
        "{rows} rows, billing API down: {engaged_after} attempts until it counted as down, then \
         {during} in {secs} s ({:.2} a second); most in flight at once {}",
        during as f64 / started_at.elapsed().as_secs_f64(),
        billing.intake.most_answering.load(Ordering::SeqCst)
    );

    billing.answer(200);
    let back_at = Instant::now();
    let mut last = 0;
    let mut first_accepted = None;
    let mut full_rate_at = None;
    for second in 1..=30 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let accepted = billing.written().len();
        if accepted > 0 && first_accepted.is_none() {
            first_accepted = Some(second);
        }
        // Eight places and 20 ms an answer: 400 a second at the most.
        if accepted - last >= 300 && full_rate_at.is_none() {
            full_rate_at = Some(second);
        }
        println!(
            "{second:>2} s after it is back: {} accepted in that second",
            accepted - last
        );
        last = accepted;
        if full_rate_at.is_some_and(|at| second >= at + 5) {
            break;
        }
    }
    println!(
        "back for {:?}: first report accepted within {first_accepted:?} s, full rate within \
         {full_rate_at:?} s; accepted so far {}, sent twice {}",
        back_at.elapsed(),
        billing.written().len(),
        billing.written().len() - billing.written().into_iter().collect::<BTreeSet<_>>().len()
    );
}
