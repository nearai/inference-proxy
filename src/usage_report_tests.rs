//! Delivery of usage reports against a stand-in billing API on a real socket.
//!
//! The stand-in counts the requests it is handling at once and, like the real
//! intake, stops working on a request whose caller hung up, so "accepted" here
//! means what it means in production: the write happened.

use std::sync::atomic::{AtomicUsize, Ordering};

use axum::extract::State as Shared;
use axum::http::{HeaderMap, StatusCode};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusRecorder};

use super::*;
use crate::auth::{AuthPath, IngressRouteKind, RequestSource};

/// How the stand-in answers.
enum Answers {
    /// The n-th request gets the n-th `(delay, status)`; the last one repeats.
    Script(Vec<(Duration, u16)>),
    /// One write at a time, each taking this long: an intake that serializes
    /// the writes of one organization.
    OneWriteAtATime(Duration),
    /// 503 at once for this long, counted from the first request, 200 after.
    DownFor(Duration),
}

struct Received {
    body: bytes::Bytes,
    authorization: Option<String>,
    request_id: Option<String>,
    content_type: Option<String>,
}

struct Intake {
    answers: Answers,
    /// Every request, as it arrived.
    received: Mutex<Vec<Received>>,
    /// The `id` of every report answered with a success: the writes made.
    written: Mutex<Vec<String>>,
    handling: AtomicUsize,
    most_handled_at_once: AtomicUsize,
    writer: tokio::sync::Mutex<()>,
    first_request_at: std::sync::OnceLock<Instant>,
}

/// One request being handled; it stops counting when the handler ends or is
/// dropped because its caller hung up.
struct Handling<'a>(&'a Intake);

impl<'a> Handling<'a> {
    fn start(intake: &'a Intake) -> Self {
        let now = intake.handling.fetch_add(1, Ordering::SeqCst) + 1;
        intake.most_handled_at_once.fetch_max(now, Ordering::SeqCst);
        Self(intake)
    }
}

impl Drop for Handling<'_> {
    fn drop(&mut self) {
        self.0.handling.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn usage(
    Shared(intake): Shared<Arc<Intake>>,
    headers: HeaderMap,
    body: bytes::Bytes,
) -> StatusCode {
    let _handling = Handling::start(&intake);
    let header = |name: &str| {
        headers
            .get(name)
            .map(|value| value.to_str().unwrap().to_string())
    };
    let index = {
        let mut received = intake.received.lock().unwrap();
        received.push(Received {
            body: body.clone(),
            authorization: header("authorization"),
            request_id: header("x-request-id"),
            content_type: header("content-type"),
        });
        received.len() - 1
    };
    let status = match &intake.answers {
        Answers::Script(script) => {
            let (delay, status) = script[index.min(script.len() - 1)];
            tokio::time::sleep(delay).await;
            status
        }
        Answers::OneWriteAtATime(per_write) => {
            let _writer = intake.writer.lock().await;
            tokio::time::sleep(*per_write).await;
            200
        }
        Answers::DownFor(outage) => {
            let first_request_at = intake.first_request_at.get_or_init(Instant::now);
            if first_request_at.elapsed() < *outage {
                503
            } else {
                200
            }
        }
    };
    let status = StatusCode::from_u16(status).unwrap();
    if status.is_success() {
        let report: serde_json::Value = serde_json::from_slice(&body).unwrap();
        intake
            .written
            .lock()
            .unwrap()
            .push(report["id"].as_str().unwrap().to_string());
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
    async fn start(answers: Answers) -> Self {
        let intake = Arc::new(Intake {
            answers,
            received: Mutex::default(),
            written: Mutex::default(),
            handling: AtomicUsize::new(0),
            most_handled_at_once: AtomicUsize::new(0),
            writer: tokio::sync::Mutex::new(()),
            first_request_at: std::sync::OnceLock::new(),
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

    /// Every request answers `status` at once.
    async fn answering(status: u16) -> Self {
        Self::start(Answers::Script(vec![(Duration::ZERO, status)])).await
    }

    /// No billing API at all: nothing listens where reports are sent, so
    /// every connection is refused.
    fn unreachable() -> Self {
        Self {
            // Port 1 needs privileges to bind and is not in the range the
            // system hands out, so no other test can be listening there.
            url: "http://127.0.0.1:1".to_string(),
            intake: Arc::new(Intake {
                answers: Answers::Script(Vec::new()),
                received: Mutex::default(),
                written: Mutex::default(),
                handling: AtomicUsize::new(0),
                most_handled_at_once: AtomicUsize::new(0),
                writer: tokio::sync::Mutex::new(()),
                first_request_at: std::sync::OnceLock::new(),
            }),
            client: reqwest::Client::new(),
        }
    }

    fn received(&self) -> usize {
        self.intake.received.lock().unwrap().len()
    }

    fn written(&self) -> Vec<String> {
        self.intake.written.lock().unwrap().clone()
    }

    fn most_handled_at_once(&self) -> usize {
        self.intake.most_handled_at_once.load(Ordering::SeqCst)
    }

    /// Hand a report for completion `id` to `delivery`, the way a finished
    /// request does.
    fn report(&self, delivery: &Arc<UsageReportDelivery>, id: &str, model: ModelLabel) {
        let reporter = UsageReporter {
            http_client: self.client.clone(),
            cloud_api_url: self.url.clone(),
            model_name: model.unwrap_or("test-model").to_string(),
            cloud_api_usage_token: Some("usage-secret".to_string()),
            org_id: Some("org-1".to_string()),
            workspace_id: Some("ws-1".to_string()),
            api_key_id: Some("key-1".to_string()),
            discount_to_user: Some(0.2),
            model_label: model,
            request_id: Some(format!("request-{id}")),
            request_source: RequestSource {
                auth_path: AuthPath::CloudApiKey,
                ingress_route: IngressRouteKind::Canonical,
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
}

/// A policy that differs from the default only where a test says so.
fn policy(change: impl FnOnce(&mut UsageReportPolicy)) -> Arc<UsageReportDelivery> {
    let mut policy = UsageReportPolicy::default();
    change(&mut policy);
    UsageReportDelivery::new(policy)
}

/// Wait until nothing is pending. A test that expects reports to be left
/// behind calls `drain` itself.
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

/// The names of the usage-report series that exist, without their labels.
fn usage_report_families(recorder: &PrometheusRecorder) -> std::collections::BTreeSet<String> {
    recorder
        .handle()
        .render()
        .lines()
        .filter(|line| !line.starts_with('#') && line.contains("usage_report"))
        .map(|line| line.split(['{', ' ']).next().unwrap().to_string())
        .collect()
}

const REPORTS: &str = "inference_proxy_usage_reports_total";
const DURATION_COUNT: &str = "inference_proxy_usage_report_duration_seconds_count";
const ATTEMPTS: &str = "inference_proxy_usage_report_attempts_total";
const RETRIES: &str = "inference_proxy_usage_report_retries_total";
const DROPPED: &str = "inference_proxy_usage_reports_dropped_total";
const WAITING: &str = "inference_proxy_usage_report_queue_depth";
const IN_FLIGHT: &str = "inference_proxy_usage_reports_in_flight";
const TIME_TO_ACCEPTED_COUNT: &str = "inference_proxy_usage_report_time_to_accepted_seconds_count";

// ---------------------------------------------------------------------------
// Nothing set: delivery as it has always been
// ---------------------------------------------------------------------------

#[test]
fn the_default_policy_is_one_attempt_of_five_seconds_and_nothing_else() {
    assert_eq!(
        UsageReportPolicy::default(),
        UsageReportPolicy {
            attempt_timeout: Duration::from_secs(5),
            max_attempts: 1,
            initial_backoff: Duration::from_millis(500),
            deadline: None,
            max_in_flight: 0,
            max_queued: 10_000,
            shutdown_drain: Duration::ZERO,
        }
    );
}

#[tokio::test]
async fn by_default_a_report_is_sent_once_whatever_the_answer_and_no_series_is_added() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = Arc::new(UsageReportDelivery::default());

    // What the new policy would retry (503, 429, an unreachable intake) and
    // what nothing retries (400) all end after one attempt.
    let mut intakes = Vec::new();
    for status in [200, 503, 429, 400] {
        let billing = Billing::answering(status).await;
        billing.report(&delivery, &format!("chatcmpl-{status}"), None);
        intakes.push(billing);
    }
    Billing::unreachable().report(&delivery, "chatcmpl-unreachable", None);
    delivered(&delivery).await;
    // Nothing follows later either.
    tokio::time::sleep(Duration::from_millis(300)).await;
    for billing in &intakes {
        assert_eq!(billing.received(), 1);
    }

    // The request is the one that has always been sent: these headers, and
    // the report with its identity as the body.
    {
        let received = intakes[0].intake.received.lock().unwrap();
        assert_eq!(
            received[0].authorization.as_deref(),
            Some("Bearer usage-secret")
        );
        assert_eq!(
            received[0].content_type.as_deref(),
            Some("application/json")
        );
        assert_eq!(
            received[0].request_id.as_deref(),
            Some("request-chatcmpl-200")
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&received[0].body).unwrap(),
            expected_report("chatcmpl-200")
        );
    }

    let source = "auth_path=\"cloud_api_key\",ingress_route=\"canonical\"}";
    for (outcome, count) in [
        ("accepted", 1.0),
        ("http_4xx", 2.0),
        ("http_5xx", 1.0),
        ("connect_error", 1.0),
    ] {
        let series = format!("{{outcome=\"{outcome}\",{source}");
        assert_eq!(value(&recorder, REPORTS, &[&series]), count, "{outcome}");
        assert_eq!(
            value(&recorder, DURATION_COUNT, &[&series]),
            count,
            "{outcome}"
        );
    }
    // The two families a report has always had, and no other.
    assert_eq!(
        usage_report_families(&recorder),
        [
            "inference_proxy_usage_report_duration_seconds",
            "inference_proxy_usage_report_duration_seconds_count",
            "inference_proxy_usage_report_duration_seconds_sum",
            "inference_proxy_usage_reports_total",
        ]
        .map(String::from)
        .into()
    );
    // Nor does shutdown wait for anything or say anything.
    assert_eq!(delivery.drain_at_shutdown().await, None);
}

/// The report for completion `id` as `/v1/internal/usage` takes it, identity
/// and discount included.
fn expected_report(id: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "chat_completion",
        "model": "test-model",
        "input_tokens": 1200,
        "output_tokens": 340,
        "cache_read_tokens": 800,
        "id": id,
        "organization_id": "org-1",
        "workspace_id": "ws-1",
        "api_key_id": "key-1",
        "discount_to_user": 0.2,
    })
}

// ---------------------------------------------------------------------------
// Patient: a longer timeout, retries, a deadline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_answer_slower_than_five_seconds_is_lost_by_default_and_accepted_with_a_longer_timeout()
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let slow = || Billing::start(Answers::Script(vec![(Duration::from_secs(7), 200)]));

    let (before, default) = (slow().await, Arc::new(UsageReportDelivery::default()));
    before.report(&default, "chatcmpl-slow", None);
    let (after, patient) = (
        slow().await,
        policy(|policy| policy.attempt_timeout = Duration::from_secs(30)),
    );
    after.report(&patient, "chatcmpl-slow", None);
    delivered(&default).await;
    delivered(&patient).await;

    // By default the caller hung up after 5 seconds and the write never
    // happened; nothing was sent again.
    assert_eq!(before.received(), 1);
    assert_eq!(before.written(), Vec::<String>::new());
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"timeout\""]), 1.0);
    // The same answer inside the longer timeout is a write.
    assert_eq!(after.received(), 1);
    assert_eq!(after.written(), ["chatcmpl-slow"]);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 1.0);
    assert_eq!(value(&recorder, TIME_TO_ACCEPTED_COUNT, &[]), 1.0);
}

#[tokio::test]
async fn a_timeout_is_retried_with_the_bytes_of_the_first_attempt() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    // The first request outlasts the timeout, the second is answered.
    let billing = Billing::start(Answers::Script(vec![
        (Duration::from_secs(10), 200),
        (Duration::ZERO, 200),
    ]))
    .await;
    let delivery = policy(|policy| {
        policy.attempt_timeout = Duration::from_secs(1);
        policy.max_attempts = 3;
        policy.initial_backoff = Duration::from_millis(40);
        policy.max_in_flight = 4;
    });
    billing.report(&delivery, "chatcmpl-retried", None);
    delivered(&delivery).await;

    {
        let received = billing.intake.received.lock().unwrap();
        assert_eq!(received.len(), 2);
        assert_eq!(received[0].body, received[1].body);
        for attempt in received.iter() {
            assert_eq!(
                attempt.authorization.as_deref(),
                Some("Bearer usage-secret")
            );
            assert_eq!(
                attempt.request_id.as_deref(),
                Some("request-chatcmpl-retried")
            );
            assert_eq!(attempt.content_type.as_deref(), Some("application/json"));
        }
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&received[0].body).unwrap(),
            expected_report("chatcmpl-retried")
        );
    }
    // One write, and the two attempts never overlapped: the first was
    // abandoned before the second was sent.
    assert_eq!(billing.written(), ["chatcmpl-retried"]);
    assert_eq!(billing.most_handled_at_once(), 1);

    // One report, accepted; the timeout shows as an attempt and a retry.
    assert_eq!(value(&recorder, REPORTS, &[]), 1.0);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 1.0);
    assert_eq!(value(&recorder, ATTEMPTS, &["outcome=\"timeout\""]), 1.0);
    assert_eq!(value(&recorder, ATTEMPTS, &["outcome=\"accepted\""]), 1.0);
    assert_eq!(value(&recorder, RETRIES, &["reason=\"timeout\""]), 1.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    assert_eq!(value(&recorder, TIME_TO_ACCEPTED_COUNT, &[]), 1.0);
}

#[tokio::test]
async fn a_refusal_of_the_report_itself_is_never_retried() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = policy(|policy| {
        policy.max_attempts = 5;
        policy.initial_backoff = Duration::from_millis(5);
        policy.max_in_flight = 4;
    });
    let refusals = [400, 401, 402, 403, 404, 409, 422];
    let mut intakes = Vec::new();
    for status in refusals {
        let billing = Billing::answering(status).await;
        billing.report(&delivery, &format!("chatcmpl-{status}"), None);
        intakes.push(billing);
    }
    delivered(&delivery).await;
    tokio::time::sleep(Duration::from_millis(200)).await;

    for (billing, status) in intakes.iter().zip(refusals) {
        assert_eq!(billing.received(), 1, "{status}");
    }
    let refused = refusals.len() as f64;
    assert_eq!(
        value(&recorder, REPORTS, &["outcome=\"http_4xx\""]),
        refused
    );
    assert_eq!(
        value(&recorder, ATTEMPTS, &["outcome=\"http_4xx\""]),
        refused
    );
    assert_eq!(value(&recorder, DROPPED, &["reason=\"rejected\""]), refused);
    assert_eq!(value(&recorder, RETRIES, &[]), 0.0);
}

#[tokio::test]
async fn rate_limits_server_errors_and_connection_failures_are_retried() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let delivery = policy(|policy| {
        policy.max_attempts = 3;
        policy.initial_backoff = Duration::from_millis(5);
        policy.max_in_flight = 16;
    });

    // Once, then accepted.
    let passing = [429, 500, 502, 503, 504];
    let mut intakes = Vec::new();
    for status in passing {
        let billing = Billing::start(Answers::Script(vec![
            (Duration::ZERO, status),
            (Duration::ZERO, 200),
        ]))
        .await;
        billing.report(&delivery, &format!("chatcmpl-{status}"), None);
        intakes.push(billing);
    }
    // Every time: the attempts run out.
    let down = Billing::answering(503).await;
    down.report(&delivery, "chatcmpl-down", None);
    Billing::unreachable().report(&delivery, "chatcmpl-unreachable", None);
    delivered(&delivery).await;

    for (billing, status) in intakes.iter().zip(passing) {
        assert_eq!(billing.received(), 2, "{status}");
        assert_eq!(billing.written(), [format!("chatcmpl-{status}")]);
    }
    assert_eq!(down.received(), 3);

    // Final outcomes: five accepted, two lost to the last error they met.
    assert_eq!(value(&recorder, REPORTS, &[]), 7.0);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 5.0);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"http_5xx\""]), 1.0);
    assert_eq!(
        value(&recorder, REPORTS, &["outcome=\"connect_error\""]),
        1.0
    );
    assert_eq!(
        value(&recorder, DROPPED, &["reason=\"attempts_exhausted\""]),
        2.0
    );
    assert_eq!(value(&recorder, DROPPED, &[]), 2.0);
    // Attempts and retries, by what they met.
    assert_eq!(value(&recorder, ATTEMPTS, &["outcome=\"accepted\""]), 5.0);
    assert_eq!(value(&recorder, ATTEMPTS, &["outcome=\"http_4xx\""]), 1.0);
    assert_eq!(
        value(&recorder, ATTEMPTS, &["outcome=\"http_5xx\""]),
        4.0 + 3.0
    );
    assert_eq!(
        value(&recorder, ATTEMPTS, &["outcome=\"connect_error\""]),
        3.0
    );
    assert_eq!(value(&recorder, RETRIES, &["reason=\"http_429\""]), 1.0);
    assert_eq!(
        value(&recorder, RETRIES, &["reason=\"http_5xx\""]),
        4.0 + 2.0
    );
    assert_eq!(
        value(&recorder, RETRIES, &["reason=\"connect_error\""]),
        2.0
    );
}

/// Twenty reports, two places and five attempts each, against an intake that
/// answers 503 at once for the two seconds after its first request: how many
/// were written.
async fn written_through_an_outage(initial_backoff: Duration) -> usize {
    let billing = Billing::start(Answers::DownFor(Duration::from_secs(2))).await;
    let delivery = policy(|policy| {
        policy.max_attempts = 5;
        policy.initial_backoff = initial_backoff;
        policy.max_in_flight = 2;
    });
    for n in 0..20 {
        billing.report(&delivery, &format!("chatcmpl-{n}"), None);
    }
    delivered(&delivery).await;
    billing.written().len()
}

#[tokio::test]
async fn failures_that_come_at_once_use_up_the_attempts_unless_the_backoff_outlasts_them() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let exhausted = || value(&recorder, DROPPED, &["reason=\"attempts_exhausted\""]);

    // A failed attempt takes no time here, so the backoff is all there is
    // between two attempts. A short one (15 ms in all) spends the five
    // attempts of one report after another inside the outage; no deadline
    // would have saved them.
    let written = written_through_an_outage(Duration::from_millis(1)).await;
    assert!(written < 20, "{written}");
    assert_eq!(exhausted(), (20 - written) as f64);

    // With a backoff whose first three steps outlast the outage (at least
    // 300 + 600 + 1200 ms), the fourth attempt of every report finds the
    // intake back, whenever its first one was made.
    let lost_before = exhausted();
    let written = written_through_an_outage(Duration::from_millis(600)).await;
    assert_eq!(written, 20);
    assert_eq!(exhausted(), lost_before);
}

#[test]
fn the_backoff_doubles_from_the_initial_value_is_spread_and_never_immediate() {
    let delivery = UsageReportDelivery::with(UsageReportPolicy {
        initial_backoff: Duration::from_millis(500),
        ..Default::default()
    });
    for (attempts, ceiling_ms) in [
        (1, 500),
        (2, 1_000),
        (3, 2_000),
        (4, 4_000),
        (7, 30_000),
        (40, 30_000),
    ] {
        let delays: Vec<u128> = (0..200)
            .map(|_| delivery.backoff(attempts).as_millis())
            .collect();
        assert!(
            delays
                .iter()
                .all(|delay| (ceiling_ms / 2..=ceiling_ms).contains(delay)),
            "after {attempts} attempts: {delays:?}"
        );
        assert!(delays.iter().any(|delay| delay != &delays[0]), "no jitter");
    }
    // Down to the smallest value the configuration accepts.
    let smallest = UsageReportDelivery::with(UsageReportPolicy {
        initial_backoff: Duration::from_millis(1),
        ..Default::default()
    });
    for _ in 0..200 {
        assert_eq!(smallest.backoff(1), Duration::from_millis(1));
        assert!(smallest.backoff(2) >= Duration::from_millis(1));
    }
}

#[tokio::test]
async fn a_report_that_cannot_be_sent_before_its_deadline_is_dropped_and_counted() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);

    // Waiting: one place, 200 ms a write, and twelve reports at once. The
    // ones whose turn comes with less than a whole timeout left before the
    // deadline are dropped unsent.
    let billing = Billing::start(Answers::Script(vec![(Duration::from_millis(200), 200)])).await;
    let waiting = policy(|policy| {
        policy.attempt_timeout = Duration::from_secs(1);
        policy.deadline = Some(Duration::from_secs(2));
        policy.max_in_flight = 1;
    });
    for n in 0..12 {
        billing.report(&waiting, &format!("chatcmpl-{n}"), None);
    }
    delivered(&waiting).await;
    let accepted = value(&recorder, REPORTS, &["outcome=\"accepted\""]);
    let too_late = value(&recorder, REPORTS, &["outcome=\"deadline_exceeded\""]);
    // About six fit (a turn every 200 ms, the last one a second before the
    // deadline); the others were too late and are counted as that.
    assert!((3.0..=7.0).contains(&accepted), "{accepted}");
    assert_eq!(accepted + too_late, 12.0);
    assert_eq!(
        value(&recorder, DROPPED, &["reason=\"deadline\""]),
        too_late
    );
    assert_eq!(value(&recorder, REPORTS, &[]), 12.0);
    // No attempt was made with less than its timeout: everything that was
    // sent was accepted, and what was dropped was never sent.
    assert_eq!(billing.received() as f64, accepted);
    assert_eq!(billing.written().len() as f64, accepted);
    assert_eq!(value(&recorder, ATTEMPTS, &[]), accepted);

    // Retrying: an intake that keeps failing is not retried past the
    // deadline, however many attempts are left.
    let down = Billing::answering(503).await;
    let retrying = policy(|policy| {
        policy.attempt_timeout = Duration::from_millis(200);
        policy.max_attempts = 10;
        policy.initial_backoff = Duration::from_millis(200);
        policy.deadline = Some(Duration::from_millis(900));
        policy.max_in_flight = 1;
    });
    down.report(&retrying, "chatcmpl-down", None);
    delivered(&retrying).await;
    let attempts = down.received() as f64;
    assert!((1.0..=5.0).contains(&attempts), "{attempts}");
    assert_eq!(
        value(&recorder, REPORTS, &["outcome=\"deadline_exceeded\""]),
        too_late + 1.0
    );
    assert_eq!(
        value(&recorder, DROPPED, &["reason=\"deadline\""]),
        too_late + 1.0
    );
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"http_5xx\""]), 0.0);
    assert_eq!(value(&recorder, REPORTS, &[]), 13.0);
    // The report dropped after its attempts has the duration of the last
    // one; the reports dropped unsent have none.
    assert_eq!(
        value(
            &recorder,
            DURATION_COUNT,
            &["outcome=\"deadline_exceeded\""]
        ),
        1.0
    );
    // A retry is counted when it is made, not when it is planned.
    assert_eq!(
        value(&recorder, ATTEMPTS, &["outcome=\"http_5xx\""]),
        attempts
    );
    assert_eq!(value(&recorder, RETRIES, &[]), attempts - 1.0);
}

// ---------------------------------------------------------------------------
// Polite: a cap on reports in flight, a bounded queue
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_burst_never_has_more_reports_in_flight_than_the_cap() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let billing = Billing::start(Answers::Script(vec![(Duration::from_millis(40), 200)])).await;
    let delivery = policy(|policy| policy.max_in_flight = 4);

    // All sixty are handed over before any of them is on the wire: four
    // take a place, the others wait.
    for n in 0..60 {
        billing.report(&delivery, &format!("chatcmpl-{n}"), None);
    }
    assert_eq!(delivery.pending(), (56, 4));
    assert_eq!(billing.received(), 0);
    assert_eq!(value(&recorder, WAITING, &[]), 56.0);
    assert_eq!(value(&recorder, IN_FLIGHT, &[]), 4.0);
    delivered(&delivery).await;

    assert_eq!(billing.most_handled_at_once(), 4);
    assert_eq!(billing.received(), 60);
    // In the order they were handed over, within what four places allow.
    let written = billing.written();
    assert_eq!(written.len(), 60);
    for (position, id) in written.iter().enumerate() {
        let n: usize = id.trim_start_matches("chatcmpl-").parse().unwrap();
        assert!(n.abs_diff(position) < 8, "{id} written at {position}");
    }
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 60.0);
    assert_eq!(value(&recorder, WAITING, &[]), 0.0);
    assert_eq!(value(&recorder, IN_FLIGHT, &[]), 0.0);
    assert_eq!(delivery.pending(), (0, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reports_handed_over_from_several_threads_are_each_delivered_once_under_the_cap() {
    let billing = Arc::new(
        Billing::start(Answers::Script(vec![
            // A few failures on the way, so places are also held by retries.
            (Duration::from_millis(2), 503),
            (Duration::from_millis(2), 429),
            (Duration::from_millis(2), 200),
        ]))
        .await,
    );
    let delivery = policy(|policy| {
        policy.max_attempts = 3;
        policy.initial_backoff = Duration::from_millis(2);
        policy.max_in_flight = 4;
    });
    let completing: Vec<_> = (0..8)
        .map(|task| {
            let (billing, delivery) = (billing.clone(), delivery.clone());
            tokio::spawn(async move {
                for n in 0..50 {
                    billing.report(&delivery, &format!("chatcmpl-{task}-{n}"), None);
                    if n % 7 == 0 {
                        tokio::task::yield_now().await;
                    }
                }
            })
        })
        .collect();
    for task in completing {
        task.await.unwrap();
    }
    delivered(&delivery).await;

    let mut written = billing.written();
    assert_eq!(written.len(), 400);
    written.sort();
    written.dedup();
    assert_eq!(written.len(), 400);
    assert!(
        billing.most_handled_at_once() <= 4,
        "{}",
        billing.most_handled_at_once()
    );
    assert_eq!(delivery.pending(), (0, 0));
}

#[tokio::test]
async fn a_full_queue_drops_the_report_that_waited_longest_and_counts_it() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let billing = Billing::start(Answers::Script(vec![(Duration::from_millis(50), 200)])).await;
    let delivery = policy(|policy| {
        policy.max_in_flight = 1;
        policy.max_queued = 3;
    });

    // One in flight, three waiting; the fifth and the sixth each push out
    // the oldest of the waiting ones.
    for id in ["one", "two", "three", "four", "five", "six"] {
        billing.report(&delivery, id, None);
    }
    assert_eq!(delivery.pending(), (3, 1));
    delivered(&delivery).await;

    assert_eq!(billing.written(), ["one", "four", "five", "six"]);
    assert_eq!(billing.received(), 4);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"queue_full\""]), 2.0);
    assert_eq!(value(&recorder, DROPPED, &["reason=\"queue_full\""]), 2.0);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 4.0);
    // Every report has exactly one final outcome.
    assert_eq!(value(&recorder, REPORTS, &[]), 6.0);
    assert_eq!(value(&recorder, WAITING, &[]), 0.0);
}

#[tokio::test]
async fn every_report_dropped_from_a_full_queue_is_counted_however_many_at_once() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let billing = Billing::start(Answers::Script(vec![(Duration::from_millis(50), 200)])).await;
    let delivery = policy(|policy| {
        policy.max_in_flight = 1;
        policy.max_queued = 1;
    });
    // More drops in one go than get a task of their own for their line: the
    // ones past that are accounted for on the spot.
    let reports = MAX_DROPS_BEING_LOGGED + 500;
    for n in 0..reports {
        billing.report(&delivery, &format!("chatcmpl-{n}"), None);
    }
    assert_eq!(delivery.pending(), (1, 1));
    assert_eq!(
        value(&recorder, REPORTS, &["outcome=\"queue_full\""]),
        498.0
    );
    delivered(&delivery).await;

    // The first took the place, the last is the one still waiting; every
    // report in between was dropped, and each has its one final outcome.
    assert_eq!(
        billing.written(),
        [
            "chatcmpl-0".to_string(),
            format!("chatcmpl-{}", reports - 1)
        ]
    );
    let dropped = (reports - 2) as f64;
    assert_eq!(
        value(&recorder, REPORTS, &["outcome=\"queue_full\""]),
        dropped
    );
    assert_eq!(
        value(&recorder, DROPPED, &["reason=\"queue_full\""]),
        dropped
    );
    assert_eq!(value(&recorder, REPORTS, &[]), reports as f64);
    assert_eq!(value(&recorder, WAITING, &[]), 0.0);
}

#[tokio::test]
async fn a_worker_that_panics_gives_its_place_back() {
    let billing = Billing::answering(200).await;
    let delivery = policy(|policy| policy.max_in_flight = 1);
    // The one place is taken, the way `submit` takes it, by a worker that is
    // about to die, and a report that arrives meanwhile has to wait.
    delivery.state().in_flight += 1;
    let place = Place {
        delivery: delivery.clone(),
        model: None,
        held: true,
    };
    billing.report(&delivery, "waiting", None);
    assert_eq!(delivery.pending(), (1, 1));
    let worker = tokio::spawn(async move {
        let _place = place;
        panic!("the worker died");
    });
    assert!(worker.await.unwrap_err().is_panic());

    // The place is free again. Had it gone with the worker, nothing would
    // ever be sent under this cap.
    assert_eq!(delivery.pending(), (1, 0));
    // The next report takes it, and its worker goes on with what waited.
    billing.report(&delivery, "next", None);
    delivered(&delivery).await;
    assert_eq!(billing.written(), ["next", "waiting"]);
    assert_eq!(delivery.pending(), (0, 0));
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

#[tokio::test]
async fn shutdown_waits_for_pending_reports_and_says_how_many_it_left() {
    // Everything pending finishes inside the drain time.
    let billing = Billing::start(Answers::Script(vec![(Duration::from_millis(60), 200)])).await;
    let delivery = policy(|policy| {
        policy.max_in_flight = 2;
        policy.shutdown_drain = Duration::from_secs(30);
    });
    for n in 0..10 {
        billing.report(&delivery, &format!("chatcmpl-{n}"), None);
    }
    let drained = delivery.drain_at_shutdown().await.unwrap();
    assert_eq!((drained.pending, drained.left()), (10, 0));
    assert_eq!(billing.written().len(), 10);
    assert!(
        drained.waited >= Duration::from_millis(5 * 60),
        "{drained:?}"
    );

    // An intake that does not answer in time: the drain gives up when its
    // time is over and reports what is still queued and still in flight.
    let stuck = Billing::start(Answers::Script(vec![(Duration::from_secs(60), 200)])).await;
    let delivery = policy(|policy| {
        policy.attempt_timeout = Duration::from_secs(120);
        policy.max_in_flight = 2;
        policy.shutdown_drain = Duration::from_millis(300);
    });
    for n in 0..5 {
        stuck.report(&delivery, &format!("chatcmpl-{n}"), None);
    }
    let drained = delivery.drain_at_shutdown().await.unwrap();
    assert_eq!(
        (
            drained.pending,
            drained.left_waiting,
            drained.left_in_flight
        ),
        (5, 3, 2)
    );
    assert!(drained.waited >= Duration::from_millis(300), "{drained:?}");
    assert!(drained.waited < Duration::from_secs(10), "{drained:?}");

    // No drain time set: nothing is awaited, the count is still given.
    let delivery = policy(|policy| policy.max_in_flight = 2);
    for n in 0..3 {
        stuck.report(&delivery, &format!("chatcmpl-late-{n}"), None);
    }
    let drained = delivery.drain_at_shutdown().await.unwrap();
    assert_eq!((drained.left_waiting, drained.left_in_flight), (1, 2));
    assert!(drained.waited < Duration::from_millis(250), "{drained:?}");
}

// ---------------------------------------------------------------------------
// The `model` label
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_delivery_series_carry_the_model_label_of_a_listed_model_and_none_otherwise() {
    for model in [None, Some("example/alpha")] {
        let recorder = PrometheusBuilder::new().build_recorder();
        let _metrics = metrics::set_default_local_recorder(&recorder);
        // A failure first, so there is a retry; one place and one waiting
        // position, so there is a wait and a drop.
        let billing = Billing::start(Answers::Script(vec![
            (Duration::from_millis(30), 503),
            (Duration::from_millis(30), 200),
        ]))
        .await;
        let delivery = policy(|policy| {
            policy.max_attempts = 2;
            policy.initial_backoff = Duration::from_millis(5);
            policy.max_in_flight = 1;
            policy.max_queued = 1;
        });
        for id in ["one", "two", "three"] {
            billing.report(&delivery, id, model);
        }
        delivered(&delivery).await;
        assert_eq!(billing.written(), ["one", "three"]);

        let rendered = recorder.handle().render();
        let source = "auth_path=\"cloud_api_key\",ingress_route=\"canonical\"";
        let model_label = model.map_or(String::new(), |model| format!(",model=\"{model}\""));
        let gauge_labels = model.map_or(String::new(), |model| format!("{{model=\"{model}\"}}"));
        for expected in [
            format!("{REPORTS}{{outcome=\"accepted\",{source}{model_label}}} 2"),
            format!("{REPORTS}{{outcome=\"queue_full\",{source}{model_label}}} 1"),
            format!("{ATTEMPTS}{{outcome=\"http_5xx\",{source}{model_label}}} 1"),
            format!("{ATTEMPTS}{{outcome=\"accepted\",{source}{model_label}}} 2"),
            format!("{RETRIES}{{reason=\"http_5xx\",{source}{model_label}}} 1"),
            format!("{DROPPED}{{reason=\"queue_full\",{source}{model_label}}} 1"),
            format!("{TIME_TO_ACCEPTED_COUNT}{{{source}{model_label}}} 2"),
            format!("{WAITING}{gauge_labels} 0"),
            format!("{IN_FLIGHT}{gauge_labels} 0"),
        ] {
            assert!(
                rendered.lines().any(|line| line == expected),
                "{expected} missing:\n{rendered}"
            );
        }
        // The same rule for every usage-report series, old and new: `model`
        // is there, last, exactly when the report is for a listed model.
        for line in rendered.lines() {
            if !line.starts_with('#') && line.contains("usage_report") {
                let series = line.rsplit_once(' ').unwrap().0;
                // (A summary's `quantile` comes after it.)
                match model {
                    Some(model) => {
                        assert!(series.contains(&format!("model=\"{model}\"")), "{line}")
                    }
                    None => assert!(!series.contains("model="), "{line}"),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The shape seen in production: a wave of completions against an intake that
// writes one report at a time
// ---------------------------------------------------------------------------

/// The intake writes about 17 reports a second.
const PER_WRITE: Duration = Duration::from_millis(1_000 / 17);

/// 200 requests complete within a second, a few at a time, and hand over
/// their reports.
async fn wave(billing: &Billing, delivery: &Arc<UsageReportDelivery>) {
    for batch in 0..20 {
        for n in 0..10 {
            billing.report(delivery, &format!("chatcmpl-{}", batch * 10 + n), None);
        }
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
}

#[tokio::test]
async fn by_default_the_tail_of_a_wave_times_out_and_is_never_written() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let billing = Billing::start(Answers::OneWriteAtATime(PER_WRITE)).await;
    let delivery = Arc::new(UsageReportDelivery::default());
    wave(&billing, &delivery).await;
    delivered(&delivery).await;

    // Every report went out at once, and the intake had almost all of them
    // open at the same time.
    assert_eq!(billing.received(), 200);
    assert!(
        billing.most_handled_at_once() > 100,
        "{}",
        billing.most_handled_at_once()
    );
    // It writes about 17 a second, so only what it reached in 5 seconds was
    // accepted. The rest timed out, and their writes never happened.
    let accepted = value(&recorder, REPORTS, &["outcome=\"accepted\""]);
    let timed_out = value(&recorder, REPORTS, &["outcome=\"timeout\""]);
    assert_eq!(accepted + timed_out, 200.0);
    assert!((20.0..=140.0).contains(&accepted), "{accepted}");
    let written = billing.written().len() as f64;
    assert!(
        (accepted..=accepted + 3.0).contains(&written),
        "{written} of {accepted}"
    );
}

#[tokio::test]
async fn with_the_lane_settings_the_whole_wave_is_accepted_and_the_cap_holds() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    let billing = Billing::start(Answers::OneWriteAtATime(PER_WRITE)).await;
    // The settings docs/gateway-mode.md suggests for a lane.
    let delivery = policy(|policy| {
        policy.attempt_timeout = Duration::from_secs(30);
        policy.max_attempts = 5;
        policy.initial_backoff = Duration::from_secs(5);
        policy.deadline = Some(Duration::from_secs(300));
        policy.max_in_flight = 8;
    });
    let started_at = Instant::now();
    wave(&billing, &delivery).await;
    delivered(&delivery).await;
    let took = started_at.elapsed();

    // All 200 written, each exactly once, and never more than 8 in flight.
    let mut written = billing.written();
    assert_eq!(written.len(), 200);
    written.sort();
    written.dedup();
    assert_eq!(written.len(), 200);
    assert_eq!(billing.most_handled_at_once(), 8);
    // No report needed a second attempt, and none was dropped.
    assert_eq!(billing.received(), 200);
    assert_eq!(value(&recorder, REPORTS, &["outcome=\"accepted\""]), 200.0);
    assert_eq!(value(&recorder, REPORTS, &[]), 200.0);
    assert_eq!(value(&recorder, ATTEMPTS, &[]), 200.0);
    assert_eq!(value(&recorder, RETRIES, &[]), 0.0);
    assert_eq!(value(&recorder, DROPPED, &[]), 0.0);
    assert_eq!(value(&recorder, TIME_TO_ACCEPTED_COUNT, &[]), 200.0);
    // The wave clears at the intake's own pace: 200 writes at 17 a second.
    assert!(took >= PER_WRITE * 200, "{took:?}");
    assert!(took < Duration::from_secs(25), "{took:?}");
}

// ---------------------------------------------------------------------------
// Log lines
// ---------------------------------------------------------------------------

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
        // tracing caches every callsite's interest for the whole process.
        // While exactly one subscriber is registered it derives that from the
        // calling thread's default, so a test on another thread, which has
        // none, would switch a shared callsite off for this capture as well.
        // With a second one registered it asks every registered subscriber.
        let registered = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        LogCapture {
            _subscriber: subscriber,
            _registered: registered,
        }
    }

    fn contents(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }

    /// Every line with this message, oldest first.
    fn lines(&self, message: &str) -> Vec<serde_json::Value> {
        self.contents()
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|line| line["fields"]["message"] == message)
            .collect()
    }
}

/// The names of a line's fields, its message aside.
fn field_names(line: &serde_json::Value) -> std::collections::BTreeSet<&str> {
    line["fields"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .filter(|name| *name != "message")
        .collect()
}

/// Whether a line was written inside a span.
fn in_a_span(line: &serde_json::Value) -> bool {
    line.get("span").is_some() || line.get("spans").is_some()
}

/// What every line about one report carries: who it is for and where it came
/// from, as ids and labels.
const WHO: [&str; 7] = [
    "request_id",
    "org_id",
    "workspace_id",
    "api_key_id",
    "model",
    "auth_path",
    "ingress_route",
];

fn assert_who(line: &serde_json::Value, request_id: &str) {
    let fields = &line["fields"];
    assert_eq!(fields["request_id"], request_id, "{line}");
    assert_eq!(fields["org_id"], "org-1", "{line}");
    assert_eq!(fields["workspace_id"], "ws-1", "{line}");
    assert_eq!(fields["api_key_id"], "key-1", "{line}");
    assert_eq!(fields["model"], "test-model", "{line}");
    assert_eq!(fields["auth_path"], "cloud_api_key", "{line}");
    assert_eq!(fields["ingress_route"], "canonical", "{line}");
}

#[tokio::test]
async fn the_lines_a_report_ends_with_are_the_old_ones_by_default_and_gain_two_fields_otherwise() {
    for (delivery, added) in [
        (Arc::new(UsageReportDelivery::default()), &[][..]),
        (
            policy(|policy| policy.max_in_flight = 4),
            &["attempts", "since_completion_ms"][..],
        ),
    ] {
        let logs = Logs::default();
        let _capture = logs.capture();
        let (accepting, refusing) = (Billing::answering(200).await, Billing::answering(503).await);
        // Handed over inside a request's span, as a handler does it.
        tracing::info_span!("request", request_id = "the-request").in_scope(|| {
            accepting.report(&delivery, "accepted", None);
            refusing.report(&delivery, "refused", None);
            Billing::unreachable().report(&delivery, "unreachable", None);
        });
        delivered(&delivery).await;

        for (message, level, id, own) in [
            (
                "Direct-key usage report accepted by Cloud API",
                "INFO",
                "accepted",
                &["status", "duration_ms"][..],
            ),
            (
                "Usage reporting returned non-success",
                "WARN",
                "refused",
                &["status", "duration_ms", "outcome"][..],
            ),
            (
                "Usage reporting failed",
                "WARN",
                "unreachable",
                &["error", "duration_ms", "outcome"][..],
            ),
        ] {
            let lines = logs.lines(message);
            assert_eq!(lines.len(), 1, "{message}: {}", logs.contents());
            let line = &lines[0];
            assert_eq!(line["level"], level, "{line}");
            // Where `proxy.rs` wrote them, and in no request's span.
            assert_eq!(line["target"], "vllm_proxy_rs::proxy", "{line}");
            assert!(!in_a_span(line), "{line}");
            assert_who(line, &format!("request-{id}"));
            // Exactly the fields these lines have always had, and under a
            // changed policy two more.
            let expected: std::collections::BTreeSet<&str> =
                WHO.iter().chain(own).chain(added).copied().collect();
            assert_eq!(field_names(line), expected, "{message}");
            if !added.is_empty() {
                assert_eq!(line["fields"]["attempts"], 1, "{line}");
                assert!(line["fields"]["since_completion_ms"].is_u64(), "{line}");
            }
        }
        assert_eq!(
            logs.lines("Direct-key usage report accepted by Cloud API")[0]["fields"]["status"],
            "200 OK"
        );
        assert_eq!(
            logs.lines("Usage reporting returned non-success")[0]["fields"]["outcome"],
            "http_5xx"
        );
        assert_eq!(
            logs.lines("Usage reporting failed")[0]["fields"]["outcome"],
            "connect_error"
        );
        // The bearer is in no line.
        assert!(!logs.contents().contains("usage-secret"));
    }
}

#[tokio::test]
async fn a_dropped_report_is_logged_with_its_ids_as_usage_not_billed() {
    const QUEUE_FULL: &str = "Usage report dropped: the waiting queue is full and this report \
                              waited longest — usage NOT billed";
    const DEADLINE: &str =
        "Usage report dropped: not accepted before its deadline — usage NOT billed";
    let logs = Logs::default();
    let _capture = logs.capture();
    let names = |own: &[&'static str]| -> std::collections::BTreeSet<&'static str> {
        WHO.iter()
            .chain(&["outcome", "since_completion_ms"])
            .chain(own)
            .copied()
            .collect()
    };

    // A full queue: "two" has waited longest when "three" is handed over.
    let billing = Billing::start(Answers::Script(vec![(Duration::from_millis(50), 200)])).await;
    let full = policy(|policy| {
        policy.max_in_flight = 1;
        policy.max_queued = 1;
    });
    tracing::info_span!("request", request_id = "request-three").in_scope(|| {
        for id in ["one", "two", "three"] {
            billing.report(&full, id, None);
        }
    });
    delivered(&full).await;
    let lines = logs.lines(QUEUE_FULL);
    assert_eq!(lines.len(), 1, "{}", logs.contents());
    assert_eq!(lines[0]["level"], "WARN");
    assert_who(&lines[0], "request-two");
    assert_eq!(lines[0]["fields"]["outcome"], "queue_full");
    assert_eq!(lines[0]["fields"]["max_queued"], 1);
    assert_eq!(field_names(&lines[0]), names(&["max_queued"]));
    // It is about "two", so it is not in the span of the request that
    // handed "three" over.
    assert!(!in_a_span(&lines[0]), "{}", lines[0]);

    // A deadline, for a report that was never sent: its turn comes 700 ms
    // after it was handed over, with less than the 1 s timeout left of its
    // 1.5 s.
    let slow = Billing::start(Answers::Script(vec![(Duration::from_millis(700), 200)])).await;
    let waiting = policy(|policy| {
        policy.attempt_timeout = Duration::from_secs(1);
        policy.deadline = Some(Duration::from_millis(1_500));
        policy.max_in_flight = 1;
    });
    slow.report(&waiting, "in-time", None);
    slow.report(&waiting, "late", None);
    delivered(&waiting).await;
    assert_eq!(slow.written(), ["in-time"]);
    let lines = logs.lines(DEADLINE);
    assert_eq!(lines.len(), 1, "{}", logs.contents());
    assert_eq!(lines[0]["level"], "WARN");
    assert_who(&lines[0], "request-late");
    assert_eq!(lines[0]["fields"]["outcome"], "deadline_exceeded");
    assert_eq!(lines[0]["fields"]["attempts"], 0);
    assert_eq!(field_names(&lines[0]), names(&["attempts"]));

    // A deadline, for a report whose attempts all failed: the line also
    // says how the last one ended.
    let down = Billing::answering(503).await;
    let retrying = policy(|policy| {
        policy.attempt_timeout = Duration::from_millis(200);
        policy.max_attempts = 5;
        policy.initial_backoff = Duration::from_millis(300);
        policy.deadline = Some(Duration::from_millis(600));
        policy.max_in_flight = 1;
    });
    down.report(&retrying, "failing", None);
    delivered(&retrying).await;
    let lines = logs.lines(DEADLINE);
    assert_eq!(lines.len(), 2, "{}", logs.contents());
    assert_who(&lines[1], "request-failing");
    assert_eq!(lines[1]["fields"]["attempts"], down.received());
    assert_eq!(lines[1]["fields"]["last_attempt"], "http_5xx");
    assert_eq!(field_names(&lines[1]), names(&["attempts", "last_attempt"]));
    assert!(!logs.contents().contains("usage-secret"));
}

#[tokio::test]
async fn shutdown_says_how_many_reports_it_waited_for_and_how_many_it_left() {
    let logs = Logs::default();
    let _capture = logs.capture();
    let names = |line: &serde_json::Value| -> Vec<String> {
        field_names(line).into_iter().map(String::from).collect()
    };

    let billing = Billing::start(Answers::Script(vec![(Duration::from_millis(50), 200)])).await;
    let delivery = policy(|policy| {
        policy.max_in_flight = 1;
        policy.shutdown_drain = Duration::from_secs(30);
    });
    for id in ["one", "two", "three"] {
        billing.report(&delivery, id, None);
    }
    delivery.drain_at_shutdown().await.unwrap();
    let lines = logs.lines("Usage reports drained before shutdown");
    assert_eq!(lines.len(), 1, "{}", logs.contents());
    assert_eq!(lines[0]["level"], "INFO");
    assert_eq!(lines[0]["fields"]["pending"], 3);
    assert_eq!(names(&lines[0]), ["pending", "waited_ms"]);

    let stuck = Billing::start(Answers::Script(vec![(Duration::from_secs(60), 200)])).await;
    let delivery = policy(|policy| {
        policy.attempt_timeout = Duration::from_secs(120);
        policy.max_in_flight = 1;
        policy.shutdown_drain = Duration::from_millis(200);
    });
    for id in ["one", "two", "three"] {
        stuck.report(&delivery, id, None);
    }
    delivery.drain_at_shutdown().await.unwrap();
    let lines = logs.lines("Usage reports left undelivered at shutdown — usage NOT billed");
    assert_eq!(lines.len(), 1, "{}", logs.contents());
    assert_eq!(lines[0]["level"], "WARN");
    assert_eq!(lines[0]["fields"]["pending"], 3);
    assert_eq!(lines[0]["fields"]["left_waiting"], 2);
    assert_eq!(lines[0]["fields"]["left_in_flight"], 1);
    assert_eq!(
        names(&lines[0]),
        ["left_in_flight", "left_waiting", "pending", "waited_ms"]
    );
    // One line for either ending, and none under the default policy.
    assert_eq!(logs.lines("Usage reports drained before shutdown").len(), 1);
    let default = Arc::new(UsageReportDelivery::default());
    stuck.report(&default, "four", None);
    assert_eq!(default.drain_at_shutdown().await, None);
    assert_eq!(
        logs.contents().matches("Usage reports").count(),
        2,
        "{}",
        logs.contents()
    );
}
