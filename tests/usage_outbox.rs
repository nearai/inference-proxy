//! The usage report outbox (`VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH`) in the
//! process the deploy starts: the built binary as a gateway, wiremock for the
//! engine and for cloud-api, and the outbox file read by a connection of the
//! test's own, the way an operator reads it.
//!
//! Everything a single delivery does is tested next to it
//! (`src/usage_report_outbox_tests.rs`). What is here needs real processes:
//! one that is killed, one that is stopped in good order, two on one file.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use vllm_proxy_rs::*;

const MODEL: &str = "example/alpha";
const KEY: &str = "sk-live-customer";
const USAGE_TOKEN: &str = "usage-secret";
const OUTBOX_PATH: &str = "VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH";
const ACCEPTED: &str = "Direct-key usage report accepted by Cloud API";

/// An engine that gives every completion an id of its own, as engines do.
struct Engine(AtomicUsize);

impl Respond for Engine {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let n = self.0.fetch_add(1, Ordering::SeqCst);
        ResponseTemplate::new(200).set_body_json(json!({
            "id": format!("chatcmpl-{n}"),
            "object": "chat.completion",
            "model": "served",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        }))
    }
}

async fn engine() -> MockServer {
    let engine = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(routes::ROUTE_CHAT_COMPLETIONS))
        .respond_with(Engine(AtomicUsize::new(0)))
        .mount(&engine)
        .await;
    // Whatever the pool's health checker asks.
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .mount(&engine)
        .await;
    engine
}

/// cloud-api's usage intake: what it answers can be changed while it runs,
/// and it remembers which reports it accepted.
#[derive(Clone)]
struct Intake(Arc<IntakeState>);

struct IntakeState {
    status: AtomicU16,
    delay_ms: u64,
    attempts: AtomicUsize,
    /// The completion id of every report answered with a success, in order.
    accepted: Mutex<Vec<String>>,
    bearers: Mutex<BTreeSet<String>>,
}

impl Respond for Intake {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let state = &self.0;
        state.attempts.fetch_add(1, Ordering::SeqCst);
        if let Some(bearer) = request.headers.get("authorization") {
            state
                .bearers
                .lock()
                .unwrap()
                .insert(bearer.to_str().unwrap().to_string());
        }
        let status = state.status.load(Ordering::SeqCst);
        if (200..300).contains(&status) {
            let report: Value = serde_json::from_slice(&request.body).unwrap();
            state
                .accepted
                .lock()
                .unwrap()
                .push(report["id"].as_str().unwrap().to_string());
        }
        ResponseTemplate::new(status).set_delay(Duration::from_millis(state.delay_ms))
    }
}

impl Intake {
    fn answer(&self, status: u16) {
        self.0.status.store(status, Ordering::SeqCst);
    }

    fn attempts(&self) -> usize {
        self.0.attempts.load(Ordering::SeqCst)
    }

    fn accepted(&self) -> Vec<String> {
        self.0.accepted.lock().unwrap().clone()
    }
}

/// cloud-api: a key check that knows `KEY`, and the usage intake.
async fn cloud_api(status: u16, delay_ms: u64) -> (MockServer, Intake) {
    let cloud = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/check_api_key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "valid": true,
            "organization_id": "org",
            "workspace_id": "ws",
            "api_key_id": "k"
        })))
        .mount(&cloud)
        .await;
    let intake = Intake(Arc::new(IntakeState {
        status: AtomicU16::new(status),
        delay_ms,
        attempts: AtomicUsize::new(0),
        accepted: Mutex::default(),
        bearers: Mutex::default(),
    }));
    Mock::given(method("POST"))
        .and(path("/v1/internal/usage"))
        .respond_with(intake.clone())
        .mount(&cloud)
        .await;
    (cloud, intake)
}

/// The built binary serving `MODEL` from a one-model list, until dropped.
struct Gateway {
    child: std::process::Child,
    base: String,
    log: tempfile::NamedTempFile,
    _list: tempfile::NamedTempFile,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Gateway {
    /// Start `vllm-proxy-rs` on a free loopback port in front of `engine`,
    /// with `cloud` as its cloud-api and `env` on top, and wait until it
    /// answers.
    async fn start(engine: &MockServer, cloud: &MockServer, env: &[(&str, &str)]) -> Gateway {
        let list = json!({"models": [{"id": MODEL, "backend_urls": [engine.uri()]}]});
        let mut last_log = String::new();
        // The port is picked by binding and releasing it; if something else
        // takes it in between, the binary exits and another one is tried.
        for _ in 0..5 {
            let port = {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                listener.local_addr().unwrap().port()
            };
            let mut list_file = tempfile::NamedTempFile::new().unwrap();
            list_file.write_all(list.to_string().as_bytes()).unwrap();
            let log = tempfile::NamedTempFile::new().unwrap();
            let child = std::process::Command::new(env!("CARGO_BIN_EXE_vllm-proxy-rs"))
                .env_clear()
                .envs([
                    ("TOKEN", "test-token"),
                    ("NON_TEE_DEPLOYMENT", "1"),
                    ("DEV", "1"),
                    ("GPU_NO_HW_MODE", "1"),
                    ("LOG_FORMAT", "json"),
                    ("LISTEN_ADDR", "127.0.0.1"),
                    ("CLOUD_API_AUTH_MAX_ATTEMPTS", "1"),
                    ("VLLM_PROXY_IMAGE_VALIDATION_DISABLED", "1"),
                    ("CLOUD_API_USAGE_TOKEN", USAGE_TOKEN),
                ])
                .env("LISTEN_PORT", port.to_string())
                .env("CLOUD_API_URL", cloud.uri())
                .env(
                    "VLLM_PROXY_MODELS_DOCUMENT_URL",
                    format!("{}/v1/models", cloud.uri()),
                )
                .env(model_list::MODEL_LIST_FILE_ENV, list_file.path())
                .envs(env.iter().copied())
                .stdin(std::process::Stdio::null())
                .stdout(log.reopen().unwrap())
                .stderr(log.reopen().unwrap())
                .spawn()
                .expect("the binary starts");
            let mut gateway = Gateway {
                child,
                base: format!("http://127.0.0.1:{port}"),
                log,
                _list: list_file,
            };
            let client = reqwest::Client::new();
            for _ in 0..400 {
                if gateway.child.try_wait().unwrap().is_some() {
                    break;
                }
                let version = client.get(format!("{}/version", gateway.base)).send().await;
                if version.is_ok_and(|response| response.status().is_success()) {
                    return gateway;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            last_log = gateway.log();
        }
        panic!("the binary did not come up: {last_log}");
    }

    fn log(&self) -> String {
        std::fs::read_to_string(self.log.path()).unwrap()
    }

    /// The `fields` of every JSON log line with this message.
    fn logged(&self, message: &str) -> Vec<Value> {
        self.log()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|line| line["fields"]["message"] == message)
            .map(|mut line| line["fields"].take())
            .collect()
    }

    /// One customer request, answered whatever becomes of its report.
    async fn chat(&self) {
        let response = reqwest::Client::new()
            .post(format!("{}{}", self.base, routes::ROUTE_CHAT_COMPLETIONS))
            .bearer_auth(KEY)
            .json(&json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK, "{}", self.log());
        response.bytes().await.unwrap();
    }

    async fn metrics(&self) -> String {
        reqwest::get(format!("{}{}", self.base, routes::ROUTE_METRICS))
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    /// Send the process `signal` and wait for it to end.
    async fn stop(&mut self, signal: &str) -> std::process::ExitStatus {
        let sent = std::process::Command::new("kill")
            .args([signal, &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(sent.success());
        for _ in 0..800 {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("the process did not end: {}", self.log());
    }
}

/// One value read from the outbox; 0 or empty while the file or its tables
/// are not there.
fn read<T: rusqlite::types::FromSql + Default>(outbox: &Path, sql: &str) -> T {
    let Ok(conn) = rusqlite::Connection::open_with_flags(
        outbox,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) else {
        return T::default();
    };
    let _ = conn.busy_timeout(Duration::from_secs(5));
    conn.query_row(sql, [], |row| row.get(0))
        .unwrap_or_default()
}

fn pending_rows(outbox: &Path) -> i64 {
    read(outbox, "SELECT COUNT(*) FROM pending")
}

/// Wait until `done`, for at most 60 seconds.
async fn eventually(what: &str, mut done: impl FnMut() -> bool) {
    for _ in 0..2_400 {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("never happened: {what}");
}

/// The value of the first series of `/metrics` output that starts with
/// `series`.
fn metric(metrics: &str, series: &str) -> Option<f64> {
    metrics
        .lines()
        .find(|line| line.starts_with(series))
        .and_then(|line| line.rsplit(' ').next()?.parse().ok())
}

/// A process is killed with reports it could not deliver, the next one is
/// stopped in good order with more of them, and the third delivers them all:
/// a report is in the file from the moment its request completed until
/// cloud-api has taken it, whatever happens to the processes in between.
#[tokio::test]
async fn reports_survive_a_kill_a_restart_and_an_outage_of_the_billing_api() {
    let engine = engine().await;
    let (cloud, intake) = cloud_api(503, 0).await;
    let dir = tempfile::tempdir().unwrap();
    let outbox = dir.path().join("usage-outbox.db");
    // One report in flight at a time and a pause of a second or more after a
    // failure, so that the moment a process is killed can be picked.
    let env = [
        (OUTBOX_PATH, outbox.to_str().unwrap()),
        ("VLLM_PROXY_USAGE_REPORT_MAX_IN_FLIGHT", "1"),
        ("VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS", "2000"),
    ];

    // The first process serves three requests whose reports cloud-api does
    // not take.
    let mut first = Gateway::start(&engine, &cloud, &env).await;
    for _ in 0..3 {
        first.chat().await;
    }
    eventually("three reports in the file, one of them tried", || {
        pending_rows(&outbox) == 3 && intake.attempts() >= 1
    })
    .await;
    eventually("no lease is held while the place pauses", || {
        read::<i64>(
            &outbox,
            "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL",
        ) == 0
    })
    .await;
    let enabled = first.logged(
        "Usage report outbox enabled: a report is kept on disk until the billing API accepts \
         it or refuses it for good",
    );
    assert_eq!(enabled.len(), 1, "{}", first.log());
    assert_eq!(enabled[0]["until_accepted"], true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&outbox).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    // It is killed outright: no shutdown code runs.
    let killed = first.stop("-KILL").await;
    assert!(!killed.success());
    assert!(first.logged("Server shut down").is_empty());
    assert_eq!(pending_rows(&outbox), 3);

    // The second process finds them, takes two more requests, and is stopped
    // the way a deploy stops a process.
    let mut second = Gateway::start(&engine, &cloud, &env).await;
    for _ in 0..2 {
        second.chat().await;
    }
    eventually("five reports in the file", || pending_rows(&outbox) == 5).await;
    let stopped = second.stop("-TERM").await;
    assert!(stopped.success(), "{}", second.log());
    let closed = second.logged(
        "Usage report outbox closed for shutdown: what it holds is sent by the next process",
    );
    assert_eq!(closed.len(), 1, "{}", second.log());
    assert_eq!(closed[0]["in_outbox"], 5);
    assert_eq!(second.logged("Server shut down").len(), 1);
    assert_eq!(pending_rows(&outbox), 5);
    assert_eq!(
        read::<i64>(
            &outbox,
            "SELECT COUNT(*) FROM pending WHERE lease_until_ms > 0"
        ),
        0
    );
    assert_eq!(intake.accepted(), Vec::<String>::new());

    // cloud-api is back, and so is a third process. Every report is taken,
    // once.
    intake.answer(200);
    let third = Gateway::start(&engine, &cloud, &env).await;
    eventually("all five reports are accepted", || {
        intake.accepted().len() >= 5 && pending_rows(&outbox) == 0
    })
    .await;
    let accepted = intake.accepted();
    let distinct: BTreeSet<&String> = accepted.iter().collect();
    assert_eq!((accepted.len(), distinct.len()), (5, 5), "{accepted:?}");
    assert_eq!(read::<i64>(&outbox, "SELECT COUNT(*) FROM rejected"), 0);

    // The third process says so in the lines and series a report has always
    // ended with, for requests it never served.
    let lines = third.logged(ACCEPTED);
    assert_eq!(lines.len(), 5, "{}", third.log());
    for line in &lines {
        assert_eq!(line["model"], MODEL);
        assert_eq!(line["org_id"], "org");
        assert_eq!(line["workspace_id"], "ws");
        assert_eq!(line["api_key_id"], "k");
        assert!(line["request_id"].as_str().is_some_and(|id| !id.is_empty()));
        assert!(line["attempts"].as_u64().unwrap() >= 1);
    }
    let metrics = third.metrics().await;
    let accepted_series = format!(
        "inference_proxy_usage_reports_total{{outcome=\"accepted\",auth_path=\"cloud_api_key\",\
         ingress_route=\"other\",model=\"{MODEL}\"}}"
    );
    assert_eq!(metric(&metrics, &accepted_series), Some(5.0), "{metrics}");
    assert_eq!(
        metric(&metrics, "inference_proxy_usage_report_outbox_available "),
        Some(1.0)
    );
    assert_eq!(
        metric(
            &metrics,
            &format!("inference_proxy_usage_report_outbox_pending{{model=\"{MODEL}\"}}")
        ),
        Some(0.0),
        "{metrics}"
    );

    // The bearer went out with every attempt and is in no log and no file.
    assert_eq!(
        *intake.0.bearers.lock().unwrap(),
        BTreeSet::from([format!("Bearer {USAGE_TOKEN}")])
    );
    for log in [first.log(), second.log(), third.log()] {
        assert!(!log.contains(USAGE_TOKEN));
        assert!(!log.contains(KEY));
    }
    for entry in std::fs::read_dir(dir.path()).unwrap() {
        let file = entry.unwrap().path();
        let bytes = std::fs::read(&file).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        for secret in [USAGE_TOKEN, KEY, "Bearer"] {
            assert!(!text.contains(secret), "{secret} in {}", file.display());
        }
    }
}

/// The overlap of a blue/green switch: the old process and the new one both
/// serve, with the same file as their outbox.
#[tokio::test]
async fn two_gateways_on_one_outbox_deliver_every_report_once() {
    let engine = engine().await;
    let (cloud, intake) = cloud_api(200, 20).await;
    let dir = tempfile::tempdir().unwrap();
    let outbox = dir.path().join("usage-outbox.db");
    let env = [(OUTBOX_PATH, outbox.to_str().unwrap())];
    let old = Gateway::start(&engine, &cloud, &env).await;
    let new = Gateway::start(&engine, &cloud, &env).await;

    // Requests to both at once, so that reports are written and claimed
    // from both sides at the same time.
    let requests = 20;
    tokio::join!(
        async {
            for _ in 0..requests {
                old.chat().await;
            }
        },
        async {
            for _ in 0..requests {
                new.chat().await;
            }
        }
    );
    eventually("every report is accepted", || {
        intake.accepted().len() >= 2 * requests && pending_rows(&outbox) == 0
    })
    .await;
    // A moment more, in which a second copy of a report would arrive.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let accepted = intake.accepted();
    let distinct: BTreeSet<&String> = accepted.iter().collect();
    assert_eq!(
        (accepted.len(), distinct.len()),
        (2 * requests, 2 * requests),
        "{accepted:?}"
    );
    assert_eq!(intake.attempts(), 2 * requests);
    // Each report was sent by one of the two, whichever claimed it.
    assert_eq!(
        old.logged(ACCEPTED).len() + new.logged(ACCEPTED).len(),
        2 * requests
    );
    // Neither ever found the file busy for longer than it would wait.
    for gateway in [&old, &new] {
        let log = gateway.log();
        assert!(!log.contains("Usage report outbox unavailable"), "{log}");
    }
}

/// The outbox must never be why a gateway does not serve: one whose file
/// cannot be opened starts, answers and bills, and says what it is missing.
#[tokio::test]
async fn a_gateway_whose_outbox_cannot_be_opened_serves_and_reports_all_the_same() {
    const UNAVAILABLE: &str = "Usage report outbox unavailable: reports are delivered from memory \
                               and not kept across a restart until it is back";
    let engine = engine().await;
    let (cloud, intake) = cloud_api(200, 0).await;
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("not-there-yet");
    let outbox = missing.join("usage-outbox.db");
    let mut gateway =
        Gateway::start(&engine, &cloud, &[(OUTBOX_PATH, outbox.to_str().unwrap())]).await;

    gateway.chat().await;
    eventually("the report is accepted", || intake.accepted().len() == 1).await;
    let metrics = gateway.metrics().await;
    assert_eq!(
        metric(&metrics, "inference_proxy_usage_report_outbox_available "),
        Some(0.0),
        "{metrics}"
    );
    assert!(
        metric(
            &metrics,
            "inference_proxy_usage_report_outbox_errors_total{op=\"open\"}"
        )
        .is_some_and(|errors| errors >= 1.0),
        "{metrics}"
    );
    assert!(metrics.contains("inference_proxy_usage_report_outbox_bypassed_total{reason="));
    let said = gateway.logged(UNAVAILABLE);
    assert_eq!(said.len(), 1, "{}", gateway.log());
    assert_eq!(said[0]["op"], "open");
    assert!(!outbox.exists());

    // The directory is made: the outbox comes up by itself, and the next
    // report is kept in it for as long as cloud-api does not take it.
    std::fs::create_dir(&missing).unwrap();
    let mut metrics = String::new();
    for _ in 0..800 {
        metrics = gateway.metrics().await;
        if metric(&metrics, "inference_proxy_usage_report_outbox_available ") == Some(1.0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(
        metric(&metrics, "inference_proxy_usage_report_outbox_available "),
        Some(1.0),
        "{metrics}"
    );
    assert_eq!(
        gateway.logged("Usage report outbox available again").len(),
        1
    );
    intake.answer(503);
    gateway.chat().await;
    eventually("the report is in the file", || pending_rows(&outbox) == 1).await;
    assert!(gateway.stop("-TERM").await.success(), "{}", gateway.log());
    assert_eq!(pending_rows(&outbox), 1);
    assert_eq!(gateway.logged(UNAVAILABLE).len(), 1);
}

/// Without the variable there is no outbox: no file, no series, no line.
/// And with it but without what sending takes, the file is not opened.
#[tokio::test]
async fn nothing_of_the_outbox_exists_unless_it_is_configured() {
    let engine = engine().await;
    let (cloud, intake) = cloud_api(200, 0).await;
    let dir = tempfile::tempdir().unwrap();

    let mut gateway = Gateway::start(&engine, &cloud, &[]).await;
    gateway.chat().await;
    eventually("the report is accepted", || intake.accepted().len() == 1).await;
    let metrics = gateway.metrics().await;
    assert!(!metrics.contains("outbox"), "{metrics}");
    // The series a process without any delivery setting has, and no other.
    let families: BTreeSet<&str> = metrics
        .lines()
        .filter(|line| !line.starts_with('#') && line.contains("usage_report"))
        .map(|line| line.split(['{', ' ']).next().unwrap())
        .collect();
    assert_eq!(
        families,
        BTreeSet::from([
            "inference_proxy_usage_report_duration_seconds",
            "inference_proxy_usage_report_duration_seconds_count",
            "inference_proxy_usage_report_duration_seconds_sum",
            "inference_proxy_usage_reports_total",
        ])
    );
    assert!(gateway.stop("-TERM").await.success());
    assert!(
        !gateway.log().to_lowercase().contains("outbox"),
        "{}",
        gateway.log()
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);

    // A path, and no usage token: nothing can be reported, so nothing is
    // kept, and the process starts and says so.
    let outbox = dir.path().join("usage-outbox.db");
    let mut gateway = Gateway::start(
        &engine,
        &cloud,
        &[
            (OUTBOX_PATH, outbox.to_str().unwrap()),
            ("CLOUD_API_USAGE_TOKEN", ""),
        ],
    )
    .await;
    gateway.chat().await;
    let not_opened = gateway.logged(
        "Usage report outbox not opened: CLOUD_API_URL and CLOUD_API_USAGE_TOKEN are both \
         needed to send what it holds",
    );
    assert_eq!(not_opened.len(), 1, "{}", gateway.log());
    assert_eq!(
        metric(
            &gateway.metrics().await,
            "inference_proxy_usage_report_outbox_available "
        ),
        Some(0.0)
    );
    assert!(gateway.stop("-TERM").await.success());
    assert!(!outbox.exists());
}
