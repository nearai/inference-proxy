//! The usage report outbox (`VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH`) in the
//! process the deploy starts: the built binary as a gateway, wiremock for the
//! engine and for cloud-api, and the outbox file read by a connection of the
//! test's own, the way an operator reads it.
//!
//! Everything a single delivery does is tested next to it
//! (`src/usage_report_outbox_tests.rs`). What is here needs real processes:
//! one that is killed, one that is stopped in good order, two on one file,
//! one whose file is damaged under it and one whose volume fills up, which
//! must both go on serving and never end with a signal.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use vllm_proxy_rs::*;

const MODEL: &str = "example/alpha";
const KEY: &str = "sk-live-customer";
const USAGE_TOKEN: &str = "usage-secret";
const OUTBOX_PATH: &str = "VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH";
const ACCEPTED: &str = "Direct-key usage report accepted by Cloud API";
const ENABLED: &str = "Usage report outbox enabled: a report is kept on disk until the billing \
                       API accepts it or refuses it for good";
const CLOSED: &str =
    "Usage report outbox closed for shutdown: what it holds is sent by the next process";
const UNAVAILABLE: &str = "Usage report outbox unavailable: reports are held in memory, sent \
                           from there, and written to the file when it is back";
const AVAILABLE_AGAIN: &str = "Usage report outbox available again";
/// The name of the thread that owns the outbox file.
const WRITER_THREAD: &str = "usage-outbox";

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
    /// The client of its customers: one, as a customer has.
    client: reqwest::Client,
    log: tempfile::NamedTempFile,
    _list: tempfile::NamedTempFile,
    /// Its working directory, its home and where its temporary files go:
    /// empty when it starts, so that whatever it creates without being told
    /// where is found here.
    home: tempfile::TempDir,
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
        Self::start_in(engine, cloud, env, None).await
    }

    /// `start`, with the binary run by `sh -c "<setup> && exec <binary>"`
    /// under `unshare` when `setup` is given: in mount and user namespaces of
    /// its own, where `setup` can mount a small volume for it.
    async fn start_in(
        engine: &MockServer,
        cloud: &MockServer,
        env: &[(&str, &str)],
        setup: Option<&str>,
    ) -> Gateway {
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
            let home = tempfile::tempdir().unwrap();
            let binary = env!("CARGO_BIN_EXE_vllm-proxy-rs");
            let mut command = match setup {
                Some(setup) => {
                    let mut command = std::process::Command::new("unshare");
                    command.args(["-Urm", "sh", "-c"]);
                    command.arg(format!("{setup} && exec {binary}"));
                    command
                }
                None => std::process::Command::new(binary),
            };
            command.env_clear();
            if setup.is_some() {
                // For the commands of `setup`; the binary reads none of it.
                command.env("PATH", std::env::var_os("PATH").unwrap_or_default());
            }
            let child = command
                .current_dir(home.path())
                .env("HOME", home.path())
                .env("TMPDIR", home.path())
                .envs([
                    ("TOKEN", "test-token"),
                    ("NON_TEE_DEPLOYMENT", "1"),
                    ("DEV", "1"),
                    ("GPU_NO_HW_MODE", "1"),
                    ("LOG_FORMAT", "json"),
                    ("LISTEN_ADDR", "127.0.0.1"),
                    ("CLOUD_API_AUTH_MAX_ATTEMPTS", "1"),
                    // The tests are one client, and some ask a lot of it.
                    ("RATE_LIMIT_PER_SECOND", "100000"),
                    ("RATE_LIMIT_BURST_SIZE", "100000"),
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
                client: reqwest::Client::new(),
                base: format!("http://127.0.0.1:{port}"),
                log,
                _list: list_file,
                home,
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

    /// What the log says about the outbox itself, with the time of each line.
    fn outbox_lines(&self) -> Vec<String> {
        self.log()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|line| {
                let message = line["fields"]["message"].as_str().unwrap_or_default();
                message.starts_with("Usage report outbox") || message.starts_with("The billing API")
            })
            .map(|line| format!("{} {}", line["timestamp"], line["fields"]))
            .collect()
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

    /// The names of the threads of the process.
    #[cfg(target_os = "linux")]
    fn threads(&self) -> BTreeSet<String> {
        std::fs::read_dir(format!("/proc/{}/task", self.child.id()))
            .unwrap()
            .filter_map(|task| std::fs::read_to_string(task.ok()?.path().join("comm")).ok())
            .map(|name| name.trim().to_string())
            .collect()
    }

    /// The files the process has open, and those it has mapped into its
    /// memory, by path, as far as they could be a database or belong to one.
    #[cfg(target_os = "linux")]
    fn database_files(&self) -> (BTreeSet<String>, BTreeSet<String>) {
        let pid = self.child.id();
        let of_a_database = |path: &String| {
            [".db", "-journal", "-wal", "-shm"]
                .iter()
                .any(|end| path.ends_with(end) || path.ends_with(&format!("{end} (deleted)")))
        };
        let open = std::fs::read_dir(format!("/proc/{pid}/fd"))
            .unwrap()
            .filter_map(|fd| std::fs::read_link(fd.ok()?.path()).ok())
            .map(|target| target.to_string_lossy().into_owned())
            .filter(of_a_database)
            .collect();
        let mapped = std::fs::read_to_string(format!("/proc/{pid}/maps"))
            .unwrap()
            .lines()
            .filter_map(|line| line.split_once('/').map(|(_, path)| format!("/{path}")))
            .filter(of_a_database)
            .collect();
        (open, mapped)
    }

    /// Whether the process is still running, and how it ended if not.
    fn ended(&mut self) -> Option<std::process::ExitStatus> {
        self.child.try_wait().unwrap()
    }

    /// One customer request, answered whatever becomes of its report.
    async fn chat(&self) {
        let response = self
            .client
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

    /// `/metrics`, once it says `done`, waited for at most 60 seconds.
    async fn metrics_until(&self, what: &str, done: impl Fn(&str) -> bool) -> String {
        let mut metrics = String::new();
        for _ in 0..2_400 {
            metrics = self.metrics().await;
            if done(&metrics) {
                return metrics;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("never happened: {what}: {metrics}\n{}", self.log());
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
/// are not there. The connection may write, as the `sqlite3` shell's does:
/// the journal a killed process left in the middle of a transaction is
/// played back by whoever opens the file next, and that takes writing.
fn read<T: rusqlite::types::FromSql + Default>(outbox: &Path, sql: &str) -> T {
    let Ok(conn) = rusqlite::Connection::open_with_flags(
        outbox,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
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
    // One report in flight at a time and a backoff of a second or more after
    // a failure, so that the moment a process is killed can be picked.
    let env = [
        (OUTBOX_PATH, outbox.to_str().unwrap()),
        ("VLLM_PROXY_USAGE_REPORT_MAX_IN_FLIGHT", "1"),
        ("VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS", "2000"),
        // Ignored with an outbox, and said to be.
        ("VLLM_PROXY_USAGE_REPORT_MAX_ATTEMPTS", "3"),
    ];

    // The first process serves three requests whose reports cloud-api does
    // not take.
    let mut first = Gateway::start(&engine, &cloud, &env).await;
    for _ in 0..3 {
        first.chat().await;
    }
    eventually("three reports in the file, each of them tried", || {
        read::<i64>(&outbox, "SELECT COUNT(*) FROM pending WHERE attempts > 0") == 3
    })
    .await;
    eventually(
        "no lease is held while they wait for their next attempt",
        || {
            read::<i64>(
                &outbox,
                "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL",
            ) == 0
        },
    )
    .await;
    let enabled = first.logged(ENABLED);
    assert_eq!(enabled.len(), 1, "{}", first.log());
    assert_eq!(enabled[0]["max_bytes"], 1u64 << 30);
    assert_eq!(enabled[0]["max_age_secs"], 7 * 24 * 3600);
    assert_eq!(enabled[0]["max_in_flight"], 1);
    assert_eq!(
        first
            .logged(
                "VLLM_PROXY_USAGE_REPORT_MAX_ATTEMPTS has no effect with a usage report outbox: \
                 a report is sent until it is accepted, refused for good or older than \
                 VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_AGE_SECS"
            )
            .len(),
        1,
        "{}",
        first.log()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&outbox).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
    // The file has a rollback journal beside it and nothing else, and the
    // process, which has both open, maps neither into its memory: a mapped
    // file that is damaged ends a process with a signal.
    let mut beside: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    beside.sort();
    assert_eq!(beside, ["usage-outbox.db", "usage-outbox.db-journal"]);
    #[cfg(target_os = "linux")]
    {
        // (The journal is open only while a transaction runs.)
        let (mut open, mapped) = first.database_files();
        open.remove(&format!("{}-journal", outbox.display()));
        assert_eq!(open, BTreeSet::from([outbox.display().to_string()]));
        assert_eq!(mapped, BTreeSet::new());
        assert!(
            first.threads().contains(WRITER_THREAD),
            "{:?}",
            first.threads()
        );
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
    let closed = second.logged(CLOSED);
    assert_eq!(closed.len(), 1, "{}", second.log());
    assert_eq!(closed[0]["in_outbox"], 5);
    assert_eq!(closed[0]["left_unwritten"], 0);
    assert_eq!(second.logged("Server shut down").len(), 1);
    assert_eq!(pending_rows(&outbox), 5);
    assert_eq!(
        read::<i64>(
            &outbox,
            "SELECT COUNT(*) FROM pending WHERE lease_owner IS NOT NULL"
        ),
        0
    );
    assert_eq!(intake.accepted(), Vec::<String>::new());
    // More than three attempts were made on the first three by now, and none
    // of them was given up on.
    assert!(intake.attempts() > 3, "{}", intake.attempts());
    assert_eq!(read::<i64>(&outbox, "SELECT COUNT(*) FROM rejected"), 0);

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
    for (series, value) in [
        ("inference_proxy_usage_report_outbox_available ", 1.0),
        ("inference_proxy_usage_report_outbox_full ", 0.0),
        ("inference_proxy_usage_report_outbox_billing_paused ", 0.0),
        ("inference_proxy_usage_report_outbox_unwritten ", 0.0),
        ("inference_proxy_usage_report_outbox_in_memory ", 0.0),
        (
            "inference_proxy_usage_report_outbox_rejected{reason=\"rejected\"} ",
            0.0,
        ),
        (
            "inference_proxy_usage_report_outbox_rejected{reason=\"max_age\"} ",
            0.0,
        ),
    ] {
        assert_eq!(metric(&metrics, series), Some(value), "{series}: {metrics}");
    }
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
    assert_eq!(gateway.logged(AVAILABLE_AGAIN).len(), 1);
    intake.answer(503);
    gateway.chat().await;
    eventually("the report is in the file", || pending_rows(&outbox) == 1).await;
    assert!(gateway.stop("-TERM").await.success(), "{}", gateway.log());
    assert_eq!(pending_rows(&outbox), 1);
    // Said when it began and when it ended, not once per try in between.
    assert_eq!(gateway.logged(UNAVAILABLE).len(), 1);
    assert_eq!(gateway.logged(AVAILABLE_AGAIN).len(), 1);
}

/// Without the variable there is no outbox: no thread, no file, no series, no
/// line. And with it but without what sending takes, the file is not opened.
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
    // No thread for the file, no file open or mapped, and nothing written
    // where a process puts what it was not told where to put.
    #[cfg(target_os = "linux")]
    {
        assert!(
            !gateway.threads().contains(WRITER_THREAD),
            "{:?}",
            gateway.threads()
        );
        assert_eq!(gateway.database_files(), (BTreeSet::new(), BTreeSet::new()));
    }
    assert!(gateway.stop("-TERM").await.success());
    assert!(
        !gateway.log().to_lowercase().contains("outbox"),
        "{}",
        gateway.log()
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    assert_eq!(std::fs::read_dir(gateway.home.path()).unwrap().count(), 0);

    // The same checks find the outbox of a process that has one: a thread by
    // that name, the file open, the file where it was told to be.
    let outbox = dir.path().join("usage-outbox.db");
    let mut with_outbox =
        Gateway::start(&engine, &cloud, &[(OUTBOX_PATH, outbox.to_str().unwrap())]).await;
    with_outbox.chat().await;
    eventually("the report is accepted", || intake.accepted().len() == 2).await;
    #[cfg(target_os = "linux")]
    {
        assert!(with_outbox.threads().contains(WRITER_THREAD));
        let (open, mapped) = with_outbox.database_files();
        assert!(open.contains(&outbox.display().to_string()), "{open:?}");
        assert_eq!(mapped, BTreeSet::new());
    }
    assert!(with_outbox.metrics().await.contains("outbox"));
    assert!(with_outbox.stop("-TERM").await.success());
    assert!(with_outbox.log().to_lowercase().contains("outbox"));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    assert_eq!(
        std::fs::read_dir(with_outbox.home.path()).unwrap().count(),
        0
    );
    for entry in std::fs::read_dir(dir.path()).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }

    // A path, and no usage token: nothing can be reported, so nothing is
    // kept, and the process starts and says so.
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
    #[cfg(target_os = "linux")]
    assert!(!gateway.threads().contains(WRITER_THREAD));
    assert!(gateway.stop("-TERM").await.success());
    assert!(!outbox.exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

/// What is done to the file under a running gateway, by a person, a script
/// or a disk. None of it may end the process or cost a request.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Damage {
    /// Cut to nothing, which to SQLite is a database without tables.
    Emptied,
    /// Cut in half.
    Truncated,
    /// Everything after its header overwritten.
    Scribbled,
    /// The file and its journal deleted.
    Deleted,
    /// Made unreadable and unwritable.
    #[cfg(unix)]
    Forbidden,
}

impl Damage {
    fn apply(self, outbox: &Path) {
        use std::io::{Read, Seek, SeekFrom};
        let open = || {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(outbox)
                .unwrap()
        };
        match self {
            Self::Emptied => open().set_len(0).unwrap(),
            Self::Truncated => {
                let file = open();
                let len = file.metadata().unwrap().len();
                file.set_len(len / 2 / 4096 * 4096 + 1000).unwrap();
            }
            Self::Scribbled => {
                // With the change counter of the header moved on, as any
                // writer would leave it: what the gateway remembers of the
                // file is then read again, and is not there.
                let mut file = open();
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
            Self::Deleted => {
                for entry in std::fs::read_dir(outbox.parent().unwrap()).unwrap() {
                    std::fs::remove_file(entry.unwrap().path()).unwrap();
                }
            }
            #[cfg(unix)]
            Self::Forbidden => {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(outbox, std::fs::Permissions::from_mode(0o000)).unwrap();
            }
        }
    }
}

#[tokio::test]
async fn a_file_damaged_under_a_running_gateway_never_ends_it_or_costs_a_request() {
    for damage in [
        Damage::Emptied,
        Damage::Truncated,
        Damage::Scribbled,
        Damage::Deleted,
        #[cfg(unix)]
        Damage::Forbidden,
    ] {
        // An engine of its own, so that the completions are numbered from 0.
        let engine = engine().await;
        let (cloud, intake) = cloud_api(503, 0).await;
        let dir = tempfile::tempdir().unwrap();
        let outbox = dir.path().join("usage-outbox.db");
        let env = [
            (OUTBOX_PATH, outbox.to_str().unwrap()),
            ("VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS", "100"),
        ];
        let mut gateway = Gateway::start(&engine, &cloud, &env).await;
        // A backlog cloud-api does not take, so that the file is in use.
        for _ in 0..40 {
            gateway.chat().await;
        }
        eventually("the backlog is in the file", || pending_rows(&outbox) == 40).await;

        damage.apply(&outbox);
        // Every request is answered, before and after cloud-api is back...
        for _ in 0..40 {
            gateway.chat().await;
        }
        intake.answer(200);
        for _ in 0..10 {
            gateway.chat().await;
        }
        // ... and the report of every request that completed since the file
        // was damaged is delivered.
        let since: BTreeSet<String> = (40..90).map(|n| format!("chatcmpl-{n}")).collect();
        let accepted = || intake.accepted().into_iter().collect::<BTreeSet<String>>();
        eventually(
            &format!("{damage:?}: the reports since are accepted"),
            || since.is_subset(&accepted()),
        )
        .await;
        assert_eq!(
            gateway.ended(),
            None,
            "{damage:?}: it ended: {}",
            gateway.log()
        );

        // What became of the backlog depends on what was done to the file,
        // and is said either way.
        let replaced = |why: &'static str| {
            let series =
                format!("inference_proxy_usage_report_outbox_replaced_total{{why=\"{why}\"}}");
            move |metrics: &str| metric(metrics, &series) == Some(1.0)
        };
        match damage {
            // The gateway still had all of it, and wrote the file back.
            Damage::Deleted => {
                gateway
                    .metrics_until("it is written back", replaced("deleted"))
                    .await;
                eventually("the whole backlog is accepted", || accepted().len() == 90).await;
            }
            // The database is gone, and a new one is started in the file.
            Damage::Emptied => {
                gateway
                    .metrics_until("a new one is started", replaced("lost"))
                    .await;
            }
            // What is left of it is moved aside for a person to look at.
            Damage::Truncated | Damage::Scribbled => {
                gateway
                    .metrics_until("it is moved aside", replaced("corrupt"))
                    .await;
                let aside = std::fs::read_dir(dir.path())
                    .unwrap()
                    .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                    .filter(|name| name.starts_with("usage-outbox.db.corrupt-"))
                    .count();
                assert!(aside >= 1, "{damage:?}");
            }
            // A file that is open stays usable, whatever its mode says.
            #[cfg(unix)]
            Damage::Forbidden => {
                eventually("the whole backlog is accepted", || accepted().len() == 90).await;
            }
        }
        let metrics = gateway
            .metrics_until("the file is in use again", |metrics| {
                metric(metrics, "inference_proxy_usage_report_outbox_available ") == Some(1.0)
            })
            .await;
        println!(
            "{damage:?}: alive, 90 requests answered, {} of 90 reports accepted ({} sends); {:?}",
            accepted().len(),
            intake.accepted().len(),
            metrics
                .lines()
                .filter(|line| line.contains("outbox_replaced_total{"))
                .collect::<Vec<_>>()
        );
        // And the outbox works: a report is kept in it again.
        #[cfg(unix)]
        if damage == Damage::Forbidden {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&outbox, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        intake.answer(503);
        gateway.chat().await;
        eventually(&format!("{damage:?}: a new report is kept"), || {
            read::<i64>(
                &outbox,
                "SELECT COUNT(*) FROM pending WHERE json_extract(body, '$.id') = 'chatcmpl-90'",
            ) == 1
        })
        .await;
        let stopped = gateway.stop("-TERM").await;
        assert!(
            stopped.success(),
            "{damage:?}: {stopped:?} {}",
            gateway.log()
        );
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(stopped.signal(), None);
        }
    }
}

/// `requests` customer requests, sixteen at a time, each of them answered.
async fn many_chats(gateway: &Gateway, requests: usize) {
    let left = AtomicUsize::new(requests);
    futures_util::future::join_all((0..16).map(|_| async {
        while left
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok()
        {
            gateway.chat().await;
        }
    }))
    .await;
}

/// The file reaches the size it was given (`MAX_BYTES`) on a volume that has
/// room: it takes no new report, and goes on with the ones it holds.
///
/// The smallest size there is takes more than ten thousand requests to fill,
/// which is minutes in a debug build: run by hand, like the measurements
/// below. What a delivery does with a file at its size is tested next to it
/// on every run (`a_file_at_its_size_keeps_sending_what_it_holds...`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ten thousand requests, run by hand"]
async fn a_gateway_whose_outbox_is_at_its_size_keeps_serving_and_sending() {
    let engine = engine().await;
    let (cloud, intake) = cloud_api(503, 10).await;
    let dir = tempfile::tempdir().unwrap();
    let outbox = dir.path().join("usage-outbox.db");
    // The smallest size there is, 4 MiB: some seven thousand reports.
    let max_bytes = 4 << 20;
    let env = [
        (OUTBOX_PATH, outbox.to_str().unwrap()),
        ("VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_BYTES", "4194304"),
        ("VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS", "100"),
    ];
    let mut gateway = Gateway::start(&engine, &cloud, &env).await;
    let gauge = |metrics: &str, name: &str| {
        metric(
            metrics,
            &format!("inference_proxy_usage_report_outbox_{name}"),
        )
    };

    // cloud-api is down and requests keep completing, until the file is
    // full. Every one of them is answered.
    let mut requests = 0;
    let mut metrics = String::new();
    while gauge(&metrics, "full ") != Some(1.0) {
        many_chats(&gateway, 1_000).await;
        requests += 1_000;
        assert!(requests <= 40_000, "the file never filled up: {metrics}");
        metrics = gateway.metrics().await;
    }
    many_chats(&gateway, 500).await;
    requests += 500;
    let metrics = gateway
        .metrics_until("every report is somewhere", |metrics| {
            gauge(metrics, "unwritten ") == Some(0.0)
        })
        .await;
    let in_memory = gauge(&metrics, "in_memory ").unwrap();
    let in_file = pending_rows(&outbox);
    println!(
        "at its size: {requests} requests answered; {in_file} reports in the file ({} bytes of \
         {max_bytes}), {in_memory} held in memory",
        std::fs::metadata(&outbox).unwrap().len()
    );
    assert!(in_memory >= 500.0, "{metrics}");
    assert_eq!(in_file as f64 + in_memory, requests as f64, "{metrics}");
    assert!(std::fs::metadata(&outbox).unwrap().len() <= max_bytes);
    // Full is not unavailable: the file works.
    assert_eq!(gauge(&metrics, "available "), Some(1.0), "{metrics}");
    assert_eq!(gauge(&metrics, "full "), Some(1.0), "{metrics}");
    assert_eq!(gateway.ended(), None);

    // cloud-api is back. The rows of the file are sent and removed while it
    // is still full, and in the end every report is accepted, once.
    intake.answer(200);
    let mut sent_while_full = false;
    let started_at = Instant::now();
    loop {
        let metrics = gateway.metrics().await;
        let rows = pending_rows(&outbox);
        sent_while_full |= gauge(&metrics, "full ") == Some(1.0) && rows < in_file;
        if rows == 0 && gauge(&metrics, "in_memory ") == Some(0.0) {
            break;
        }
        assert!(started_at.elapsed() < Duration::from_secs(240), "{metrics}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        sent_while_full,
        "rows of the file were sent only once it had room"
    );
    eventually("every report is accepted", || {
        intake.accepted().len() >= requests
    })
    .await;
    let accepted = intake.accepted();
    let distinct: BTreeSet<&String> = accepted.iter().collect();
    assert_eq!((accepted.len(), distinct.len()), (requests, requests));
    let metrics = gateway
        .metrics_until("it has room again", |metrics| {
            gauge(metrics, "full ") == Some(0.0)
        })
        .await;
    assert_eq!(gauge(&metrics, "available "), Some(1.0));
    // Said once each way, and never that it was unavailable.
    const FULL: &str = "Usage report outbox is full: new reports are held in memory and sent \
                        from there until it has room. What it holds is still sent";
    assert_eq!(
        gateway.logged(FULL).len(),
        1,
        "{:#?}",
        gateway.outbox_lines()
    );
    assert_eq!(
        gateway.logged("Usage report outbox has room again").len(),
        1
    );
    assert!(gateway.logged(UNAVAILABLE).is_empty());
    assert!(!gateway.log().contains("usage NOT billed"));
    assert!(gateway.stop("-TERM").await.success());
}

/// `unshare` can give a process a mount namespace of its own here (and so a
/// small volume, without root). Where it cannot, the tests that need a full
/// volume say so and are skipped.
fn can_mount_a_volume() -> bool {
    std::process::Command::new("unshare")
        .args(["-Urm", "sh", "-c"])
        .arg("d=$(mktemp -d) && mount -t tmpfs -o size=1m tmpfs \"$d\"")
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Something else fills the volume the outbox is on, to the last block, and
/// later frees it again. The gateway serves throughout, delivers from memory
/// what the file cannot take, and uses the file again when there is space,
/// without a restart.
#[tokio::test]
async fn a_gateway_whose_volume_fills_up_keeps_serving_and_recovers_when_there_is_space() {
    if !can_mount_a_volume() {
        eprintln!(
            "SKIPPED a_gateway_whose_volume_fills_up_keeps_serving_and_recovers_when_there_is_space: \
             `unshare -Urm` with a tmpfs mount does not work here, so no volume could be filled"
        );
        return;
    }
    let engine = engine().await;
    let (cloud, intake) = cloud_api(503, 0).await;
    // The volume: 2 MiB, mounted where only the gateway (and the helper
    // beside it) sees it, gone with them. The helper fills it up and frees
    // it when told to through files in `control`, which both sides see.
    let dir = tempfile::tempdir().unwrap();
    let volume = dir.path().join("volume");
    let control = dir.path().join("control");
    std::fs::create_dir(&volume).unwrap();
    std::fs::create_dir(&control).unwrap();
    let outbox = volume.join("usage-outbox.db");
    let (volume, control) = (volume.to_str().unwrap(), control.to_str().unwrap());
    let setup = format!(
        "mount -t tmpfs -o size=2m tmpfs '{volume}' && {{ ( \
           while kill -0 $$ 2>/dev/null; do \
             if [ -e '{control}/fill' ] && [ ! -e '{control}/filled' ]; then \
               dd if=/dev/zero of='{volume}/ballast' bs=4096 2>/dev/null; touch '{control}/filled'; fi; \
             if [ -e '{control}/free' ] && [ ! -e '{control}/freed' ]; then \
               rm -f '{volume}/ballast'; touch '{control}/freed'; fi; \
             sleep 0.1; \
           done ) > /dev/null 2>&1 & }}"
    );
    let env = [
        (OUTBOX_PATH, outbox.to_str().unwrap()),
        ("VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS", "100"),
    ];
    let mut gateway = Gateway::start_in(&engine, &cloud, &env, Some(&setup)).await;
    let gauge = |metrics: &str, name: &str| {
        metric(
            metrics,
            &format!("inference_proxy_usage_report_outbox_{name}"),
        )
    };
    let pending =
        |metrics: &str| gauge(metrics, &format!("pending{{model=\"{MODEL}\"}}")).unwrap_or(0.0);
    let told = |what: &str| {
        std::fs::write(Path::new(control).join(what), b"").unwrap();
    };
    let done = |what: &'static str| {
        let done = Path::new(control).join(what);
        move || done.exists()
    };

    // A backlog in the file, while cloud-api is down.
    many_chats(&gateway, 300).await;
    gateway
        .metrics_until("the backlog is in the file", |metrics| {
            pending(metrics) == 300.0
        })
        .await;

    // The volume fills up. Requests go on being answered.
    told("fill");
    eventually("the volume is full", done("filled")).await;
    many_chats(&gateway, 400).await;
    let metrics = gateway
        .metrics_until("the reports are held in memory", |metrics| {
            gauge(metrics, "in_memory ").is_some_and(|held| held >= 390.0)
                && gauge(metrics, "full ") == Some(1.0)
        })
        .await;
    println!(
        "volume full: 700 requests answered; available {:?}, full {:?}, rows {}, in memory {:?}, \
         unwritten {:?}",
        gauge(&metrics, "available "),
        gauge(&metrics, "full "),
        pending(&metrics),
        gauge(&metrics, "in_memory "),
        gauge(&metrics, "unwritten "),
    );
    assert_eq!(gateway.ended(), None, "{}", gateway.log());

    // Nothing can be written to the file now. The first write tried on it
    // (a lease, an outcome) says so, and from then on it is tried again
    // every five seconds, each time in vain. That is one state: `available`
    // never goes back to 1 while the volume is full, and the log says it
    // once, however long it lasts. (A transaction that writes nothing still
    // succeeds on a full volume, and proves nothing.)
    let mut seen_unavailable = false;
    let watched_from = Instant::now();
    while watched_from.elapsed() < Duration::from_secs(12) {
        let metrics = gateway.metrics().await;
        let available = gauge(&metrics, "available ") == Some(1.0);
        assert!(
            !(seen_unavailable && available),
            "available again on a volume that is full: {:#?}",
            gateway.outbox_lines()
        );
        seen_unavailable |= !available;
        assert_eq!(gauge(&metrics, "full "), Some(1.0), "{metrics}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        gateway.logged(UNAVAILABLE).len() <= 1,
        "{:#?}",
        gateway.outbox_lines()
    );
    assert!(
        gateway.logged(AVAILABLE_AGAIN).is_empty(),
        "{:#?}",
        gateway.outbox_lines()
    );

    // cloud-api is back while the volume is still full: what memory holds
    // is delivered from there.
    intake.answer(200);
    eventually("the reports held in memory are accepted", || {
        intake.accepted().len() >= 400
    })
    .await;
    let metrics = gateway.metrics().await;
    println!(
        "volume full, cloud-api back: {} of 700 accepted; available {:?}, full {:?}, rows {}, in \
         memory {:?}",
        intake.accepted().len(),
        gauge(&metrics, "available "),
        gauge(&metrics, "full "),
        pending(&metrics),
        gauge(&metrics, "in_memory "),
    );
    assert_eq!(gateway.ended(), None, "{}", gateway.log());

    // Space returns. The file is found usable again within its retry
    // interval, without a restart, and every report is accepted, once.
    told("free");
    eventually("there is space again", done("freed")).await;
    let freed_at = Instant::now();
    eventually("every report is accepted", || {
        intake.accepted().len() >= 700
    })
    .await;
    let metrics = gateway
        .metrics_until("the file is in use again and has room", |metrics| {
            gauge(metrics, "available ") == Some(1.0)
                && gauge(metrics, "full ") == Some(0.0)
                && gauge(metrics, "in_memory ") == Some(0.0)
                && pending(metrics) == 0.0
        })
        .await;
    println!(
        "space back: everything accepted and the file in use {:?} later",
        freed_at.elapsed()
    );
    let accepted = intake.accepted();
    let distinct: BTreeSet<&String> = accepted.iter().collect();
    assert_eq!(distinct.len(), 700);
    // A report whose outcome could not be written while the volume was full
    // may have been sent once more; cloud-api tells those apart.
    assert!(
        accepted.len() <= 700 + 40,
        "{} sends for 700 reports",
        accepted.len()
    );
    assert_eq!(gauge(&metrics, "unwritten "), Some(0.0));
    // And a new report is kept in the file again.
    intake.answer(503);
    gateway.chat().await;
    gateway
        .metrics_until("a new report is kept", |metrics| pending(metrics) == 1.0)
        .await;
    assert!(!gateway.log().contains("usage NOT billed"));
    // The rows of the file could not be leased once cloud-api was back and
    // the volume still full: unavailable, said once, and once that it ended.
    assert_eq!(
        (
            gateway.logged(UNAVAILABLE).len(),
            gateway.logged(AVAILABLE_AGAIN).len()
        ),
        (1, 1),
        "{:#?}",
        gateway.outbox_lines()
    );
    let stopped = gateway.stop("-TERM").await;
    assert!(stopped.success(), "{stopped:?} {}", gateway.log());
}

/// The outbox is given a volume far smaller than the size its file may grow
/// to (a megabyte, and nothing said about the size: a gigabyte). Left to
/// fill the volume, the file could never be written to again, not even to
/// remove a report cloud-api accepted, because every change needs room for
/// its journal first. So the file is kept to what the volume has room for:
/// it is full by its own size, the volume keeps room, and what the file
/// holds is sent as soon as cloud-api takes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_gateway_on_a_volume_smaller_than_its_outbox_may_grow_keeps_sending_what_it_holds() {
    if !can_mount_a_volume() {
        eprintln!(
            "SKIPPED a_gateway_on_a_volume_smaller_than_its_outbox_may_grow_keeps_sending_what_it_holds: \
             `unshare -Urm` with a tmpfs mount does not work here, so there is no small volume"
        );
        return;
    }
    const LIMITED: &str = "The volume of the usage report outbox has no room for \
                           VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_BYTES and the journal beside it: \
                           the file is kept to what the volume has room for";
    const FULL: &str = "Usage report outbox is full: new reports are held in memory and sent \
                        from there until it has room. What it holds is still sent";
    let engine = engine().await;
    let (cloud, intake) = cloud_api(503, 0).await;
    let dir = tempfile::tempdir().unwrap();
    let volume = dir.path().join("volume");
    std::fs::create_dir(&volume).unwrap();
    let outbox = volume.join("usage-outbox.db");
    let setup = format!(
        "mount -t tmpfs -o size=1m tmpfs '{}'",
        volume.to_str().unwrap()
    );
    let env = [
        (OUTBOX_PATH, outbox.to_str().unwrap()),
        ("VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS", "100"),
    ];
    let mut gateway = Gateway::start_in(&engine, &cloud, &env, Some(&setup)).await;
    let gauge = |metrics: &str, name: &str| {
        metric(
            metrics,
            &format!("inference_proxy_usage_report_outbox_{name}"),
        )
    };
    let pending =
        |metrics: &str| gauge(metrics, &format!("pending{{model=\"{MODEL}\"}}")).unwrap_or(0.0);

    // It says so when it opens the file: three quarters of the megabyte are
    // the file's, the rest is the journal's.
    let limited = gateway.logged(LIMITED);
    assert_eq!(limited.len(), 1, "{:#?}", gateway.outbox_lines());
    assert_eq!(limited[0]["max_bytes"], 1_u64 << 30);
    let limited_to = limited[0]["limited_to"].as_u64().unwrap();
    assert!(
        (512 << 10..=768 << 10).contains(&limited_to),
        "{limited_to}"
    );

    // cloud-api is down and requests keep completing, until the file is
    // full: at its own size, on a volume that still has room.
    let mut requests = 0;
    let mut metrics = String::new();
    while gauge(&metrics, "full ") != Some(1.0) {
        many_chats(&gateway, 500).await;
        requests += 500;
        assert!(requests <= 20_000, "the file never filled up: {metrics}");
        metrics = gateway.metrics().await;
    }
    many_chats(&gateway, 300).await;
    requests += 300;
    let metrics = gateway
        .metrics_until("every report is somewhere", |metrics| {
            gauge(metrics, "unwritten ") == Some(0.0)
        })
        .await;
    let (in_file, in_memory, bytes) = (
        pending(&metrics),
        gauge(&metrics, "in_memory ").unwrap(),
        gauge(&metrics, "bytes ").unwrap(),
    );
    println!(
        "a 1 MiB volume: {requests} requests answered; {in_file} reports in the file ({bytes} \
         bytes, it may have {limited_to}), {in_memory} held in memory"
    );
    assert!(in_file >= 500.0 && in_memory >= 300.0, "{metrics}");
    assert_eq!(in_file + in_memory, requests as f64, "{metrics}");
    assert!(bytes <= limited_to as f64, "{metrics}");
    // Full is not unavailable: the file works.
    assert_eq!(gauge(&metrics, "available "), Some(1.0), "{metrics}");
    assert_eq!(gateway.ended(), None, "{}", gateway.log());

    // cloud-api is back. What the file holds is sent, and what memory
    // holds: every report is accepted, once, and nothing had to be done to
    // the volume for it.
    intake.answer(200);
    eventually("every report is accepted", || {
        intake.accepted().len() >= requests
    })
    .await;
    let accepted = intake.accepted();
    let distinct: BTreeSet<&String> = accepted.iter().collect();
    assert_eq!((accepted.len(), distinct.len()), (requests, requests));
    let metrics = gateway
        .metrics_until("the file is empty and has room", |metrics| {
            gauge(metrics, "full ") == Some(0.0)
                && gauge(metrics, "in_memory ") == Some(0.0)
                && pending(metrics) == 0.0
        })
        .await;
    assert_eq!(gauge(&metrics, "available "), Some(1.0), "{metrics}");
    // It was never unavailable, it was full once, and it said once what
    // its volume is.
    assert!(
        gateway.logged(UNAVAILABLE).is_empty(),
        "{:#?}",
        gateway.outbox_lines()
    );
    assert_eq!(
        (gateway.logged(FULL).len(), gateway.logged(LIMITED).len()),
        (1, 1),
        "{:#?}",
        gateway.outbox_lines()
    );
    assert!(!gateway.log().contains("usage NOT billed"));
    assert!(gateway.stop("-TERM").await.success());
}

// ---------------------------------------------------------------------------
// Measurements, run by hand:
//   cargo test --release --test usage_outbox -- --ignored --nocapture --test-threads 1
// ---------------------------------------------------------------------------

/// What `kill -9` costs: a gateway that serves as fast as eight clients ask
/// is killed at some moment, again and again, and the reports of the requests
/// it had answered are counted in its file.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement, run by hand"]
async fn what_a_kill_loses() {
    let rounds: usize = std::env::var("ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let engine = engine().await;
    // cloud-api takes nothing: every report stays where it was written.
    let (cloud, _intake) = cloud_api(503, 0).await;
    let (mut answered_in_all, mut lost_in_all, mut most_lost) = (0, 0, 0);
    for round in 0..rounds {
        let dir = tempfile::tempdir().unwrap();
        let outbox = dir.path().join("usage-outbox.db");
        let mut gateway =
            Gateway::start(&engine, &cloud, &[(OUTBOX_PATH, outbox.to_str().unwrap())]).await;
        let gateway_ref = &gateway;
        // When each request was answered, as its client saw it.
        let answered = Mutex::new(Vec::<Instant>::new());
        let stop = std::sync::atomic::AtomicBool::new(false);
        let kill_after = Duration::from_millis(400 + 97 * (round as u64 % 7));
        let pid = gateway.child.id().to_string();
        let killed_at = {
            let clients = futures_util::future::join_all((0..8).map(|_| async {
                while !stop.load(Ordering::SeqCst) {
                    let response = gateway_ref
                        .client
                        .post(format!(
                            "{}{}",
                            gateway_ref.base,
                            routes::ROUTE_CHAT_COMPLETIONS
                        ))
                        .bearer_auth(KEY)
                        .json(&json!({
                            "model": MODEL,
                            "messages": [{"role": "user", "content": "hello"}]
                        }))
                        .send()
                        .await;
                    let Ok(response) = response else { break };
                    if response.status().is_success() && response.bytes().await.is_ok() {
                        answered.lock().unwrap().push(Instant::now());
                    }
                }
            }));
            let killer = async {
                tokio::time::sleep(kill_after).await;
                let killed_at = Instant::now();
                std::process::Command::new("kill")
                    .args(["-KILL", &pid])
                    .status()
                    .unwrap();
                stop.store(true, Ordering::SeqCst);
                killed_at
            };
            tokio::join!(clients, killer).1
        };
        while gateway.ended().is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let answered = answered.into_inner().unwrap();
        let before_kill = answered.iter().filter(|at| **at <= killed_at).count();
        let in_last = |window: Duration| {
            answered
                .iter()
                .filter(|at| **at <= killed_at && killed_at.duration_since(**at) <= window)
                .count()
        };
        let in_file = pending_rows(&outbox) as usize;
        let lost = before_kill.saturating_sub(in_file);
        println!(
            "round {round:>2}: killed after {kill_after:?}; answered before the kill {before_kill} \
             (in its last 50 ms {}, last 100 ms {}), reports in the file {in_file}, lost {lost}",
            in_last(Duration::from_millis(50)),
            in_last(Duration::from_millis(100)),
        );
        answered_in_all += before_kill;
        lost_in_all += lost;
        most_lost = most_lost.max(lost);
        // The file a killed process leaves is a database the next one opens.
        assert_eq!(
            read::<String>(&outbox, "PRAGMA integrity_check"),
            "ok",
            "round {round}"
        );
    }
    println!(
        "{rounds} kills: {answered_in_all} requests answered before them, {lost_in_all} reports \
         lost ({:.3} %), at most {most_lost} by one kill",
        100.0 * lost_in_all as f64 / answered_in_all.max(1) as f64
    );
}

/// Two gateways on one file, serving as fast as their clients ask, with a
/// cloud-api that fails now and then, and one of them stopped and started
/// again in the middle of it: every report is accepted, and how many were
/// sent twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement, run by hand"]
async fn two_gateways_under_load() {
    let requests: usize = std::env::var("REQUESTS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5_000);
    let engine = engine().await;
    let (cloud, intake) = cloud_api(200, 5).await;
    let dir = tempfile::tempdir().unwrap();
    let outbox = dir.path().join("usage-outbox.db");
    let env = [(OUTBOX_PATH, outbox.to_str().unwrap())];
    let old = Gateway::start(&engine, &cloud, &env).await;
    let mut new = Gateway::start(&engine, &cloud, &env).await;
    let started_at = Instant::now();

    // cloud-api fails for 300 ms out of every two seconds.
    let flapping = async {
        for _ in 0..1_000 {
            tokio::time::sleep(Duration::from_millis(1_700)).await;
            intake.answer(503);
            tokio::time::sleep(Duration::from_millis(300)).await;
            intake.answer(200);
        }
    };
    let load = async {
        tokio::join!(many_chats(&old, requests), async {
            many_chats(&new, requests / 2).await;
            // A deploy: the new one is stopped in good order and started
            // again, on the same file, while the old one serves.
            assert!(new.stop("-TERM").await.success());
            new = Gateway::start(&engine, &cloud, &env).await;
            many_chats(&new, requests - requests / 2).await;
        });
    };
    tokio::select! {
        _ = flapping => unreachable!("the load ends first"),
        _ = load => {}
    }
    let served_in = started_at.elapsed();
    intake.answer(200);
    eventually("every report is accepted", || {
        intake.accepted().into_iter().collect::<BTreeSet<_>>().len() >= 2 * requests
            && pending_rows(&outbox) == 0
    })
    .await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    let accepted = intake.accepted();
    let distinct: BTreeSet<&String> = accepted.iter().collect();
    println!(
        "two gateways on one file, {} requests in {served_in:?} ({:.0} a second): {} reports \
         accepted, {} sent twice, {} attempts in all; rows left {}, rejected {}; the file was \
         unavailable to: old {}, new {}",
        2 * requests,
        2.0 * requests as f64 / served_in.as_secs_f64(),
        distinct.len(),
        accepted.len() - distinct.len(),
        intake.attempts(),
        pending_rows(&outbox),
        read::<i64>(&outbox, "SELECT COUNT(*) FROM rejected"),
        old.logged(UNAVAILABLE).len(),
        new.logged(UNAVAILABLE).len(),
    );
    assert_eq!(distinct.len(), 2 * requests);
    assert_eq!(read::<String>(&outbox, "PRAGMA integrity_check"), "ok");
}
