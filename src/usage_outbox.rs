//! Durable, retrying delivery for direct-key usage reports.
//!
//! Each event is fsynced as one JSON file before the asynchronous sender is
//! notified. Cloud API de-duplicates by `(organization_id, inference id)`, so
//! replay after a crash between its commit and our local unlink is safe.

use rand::RngExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::task::AbortHandle;
use tracing::{error, info, warn};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

const EVENT_VERSION: u8 = 1;

#[derive(Clone)]
pub struct UsageOutboxConfig {
    pub directory: PathBuf,
    pub cloud_api_url: String,
    pub cloud_api_usage_token: String,
    pub request_timeout: Duration,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    /// Test-only convenience for ephemeral spools. Production must leave this false.
    pub delete_on_drop: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UsageEvent {
    version: u8,
    pub body: serde_json::Value,
    pub request_id: Option<String>,
    pub auth_path: String,
    pub ingress_route: String,
}

impl UsageEvent {
    pub fn new(
        body: serde_json::Value,
        request_id: Option<String>,
        auth_path: &str,
        ingress_route: &str,
    ) -> Self {
        Self {
            version: EVENT_VERSION,
            body,
            request_id,
            auth_path: auth_path.to_string(),
            ingress_route: ingress_route.to_string(),
        }
    }

    fn string_field(&self, name: &str) -> &str {
        self.body
            .get(name)
            .and_then(|value| value.as_str())
            .unwrap_or("")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueResult {
    Stored,
    AlreadyPending,
}

#[derive(Debug, thiserror::Error)]
pub enum EnqueueError {
    #[error("usage event is missing a non-empty {0} string")]
    InvalidEvent(&'static str),
    #[error("failed to persist usage event: {0}")]
    Io(#[from] io::Error),
}

pub struct UsageOutbox {
    directory: PathBuf,
    notify: Arc<Notify>,
    pending: Arc<AtomicUsize>,
    worker_abort: Mutex<Option<AbortHandle>>,
    delete_on_drop: bool,
}

impl UsageOutbox {
    pub fn open(config: UsageOutboxConfig, http_client: reqwest::Client) -> io::Result<Arc<Self>> {
        std::fs::create_dir_all(&config.directory)?;
        #[cfg(unix)]
        std::fs::set_permissions(&config.directory, std::fs::Permissions::from_mode(0o700))?;

        let initial_pending = pending_paths(&config.directory)?.len();
        let notify = Arc::new(Notify::new());
        let pending = Arc::new(AtomicUsize::new(initial_pending));
        record_pending(initial_pending);

        let outbox = Arc::new(Self {
            directory: config.directory.clone(),
            notify: notify.clone(),
            pending: pending.clone(),
            worker_abort: Mutex::new(None),
            delete_on_drop: config.delete_on_drop,
        });
        let worker = Worker {
            directory: config.directory,
            http_client,
            url: format!(
                "{}/v1/internal/usage",
                config.cloud_api_url.trim_end_matches('/')
            ),
            auth: format!("Bearer {}", config.cloud_api_usage_token),
            request_timeout: config.request_timeout,
            initial_backoff: config.initial_backoff.max(Duration::from_millis(1)),
            max_backoff: config
                .max_backoff
                .max(config.initial_backoff)
                .max(Duration::from_millis(1)),
            notify,
            pending,
        };
        let task = tokio::spawn(worker.run());
        *outbox
            .worker_abort
            .lock()
            .expect("usage outbox lock poisoned") = Some(task.abort_handle());
        Ok(outbox)
    }

    /// Atomically persist one event. The final file is never overwritten: a
    /// duplicate provider ID maps to the same name and reuses the pending copy.
    pub fn enqueue(&self, event: UsageEvent) -> Result<EnqueueResult, EnqueueError> {
        let organization_id = required_string(&event.body, "organization_id")?;
        let inference_id = required_string(&event.body, "id")?;
        let mut hasher = Sha256::new();
        hasher.update(organization_id.as_bytes());
        hasher.update([0]);
        hasher.update(inference_id.as_bytes());
        let final_path = self
            .directory
            .join(format!("{}.json", hex::encode(hasher.finalize())));

        if final_path.exists() {
            record_enqueue("already_pending");
            self.notify.notify_one();
            return Ok(EnqueueResult::AlreadyPending);
        }

        let temporary_path = self
            .directory
            .join(format!(".{}.tmp", uuid::Uuid::new_v4()));
        let persist_result = persist_noclobber(&temporary_path, &final_path, &event);
        let _ = std::fs::remove_file(&temporary_path);

        match persist_result {
            Ok(true) => {
                let pending = self.pending.fetch_add(1, Ordering::Relaxed) + 1;
                record_pending(pending);
                record_enqueue("stored");
                self.notify.notify_one();
                Ok(EnqueueResult::Stored)
            }
            Ok(false) => {
                record_enqueue("already_pending");
                self.notify.notify_one();
                Ok(EnqueueResult::AlreadyPending)
            }
            Err(error) => {
                record_enqueue("write_error");
                Err(EnqueueError::Io(error))
            }
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }
}

impl Drop for UsageOutbox {
    fn drop(&mut self) {
        if let Some(abort) = self
            .worker_abort
            .lock()
            .expect("usage outbox lock poisoned")
            .take()
        {
            abort.abort();
        }
        if self.delete_on_drop {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }
}

fn required_string<'a>(
    body: &'a serde_json::Value,
    field: &'static str,
) -> Result<&'a str, EnqueueError> {
    body.get(field)
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or(EnqueueError::InvalidEvent(field))
}

fn persist_noclobber(
    temporary_path: &Path,
    final_path: &Path,
    event: &UsageEvent,
) -> io::Result<bool> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(temporary_path)?;
    serde_json::to_writer(&mut file, event).map_err(io::Error::other)?;
    file.write_all(b"\n")?;
    file.sync_all()?;

    match std::fs::hard_link(temporary_path, final_path) {
        Ok(()) => {
            sync_directory(final_path.parent().expect("outbox event has parent"))?;
            Ok(true)
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error),
    }
}

fn sync_directory(directory: &Path) -> io::Result<()> {
    File::open(directory)?.sync_all()
}

fn pending_paths(directory: &Path) -> io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            paths.push(path);
        }
    }
    paths.sort_unstable();
    Ok(paths)
}

fn decrement_pending(pending: &AtomicUsize) -> usize {
    let previous = pending
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            Some(value.saturating_sub(1))
        })
        .expect("saturating pending update always succeeds");
    previous.saturating_sub(1)
}

fn record_enqueue(outcome: &'static str) {
    metrics::counter!("inference_proxy_usage_outbox_enqueues_total", "outcome" => outcome)
        .increment(1);
}

fn record_pending(pending: usize) {
    metrics::gauge!("inference_proxy_usage_outbox_pending").set(pending as f64);
}

#[derive(Clone)]
struct Worker {
    directory: PathBuf,
    http_client: reqwest::Client,
    url: String,
    auth: String,
    request_timeout: Duration,
    initial_backoff: Duration,
    max_backoff: Duration,
    notify: Arc<Notify>,
    pending: Arc<AtomicUsize>,
}

enum DrainResult {
    Idle,
    Progress,
    Blocked,
}

impl Worker {
    async fn run(self) {
        let mut backoff = self.initial_backoff;
        loop {
            match self.drain_once().await {
                DrainResult::Idle => {
                    backoff = self.initial_backoff;
                    self.notify.notified().await;
                }
                DrainResult::Progress => {
                    backoff = self.initial_backoff;
                }
                DrainResult::Blocked => {
                    let delay = if backoff <= Duration::from_millis(1) {
                        backoff
                    } else {
                        let upper = backoff.as_millis().min(u64::MAX as u128) as u64;
                        Duration::from_millis(rand::rng().random_range(1..=upper))
                    };
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        _ = self.notify.notified() => {}
                    }
                    backoff = backoff.saturating_mul(2).min(self.max_backoff);
                }
            }
        }
    }

    async fn drain_once(&self) -> DrainResult {
        let paths = match pending_paths(&self.directory) {
            Ok(paths) => paths,
            Err(error) => {
                error!(error = %error, "Failed to scan durable usage outbox");
                return DrainResult::Blocked;
            }
        };
        self.pending.store(paths.len(), Ordering::Relaxed);
        record_pending(paths.len());
        if paths.is_empty() {
            return DrainResult::Idle;
        }

        let mut made_progress = false;
        for path in paths {
            let bytes = match tokio::fs::read(&path).await {
                Ok(bytes) => bytes,
                Err(error) => {
                    warn!(error = %error, "Failed to read durable usage event");
                    return DrainResult::Blocked;
                }
            };
            let event: UsageEvent = match serde_json::from_slice::<UsageEvent>(&bytes) {
                Ok(event) if event.version == EVENT_VERSION => event,
                Ok(event) => {
                    if !self
                        .quarantine(
                            &path,
                            &format!("unsupported version {}", event.version),
                            QuarantineKind::Corrupt,
                        )
                        .await
                    {
                        return DrainResult::Blocked;
                    }
                    made_progress = true;
                    continue;
                }
                Err(error) => {
                    if !self
                        .quarantine(&path, &error.to_string(), QuarantineKind::Corrupt)
                        .await
                    {
                        return DrainResult::Blocked;
                    }
                    made_progress = true;
                    continue;
                }
            };

            let started_at = Instant::now();
            let mut request = self
                .http_client
                .post(&self.url)
                .header("authorization", &self.auth)
                .json(&event.body)
                .timeout(self.request_timeout);
            if let Some(request_id) = event.request_id.as_deref() {
                request = request.header("x-request-id", request_id);
            }
            match request.send().await {
                Ok(response) if response.status().is_success() => {
                    let elapsed = started_at.elapsed();
                    record_delivery(&event, DeliveryOutcome::Accepted, elapsed);
                    if let Err(error) = tokio::fs::remove_file(&path).await {
                        warn!(error = %error, "Cloud API accepted usage but outbox unlink failed; replay is idempotent");
                        return DrainResult::Blocked;
                    }
                    if let Err(error) = sync_directory(&self.directory) {
                        warn!(error = %error, "Failed to fsync usage outbox after accepted-event unlink; replay remains idempotent");
                    }
                    let pending = decrement_pending(&self.pending);
                    record_pending(pending);
                    info!(
                        request_id = %event.request_id.as_deref().unwrap_or(""),
                        org_id = %event.string_field("organization_id"),
                        workspace_id = %event.string_field("workspace_id"),
                        api_key_id = %event.string_field("api_key_id"),
                        model = %event.string_field("model"),
                        status = %response.status(),
                        duration_ms = elapsed.as_millis() as u64,
                        auth_path = %event.auth_path,
                        ingress_route = %event.ingress_route,
                        "Durable usage report accepted by Cloud API"
                    );
                    made_progress = true;
                }
                Ok(response) if response.status() == StatusCode::BAD_REQUEST => {
                    let elapsed = started_at.elapsed();
                    let outcome = DeliveryOutcome::from_status(response.status());
                    record_delivery(&event, outcome, elapsed);
                    let status = response.status();
                    // Cloud API parsed this authenticated request and rejected
                    // its immutable payload. Retrying can never help and must
                    // not head-of-line block every later billing event.
                    if !self
                        .quarantine(
                            &path,
                            &format!("Cloud API permanently rejected event with {status}"),
                            QuarantineKind::Rejected,
                        )
                        .await
                    {
                        return DrainResult::Blocked;
                    }
                    error!(
                        request_id = %event.request_id.as_deref().unwrap_or(""),
                        org_id = %event.string_field("organization_id"),
                        workspace_id = %event.string_field("workspace_id"),
                        api_key_id = %event.string_field("api_key_id"),
                        model = %event.string_field("model"),
                        status = %status,
                        duration_ms = elapsed.as_millis() as u64,
                        auth_path = %event.auth_path,
                        ingress_route = %event.ingress_route,
                        outcome = outcome.as_label(),
                        "Cloud API permanently rejected durable usage event; quarantined for inspection"
                    );
                    made_progress = true;
                }
                Ok(response) => {
                    let elapsed = started_at.elapsed();
                    let outcome = DeliveryOutcome::from_status(response.status());
                    record_delivery(&event, outcome, elapsed);
                    warn!(
                        request_id = %event.request_id.as_deref().unwrap_or(""),
                        org_id = %event.string_field("organization_id"),
                        workspace_id = %event.string_field("workspace_id"),
                        api_key_id = %event.string_field("api_key_id"),
                        model = %event.string_field("model"),
                        status = %response.status(),
                        duration_ms = elapsed.as_millis() as u64,
                        auth_path = %event.auth_path,
                        ingress_route = %event.ingress_route,
                        outcome = outcome.as_label(),
                        "Durable usage delivery failed; event retained for retry"
                    );
                    return DrainResult::Blocked;
                }
                Err(error) => {
                    let elapsed = started_at.elapsed();
                    let outcome = DeliveryOutcome::from_error(&error);
                    record_delivery(&event, outcome, elapsed);
                    warn!(
                        request_id = %event.request_id.as_deref().unwrap_or(""),
                        org_id = %event.string_field("organization_id"),
                        workspace_id = %event.string_field("workspace_id"),
                        api_key_id = %event.string_field("api_key_id"),
                        model = %event.string_field("model"),
                        error = %error,
                        duration_ms = elapsed.as_millis() as u64,
                        auth_path = %event.auth_path,
                        ingress_route = %event.ingress_route,
                        outcome = outcome.as_label(),
                        "Durable usage delivery failed; event retained for retry"
                    );
                    return DrainResult::Blocked;
                }
            }
        }

        if made_progress {
            DrainResult::Progress
        } else {
            DrainResult::Idle
        }
    }

    async fn quarantine(&self, path: &Path, reason: &str, kind: QuarantineKind) -> bool {
        let quarantine =
            path.with_extension(format!("{}-{}", kind.as_extension(), uuid::Uuid::new_v4()));
        match tokio::fs::rename(path, &quarantine).await {
            Ok(()) => {
                if let Err(error) = sync_directory(&self.directory) {
                    warn!(error = %error, "Failed to fsync usage outbox after quarantine rename");
                }
                let pending = decrement_pending(&self.pending);
                record_pending(pending);
                metrics::counter!(
                    "inference_proxy_usage_outbox_quarantined_total",
                    "reason" => kind.as_label()
                )
                .increment(1);
                error!(
                    reason,
                    quarantine_reason = kind.as_label(),
                    "Quarantined durable usage event"
                );
                true
            }
            Err(error) => {
                error!(reason, error = %error, quarantine_reason = kind.as_label(), "Failed to quarantine durable usage event");
                false
            }
        }
    }
}

#[derive(Clone, Copy)]
enum QuarantineKind {
    Corrupt,
    Rejected,
}

impl QuarantineKind {
    fn as_extension(self) -> &'static str {
        match self {
            Self::Corrupt => "invalid",
            Self::Rejected => "rejected",
        }
    }

    fn as_label(self) -> &'static str {
        match self {
            Self::Corrupt => "corrupt",
            Self::Rejected => "http_400",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeliveryOutcome {
    Accepted,
    Http4xx,
    Http5xx,
    HttpOther,
    Timeout,
    ConnectError,
    TransportError,
}

impl DeliveryOutcome {
    fn from_status(status: StatusCode) -> Self {
        if status.is_success() {
            Self::Accepted
        } else if status.is_client_error() {
            Self::Http4xx
        } else if status.is_server_error() {
            Self::Http5xx
        } else {
            Self::HttpOther
        }
    }

    fn from_error(error: &reqwest::Error) -> Self {
        if error.is_timeout() {
            Self::Timeout
        } else if error.is_connect() {
            Self::ConnectError
        } else {
            Self::TransportError
        }
    }

    fn as_label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Http4xx => "http_4xx",
            Self::Http5xx => "http_5xx",
            Self::HttpOther => "http_other",
            Self::Timeout => "timeout",
            Self::ConnectError => "connect_error",
            Self::TransportError => "transport_error",
        }
    }
}

fn record_delivery(event: &UsageEvent, outcome: DeliveryOutcome, duration: Duration) {
    let labels = [
        ("outcome", outcome.as_label().to_string()),
        ("auth_path", event.auth_path.clone()),
        ("ingress_route", event.ingress_route.clone()),
    ];
    metrics::counter!("inference_proxy_usage_reports_total", &labels).increment(1);
    metrics::histogram!("inference_proxy_usage_report_duration_seconds", &labels)
        .record(duration.as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(directory: &Path, cloud_api_url: String) -> UsageOutboxConfig {
        UsageOutboxConfig {
            directory: directory.to_path_buf(),
            cloud_api_url,
            cloud_api_usage_token: "test-usage-token".to_string(),
            request_timeout: Duration::from_millis(100),
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(20),
            delete_on_drop: false,
        }
    }

    fn event(id: &str) -> UsageEvent {
        UsageEvent::new(
            serde_json::json!({
                "type": "chat_completion",
                "model": "test-model",
                "input_tokens": 3,
                "output_tokens": 1,
                "id": id,
                "organization_id": "00000000-0000-0000-0000-000000000001",
                "workspace_id": "00000000-0000-0000-0000-000000000002",
                "api_key_id": "00000000-0000-0000-0000-000000000003"
            }),
            Some("test-request-id".to_string()),
            "cloud_api_key",
            "canonical",
        )
    }

    async fn wait_for<F: Fn() -> bool>(predicate: F) {
        for _ in 0..100 {
            if predicate() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("condition did not become true");
    }

    #[tokio::test]
    async fn persists_and_replays_after_reporter_restart() {
        let directory = tempfile::tempdir().unwrap();
        let failing = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/internal/usage"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&failing)
            .await;

        let first = UsageOutbox::open(
            config(directory.path(), failing.uri()),
            reqwest::Client::new(),
        )
        .unwrap();
        assert_eq!(
            first.enqueue(event("replay-id")).unwrap(),
            EnqueueResult::Stored
        );
        wait_for(|| first.pending_count() == 1).await;
        for _ in 0..100 {
            if !failing.received_requests().await.unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!failing.received_requests().await.unwrap().is_empty());
        drop(first);

        let accepting = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/internal/usage"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&accepting)
            .await;
        let second = UsageOutbox::open(
            config(directory.path(), accepting.uri()),
            reqwest::Client::new(),
        )
        .unwrap();
        wait_for(|| second.pending_count() == 0).await;

        let requests = accepting.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["id"], "replay-id");
        assert_eq!(body["input_tokens"], 3);
        assert_eq!(
            requests[0].headers["authorization"],
            "Bearer test-usage-token"
        );
    }

    #[tokio::test]
    async fn duplicate_pending_event_is_not_written_twice() {
        let directory = tempfile::tempdir().unwrap();
        let failing = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&failing)
            .await;
        let outbox = UsageOutbox::open(
            config(directory.path(), failing.uri()),
            reqwest::Client::new(),
        )
        .unwrap();

        assert_eq!(
            outbox.enqueue(event("same-id")).unwrap(),
            EnqueueResult::Stored
        );
        assert_eq!(
            outbox.enqueue(event("same-id")).unwrap(),
            EnqueueResult::AlreadyPending
        );
        assert_eq!(pending_paths(directory.path()).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn permanently_rejected_event_does_not_block_later_usage() {
        let directory = tempfile::tempdir().unwrap();
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/internal/usage"))
            .and(body_partial_json(serde_json::json!({"id": "bad-id"})))
            .respond_with(ResponseTemplate::new(400))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/internal/usage"))
            .and(body_partial_json(serde_json::json!({"id": "good-id"})))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        let outbox = UsageOutbox::open(
            config(directory.path(), server.uri()),
            reqwest::Client::new(),
        )
        .unwrap();

        outbox.enqueue(event("bad-id")).unwrap();
        outbox.enqueue(event("good-id")).unwrap();
        wait_for(|| outbox.pending_count() == 0).await;

        server.verify().await;
        let rejected = std::fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension.to_string_lossy().starts_with("rejected-"))
            })
            .count();
        assert_eq!(rejected, 1);
    }

    #[test]
    fn decrement_pending_saturates_at_zero() {
        let pending = AtomicUsize::new(0);
        assert_eq!(decrement_pending(&pending), 0);
        assert_eq!(pending.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn delivery_outcome_classification_is_stable() {
        assert_eq!(
            DeliveryOutcome::from_status(StatusCode::OK),
            DeliveryOutcome::Accepted
        );
        assert_eq!(
            DeliveryOutcome::from_status(StatusCode::UNAUTHORIZED),
            DeliveryOutcome::Http4xx
        );
        assert_eq!(
            DeliveryOutcome::from_status(StatusCode::SERVICE_UNAVAILABLE),
            DeliveryOutcome::Http5xx
        );
        assert_eq!(
            DeliveryOutcome::from_status(StatusCode::TEMPORARY_REDIRECT),
            DeliveryOutcome::HttpOther
        );
    }
}
