//! Only final billing receipts are durable here. Cloud API owns all balances and holds.
//! Files contain the shared typed usage contract, never tokens, prompts or media.
use super::{Job, UsageReportDelivery};
use crate::auth::{AuthPath, IngressRouteKind, RequestSource};
use crate::proxy::UsageReporter;
use anyhow::{ensure, Context};
use cloud_platform_events::InternalUsageReport;
use futures_util::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use uuid::Uuid;

const MAX_RECEIPT_BYTES: u64 = 256 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: u8,
    emitted_at: chrono::DateTime<chrono::Utc>,
    usage: InternalUsageReport,
    ingress_route: IngressRouteKind,
    request_id: Option<String>,
    model_label: Option<String>,
}

pub(super) struct Outbox {
    directory: PathBuf,
    client: reqwest::Client,
    url: Option<String>,
    token: Option<String>,
    ready: AtomicBool,
    queued_scan: AtomicBool,
    notify: Notify,
    replay_lock: tokio::sync::Mutex<()>,
    model_labels: std::sync::Mutex<std::collections::HashMap<String, &'static str>>,
}

impl Outbox {
    fn open(
        directory: PathBuf,
        client: reqwest::Client,
        url: Option<String>,
        token: Option<String>,
    ) -> anyhow::Result<Self> {
        fs::create_dir_all(&directory).context("create receipt outbox")?;
        let outbox = Self {
            directory,
            client,
            url,
            token,
            ready: AtomicBool::new(false),
            queued_scan: AtomicBool::new(false),
            notify: Notify::new(),
            replay_lock: tokio::sync::Mutex::new(()),
            model_labels: Default::default(),
        };
        outbox.check_storage()?;
        Ok(outbox)
    }

    fn check_storage(&self) -> anyhow::Result<()> {
        let path = self.directory.join(format!("{}.tmp", Uuid::new_v4()));
        let result = (|| -> anyhow::Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            // Empty inode creation can succeed under ENOSPC while receipt writes fail.
            file.write_all(&vec![0; MAX_RECEIPT_BYTES as usize])?;
            file.sync_all()?;
            Ok(())
        })();
        let removed = fs::remove_file(path);
        result?;
        removed?;
        File::open(&self.directory)?.sync_all()?;
        self.ready.store(true, Ordering::Release);
        Ok(())
    }

    fn persist(&self, job: &Job) -> anyhow::Result<()> {
        if let Some(label) = job.reporter.model_label {
            self.model_labels
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(label.into(), label);
        }
        // Decode before storing, so an arbitrary request body cannot reach this spool.
        let usage: InternalUsageReport = serde_json::from_slice(&job.body)?;
        let id = Uuid::parse_str(
            usage
                .admission_id
                .as_deref()
                .context("missing admission id")?,
        )?;
        ensure!(
            job.reporter.budget_report_id.as_deref() == usage.admission_id.as_deref(),
            "receipt admission mismatch"
        );
        let destination = self.directory.join(format!("{id}.json"));
        if destination.exists() {
            let old: Receipt = serde_json::from_slice(&fs::read(&destination)?)?;
            ensure!(
                serde_json::to_value(old.usage)? == serde_json::to_value(&usage)?,
                "conflicting receipt for admission"
            );
            return Ok(());
        }
        let receipt = Receipt {
            version: 1,
            emitted_at: job.emitted_at,
            usage,
            ingress_route: job.reporter.request_source.ingress_route,
            request_id: job.reporter.request_id.clone(),
            model_label: job.reporter.model_label.map(str::to_owned),
        };
        let body = serde_json::to_vec(&receipt)?;
        ensure!(body.len() as u64 <= MAX_RECEIPT_BYTES, "receipt too large");
        let temporary = self.directory.join(format!("{id}.{}.tmp", Uuid::new_v4()));
        let result = (|| -> anyhow::Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(&body)?;
            file.sync_all()?;
            // Link publishes a complete, fsynced receipt without overwriting an existing one.
            // A crash before this point leaves only an uncommitted .tmp file.
            fs::hard_link(&temporary, &destination)?;
            File::open(&self.directory)?.sync_all()?;
            Ok(())
        })();
        let _ = fs::remove_file(temporary);
        result?;
        self.notify.notify_one();
        Ok(())
    }

    pub(super) fn pending(&self) -> usize {
        // Used only by shutdown/drain, never on the inference path. An unreadable
        // spool is not reported as drained.
        fs::read_dir(&self.directory)
            .map(|entries| {
                entries
                    .filter(|entry| {
                        entry
                            .as_ref()
                            .map(|e| e.path().extension().is_some_and(|s| s == "json"))
                            .unwrap_or(true)
                    })
                    .count()
            })
            .unwrap_or(1)
    }

    async fn replay(&self, delivery: &Arc<UsageReportDelivery>) -> anyhow::Result<usize> {
        let _replay = self.replay_lock.lock().await;
        let mut paths = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            if entry.file_type()?.is_file() && entry.path().extension().is_some_and(|s| s == "json")
            {
                paths.push(entry.path());
            }
        }
        metrics::gauge!("inference_proxy_usage_outbox_pending").set(paths.len() as f64);
        // Every pending file gets a turn, including after a poison receipt. A bounded number
        // of sends run at once; failed receipts stay on disk for the next pass or restart.
        let acknowledged = stream::iter(paths)
            .map(|path| async move {
                let result: anyhow::Result<bool> = async {
                    ensure!(
                        fs::metadata(&path)?.len() <= MAX_RECEIPT_BYTES,
                        "receipt too large"
                    );
                    let receipt: Receipt = serde_json::from_slice(&fs::read(&path)?)?;
                    ensure!(receipt.version == 1, "unsupported receipt version");
                    let admission_id = receipt
                        .usage
                        .admission_id
                        .clone()
                        .context("missing admission id")?;
                    ensure!(
                        path.file_stem().and_then(|s| s.to_str()) == Some(&admission_id),
                        "receipt filename mismatch"
                    );
                    let value = serde_json::to_value(&receipt.usage)?;
                    let reporter = UsageReporter {
                        budget_report_id: Some(admission_id),
                        http_client: self.client.clone(),
                        cloud_api_url: self.url.clone(),
                        cloud_api_usage_token: self.token.clone(),
                        model_name: value["model"].as_str().context("missing model")?.into(),
                        org_id: Some(receipt.usage.organization_id.clone()),
                        workspace_id: Some(receipt.usage.workspace_id.clone()),
                        api_key_id: Some(receipt.usage.api_key_id.clone()),
                        discount_to_user: None,
                        model_label: receipt.model_label.as_ref().and_then(|label| {
                            self.model_labels
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .get(label)
                                .copied()
                        }),
                        request_id: receipt.request_id,
                        request_source: RequestSource {
                            auth_path: AuthPath::CloudApiKey,
                            ingress_route: receipt.ingress_route,
                        },
                        delivery: Arc::clone(delivery),
                    };
                    let mut job = Job {
                        reporter,
                        body: serde_json::to_vec(&receipt.usage)?.into(),
                        completed_at: Instant::now(),
                        emitted_at: receipt.emitted_at,
                    };
                    delivery
                        .sender
                        .prepare(&mut job)
                        .map_err(|_| anyhow::anyhow!("receipt refused by transport"))?;
                    delivery.waiting_gauge(job.reporter.model_label, super::log::Step::None);
                    delivery.in_flight_gauge(job.reporter.model_label, super::log::Step::Up);
                    let accepted = delivery.deliver(&job).await;
                    delivery.in_flight_gauge(job.reporter.model_label, super::log::Step::Down);
                    if accepted {
                        fs::remove_file(path)?;
                        File::open(&self.directory)?.sync_all()?;
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                }
                .await;
                match result {
                    Ok(accepted) => usize::from(accepted),
                    Err(_) => {
                        tracing::warn!("Durable usage receipt remains pending");
                        0
                    }
                }
            })
            .buffer_unordered(if delivery.policy.max_in_flight == 0 {
                8
            } else {
                delivery.policy.max_in_flight.min(8)
            })
            .fold(0, |total, next| async move { total + next })
            .await;
        delivery.idle.notify_waiters();
        Ok(acknowledged)
    }
}

impl UsageReportDelivery {
    /// Enable before accepting direct-key traffic. The directory must survive process/container replacement.
    pub fn enable_outbox(
        self: &Arc<Self>,
        directory: PathBuf,
        client: reqwest::Client,
        url: Option<String>,
        token: Option<String>,
    ) -> anyhow::Result<()> {
        let outbox = Arc::new(Outbox::open(directory, client, url, token)?);
        ensure!(
            self.outbox.set(outbox.clone()).is_ok(),
            "receipt outbox already enabled"
        );
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                {
                    let Some(delivery) = weak.upgrade() else {
                        break;
                    };
                    if !outbox.ready.load(Ordering::Acquire) && outbox.check_storage().is_err() {
                        tracing::error!("Receipt outbox unavailable; further admission disabled");
                    }
                    if outbox.replay(&delivery).await.is_err() {
                        outbox.ready.store(false, Ordering::Release);
                        tracing::error!("Receipt outbox scan failed; further admission disabled");
                    }
                }
                tokio::select! {
                    _ = outbox.notify.notified() => {},
                    _ = tokio::time::sleep(Duration::from_secs(5)) => {},
                }
            }
        });
        Ok(())
    }

    pub(crate) fn admission_ready(&self) -> bool {
        self.outbox
            .get()
            .is_some_and(|outbox| outbox.ready.load(Ordering::Acquire))
    }

    pub(crate) fn register_outbox_model_labels(&self, labels: impl Iterator<Item = &'static str>) {
        if let Some(outbox) = self.outbox.get() {
            outbox
                .model_labels
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(labels.map(|label| (label.into(), label)));
        }
    }

    pub(super) fn persist_admitted(self: &Arc<Self>, job: &Job) -> anyhow::Result<()> {
        let outbox = self.outbox.get().context("receipt outbox unavailable")?;
        let result = outbox.persist(job);
        if result.is_err() {
            outbox.ready.store(false, Ordering::Release);
        } else if !outbox.queued_scan.swap(true, Ordering::AcqRel) {
            // Keep the delivery alive for the first attempt even if the handler
            // owned the last AppState (e.g. a completed one-shot connection).
            let delivery = Arc::clone(self);
            let outbox = Arc::clone(outbox);
            tokio::spawn(async move {
                let _ = outbox.replay(&delivery).await;
                outbox.queued_scan.store(false, Ordering::Release);
            });
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proxy::UsageReportOutcome;
    use crate::usage_report::{AttemptLog, SendOutcome, TransportKind, UsageSender};
    use futures_util::future::BoxFuture;
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Ledger {
        written: Mutex<BTreeSet<String>>,
        replies_lost: AtomicBool,
    }
    impl UsageSender for Ledger {
        fn kind(&self) -> TransportKind {
            TransportKind::Http
        }
        fn send<'a>(
            &'a self,
            job: &'a Job,
            _: Duration,
        ) -> BoxFuture<'a, (SendOutcome, AttemptLog)> {
            Box::pin(async move {
                let usage: InternalUsageReport = serde_json::from_slice(&job.body).unwrap();
                self.written
                    .lock()
                    .unwrap()
                    .insert(usage.admission_id.unwrap());
                let lost = self.replies_lost.load(Ordering::Acquire);
                (
                    if lost {
                        SendOutcome::Retryable("timeout")
                    } else {
                        SendOutcome::Accepted
                    },
                    AttemptLog {
                        answer: if lost {
                            Err("lost ACK".into())
                        } else {
                            Ok(reqwest::StatusCode::OK)
                        },
                        outcome: if lost {
                            UsageReportOutcome::Timeout
                        } else {
                            UsageReportOutcome::Accepted
                        },
                        duration: Duration::ZERO,
                    },
                )
            })
        }
    }
    fn job(delivery: &Arc<UsageReportDelivery>) -> Job {
        let admission = Uuid::new_v4().to_string();
        let reporter = UsageReporter {
            budget_report_id: Some(admission.clone()),
            http_client: reqwest::Client::new(),
            cloud_api_url: None,
            cloud_api_usage_token: Some("secret-not-for-disk".into()),
            model_name: "test".into(),
            org_id: Some(Uuid::new_v4().to_string()),
            workspace_id: Some(Uuid::new_v4().to_string()),
            api_key_id: Some(Uuid::new_v4().to_string()),
            discount_to_user: None,
            model_label: None,
            request_id: None,
            request_source: RequestSource {
                auth_path: AuthPath::CloudApiKey,
                ingress_route: IngressRouteKind::Missing,
            },
            delivery: Arc::clone(delivery),
        };
        Job { body: serde_json::to_vec(&serde_json::json!({
            "organization_id": reporter.org_id, "workspace_id": reporter.workspace_id, "api_key_id": reporter.api_key_id,
            "admission_id":admission, "type":"chat_completion", "model":"test", "id":"provider-id", "input_tokens":2,"output_tokens":3
        })).unwrap().into(), reporter, completed_at: Instant::now(), emitted_at: chrono::Utc::now() }
    }

    #[tokio::test]
    async fn restart_replays_persisted_receipts_and_lost_ack_without_double_billing() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Arc::new(Ledger::default());
        ledger.replies_lost.store(true, Ordering::Release);
        let delivery = UsageReportDelivery::with_sender(Default::default(), ledger.clone());
        let outbox =
            Outbox::open(directory.path().into(), reqwest::Client::new(), None, None).unwrap();
        let receipt = job(&delivery);
        outbox.persist(&receipt).unwrap();
        outbox.persist(&receipt).unwrap();
        let path = directory.path().join(format!(
            "{}.json",
            receipt.reporter.budget_report_id.as_ref().unwrap()
        ));
        assert!(!fs::read_to_string(&path)
            .unwrap()
            .contains("secret-not-for-disk"));
        assert_eq!(outbox.replay(&delivery).await.unwrap(), 0);
        assert!(path.exists(), "lost ACK must retain durable receipt");
        drop(outbox);
        drop(receipt);
        drop(delivery);
        let restarted = UsageReportDelivery::with_sender(Default::default(), ledger.clone());
        let restored =
            Outbox::open(directory.path().into(), reqwest::Client::new(), None, None).unwrap();
        ledger.replies_lost.store(false, Ordering::Release);
        assert_eq!(restored.replay(&restarted).await.unwrap(), 1);
        assert_eq!(ledger.written.lock().unwrap().len(), 1);
        assert!(!path.exists());
        assert_eq!(restored.replay(&restarted).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn poison_receipt_cannot_starve_valid_receipts_and_failed_persistence_disables_admission()
    {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Arc::new(Ledger::default());
        let delivery = UsageReportDelivery::with_sender(Default::default(), ledger.clone());
        let outbox = Arc::new(
            Outbox::open(directory.path().into(), reqwest::Client::new(), None, None).unwrap(),
        );
        assert!(delivery.outbox.set(outbox.clone()).is_ok());
        for _ in 0..101 {
            fs::write(
                directory.path().join(format!("{}.json", Uuid::new_v4())),
                b"invalid",
            )
            .unwrap();
        }
        let receipt = job(&delivery);
        outbox.persist(&receipt).unwrap();
        assert_eq!(outbox.replay(&delivery).await.unwrap(), 1);
        assert_eq!(ledger.written.lock().unwrap().len(), 1);
        assert!(delivery.admission_ready());
        fs::remove_dir_all(directory.path()).unwrap();
        assert!(delivery.persist_admitted(&job(&delivery)).is_err());
        assert!(!delivery.admission_ready());
    }
}
