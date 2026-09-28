//! Per-tick publisher: reads every replica's engine load, then builds one
//! signed frame per replica, all sharing one `seq`.
//!
//! Frames carry IDs and numbers only. Nothing here logs replica URLs, frame
//! contents or key material.

use std::sync::Arc;
use std::time::Duration;

use super::report::{self, Engine, Envelope, Lifecycle, Load, ReplicaReport};
use super::report_key::ReportKey;
use super::sglang::{self, ReplicaLoad};

/// Consecutive failed reads (after at least one success) that mark a replica
/// `Unhealthy`.
pub const UNHEALTHY_AFTER_FAILURES: u32 = 3;

struct ReplicaSlot {
    id: String,
    base_url: String,
    last: Option<ReplicaLoad>,
    failures: u32,
    lifecycle: Lifecycle,
}

pub struct Publisher {
    host_id: String,
    model: String,
    key: Arc<ReportKey>,
    slots: Vec<ReplicaSlot>,
    seq: u64,
    client: reqwest::Client,
    read_timeout: Duration,
}

impl Publisher {
    /// `replicas` is `(replica_id, base_url)` in backend-pool order. Reads are
    /// bounded by 4/5 of `interval` so a tick never overruns its slot.
    pub fn new(
        host_id: String,
        model: String,
        key: Arc<ReportKey>,
        replicas: Vec<(String, String)>,
        client: reqwest::Client,
        interval: Duration,
    ) -> Self {
        let slots = replicas
            .into_iter()
            .map(|(id, base_url)| ReplicaSlot {
                id,
                base_url,
                last: None,
                failures: 0,
                lifecycle: Lifecycle::Warming,
            })
            .collect();
        Self {
            host_id,
            model,
            key,
            slots,
            seq: 0,
            client,
            read_timeout: interval * 4 / 5,
        }
    }

    /// Read every replica concurrently, then build one frame per replica
    /// sharing one seq. `now_ms` is called once, after all reads complete, so
    /// `reported_at_ms` is the seal time.
    pub async fn tick(
        &mut self,
        inflight: impl Fn(usize) -> u32,
        now_ms: impl Fn() -> u64,
    ) -> Vec<(String, Envelope)> {
        let reads = futures_util::future::join_all(
            self.slots
                .iter()
                .map(|slot| sglang::read_replica(&self.client, &slot.base_url, self.read_timeout)),
        )
        .await;
        let reported_at_ms = now_ms();

        self.seq += 1;
        let seq = self.seq;
        let mut frames = Vec::with_capacity(self.slots.len());
        for (i, (slot, read)) in self.slots.iter_mut().zip(reads).enumerate() {
            let (load, limits, engine_version, engine_sampled_at_ms) = match read {
                Some(read) => {
                    let recovering = slot.lifecycle == Lifecycle::Unhealthy
                        || (slot.lifecycle == Lifecycle::Warming && slot.failures > 0);
                    if recovering {
                        tracing::info!(replica_id = %slot.id, "replica state: replica ready");
                    }
                    slot.failures = 0;
                    slot.lifecycle = Lifecycle::Ready;
                    let out = (
                        read.load.clone(),
                        read.limits.clone(),
                        read.engine_version.clone(),
                        Some(read.sampled_at_ms),
                    );
                    slot.last = Some(read);
                    out
                }
                None => {
                    slot.failures = slot.failures.saturating_add(1);
                    if slot.failures == UNHEALTHY_AFTER_FAILURES {
                        tracing::warn!(
                            replica_id = %slot.id,
                            failures = slot.failures,
                            "replica state: replica unhealthy"
                        );
                    }
                    if slot.last.is_none() {
                        slot.lifecycle = Lifecycle::Warming;
                    } else if slot.failures >= UNHEALTHY_AFTER_FAILURES {
                        slot.lifecycle = Lifecycle::Unhealthy;
                    }
                    match &slot.last {
                        Some(last) => (
                            Load::default(),
                            last.limits.clone(),
                            last.engine_version.clone(),
                            Some(last.sampled_at_ms),
                        ),
                        None => (Load::default(), Default::default(), None, None),
                    }
                }
            };
            let report = ReplicaReport {
                schema: 1,
                host_id: self.host_id.clone(),
                replica_id: slot.id.clone(),
                boot_id: self.key.boot_id.clone(),
                seq,
                engine_sampled_at_ms,
                reported_at_ms,
                lifecycle_state: slot.lifecycle,
                model: self.model.clone(),
                engine: Engine::Sglang,
                engine_version,
                limits,
                load,
                proxy_inflight: inflight(i),
                report_key_id: self.key.key_id.clone(),
            };
            frames.push((
                slot.id.clone(),
                report::seal(&report, self.key.signing_key()),
            ));
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replica_state::report::open;
    use std::time::Instant;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const NOW: u64 = 1_790_000_005_000;

    fn body(ts: f64) -> serde_json::Value {
        serde_json::json!({"version":"0.5.9","loads":[{"timestamp":ts,"num_running_reqs":3,
            "num_waiting_reqs":1,"num_waiting_uncached_tokens":100,"num_used_tokens":10,
            "max_total_num_tokens":100,"gen_throughput":5.0,"cache_hit_rate":0.5,
            "max_running_requests":16}]})
    }

    async fn serve(server: &MockServer, status: u16, ts: f64) {
        Mock::given(method("GET"))
            .and(path("/v1/loads"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body(ts)))
            .mount(server)
            .await;
    }

    fn publisher(key: ReportKey, replicas: &[(&str, &str)], interval: Duration) -> Publisher {
        Publisher::new(
            "host-a".into(),
            "m".into(),
            Arc::new(key),
            replicas
                .iter()
                .map(|(id, url)| (id.to_string(), url.to_string()))
                .collect(),
            reqwest::Client::new(),
            interval,
        )
    }

    fn open_all(
        frames: &[(String, Envelope)],
        vk: &ed25519_dalek::VerifyingKey,
    ) -> Vec<ReplicaReport> {
        frames
            .iter()
            .map(|(id, env)| {
                let r = open(env, vk).expect("frame verifies under the report key");
                assert_eq!(&r.replica_id, id);
                r
            })
            .collect()
    }

    #[tokio::test]
    async fn one_signed_frame_per_replica_sharing_one_seq_per_tick() {
        let (s1, s2) = (MockServer::start().await, MockServer::start().await);
        serve(&s1, 200, 1_790_000_000.0).await;
        serve(&s2, 200, 1_790_000_000.0).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let boot_id = key.boot_id.clone();
        let key_id = key.key_id.clone();
        let mut p = publisher(
            key,
            &[("r1", &s1.uri()), ("r2", &s2.uri())],
            Duration::from_millis(500),
        );

        let frames = p.tick(|i| 10 + i as u32, || NOW).await;
        let ids: Vec<_> = frames.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["r1", "r2"]);
        let reports = open_all(&frames, &vk);
        for (i, r) in reports.iter().enumerate() {
            assert_eq!(r.seq, 1);
            assert_eq!(r.schema, 1);
            assert_eq!(r.lifecycle_state, Lifecycle::Ready);
            assert_eq!(r.boot_id, boot_id);
            assert_eq!(r.report_key_id, key_id);
            assert_eq!(r.host_id, "host-a");
            assert_eq!(r.reported_at_ms, NOW);
            assert!(r.reported_at_ms >= r.engine_sampled_at_ms.unwrap());
            assert_eq!(r.proxy_inflight, 10 + i as u32);
            assert_eq!(r.load.running, Some(3));
        }

        let reports = open_all(&p.tick(|_| 0, || NOW + 500).await, &vk);
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|r| r.seq == 2));
    }

    #[tokio::test]
    async fn engine_timestamp_passes_through_even_when_stale() {
        let s = MockServer::start().await;
        serve(&s, 200, 1000.0).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[("r1", &s.uri())], Duration::from_millis(500));
        let r = &open_all(&p.tick(|_| 0, || NOW).await, &vk)[0];
        assert_eq!(r.engine_sampled_at_ms, Some(1_000_000));
        assert_eq!(r.reported_at_ms, NOW);
        assert_eq!(r.lifecycle_state, Lifecycle::Ready);
    }

    #[tokio::test]
    async fn failed_read_keeps_last_sample_time_and_nulls_load() {
        let s = MockServer::start().await;
        serve(&s, 200, 1_790_000_000.0).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[("r1", &s.uri())], Duration::from_millis(500));
        let ok = &open_all(&p.tick(|_| 0, || NOW).await, &vk)[0];
        assert_eq!(ok.load.running, Some(3));
        assert!(ok.reported_at_ms >= ok.engine_sampled_at_ms.unwrap());

        s.reset().await;
        serve(&s, 500, 0.0).await;
        let r = &open_all(&p.tick(|_| 0, || NOW + 500).await, &vk)[0];
        assert_eq!(r.load, Load::default());
        assert_eq!(r.engine_sampled_at_ms, Some(1_790_000_000_000));
        assert_eq!(r.limits.max_running, Some(16));
        assert_eq!(r.engine_version.as_deref(), Some("0.5.9"));
        assert_eq!(r.lifecycle_state, Lifecycle::Ready);

        let r = &open_all(&p.tick(|_| 0, || NOW + 1000).await, &vk)[0];
        assert_eq!(r.lifecycle_state, Lifecycle::Ready);
        let r = &open_all(&p.tick(|_| 0, || NOW + 1500).await, &vk)[0];
        assert_eq!(r.lifecycle_state, Lifecycle::Unhealthy);
        assert_eq!(r.load, Load::default());

        s.reset().await;
        serve(&s, 200, 1_790_000_002.0).await;
        let r = &open_all(&p.tick(|_| 0, || NOW + 2000).await, &vk)[0];
        assert_eq!(r.lifecycle_state, Lifecycle::Ready);
        assert_eq!(r.engine_sampled_at_ms, Some(1_790_000_002_000));
        assert!(r.reported_at_ms >= r.engine_sampled_at_ms.unwrap());
        assert_eq!(r.load.running, Some(3));
        assert_eq!(r.seq, 5);
    }

    #[tokio::test]
    async fn never_reachable_replica_is_warming_with_null_sample_time() {
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(
            key,
            &[("r1", "http://127.0.0.1:1")],
            Duration::from_millis(500),
        );
        let mut last = None;
        for n in 0..5 {
            last = Some(open_all(&p.tick(|_| 0, || NOW + n * 500).await, &vk).remove(0));
        }
        let r = last.unwrap();
        assert_eq!(r.lifecycle_state, Lifecycle::Warming);
        assert_eq!(r.engine_sampled_at_ms, None);
        assert_eq!(r.load, Load::default());
        assert_eq!(r.limits.max_running, None);
        assert_eq!(r.engine_version, None);
    }

    #[tokio::test]
    async fn reported_at_is_taken_once_after_reads_complete() {
        let s = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/loads"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(body(1_790_000_000.0))
                    .set_delay(Duration::from_millis(150)),
            )
            .mount(&s)
            .await;
        let key = ReportKey::generate();
        let mut p = publisher(key, &[("r1", &s.uri())], Duration::from_millis(500));
        let calls = std::cell::Cell::new(0u32);
        let called_after = std::cell::Cell::new(Duration::ZERO);
        let start = Instant::now();
        p.tick(
            |_| 0,
            || {
                calls.set(calls.get() + 1);
                called_after.set(start.elapsed());
                NOW
            },
        )
        .await;
        assert_eq!(calls.get(), 1);
        assert!(
            called_after.get() >= Duration::from_millis(150),
            "now_ms read at {:?}, before the replica read finished",
            called_after.get()
        );
    }

    #[tokio::test]
    async fn tick_is_bounded_by_interval() {
        let slow = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/loads"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(body(1_790_000_000.0))
                    .set_delay(Duration::from_secs(5)),
            )
            .mount(&slow)
            .await;
        let fast = MockServer::start().await;
        serve(&fast, 200, 1_790_000_000.0).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(
            key,
            &[("slow", &slow.uri()), ("fast", &fast.uri())],
            Duration::from_millis(200),
        );
        let start = Instant::now();
        let frames = p.tick(|_| 0, || NOW).await;
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(1), "tick took {elapsed:?}");
        let reports = open_all(&frames, &vk);
        assert_eq!(reports[0].load, Load::default());
        assert_eq!(reports[1].load.running, Some(3));
    }
}
