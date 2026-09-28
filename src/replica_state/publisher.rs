//! Per-tick publisher: reads every replica's engine load concurrently, then
//! builds one signed host frame carrying every replica's state.
//!
//! Frames carry IDs and numbers only. Nothing here logs replica URLs, frame
//! contents or key material.

use std::sync::Arc;
use std::time::Duration;

use super::report::{self, Engine, Envelope, HostReport, Lifecycle, Limits, Load, ReplicaState};
use super::report_key::ReportKey;
use super::sglang::{self, ReplicaLoad};

/// Consecutive failed reads (after at least one success) that mark a replica
/// `Unhealthy`.
pub const UNHEALTHY_AFTER_FAILURES: u32 = 3;

struct ReplicaSlot {
    base_url: String,
    last: Option<ReplicaLoad>,
    failures: u32,
    lifecycle: Lifecycle,
    max_context: Option<u64>,
}

pub struct Publisher {
    host_id: String,
    key: Arc<ReportKey>,
    slots: Vec<ReplicaSlot>,
    seq: u64,
    client: reqwest::Client,
    read_timeout: Duration,
}

impl Publisher {
    /// `base_urls` is in backend-pool order (base backends first, then
    /// long-context); slot index `i` becomes `ReplicaState.index == i as u32`.
    /// Reads are bounded by 4/5 of `interval` so a tick never overruns its
    /// slot.
    pub fn new(
        host_id: String,
        key: Arc<ReportKey>,
        base_urls: Vec<String>,
        client: reqwest::Client,
        interval: Duration,
    ) -> Self {
        let slots = base_urls
            .into_iter()
            .map(|base_url| ReplicaSlot {
                base_url,
                last: None,
                failures: 0,
                lifecycle: Lifecycle::Warming,
                max_context: None,
            })
            .collect();
        Self {
            host_id,
            key,
            slots,
            seq: 0,
            client,
            read_timeout: interval * 4 / 5,
        }
    }

    /// Read every replica concurrently — joined, per slot, with a
    /// context-length refresh for a slot that just became ready or has never
    /// resolved one — then build one host frame for the whole tick. Each
    /// slot's load read and (if needed) context read run as one joined pair,
    /// so a tick stays bounded by `read_timeout` even while a refresh is
    /// pending. `now_ms` is called once, after all reads complete, so
    /// `reported_at_ms` is the seal time.
    pub async fn tick(
        &mut self,
        inflight: impl Fn(usize) -> u32,
        now_ms: impl Fn() -> u64,
    ) -> Envelope {
        // Whether to also fire a context read for slot `i`, decided from
        // state as of the start of this tick (before any read completes):
        // no context yet, or the slot wasn't Ready coming into this tick.
        let needs_refresh: Vec<bool> = self
            .slots
            .iter()
            .map(|slot| slot.max_context.is_none() || slot.lifecycle != Lifecycle::Ready)
            .collect();

        let results: Vec<(Option<ReplicaLoad>, Option<Option<u64>>)> =
            futures_util::future::join_all(self.slots.iter().zip(needs_refresh.iter()).map(
                |(slot, &refresh)| {
                    let client = &self.client;
                    let base_url = &slot.base_url;
                    let timeout = self.read_timeout;
                    async move {
                        if refresh {
                            let (load, ctx) = futures_util::future::join(
                                sglang::read_replica(client, base_url, timeout),
                                sglang::read_max_context(client, base_url, timeout),
                            )
                            .await;
                            (load, Some(ctx))
                        } else {
                            let load = sglang::read_replica(client, base_url, timeout).await;
                            (load, None)
                        }
                    }
                },
            ))
            .await;

        let reported_at_ms = now_ms();

        self.seq += 1;
        let seq = self.seq;
        let mut replicas = Vec::with_capacity(self.slots.len());
        for (i, (slot, (read, context_read))) in self.slots.iter_mut().zip(results).enumerate() {
            // Only apply a context result once we know the load read for
            // this slot succeeded this tick; a failed load read leaves
            // `max_context` (and everything else) at its previous value.
            if read.is_some() {
                if let Some(Some(v)) = context_read {
                    slot.max_context = Some(v);
                }
            }

            let (load, limits, engine_version, engine_sampled_at_ms) = match read {
                Some(read) => {
                    let recovering = slot.lifecycle == Lifecycle::Unhealthy
                        || (slot.lifecycle == Lifecycle::Warming && slot.failures > 0);
                    if recovering {
                        tracing::info!(replica_index = i, "replica state: replica ready");
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
                            replica_index = i,
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
                        None => (Load::default(), Limits::default(), None, None),
                    }
                }
            };
            replicas.push(ReplicaState {
                index: i as u32,
                engine_sampled_at_ms,
                lifecycle_state: slot.lifecycle,
                engine_version,
                limits: Limits {
                    max_context_tokens: slot.max_context,
                    ..limits
                },
                load,
                proxy_inflight: inflight(i),
            });
        }

        let report = HostReport {
            schema: 1,
            host_id: self.host_id.clone(),
            boot_id: self.key.boot_id.clone(),
            seq,
            reported_at_ms,
            engine: Engine::Sglang,
            report_key_id: self.key.key_id.clone(),
            replicas,
        };
        report::seal(&report, self.key.signing_key())
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

    async fn serve_models(server: &MockServer, max_model_len: u64) {
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"id": "m", "max_model_len": max_model_len}],
            })))
            .mount(server)
            .await;
    }

    fn publisher(key: ReportKey, base_urls: &[&str], interval: Duration) -> Publisher {
        Publisher::new(
            "host-a".into(),
            Arc::new(key),
            base_urls.iter().map(|u| u.to_string()).collect(),
            reqwest::Client::new(),
            interval,
        )
    }

    fn open_host(env: &Envelope, vk: &ed25519_dalek::VerifyingKey) -> HostReport {
        open(env, vk).expect("frame verifies under the report key")
    }

    #[tokio::test]
    async fn one_signed_frame_per_tick_carrying_every_replica_sharing_one_seq() {
        let (s1, s2) = (MockServer::start().await, MockServer::start().await);
        serve(&s1, 200, 1_790_000_000.0).await;
        serve(&s2, 200, 1_790_000_000.0).await;
        serve_models(&s1, 131_072).await;
        serve_models(&s2, 131_072).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let boot_id = key.boot_id.clone();
        let key_id = key.key_id.clone();
        let mut p = publisher(key, &[&s1.uri(), &s2.uri()], Duration::from_millis(500));

        let host = open_host(&p.tick(|i| 10 + i as u32, || NOW).await, &vk);
        assert_eq!(
            host.replicas.iter().map(|r| r.index).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(host.schema, 1);
        assert_eq!(host.boot_id, boot_id);
        assert_eq!(host.report_key_id, key_id);
        assert_eq!(host.host_id, "host-a");
        assert_eq!(host.reported_at_ms, NOW);
        assert_eq!(host.seq, 1);
        for (i, r) in host.replicas.iter().enumerate() {
            assert_eq!(r.lifecycle_state, Lifecycle::Ready);
            assert!(host.reported_at_ms >= r.engine_sampled_at_ms.unwrap());
            assert_eq!(r.proxy_inflight, 10 + i as u32);
            assert_eq!(r.load.running, Some(3));
        }

        let host2 = open_host(&p.tick(|_| 0, || NOW + 500).await, &vk);
        assert_eq!(host2.replicas.len(), 2);
        assert_eq!(host2.seq, 2);
    }

    #[tokio::test]
    async fn engine_timestamp_passes_through_even_when_stale() {
        let s = MockServer::start().await;
        serve(&s, 200, 1000.0).await;
        serve_models(&s, 131_072).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[&s.uri()], Duration::from_millis(500));
        let host = open_host(&p.tick(|_| 0, || NOW).await, &vk);
        let r = &host.replicas[0];
        assert_eq!(r.engine_sampled_at_ms, Some(1_000_000));
        assert_eq!(host.reported_at_ms, NOW);
        assert_eq!(r.lifecycle_state, Lifecycle::Ready);
    }

    #[tokio::test]
    async fn failed_read_keeps_last_sample_time_and_nulls_load() {
        let s = MockServer::start().await;
        serve(&s, 200, 1_790_000_000.0).await;
        serve_models(&s, 16_384).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[&s.uri()], Duration::from_millis(500));
        let host = open_host(&p.tick(|_| 0, || NOW).await, &vk);
        let ok = &host.replicas[0];
        assert_eq!(ok.load.running, Some(3));
        assert!(host.reported_at_ms >= ok.engine_sampled_at_ms.unwrap());

        s.reset().await;
        serve(&s, 500, 0.0).await;
        let host = open_host(&p.tick(|_| 0, || NOW + 500).await, &vk);
        let r = &host.replicas[0];
        assert_eq!(r.load, Load::default());
        assert_eq!(r.engine_sampled_at_ms, Some(1_790_000_000_000));
        assert_eq!(r.limits.max_running, Some(16));
        assert_eq!(r.engine_version.as_deref(), Some("0.5.9"));
        assert_eq!(r.lifecycle_state, Lifecycle::Ready);

        let host = open_host(&p.tick(|_| 0, || NOW + 1000).await, &vk);
        assert_eq!(host.replicas[0].lifecycle_state, Lifecycle::Ready);
        let host = open_host(&p.tick(|_| 0, || NOW + 1500).await, &vk);
        assert_eq!(host.replicas[0].lifecycle_state, Lifecycle::Unhealthy);
        assert_eq!(host.replicas[0].load, Load::default());

        s.reset().await;
        serve(&s, 200, 1_790_000_002.0).await;
        let host = open_host(&p.tick(|_| 0, || NOW + 2000).await, &vk);
        let r = &host.replicas[0];
        assert_eq!(r.lifecycle_state, Lifecycle::Ready);
        assert_eq!(r.engine_sampled_at_ms, Some(1_790_000_002_000));
        assert!(host.reported_at_ms >= r.engine_sampled_at_ms.unwrap());
        assert_eq!(r.load.running, Some(3));
        assert_eq!(host.seq, 5);
    }

    #[tokio::test]
    async fn never_reachable_replica_is_warming_with_null_sample_time() {
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &["http://127.0.0.1:1"], Duration::from_millis(500));
        let mut last = None;
        for n in 0..5 {
            last = Some(open_host(&p.tick(|_| 0, || NOW + n * 500).await, &vk));
        }
        let host = last.unwrap();
        let r = &host.replicas[0];
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
        serve_models(&s, 131_072).await;
        let key = ReportKey::generate();
        let mut p = publisher(key, &[&s.uri()], Duration::from_millis(500));
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
        let mut p = publisher(key, &[&slow.uri(), &fast.uri()], Duration::from_millis(200));
        let start = Instant::now();
        let env = p.tick(|_| 0, || NOW).await;
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(1), "tick took {elapsed:?}");
        let host = open_host(&env, &vk);
        assert_eq!(host.replicas[0].load, Load::default());
        assert_eq!(host.replicas[1].load.running, Some(3));
    }

    #[tokio::test]
    async fn context_refresh_runs_concurrently_with_the_load_read() {
        // Interval 1000ms -> read_timeout 800ms, so both 300ms reads succeed.
        // Joined per slot, the tick takes ~300ms; a sequential (load, then
        // context) tick would take ~600ms. The 500ms bound separates the two
        // with ~200ms of scheduling slack either side.
        let s = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/loads"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(body(1_790_000_000.0))
                    .set_delay(Duration::from_millis(300)),
            )
            .mount(&s)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({
                        "object": "list",
                        "data": [{"id": "m", "max_model_len": 131_072}],
                    }))
                    .set_delay(Duration::from_millis(300)),
            )
            .mount(&s)
            .await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[&s.uri()], Duration::from_millis(1000));
        let start = Instant::now();
        let env = p.tick(|_| 0, || NOW).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "tick took {elapsed:?}, expected one read's latency, not the sum of both reads"
        );
        let host = open_host(&env, &vk);
        assert_eq!(host.replicas[0].limits.max_context_tokens, Some(131_072));
    }

    #[tokio::test]
    async fn one_host_frame_even_when_every_replica_is_down() {
        let key = Arc::new(ReportKey::generate());
        let vk = key.signing_key().verifying_key();
        let mut p = Publisher::new(
            "host-a".into(),
            key,
            vec!["http://127.0.0.1:9".into(), "http://127.0.0.1:9".into()],
            reqwest::Client::new(),
            Duration::from_millis(200),
        );
        let a = open_host(&p.tick(|_| 0, || 1).await, &vk);
        let b = open_host(&p.tick(|_| 0, || 2).await, &vk);
        assert_eq!(a.replicas.len(), 2);
        assert_eq!(b.seq, a.seq + 1);
        assert!(a
            .replicas
            .iter()
            .all(|r| r.lifecycle_state == Lifecycle::Warming));
        assert_eq!(
            a.replicas.iter().map(|r| r.index).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[tokio::test]
    async fn context_limit_is_read_once_ready_and_kept() {
        let s = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/loads"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body(1_790_000_000.0)))
            .mount(&s)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"id": "m", "max_model_len": 131_072}],
            })))
            .expect(1)
            .mount(&s)
            .await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[&s.uri()], Duration::from_millis(500));

        let host1 = open_host(&p.tick(|_| 0, || NOW).await, &vk);
        assert_eq!(host1.replicas[0].limits.max_context_tokens, Some(131_072));
        let host2 = open_host(&p.tick(|_| 0, || NOW + 500).await, &vk);
        assert_eq!(host2.replicas[0].limits.max_context_tokens, Some(131_072));

        // Dropping the server verifies the `.expect(1)` on the /v1/models
        // mock: exactly one call across both ticks.
        drop(s);
    }

    #[tokio::test]
    async fn missing_models_endpoint_publishes_null_context() {
        let s = MockServer::start().await;
        serve(&s, 200, 1_790_000_000.0).await;
        // No /v1/models mock mounted → context reads 404.
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[&s.uri()], Duration::from_millis(500));

        let host = open_host(&p.tick(|_| 0, || NOW).await, &vk);
        assert_eq!(host.replicas[0].lifecycle_state, Lifecycle::Ready);
        assert_eq!(host.replicas[0].limits.max_context_tokens, None);
        // Review Focus 3: load is unaffected by the missing /v1/models endpoint.
        assert_eq!(host.replicas[0].load.running, Some(3));
        assert_eq!(host.replicas[0].limits.max_running, Some(16));
    }

    #[tokio::test]
    async fn context_is_re_read_and_updated_after_unhealthy_to_ready_transition() {
        let s = MockServer::start().await;
        serve(&s, 200, 1_790_000_000.0).await;
        serve_models(&s, 131_072).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[&s.uri()], Duration::from_millis(500));

        let host = open_host(&p.tick(|_| 0, || NOW).await, &vk);
        assert_eq!(host.replicas[0].limits.max_context_tokens, Some(131_072));

        // Three failed /v1/loads reads move Ready -> Unhealthy.
        s.reset().await;
        serve(&s, 500, 0.0).await;
        for n in 1..=3 {
            let host = open_host(&p.tick(|_| 0, || NOW + n * 500).await, &vk);
            if n < 3 {
                assert_eq!(host.replicas[0].lifecycle_state, Lifecycle::Ready);
            } else {
                assert_eq!(host.replicas[0].lifecycle_state, Lifecycle::Unhealthy);
            }
        }

        // Recovery: /v1/loads succeeds again and /v1/models now reports a
        // different value; the Unhealthy -> Ready transition must re-read it.
        s.reset().await;
        serve(&s, 200, 1_790_000_004.0).await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"id": "m", "max_model_len": 262_144}],
            })))
            .expect(1)
            .mount(&s)
            .await;
        let host = open_host(&p.tick(|_| 0, || NOW + 2000).await, &vk);
        assert_eq!(host.replicas[0].lifecycle_state, Lifecycle::Ready);
        assert_eq!(host.replicas[0].limits.max_context_tokens, Some(262_144));

        drop(s);
    }

    #[tokio::test]
    async fn failed_context_read_after_reready_keeps_the_prior_value() {
        let s = MockServer::start().await;
        serve(&s, 200, 1_790_000_000.0).await;
        serve_models(&s, 131_072).await;
        let key = ReportKey::generate();
        let vk = key.signing_key().verifying_key();
        let mut p = publisher(key, &[&s.uri()], Duration::from_millis(500));

        let host = open_host(&p.tick(|_| 0, || NOW).await, &vk);
        assert_eq!(host.replicas[0].limits.max_context_tokens, Some(131_072));

        // Three failed /v1/loads reads move Ready -> Unhealthy.
        s.reset().await;
        serve(&s, 500, 0.0).await;
        for n in 1..=3 {
            p.tick(|_| 0, || NOW + n * 500).await;
        }

        // Recovery: /v1/loads succeeds again, but /v1/models now fails. The
        // Unhealthy -> Ready transition must still attempt the re-read
        // (verified by `.expect(1)`), and the failure must keep the prior
        // context value rather than clearing it.
        s.reset().await;
        serve(&s, 200, 1_790_000_004.0).await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(500))
            .expect(1)
            .mount(&s)
            .await;
        let host = open_host(&p.tick(|_| 0, || NOW + 2000).await, &vk);
        assert_eq!(host.replicas[0].lifecycle_state, Lifecycle::Ready);
        assert_eq!(host.replicas[0].limits.max_context_tokens, Some(131_072));

        drop(s);
    }
}
