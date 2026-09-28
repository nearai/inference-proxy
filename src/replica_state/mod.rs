pub mod config;
pub mod publisher;
pub mod redis_sink;
pub mod report;
pub mod report_key;
pub mod sglang;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::backend_pool::BackendPool;
use config::ReplicaStateConfig;
use publisher::Publisher;
use redis_sink::RedisSink;
use report_key::ReportKey;

/// Upper bound on recording the report key in the dstack event log.
const BIND_TIMEOUT: Duration = Duration::from_secs(5);
/// Cap on the pause between key binding retries.
const BIND_RETRY_MAX: Duration = Duration::from_secs(60);
/// Minimum gap between repeated "still failing" bind warnings.
const BIND_WARN_EVERY: Duration = Duration::from_secs(60);
/// Pause between initial Redis connect attempts.
const CONNECT_RETRY: Duration = Duration::from_secs(5);
/// Minimum gap between repeated "still failing" connect warnings.
const CONNECT_WARN_EVERY: Duration = Duration::from_secs(60);

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Whether a repeated failure (Redis connect, key binding) should be logged again.
fn should_warn(last: Option<Instant>, now: Instant, every: Duration) -> bool {
    last.is_none_or(|t| now.duration_since(t) >= every)
}

/// Value of the `replica_state_key_bound` gauge for a bind outcome: 1 only
/// when the key was actually recorded in the attested event log.
fn key_bound_gauge<T>(bound: &Result<anyhow::Result<bool>, T>) -> f64 {
    if matches!(bound, Ok(Ok(true))) {
        1.0
    } else {
        0.0
    }
}

/// Only a definite bind error is retried. A timeout is not: the event may
/// already be recorded, and a retry could append a duplicate
/// `nearai-replica-report-key-v1` event to RTMR3. A skip (dev/non-TEE) is final.
fn should_retry_bind<T>(bound: &Result<anyhow::Result<bool>, T>) -> bool {
    matches!(bound, Ok(Err(_)))
}

/// Pause after failed bind attempt `attempt` (1-based): 1 s, 2 s, 4 s, ...
/// capped at [`BIND_RETRY_MAX`].
fn bind_retry_delay(attempt: u32) -> Duration {
    Duration::from_secs(1u64 << attempt.saturating_sub(1).min(6)).min(BIND_RETRY_MAX)
}

/// Loggable cause of a failed key binding. With an HTTP dstack endpoint
/// (simulator) the error is a `reqwest::Error`, whose `Display` includes the
/// URL; that is stripped so no URL reaches the logs.
fn bind_error_detail(e: anyhow::Error) -> String {
    match e.downcast::<reqwest::Error>() {
        Ok(re) => format!("{:#}", anyhow::Error::from(re.without_url())),
        Err(e) => format!("{e:#}"),
    }
}

/// Records `key` in the attested event log, retrying definite errors with
/// capped backoff until it succeeds. Runs beside the publish loop, so frames
/// are published meanwhile; readers reject them until the key is bound.
async fn bind_key(key: Arc<ReportKey>, host_id: String, skip: bool) {
    metrics::gauge!("replica_state_key_bound").set(0.0);
    let mut last_warn: Option<Instant> = None;
    for attempt in 1u32.. {
        let bound = tokio::time::timeout(
            BIND_TIMEOUT,
            report_key::bind_to_attestation(&key, &host_id, skip),
        )
        .await;
        metrics::gauge!("replica_state_key_bound").set(key_bound_gauge(&bound));
        let retry = should_retry_bind(&bound);
        match bound {
            Ok(Ok(true)) if attempt > 1 => tracing::info!(
                key_id = %key.key_id,
                attempt,
                "Replica report key bound to attestation after retrying"
            ),
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                if should_warn(last_warn, Instant::now(), BIND_WARN_EVERY) {
                    tracing::warn!(
                        key_id = %key.key_id,
                        attempt,
                        error = %bind_error_detail(e),
                        "Replica report key could not be bound to attestation; retrying, readers reject its frames until bound"
                    );
                    last_warn = Some(Instant::now());
                }
            }
            Err(_) => tracing::warn!(
                key_id = %key.key_id,
                attempt,
                "Replica report key binding timed out; not retried, readers will reject its frames"
            ),
        }
        if !retry {
            return;
        }
        tokio::time::sleep(bind_retry_delay(attempt)).await;
    }
}

/// Backend base URLs in publisher order: base backends first, then
/// long-context. Slot index `i` becomes `ReplicaState.index == i as u32`.
fn publisher_urls(pool: &BackendPool) -> Vec<String> {
    pool.backends().iter().map(|b| b.base_url.clone()).collect()
}

/// Spawns the background task that publishes one signed host frame to Redis
/// every `cfg.interval`. Replica index `i` is backend `i` of `pool` (base
/// backends first, then long-context). Never blocks the caller: key binding
/// runs in its own task and the first Redis connect happens inside the
/// publish task, so neither delays the other.
pub fn spawn_replica_state_publisher(
    cfg: ReplicaStateConfig,
    pool: Arc<BackendPool>,
    client: reqwest::Client,
    skip_binding: bool,
) {
    // Process-wide rustls default, needed before any `rediss://` connect.
    redis_sink::install_crypto_provider();
    let key = Arc::new(ReportKey::generate());
    tokio::spawn(bind_key(key.clone(), cfg.host_id.clone(), skip_binding));
    tokio::spawn(async move {
        let mut last_warn: Option<Instant> = None;
        let mut sink = loop {
            match RedisSink::connect(&cfg.redis_url, cfg.redis_ca_cert.as_deref(), &cfg.host_id)
                .await
            {
                Ok(sink) => break sink,
                // RedisSink errors carry only the redis error kind, never the URL.
                Err(e) => {
                    if should_warn(last_warn, Instant::now(), CONNECT_WARN_EVERY) {
                        tracing::warn!(
                            redis = %cfg.redis_host_for_logs(),
                            error = %e,
                            "Replica state Redis connect failed; retrying"
                        );
                        last_warn = Some(Instant::now());
                    }
                    tokio::time::sleep(CONNECT_RETRY).await;
                }
            }
        };
        tracing::info!(
            redis = %cfg.redis_host_for_logs(),
            "Replica state Redis connected"
        );

        let mut publisher = Publisher::new(
            cfg.host_id.clone(),
            key,
            publisher_urls(&pool),
            client,
            cfg.interval,
        );

        let mut ticker = tokio::time::interval(cfg.interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            let started = Instant::now();
            let env = publisher
                .tick(
                    |i| {
                        pool.backends()
                            .get(i)
                            .map_or(0, |b| b.active_conns.load(Ordering::Relaxed))
                    },
                    now_ms,
                )
                .await;
            // RedisSink errors carry only the redis error kind, never the URL.
            match sink.publish(&env).await {
                Ok(()) => {
                    metrics::counter!("replica_state_frames_total").increment(1);
                }
                Err(e) => {
                    metrics::counter!("replica_state_publish_failures_total").increment(1);
                    tracing::warn!(error = %e, "Replica state publish failed");
                }
            }
            metrics::histogram!("replica_state_tick_seconds")
                .record(started.elapsed().as_secs_f64());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publisher_urls_are_base_then_long_context() {
        let pool = BackendPool::with_long_context(
            vec!["http://b0".into(), "http://b1".into()],
            vec!["http://l0".into()],
        );
        assert_eq!(
            publisher_urls(&pool),
            vec![
                "http://b0".to_string(),
                "http://b1".to_string(),
                "http://l0".to_string(),
            ]
        );
    }

    #[test]
    fn connect_warning_repeats_only_after_the_gap() {
        let t0 = Instant::now();
        let gap = Duration::from_secs(60);
        assert!(should_warn(None, t0, gap));
        assert!(!should_warn(Some(t0), t0 + Duration::from_secs(59), gap));
        assert!(should_warn(Some(t0), t0 + gap, gap));
    }

    #[tokio::test]
    async fn bind_error_detail_never_includes_the_url() {
        // Nothing listens on port 1; the path stands in for anything the
        // URL could carry.
        let err = reqwest::Client::new()
            .post("http://127.0.0.1:1/EmitEvent-url-marker")
            .send()
            .await
            .unwrap_err();
        assert!(err.to_string().contains("url-marker"));
        let detail = bind_error_detail(anyhow::Error::from(err));
        assert!(!detail.contains("url-marker"), "{detail}");
        assert!(!detail.contains("127.0.0.1"), "{detail}");
        assert!(!detail.is_empty());

        let other = bind_error_detail(anyhow::anyhow!("dstack socket missing"));
        assert_eq!(other, "dstack socket missing");
    }

    #[test]
    fn bind_retry_delay_doubles_from_one_second_up_to_the_cap() {
        let secs: Vec<u64> = (1..=9).map(|a| bind_retry_delay(a).as_secs()).collect();
        assert_eq!(secs, [1, 2, 4, 8, 16, 32, 60, 60, 60]);
        assert_eq!(bind_retry_delay(0), Duration::from_secs(1));
        assert_eq!(bind_retry_delay(u32::MAX), BIND_RETRY_MAX);
    }

    #[test]
    fn bind_is_retried_only_on_a_definite_error() {
        assert!(should_retry_bind::<()>(&Ok(Err(anyhow::anyhow!(
            "dstack down"
        )))));
        assert!(!should_retry_bind::<()>(&Ok(Ok(true))));
        assert!(!should_retry_bind::<()>(&Ok(Ok(false))));
        // A timeout may already have recorded the event; never retry it.
        assert!(!should_retry_bind(&Err::<anyhow::Result<bool>, _>(())));
    }

    #[test]
    fn key_bound_gauge_is_one_only_when_recorded() {
        assert_eq!(key_bound_gauge::<()>(&Ok(Ok(true))), 1.0);
        assert_eq!(key_bound_gauge::<()>(&Ok(Ok(false))), 0.0);
        assert_eq!(
            key_bound_gauge::<()>(&Ok(Err(anyhow::anyhow!("dstack down")))),
            0.0
        );
        assert_eq!(key_bound_gauge(&Err::<anyhow::Result<bool>, _>(())), 0.0);
    }
}
