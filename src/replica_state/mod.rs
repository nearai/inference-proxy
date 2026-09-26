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

/// `(replica_id, base_url)` pairs: replica `i` is backend `i` of `pool`
/// (base backends first, then long-context).
fn replica_targets(ids: &[String], pool: &BackendPool) -> Vec<(String, String)> {
    ids.iter()
        .cloned()
        .zip(pool.backends().iter().map(|b| b.base_url.clone()))
        .collect()
}

/// Spawns the background task that publishes signed per-replica state to
/// Redis every `cfg.interval`. Replica `i` of `cfg.replica_ids` is backend `i`
/// of `pool` (base backends first, then long-context). Never blocks the
/// caller: key binding and the first Redis connect happen inside the task.
pub fn spawn_replica_state_publisher(
    cfg: ReplicaStateConfig,
    model: String,
    pool: Arc<BackendPool>,
    client: reqwest::Client,
    skip_binding: bool,
) {
    // Process-wide rustls default, needed before any `rediss://` connect.
    redis_sink::install_crypto_provider();
    tokio::spawn(async move {
        let key = ReportKey::generate();
        let bound = tokio::time::timeout(
            BIND_TIMEOUT,
            report_key::bind_to_attestation(
                &key,
                &cfg.host_id,
                &model,
                &cfg.replica_ids,
                skip_binding,
            ),
        )
        .await;
        match bound {
            Ok(Ok(_)) => {}
            Ok(Err(_)) => tracing::warn!(
                key_id = %key.key_id,
                "Replica report key could not be bound to attestation; readers will reject its frames"
            ),
            Err(_) => tracing::warn!(
                key_id = %key.key_id,
                "Replica report key binding timed out; readers will reject its frames"
            ),
        }
        let key_bound = matches!(bound, Ok(Ok(true)));
        metrics::gauge!("replica_state_key_bound").set(if key_bound { 1.0 } else { 0.0 });

        let mut last_warn: Option<Instant> = None;
        let mut sink = loop {
            match RedisSink::connect(&cfg.redis_url, cfg.redis_ca_cert.as_deref(), &cfg.host_id)
                .await
            {
                Ok(sink) => break sink,
                Err(_) => {
                    if last_warn.is_none_or(|t| t.elapsed() >= CONNECT_WARN_EVERY) {
                        tracing::warn!(
                            redis = %cfg.redis_host_for_logs(),
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
            model,
            key,
            replica_targets(&cfg.replica_ids, &pool),
            client,
            cfg.interval,
        );

        let mut ticker = tokio::time::interval(cfg.interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            let started = Instant::now();
            let frames = publisher
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
            match sink.publish(&frames).await {
                Ok(()) => {
                    metrics::counter!("replica_state_frames_total").increment(frames.len() as u64);
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
    fn replica_targets_pair_ids_with_base_then_long_context_backends() {
        let pool = BackendPool::with_long_context(
            vec!["http://a:8000".into(), "http://b:8000".into()],
            vec!["http://a-long:8000".into()],
        );
        let ids = vec!["r1".to_string(), "r2".to_string(), "r3".to_string()];
        assert_eq!(
            replica_targets(&ids, &pool),
            vec![
                ("r1".to_string(), "http://a:8000".to_string()),
                ("r2".to_string(), "http://b:8000".to_string()),
                ("r3".to_string(), "http://a-long:8000".to_string()),
            ]
        );
    }
}
