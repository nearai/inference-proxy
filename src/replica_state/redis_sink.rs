//! Redis sink for signed replica frames: a short-TTL "latest" key per replica
//! plus a capped per-host stream.
//!
//! Never log the Redis URL (it may hold a password) or a redis error's Display
//! (it can echo the URL); only `err.kind()`.

use std::time::Duration;

use anyhow::anyhow;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};

use super::report::Envelope;

/// TTL of each replica's latest-state key.
pub const KEY_TTL_SECS: u64 = 5;
/// ~80 min at 2 replicas x 2 Hz; a buffer, not the recorder.
pub const STREAM_MAXLEN: usize = 20_000;

pub fn state_key(host_id: &str, replica_id: &str) -> String {
    format!("replica:{host_id}:{replica_id}")
}

pub fn stream_key(host_id: &str) -> String {
    format!("replica:{host_id}:frames")
}

pub struct RedisSink {
    conn: ConnectionManager,
    host_id: String,
    stream: String,
}

impl RedisSink {
    pub async fn connect(url: &str, host_id: &str) -> anyhow::Result<Self> {
        // The lock enables both ring and aws-lc-rs, so rustls has no implicit
        // process default and redis's `ClientConfig::builder()` would panic on
        // `rediss://`. Err here just means a provider is already installed.
        let _ = rustls::crypto::ring::default_provider().install_default();

        let client = redis::Client::open(url)
            .map_err(|e| anyhow!("redis connect failed: {:?}", e.kind()))?;
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(Duration::from_secs(2))
            .set_response_timeout(Duration::from_secs(1))
            .set_number_of_retries(1);
        let conn = ConnectionManager::new_with_config(client, config)
            .await
            .map_err(|e| anyhow!("redis connect failed: {:?}", e.kind()))?;
        Ok(Self {
            conn,
            host_id: host_id.to_string(),
            stream: stream_key(host_id),
        })
    }

    /// One pipeline: SET state_key json EX 5 for each frame, then
    /// XADD stream MAXLEN ~ 20000 * env json.
    pub async fn publish(&mut self, frames: &[(String, Envelope)]) -> anyhow::Result<()> {
        if frames.is_empty() {
            return Ok(());
        }
        let jsons = frames
            .iter()
            .map(|(_, env)| serde_json::to_string(env))
            .collect::<Result<Vec<_>, _>>()?;
        let mut pipe = redis::pipe();
        for ((replica_id, _), json) in frames.iter().zip(&jsons) {
            pipe.cmd("SET")
                .arg(state_key(&self.host_id, replica_id))
                .arg(json)
                .arg("EX")
                .arg(KEY_TTL_SECS)
                .ignore();
        }
        for json in &jsons {
            pipe.cmd("XADD")
                .arg(&self.stream)
                .arg("MAXLEN")
                .arg("~")
                .arg(STREAM_MAXLEN)
                .arg("*")
                .arg("env")
                .arg(json)
                .ignore();
        }
        pipe.query_async::<()>(&mut self.conn)
            .await
            .map_err(|e| anyhow!("redis publish failed: {:?}", e.kind()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_layout_is_per_host() {
        assert_eq!(state_key("glm53-gpu03", "r1"), "replica:glm53-gpu03:r1");
        assert_eq!(stream_key("glm53-gpu03"), "replica:glm53-gpu03:frames");
    }

    #[tokio::test]
    async fn unreachable_redis_fails_fast() {
        let t = std::time::Instant::now();
        assert!(RedisSink::connect("redis://127.0.0.1:1", "h")
            .await
            .is_err());
        assert!(t.elapsed() < std::time::Duration::from_secs(5));
    }

    #[tokio::test]
    async fn tls_url_does_not_panic_with_two_rustls_providers() {
        // Must return Err (nothing listening), not panic inside rustls ClientConfig::builder().
        assert!(RedisSink::connect("rediss://127.0.0.1:1", "h")
            .await
            .is_err());
    }

    /// Real Redis, only when REPLICA_STATE_TEST_REDIS_URL is set.
    #[tokio::test]
    async fn publish_sets_ttl_keys_and_appends_stream() {
        let Ok(url) = std::env::var("REPLICA_STATE_TEST_REDIS_URL") else {
            return;
        };
        let mut sink = RedisSink::connect(&url, "test-host").await.unwrap();
        let env = |n: u32| Envelope {
            frame: format!("{{\"seq\":{n}}}"),
            sig: format!("sig{n}"),
        };
        let (e1, e2) = (env(1), env(2));
        sink.publish(&[("r1".to_string(), e1.clone()), ("r2".to_string(), e2)])
            .await
            .unwrap();

        let mut plain = redis::Client::open(url.as_str())
            .unwrap()
            .get_multiplexed_async_connection()
            .await
            .unwrap();
        let got: String = redis::cmd("GET")
            .arg("replica:test-host:r1")
            .query_async(&mut plain)
            .await
            .unwrap();
        assert_eq!(got, serde_json::to_string(&e1).unwrap());
        let ttl: i64 = redis::cmd("TTL")
            .arg("replica:test-host:r1")
            .query_async(&mut plain)
            .await
            .unwrap();
        assert!((1..=5).contains(&ttl), "ttl={ttl}");
        let len: u64 = redis::cmd("XLEN")
            .arg("replica:test-host:frames")
            .query_async(&mut plain)
            .await
            .unwrap();
        assert!(len >= 2, "xlen={len}");
    }
}
