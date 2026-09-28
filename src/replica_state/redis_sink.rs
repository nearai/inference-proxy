//! Redis sink for the signed host frame: a short-TTL "latest" key per host
//! plus a capped per-host stream.
//!
//! Never log the Redis URL (it may hold a password) or a redis error's Display
//! (it can echo the URL); only `err.kind()`.

use std::time::Duration;

use anyhow::anyhow;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};

use super::report::Envelope;

/// TTL of the host's latest-state key.
pub const KEY_TTL_SECS: u64 = 5;
/// ~40 min at 2 Hz for one host; covers a full host replacement window with
/// room to spare. A buffer, not the recorder.
pub const STREAM_MAXLEN: usize = 5_000;

pub fn state_key(host_id: &str) -> String {
    format!("replica:{host_id}")
}

pub fn stream_key(host_id: &str) -> String {
    format!("replica:{host_id}:frames")
}

pub struct RedisSink {
    conn: ConnectionManager,
    host_id: String,
    stream: String,
}

/// Installs ring as the process-wide rustls provider. Call once before any
/// `rediss://` connect.
pub fn install_crypto_provider() {
    // The lock enables both ring and aws-lc-rs, so rustls has no implicit
    // process default and redis's `ClientConfig::builder()` would panic on
    // `rediss://`. Err here just means a provider is already installed.
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Builds a client for `url`. With `ca_pem`, TLS trusts only that CA (PEM)
/// instead of the system store, for a Redis serving a private-CA certificate.
pub fn client(url: &str, ca_pem: Option<&str>) -> redis::RedisResult<redis::Client> {
    match ca_pem {
        Some(pem) => redis::Client::build_with_tls(
            url,
            redis::TlsCertificates {
                client_tls: None,
                root_cert: Some(pem.as_bytes().to_vec()),
            },
        ),
        None => redis::Client::open(url),
    }
}

impl RedisSink {
    /// Requires [`install_crypto_provider`] to have run for `rediss://` URLs.
    pub async fn connect(url: &str, ca_pem: Option<&str>, host_id: &str) -> anyhow::Result<Self> {
        let client =
            client(url, ca_pem).map_err(|e| anyhow!("redis connect failed: {:?}", e.kind()))?;
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

    /// Sends the `build_pipeline` commands in one round trip.
    pub async fn publish(&mut self, env: &Envelope) -> anyhow::Result<()> {
        build_pipeline(&self.host_id, &self.stream, env)?
            .query_async::<()>(&mut self.conn)
            .await
            .map_err(|e| anyhow!("redis publish failed: {:?}", e.kind()))
    }
}

/// One pipeline: `SET replica:{host} <json> EX 5`, then
/// `XADD replica:{host}:frames MAXLEN ~ 5000 * env <json>`.
fn build_pipeline(host_id: &str, stream: &str, env: &Envelope) -> anyhow::Result<redis::Pipeline> {
    let json = serde_json::to_string(env)?;
    let mut pipe = redis::pipe();
    pipe.cmd("SET")
        .arg(state_key(host_id))
        .arg(&json)
        .arg("EX")
        .arg(KEY_TTL_SECS)
        .ignore();
    pipe.cmd("XADD")
        .arg(stream)
        .arg("MAXLEN")
        .arg("~")
        .arg(STREAM_MAXLEN)
        .arg("*")
        .arg("env")
        .arg(&json)
        .ignore();
    Ok(pipe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(cmd: &redis::Cmd) -> Vec<String> {
        cmd.args_iter()
            .map(|a| match a {
                redis::Arg::Simple(b) => String::from_utf8_lossy(b).into_owned(),
                redis::Arg::Cursor => "<cursor>".to_string(),
            })
            .collect()
    }

    fn env(n: u32) -> Envelope {
        Envelope {
            frame: format!("{{\"seq\":{n}}}"),
            sig: format!("sig{n}"),
            key_id: "0123456789abcdef".to_string(),
        }
    }

    #[test]
    fn pipeline_sets_ttl_keys_then_appends_capped_stream() {
        let e = env(1);
        let pipe = build_pipeline("gpu01", &stream_key("gpu01"), &e).unwrap();
        let cmds: Vec<Vec<String>> = pipe.cmd_iter().map(args).collect();
        let j = serde_json::to_string(&e).unwrap();
        assert_eq!(
            cmds,
            vec![
                vec![
                    "SET".into(),
                    "replica:gpu01".into(),
                    j.clone(),
                    "EX".into(),
                    "5".into()
                ],
                vec![
                    "XADD".into(),
                    "replica:gpu01:frames".into(),
                    "MAXLEN".into(),
                    "~".into(),
                    "5000".into(),
                    "*".into(),
                    "env".into(),
                    j,
                ],
            ]
        );
    }

    #[test]
    fn key_layout_is_per_host() {
        assert_eq!(state_key("h"), "replica:h");
        assert_eq!(stream_key("h"), "replica:h:frames");
    }

    #[tokio::test]
    async fn unreachable_redis_fails_fast() {
        let t = std::time::Instant::now();
        assert!(RedisSink::connect("redis://127.0.0.1:1", None, "h")
            .await
            .is_err());
        assert!(t.elapsed() < std::time::Duration::from_secs(5));
    }

    #[tokio::test]
    async fn tls_url_does_not_panic_with_two_rustls_providers() {
        // Must return Err (nothing listening), not panic inside rustls ClientConfig::builder().
        install_crypto_provider();
        assert!(RedisSink::connect("rediss://127.0.0.1:1", None, "h")
            .await
            .is_err());
    }

    /// Real Redis: `REPLICA_STATE_TEST_REDIS_URL=redis://127.0.0.1:6379 cargo test -- --ignored`.
    #[tokio::test]
    #[ignore = "needs REPLICA_STATE_TEST_REDIS_URL"]
    async fn publish_sets_ttl_key_and_appends_stream() {
        let url = std::env::var("REPLICA_STATE_TEST_REDIS_URL")
            .expect("REPLICA_STATE_TEST_REDIS_URL must point at a disposable Redis");
        install_crypto_provider();
        let host = format!("test-{}", uuid::Uuid::new_v4());
        let ca = std::env::var("REPLICA_STATE_TEST_REDIS_CA_CERT").ok();
        let mut sink = RedisSink::connect(&url, ca.as_deref(), &host)
            .await
            .unwrap();
        let e = env(1);
        sink.publish(&e).await.unwrap();

        let mut plain = client(&url, ca.as_deref())
            .unwrap()
            .get_multiplexed_async_connection()
            .await
            .unwrap();
        let got: String = redis::cmd("GET")
            .arg(state_key(&host))
            .query_async(&mut plain)
            .await
            .unwrap();
        assert_eq!(got, serde_json::to_string(&e).unwrap());
        let ttl: i64 = redis::cmd("TTL")
            .arg(state_key(&host))
            .query_async(&mut plain)
            .await
            .unwrap();
        assert!((1..=5).contains(&ttl), "ttl={ttl}");
        let len: u64 = redis::cmd("XLEN")
            .arg(stream_key(&host))
            .query_async(&mut plain)
            .await
            .unwrap();
        assert_eq!(len, 1, "xlen={len}");
    }
}
