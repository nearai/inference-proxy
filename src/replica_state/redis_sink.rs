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

    /// Sends [`build_pipeline`] in one round trip.
    pub async fn publish(&mut self, frames: &[(String, Envelope)]) -> anyhow::Result<()> {
        if frames.is_empty() {
            return Ok(());
        }
        build_pipeline(&self.host_id, &self.stream, frames)?
            .query_async::<()>(&mut self.conn)
            .await
            .map_err(|e| anyhow!("redis publish failed: {:?}", e.kind()))
    }
}

/// One pipeline: SET state_key json EX 5 for each frame, then
/// XADD stream MAXLEN ~ 20000 * env json.
fn build_pipeline(
    host_id: &str,
    stream: &str,
    frames: &[(String, Envelope)],
) -> anyhow::Result<redis::Pipeline> {
    let jsons = frames
        .iter()
        .map(|(_, env)| serde_json::to_string(env))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pipe = redis::pipe();
    for ((replica_id, _), json) in frames.iter().zip(&jsons) {
        pipe.cmd("SET")
            .arg(state_key(host_id, replica_id))
            .arg(json)
            .arg("EX")
            .arg(KEY_TTL_SECS)
            .ignore();
    }
    for json in &jsons {
        pipe.cmd("XADD")
            .arg(stream)
            .arg("MAXLEN")
            .arg("~")
            .arg(STREAM_MAXLEN)
            .arg("*")
            .arg("env")
            .arg(json)
            .ignore();
    }
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

    #[test]
    fn pipeline_sets_ttl_keys_then_appends_capped_stream() {
        let env = |n: u32| Envelope {
            frame: format!("{{\"seq\":{n}}}"),
            sig: format!("sig{n}"),
            key_id: "0123456789abcdef".to_string(),
        };
        let frames = [("r1".to_string(), env(1)), ("r2".to_string(), env(2))];
        let pipe = build_pipeline("gpu01", &stream_key("gpu01"), &frames).unwrap();
        let cmds: Vec<Vec<String>> = pipe.cmd_iter().map(args).collect();
        let j = |n: u32| serde_json::to_string(&env(n)).unwrap();
        assert_eq!(
            cmds,
            vec![
                vec![
                    "SET".into(),
                    "replica:gpu01:r1".into(),
                    j(1),
                    "EX".into(),
                    "5".into()
                ],
                vec![
                    "SET".into(),
                    "replica:gpu01:r2".into(),
                    j(2),
                    "EX".into(),
                    "5".into()
                ],
                vec![
                    "XADD".into(),
                    "replica:gpu01:frames".into(),
                    "MAXLEN".into(),
                    "~".into(),
                    "20000".into(),
                    "*".into(),
                    "env".into(),
                    j(1)
                ],
                vec![
                    "XADD".into(),
                    "replica:gpu01:frames".into(),
                    "MAXLEN".into(),
                    "~".into(),
                    "20000".into(),
                    "*".into(),
                    "env".into(),
                    j(2)
                ],
            ]
        );
    }

    #[test]
    fn key_layout_is_per_host() {
        assert_eq!(state_key("glm53-gpu03", "r1"), "replica:glm53-gpu03:r1");
        assert_eq!(stream_key("glm53-gpu03"), "replica:glm53-gpu03:frames");
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
    async fn publish_sets_ttl_keys_and_appends_stream() {
        let url = std::env::var("REPLICA_STATE_TEST_REDIS_URL")
            .expect("REPLICA_STATE_TEST_REDIS_URL must point at a disposable Redis");
        install_crypto_provider();
        let host = format!("test-{}", uuid::Uuid::new_v4());
        let ca = std::env::var("REPLICA_STATE_TEST_REDIS_CA_CERT").ok();
        let mut sink = RedisSink::connect(&url, ca.as_deref(), &host)
            .await
            .unwrap();
        let env = |n: u32| Envelope {
            frame: format!("{{\"seq\":{n}}}"),
            sig: format!("sig{n}"),
            key_id: "0123456789abcdef".to_string(),
        };
        let (e1, e2) = (env(1), env(2));
        sink.publish(&[("r1".to_string(), e1.clone()), ("r2".to_string(), e2)])
            .await
            .unwrap();

        let mut plain = client(&url, ca.as_deref())
            .unwrap()
            .get_multiplexed_async_connection()
            .await
            .unwrap();
        let got: String = redis::cmd("GET")
            .arg(state_key(&host, "r1"))
            .query_async(&mut plain)
            .await
            .unwrap();
        assert_eq!(got, serde_json::to_string(&e1).unwrap());
        let ttl: i64 = redis::cmd("TTL")
            .arg(state_key(&host, "r1"))
            .query_async(&mut plain)
            .await
            .unwrap();
        assert!((1..=5).contains(&ttl), "ttl={ttl}");
        let len: u64 = redis::cmd("XLEN")
            .arg(stream_key(&host))
            .query_async(&mut plain)
            .await
            .unwrap();
        assert_eq!(len, 2, "xlen={len}");
    }
}
