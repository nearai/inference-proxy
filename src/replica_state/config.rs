use std::time::Duration;

/// Default publish interval when `REPLICA_STATE_INTERVAL_MS` is unset.
pub const DEFAULT_INTERVAL_MS: u64 = 500;

const MIN_INTERVAL_MS: u64 = 200;
const MAX_INTERVAL_MS: u64 = 2000;

/// Opt-in configuration for publishing signed per-replica load frames to
/// Redis. The feature is enabled only when `REPLICA_STATE_REDIS_URL` is set
/// to a non-blank value; callers construct this via [`Self::from_env`] (or
/// [`Self::from_lookup`] in tests) and treat any error as "log and disable",
/// never abort the process.
pub struct ReplicaStateConfig {
    /// `REPLICA_STATE_REDIS_URL`. Presence (non-blank) is what enables the
    /// feature. Never logged or included in `Debug` output.
    pub redis_url: String,
    /// `REPLICA_STATE_REDIS_CA_CERT` (optional, `rediss://` only): PEM CA the
    /// Redis certificate chains to; replaces the system trust store for Redis
    /// only. Literal `\n` sequences are accepted for single-line env values.
    pub redis_ca_cert: Option<String>,
    /// `REPLICA_STATE_HOST_ID` (required once the feature is enabled).
    pub host_id: String,
    /// `REPLICA_STATE_REPLICA_IDS` (required; one per `VLLM_BACKEND_URLS`
    /// entry, same order).
    pub replica_ids: Vec<String>,
    /// `REPLICA_STATE_INTERVAL_MS` (optional, must be within 200..=2000
    /// (otherwise the feature is disabled with an error), default
    /// [`DEFAULT_INTERVAL_MS`]).
    pub interval: Duration,
}

impl ReplicaStateConfig {
    /// Builds the config from an arbitrary key lookup function, or returns
    /// `Ok(None)` when `REPLICA_STATE_REDIS_URL` is unset or blank (the
    /// feature stays off). Any other missing/invalid value is a hard error
    /// so callers can log and disable rather than run with bad config.
    pub fn from_lookup(
        get: impl Fn(&str) -> Option<String>,
        backend_count: usize,
    ) -> anyhow::Result<Option<Self>> {
        let redis_url = match get("REPLICA_STATE_REDIS_URL") {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => return Ok(None),
        };
        // Neither message may include the URL: it can carry a password.
        let scheme = url::Url::parse(&redis_url)
            .ok()
            .map(|u| u.scheme().to_string());
        let scheme_ok = matches!(scheme.as_deref(), Some("redis" | "rediss"));
        if !scheme_ok || redis::Client::open(redis_url.as_str()).is_err() {
            anyhow::bail!(
                "REPLICA_STATE_REDIS_URL is not a valid redis:// or rediss:// URL (value not shown)"
            );
        }
        let redis_ca_cert = match get("REPLICA_STATE_REDIS_CA_CERT") {
            Some(v) if !v.trim().is_empty() => Some(v.trim().replace("\\n", "\n")),
            _ => None,
        };
        if let Some(pem) = &redis_ca_cert {
            if scheme.as_deref() != Some("rediss") {
                anyhow::bail!("REPLICA_STATE_REDIS_CA_CERT requires a rediss:// URL");
            }
            if !pem.contains("-----BEGIN CERTIFICATE-----")
                || super::redis_sink::client(&redis_url, Some(pem)).is_err()
            {
                anyhow::bail!("REPLICA_STATE_REDIS_CA_CERT is not a valid PEM certificate");
            }
        }

        let host_id = match get("REPLICA_STATE_HOST_ID") {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => anyhow::bail!(
                "REPLICA_STATE_HOST_ID is required when REPLICA_STATE_REDIS_URL is set"
            ),
        };

        let replica_ids_raw = match get("REPLICA_STATE_REPLICA_IDS") {
            Some(v) if !v.trim().is_empty() => v,
            _ => anyhow::bail!(
                "REPLICA_STATE_REPLICA_IDS is required when REPLICA_STATE_REDIS_URL is set"
            ),
        };
        let replica_ids: Vec<String> = replica_ids_raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if replica_ids.len() != backend_count {
            anyhow::bail!(
                "REPLICA_STATE_REPLICA_IDS has {} entr{} but there are {backend_count} backend(s); they must match 1:1",
                replica_ids.len(),
                if replica_ids.len() == 1 { "y" } else { "ies" }
            );
        }
        let mut seen = std::collections::HashSet::with_capacity(replica_ids.len());
        for id in &replica_ids {
            if !seen.insert(id) {
                anyhow::bail!("REPLICA_STATE_REPLICA_IDS contains a duplicate entry: {id}");
            }
        }

        let interval_ms = match get("REPLICA_STATE_INTERVAL_MS") {
            Some(v) if !v.trim().is_empty() => v
                .trim()
                .parse::<u64>()
                .map_err(|e| anyhow::anyhow!("REPLICA_STATE_INTERVAL_MS: {e}"))?,
            _ => DEFAULT_INTERVAL_MS,
        };
        if !(MIN_INTERVAL_MS..=MAX_INTERVAL_MS).contains(&interval_ms) {
            anyhow::bail!(
                "REPLICA_STATE_INTERVAL_MS must be between {MIN_INTERVAL_MS} and {MAX_INTERVAL_MS}, got {interval_ms}"
            );
        }

        Ok(Some(Self {
            redis_url,
            redis_ca_cert,
            host_id,
            replica_ids,
            interval: Duration::from_millis(interval_ms),
        }))
    }

    /// Builds the config from real process environment variables.
    pub fn from_env(backend_count: usize) -> anyhow::Result<Option<Self>> {
        Self::from_lookup(|k| std::env::var(k).ok(), backend_count)
    }

    /// Host (and port, if present) of `redis_url`, safe to log. Never
    /// includes user info (credentials) or the path/query.
    pub fn redis_host_for_logs(&self) -> String {
        match url::Url::parse(&self.redis_url) {
            Ok(u) => match (u.host_str(), u.port()) {
                (Some(host), Some(port)) => format!("{host}:{port}"),
                (Some(host), None) => host.to_string(),
                _ => "<unknown>".to_string(),
            },
            // Unreachable in practice: `from_lookup` rejects unparseable URLs.
            Err(_) => "<unparseable>".to_string(),
        }
    }
}

impl std::fmt::Debug for ReplicaStateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplicaStateConfig")
            .field("host_id", &self.host_id)
            .field("replica_ids", &self.replica_ids)
            .field("interval", &self.interval)
            .field("redis_host", &self.redis_host_for_logs())
            .field("redis_ca_cert", &self.redis_ca_cert.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    fn look(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn off_without_redis_url() {
        assert!(
            ReplicaStateConfig::from_lookup(look(&[("REPLICA_STATE_HOST_ID", "h")]), 1)
                .unwrap()
                .is_none()
        );
        assert!(
            ReplicaStateConfig::from_lookup(look(&[("REPLICA_STATE_REDIS_URL", "  ")]), 1)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn requires_host_id_and_replica_ids() {
        let e = ReplicaStateConfig::from_lookup(
            look(&[
                ("REPLICA_STATE_REDIS_URL", "redis://r:6379"),
                ("REPLICA_STATE_REPLICA_IDS", "r1"),
            ]),
            1,
        )
        .unwrap_err();
        assert!(e.to_string().contains("REPLICA_STATE_HOST_ID"));
        let e = ReplicaStateConfig::from_lookup(
            look(&[
                ("REPLICA_STATE_REDIS_URL", "redis://r:6379"),
                ("REPLICA_STATE_HOST_ID", "h"),
            ]),
            1,
        )
        .unwrap_err();
        assert!(e.to_string().contains("REPLICA_STATE_REPLICA_IDS"));
    }

    #[test]
    fn replica_ids_must_match_backend_count_and_be_unique() {
        let base = [
            ("REPLICA_STATE_REDIS_URL", "redis://r:6379"),
            ("REPLICA_STATE_HOST_ID", "h"),
        ];
        let mk = |ids: &'static str| {
            let mut v = base.to_vec();
            v.push(("REPLICA_STATE_REPLICA_IDS", ids));
            v
        };
        assert!(ReplicaStateConfig::from_lookup(look(&mk("r1")), 2).is_err());
        assert!(ReplicaStateConfig::from_lookup(look(&mk("r1,r1")), 2).is_err());
        let ok = ReplicaStateConfig::from_lookup(look(&mk(" r1 , r2 ")), 2)
            .unwrap()
            .unwrap();
        assert_eq!(ok.replica_ids, vec!["r1", "r2"]);
        assert_eq!(
            ok.interval,
            std::time::Duration::from_millis(DEFAULT_INTERVAL_MS)
        );
    }

    #[test]
    fn interval_bounds() {
        let mk = |ms: &str| {
            look(&[
                ("REPLICA_STATE_REDIS_URL", "redis://r:6379"),
                ("REPLICA_STATE_HOST_ID", "h"),
                ("REPLICA_STATE_REPLICA_IDS", "r1"),
                ("REPLICA_STATE_INTERVAL_MS", ms),
            ])
        };
        assert!(ReplicaStateConfig::from_lookup(mk("100"), 1).is_err());
        assert!(ReplicaStateConfig::from_lookup(mk("abc"), 1).is_err());
        assert_eq!(
            ReplicaStateConfig::from_lookup(mk("250"), 1)
                .unwrap()
                .unwrap()
                .interval
                .as_millis(),
            250
        );
    }

    #[test]
    fn redis_url_must_parse_with_a_redis_scheme_and_error_hides_it() {
        let mk = |url: &'static str| {
            look(&[
                ("REPLICA_STATE_REDIS_URL", url),
                ("REPLICA_STATE_HOST_ID", "h"),
                ("REPLICA_STATE_REPLICA_IDS", "r1"),
            ])
        };
        for bad in [
            "not a url s3cret",
            "http://user:s3cret@redis.internal:6379",
            "unix:///tmp/s3cret.sock",
            "redis://user:s3cret@[::1",
        ] {
            let e = ReplicaStateConfig::from_lookup(mk(bad), 1).unwrap_err();
            let msg = format!("{e:#}");
            assert!(msg.contains("REPLICA_STATE_REDIS_URL"), "{msg}");
            assert!(!msg.contains("s3cret"), "error leaked the URL: {msg}");
        }
        for good in ["redis://r:6379", "rediss://user:pw@r:6380/0"] {
            assert!(ReplicaStateConfig::from_lookup(mk(good), 1)
                .unwrap()
                .is_some());
        }
    }

    #[test]
    fn debug_and_log_host_never_show_credentials() {
        let c = ReplicaStateConfig::from_lookup(
            look(&[
                (
                    "REPLICA_STATE_REDIS_URL",
                    "rediss://user:s3cret@redis.internal:6380/0",
                ),
                ("REPLICA_STATE_HOST_ID", "h"),
                ("REPLICA_STATE_REPLICA_IDS", "r1"),
            ]),
            1,
        )
        .unwrap()
        .unwrap();
        let d = format!("{c:?}");
        assert!(!d.contains("s3cret") && !d.contains("user:"));
        assert_eq!(c.redis_host_for_logs(), "redis.internal:6380");
    }

    /// Public test CA certificate (no key), used only to exercise PEM parsing.
    const TEST_CA: &str = "-----BEGIN CERTIFICATE-----\nMIIBmDCCAT2gAwIBAgIUBO8axf+bI+fy6nCWKxvbDQK8Vh8wCgYIKoZIzj0EAwIw\nIDEeMBwGA1UEAwwVcmVwbGljYS1zdGF0ZS10ZXN0LWNhMCAXDTI2MDkyNjA0MDQx\nOVoYDzIxMjYwOTAyMDQwNDE5WjAgMR4wHAYDVQQDDBVyZXBsaWNhLXN0YXRlLXRl\nc3QtY2EwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAQvZoRSR2lpnIRoOtWMg4/A\nC8waZ+M5FKrVPKnTbEQ9Rg8FsgLbiGo98qj8xMiCnDLBtZWu2kIS8aUkYKBTi9i/\no1MwUTAdBgNVHQ4EFgQU0sUHDnN4gedUYXUcbPvjGM8z6A8wHwYDVR0jBBgwFoAU\n0sUHDnN4gedUYXUcbPvjGM8z6A8wDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQD\nAgNJADBGAiEAmhuVv3kXXoW/L/c1OLtbstq6AbI/PA/9VQpxzr0tDoECIQDDyvF8\n2A0KtuzhoAcZZVU2L0OjHlMqrr0f57OEw5SfEQ==\n-----END CERTIFICATE-----";

    fn with_ca(url: &str, ca: &str) -> anyhow::Result<Option<ReplicaStateConfig>> {
        ReplicaStateConfig::from_lookup(
            look(&[
                ("REPLICA_STATE_REDIS_URL", url),
                ("REPLICA_STATE_REDIS_CA_CERT", ca),
                ("REPLICA_STATE_HOST_ID", "h"),
                ("REPLICA_STATE_REPLICA_IDS", "r1"),
            ]),
            1,
        )
    }

    #[test]
    fn redis_ca_cert_accepts_pem_and_escaped_newlines() {
        let cfg = with_ca("rediss://r:6379", TEST_CA).unwrap().unwrap();
        assert_eq!(cfg.redis_ca_cert.as_deref(), Some(TEST_CA));
        let escaped = TEST_CA.replace('\n', "\\n");
        let cfg = with_ca("rediss://r:6379", &escaped).unwrap().unwrap();
        assert_eq!(cfg.redis_ca_cert.as_deref(), Some(TEST_CA));
        assert!(format!("{cfg:?}").contains("redis_ca_cert: true"));
    }

    #[test]
    fn redis_ca_cert_accepts_uppercase_tls_scheme() {
        assert!(with_ca("REDISS://r:6379", TEST_CA).unwrap().is_some());
    }

    #[test]
    fn redis_ca_cert_rejects_plain_redis_and_bad_pem() {
        let e = with_ca("redis://r:6379", TEST_CA).unwrap_err();
        assert!(e.to_string().contains("rediss://"));
        for bad in [
            "not a pem",
            "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----",
        ] {
            let e = with_ca("rediss://r:6379", bad).unwrap_err();
            assert!(e.to_string().contains("not a valid PEM"), "{bad}");
        }
    }
}
