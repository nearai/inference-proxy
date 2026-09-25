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
    /// `REPLICA_STATE_HOST_ID` (required once the feature is enabled).
    pub host_id: String,
    /// `REPLICA_STATE_REPLICA_IDS` (required; one per `VLLM_BACKEND_URLS`
    /// entry, same order).
    pub replica_ids: Vec<String>,
    /// `REPLICA_STATE_INTERVAL_MS` (optional, clamped to 200..=2000,
    /// default [`DEFAULT_INTERVAL_MS`]).
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
            Err(_) => {
                // Fall back to a manual best-effort parse: substring after
                // the last '@' (strips credentials) up to the next '/'.
                let after_at = self
                    .redis_url
                    .rsplit_once('@')
                    .map(|(_, rest)| rest)
                    .unwrap_or(&self.redis_url);
                after_at
                    .split('/')
                    .next()
                    .unwrap_or("<unknown>")
                    .to_string()
            }
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
}
