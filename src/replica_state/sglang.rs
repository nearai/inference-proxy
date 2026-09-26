//! SGLang `/v1/loads` adapter: normalizes an engine-specific load snapshot
//! into the neutral `report::{Load, Limits}` view.
//!
//! Never logs response bodies or replica identity — errors are logged (at
//! `debug`) with an error kind only; the caller adds replica context.

use crate::replica_state::report::{Limits, Load};

#[derive(Clone, Debug, PartialEq)]
pub struct ReplicaLoad {
    pub load: Load,
    pub limits: Limits,
    pub engine_version: Option<String>,
    pub sampled_at_ms: u64,
}

/// Sums an unsigned integer field across ranks. `None` if any rank is
/// missing the key, the value isn't a non-negative integer, or the sum
/// overflows `u64`.
fn sum_u64(ranks: &[serde_json::Value], key: &str) -> Option<u64> {
    let mut total: u64 = 0;
    for rank in ranks {
        total = total.checked_add(rank.get(key)?.as_u64()?)?;
    }
    Some(total)
}

/// [`sum_u64`] narrowed to `u32`; `None` if the sum doesn't fit.
fn sum_u32(ranks: &[serde_json::Value], key: &str) -> Option<u32> {
    sum_u64(ranks, key).and_then(|v| u32::try_from(v).ok())
}

/// Sums a floating-point field across ranks. `None` if any rank is missing
/// the key, or the value isn't a number.
fn sum_f64(ranks: &[serde_json::Value], key: &str) -> Option<f64> {
    let mut total: f64 = 0.0;
    for rank in ranks {
        let v = rank.get(key)?.as_f64()?;
        total += v;
    }
    Some(total)
}

/// Minimum `timestamp` (float seconds) across ranks, converted to
/// milliseconds. `None` if any rank lacks a numeric `timestamp`.
fn ts_ms(ranks: &[serde_json::Value]) -> Option<u64> {
    let mut min_secs: Option<f64> = None;
    for rank in ranks {
        let t = rank.get("timestamp")?.as_f64()?;
        min_secs = Some(match min_secs {
            Some(cur) if cur <= t => cur,
            _ => t,
        });
    }
    min_secs.map(|s| (s * 1000.0).trunc() as u64)
}

/// Pure normalization of a `/v1/loads?include=core` body. `None` if it has
/// no `loads` array, the array is empty, or no rank has a `timestamp`.
pub fn parse_sglang_loads(body: &serde_json::Value) -> Option<ReplicaLoad> {
    let loads = body.get("loads")?.as_array()?;
    if loads.is_empty() {
        return None;
    }
    let sampled_at_ms = ts_ms(loads)?;

    let running = sum_u32(loads, "num_running_reqs");
    let queued = sum_u32(loads, "num_waiting_reqs");
    let prefill_backlog_tokens = sum_u64(loads, "num_waiting_uncached_tokens");
    let gen_tps = sum_f64(loads, "gen_throughput");

    let used = sum_u64(loads, "num_used_tokens");
    let max_total = sum_u64(loads, "max_total_num_tokens");
    let kv_usage = match (used, max_total) {
        (Some(used), Some(max_total)) if max_total != 0 => {
            Some((used as f64 / max_total as f64).clamp(0.0, 1.0))
        }
        _ => None,
    };

    let cached_token_ratio = if loads.len() == 1 {
        loads[0].get("cache_hit_rate").and_then(|v| v.as_f64())
    } else {
        None
    };

    let max_running = sum_u32(loads, "max_running_requests").filter(|&v| v != 0);

    let engine_version = body
        .get("version")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Some(ReplicaLoad {
        load: Load {
            running,
            queued,
            prefill_backlog_tokens,
            kv_usage,
            gen_tps,
            cached_token_ratio,
        },
        limits: Limits { max_running },
        engine_version,
        sampled_at_ms,
    })
}

/// `GET {base}/v1/loads?include=core` with a timeout, then parse. Any
/// failure (network, timeout, non-success status, bad JSON, or malformed
/// body) yields `None`. Never logs the response body or the replica's base
/// URL; the caller adds replica context.
pub async fn read_replica(
    client: &reqwest::Client,
    base_url: &str,
    timeout: std::time::Duration,
) -> Option<ReplicaLoad> {
    let url = format!("{}/v1/loads?include=core", base_url.trim_end_matches('/'));
    let resp = match client.get(&url).timeout(timeout).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(
                timed_out = e.is_timeout(),
                "replica_state: sglang /v1/loads request failed"
            );
            return None;
        }
    };
    let resp = match resp.error_for_status() {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(
                status = e.status().map(|s| s.as_u16()),
                "replica_state: sglang /v1/loads returned non-success status"
            );
            return None;
        }
    };
    let body: serde_json::Value = match resp.json().await {
        Ok(b) => b,
        Err(_) => {
            tracing::debug!("replica_state: sglang /v1/loads returned non-JSON body");
            return None;
        }
    };
    parse_sglang_loads(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_rank_maps_fields() {
        let v = serde_json::json!({"version":"0.5.9","loads":[{"timestamp":1790000000.5,"num_running_reqs":14,"num_waiting_reqs":2,
            "num_waiting_uncached_tokens":51200,"num_used_tokens":630,"max_total_num_tokens":1000,"gen_throughput":910.0,
            "cache_hit_rate":0.71,"max_running_requests":32}]});
        let r = parse_sglang_loads(&v).unwrap();
        assert_eq!(
            (r.load.running, r.load.queued, r.load.prefill_backlog_tokens),
            (Some(14), Some(2), Some(51200))
        );
        assert_eq!(r.load.kv_usage, Some(0.63));
        assert_eq!(r.load.gen_tps, Some(910.0));
        assert_eq!(r.load.cached_token_ratio, Some(0.71));
        assert_eq!(r.limits.max_running, Some(32));
        assert_eq!(r.sampled_at_ms, 1_790_000_000_500);
        assert_eq!(r.engine_version.as_deref(), Some("0.5.9"));
    }

    #[test]
    fn two_ranks_sum_counts_and_compute_ratios() {
        let v = serde_json::json!({"loads":[
            {"timestamp":1790000001.0,"num_running_reqs":3,"num_waiting_reqs":1,"num_waiting_uncached_tokens":100,"num_used_tokens":100,"max_total_num_tokens":1000,"gen_throughput":10.0,"cache_hit_rate":0.9,"max_running_requests":16},
            {"timestamp":1790000000.0,"num_running_reqs":5,"num_waiting_reqs":0,"num_waiting_uncached_tokens":50,"num_used_tokens":300,"max_total_num_tokens":1000,"gen_throughput":20.0,"cache_hit_rate":0.1,"max_running_requests":16}]});
        let r = parse_sglang_loads(&v).unwrap();
        assert_eq!(
            (r.load.running, r.load.queued, r.load.prefill_backlog_tokens),
            (Some(8), Some(1), Some(150))
        );
        assert_eq!(r.load.kv_usage, Some(0.2));
        assert_eq!(r.load.gen_tps, Some(30.0));
        assert_eq!(r.load.cached_token_ratio, None);
        assert_eq!(r.limits.max_running, Some(32));
        assert_eq!(r.sampled_at_ms, 1_790_000_000_000); // oldest rank
    }

    #[test]
    fn missing_field_is_null_not_zero() {
        let v = serde_json::json!({"loads":[{"timestamp":1790000000.0,"num_running_reqs":4}]});
        let r = parse_sglang_loads(&v).unwrap();
        assert_eq!(r.load.running, Some(4));
        assert_eq!(r.load.queued, None);
        assert_eq!(r.load.kv_usage, None);
    }

    #[test]
    fn absent_version_is_null() {
        let v = serde_json::json!({"loads":[{"timestamp":1790000000.0,"num_running_reqs":1}]});
        assert_eq!(parse_sglang_loads(&v).unwrap().engine_version, None);
    }

    #[test]
    fn kv_usage_is_clamped_when_used_exceeds_max() {
        let v = serde_json::json!({"loads":[{"timestamp":1790000000.0,
            "num_used_tokens":1500,"max_total_num_tokens":1000}]});
        assert_eq!(parse_sglang_loads(&v).unwrap().load.kv_usage, Some(1.0));
    }

    #[test]
    fn overflowing_counts_are_null() {
        // u64 sum overflow across ranks.
        let v = serde_json::json!({"loads":[
            {"timestamp":1790000000.0,"num_waiting_uncached_tokens":u64::MAX,"num_used_tokens":u64::MAX,"max_total_num_tokens":10,"max_running_requests":u64::MAX},
            {"timestamp":1790000000.0,"num_waiting_uncached_tokens":1,"num_used_tokens":1,"max_total_num_tokens":10,"max_running_requests":1}]});
        let r = parse_sglang_loads(&v).unwrap();
        assert_eq!(r.load.prefill_backlog_tokens, None);
        assert_eq!(r.load.kv_usage, None);
        assert_eq!(r.limits.max_running, None);
        // Fits u64 but not u32.
        let v = serde_json::json!({"loads":[{"timestamp":1790000000.0,
            "num_running_reqs":5_000_000_000u64,"num_waiting_reqs":4_294_967_296u64}]});
        let r = parse_sglang_loads(&v).unwrap();
        assert_eq!((r.load.running, r.load.queued), (None, None));
    }

    #[test]
    fn malformed_is_none() {
        assert!(parse_sglang_loads(&serde_json::json!({"nope":1})).is_none());
        assert!(parse_sglang_loads(&serde_json::json!({"loads":[]})).is_none());
        assert!(
            parse_sglang_loads(&serde_json::json!({"loads":[{"num_running_reqs":1}]})).is_none()
        ); // no timestamp
    }

    #[tokio::test]
    async fn read_replica_uses_timeout_and_parses() {
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Success: single-rank body maps to Some.
        let server = MockServer::start().await;
        let body = serde_json::json!({"version":"0.5.9","loads":[{"timestamp":1790000000.0,
            "num_running_reqs":1,"num_waiting_reqs":0,"num_waiting_uncached_tokens":0,
            "num_used_tokens":10,"max_total_num_tokens":100,"gen_throughput":1.0,
            "cache_hit_rate":0.5,"max_running_requests":8}]});
        Mock::given(method("GET"))
            .and(path("/v1/loads"))
            .and(query_param("include", "core"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;
        let client = reqwest::Client::new();
        let r = read_replica(&client, &server.uri(), std::time::Duration::from_secs(2)).await;
        assert!(r.is_some());
        assert_eq!(r.unwrap().sampled_at_ms, 1_790_000_000_000);

        // Timeout: a 2s server delay with a 200ms client timeout yields
        // None well before the server would answer.
        let slow_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/loads"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(&body)
                    .set_delay(std::time::Duration::from_secs(2)),
            )
            .mount(&slow_server)
            .await;
        let start = std::time::Instant::now();
        let r = read_replica(
            &client,
            &slow_server.uri(),
            std::time::Duration::from_millis(200),
        )
        .await;
        let elapsed = start.elapsed();
        assert!(r.is_none());
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "expected timeout well under 1s, took {elapsed:?}"
        );

        // HTTP 500: non-success status yields None.
        let err_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/loads"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&err_server)
            .await;
        let r = read_replica(
            &client,
            &err_server.uri(),
            std::time::Duration::from_secs(2),
        )
        .await;
        assert!(r.is_none());
    }
}
