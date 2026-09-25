# Replica State → Redis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each in-CVM host proxy reads its SGLang replicas' `/v1/loads` every 500 ms and writes one signed, normalized `ReplicaReport` v1 frame per replica to Redis: a 5 s TTL key for the latest state, plus a capped per-host stream as a short buffer. It is opt-in, off by default, and changes nothing on the request path.

**Architecture:** A new, self-contained `replica_state` module:
- `parse_sglang_loads` is a pure normalizing function (Adapter). It turns SGLang's `/v1/loads` into one engine-neutral `ReplicaLoad`.
- `Publisher::tick` reads every replica concurrently and builds one frame per replica. All frames in a tick share one `seq`. Each frame is signed with a per-boot Ed25519 report key, domain-separated.
- A concrete `RedisSink` writes the frames in one pipeline.
- `spawn_replica_state_publisher` (Facade) is the only entry point `main.rs` calls.
- The report key's public half is recorded in the dstack event log (`emit_event`), which the TDX quote covers. `report_data` stays unchanged.

**Tech Stack:** Rust (tokio, reqwest, serde_json, ed25519-dalek, base64, sha2, uuid, rand, all already dependencies). New: `redis` 1.x (BSD-3-Clause, `default-features = false`, features `tokio-comp`, `tokio-rustls-comp`, `connection-manager`), plus `rustls` as a direct dependency (already in the lock) to install the crypto provider. Tests use `wiremock` (already a dev dependency).

**Spec:** Design doc "Inference Placement Map" (https://claude.ai/artifact/T3WkANwFetyUYsHhMBNheo): "Latest recommendation" (stage 1); "Replica lifecycle and state reporting" §2 and §4 (state contract, Redis transport, who can trust a frame, staleness); gated stage 1 in "Baseline". This plan was revised after a principal-engineer review (see "Review changes" at the end).

## Global Constraints

- **Off unless `REPLICA_STATE_REDIS_URL` is set.** Unset means behaviour, startup and tests are unchanged.
- **Non-intrusive.** Existing code edits are limited to `src/lib.rs` (one `pub mod` line), `src/main.rs` (one `match` block), `Cargo.toml`/`Cargo.lock`, and docs. `Config`, `AppState`, routes, `proxy.rs`, `attestation.rs`, `backend_pool.rs` and the integration-test helpers are not touched. `BackendPool` is read only through its public `backends()`.
- **Bad configuration never stops the proxy.** An invalid `REPLICA_STATE_*` value logs an error and disables the feature.
- **IDs and numbers only** in frames: no prompts, org IDs, affinity keys, or content-derived hashes.
- **Never log the Redis URL** (it may hold a password), a connection error formatted with it, or any key material. Frame contents only at `debug`.
- **Unknown is `null`, never `0`.** A core field missing from `/v1/loads` JSON is `null`. A failed read makes every load field `null`.
- **Signing:** `sig = base64(ed25519(report_key, b"nearai-replica-report-v1\n" ++ frame_bytes))`, where `frame_bytes` is the exact UTF-8 of the `frame` string in the envelope. Readers verify the bytes they received, then parse. Nothing re-serializes.
- **`seq` is per tick:** one value shared by every frame of that tick, starting at 1 per boot.
- **Attestation `report_data` layout is not changed.**
- Reproducible build: no new required build env vars or arguments. `cargo deny check` must pass. Run `cargo fmt` before each commit.

## Design patterns (refactoring.guru catalog)

| Pattern | Where | Why |
|---|---|---|
| **Adapter** | `parse_sglang_loads(&Value) -> Option<ReplicaLoad>` | Converts SGLang's engine-specific load snapshot into the engine-neutral view readers use. It's a pure function: one engine today. When vLLM arrives it becomes a `match` on a closed `Engine` enum, not a trait. |
| **Facade** | `replica_state::spawn_replica_state_publisher` | The single entry point from `main.rs`. It hides the key, the binding, reads, retries, Redis and metrics. |

Deliberately not used: Strategy or trait objects for sinks or engines (one implementation each; `tick()` returning frames is the test seam), Template Method (a plain function is clearer), Observer, Decorator, Singleton.

## Review Focus

1. **A wedged scheduler** serves a stale snapshot from shared memory. `engine_sampled_at_ms` comes from the engine's own `timestamp`, so the frame shows its age. (Task 3 and Task 5 tests.)
2. **An engine that is unreachable or slow:** the tick finishes within the interval. That replica's frame keeps its last successful sample time, has all load fields `null`, and moves `warming → ready → unhealthy` after 3 consecutive failures. (Task 5 tests.)
3. **Redis down at boot or mid-run:** the proxy serves normally. The publisher retries the connection every 5 s, counts failures and never panics. That includes `rediss://` with two rustls providers in the lock. (Task 4 and Task 5 tests.)
4. **Several DP ranks in `/v1/loads`:** counts and throughput are summed, while ratios are computed, never summed. (Task 3 test.)
5. **A typo in a `REPLICA_STATE_*` variable:** the proxy starts, logs one error and publishes nothing. (Task 1 and Task 5 tests.)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/replica_state/mod.rs` | Submodule declarations; `spawn_replica_state_publisher` (Facade) |
| `src/replica_state/config.rs` | `ReplicaStateConfig::from_lookup(get, backend_count)`; `from_env` wrapper; manual `Debug` |
| `src/replica_state/report.rs` | `ReplicaReport` v1, `Envelope` (frame as string), `seal`, `verify` |
| `src/replica_state/sglang.rs` | `parse_sglang_loads` (Adapter), `read_replica` (HTTP + timeout) |
| `src/replica_state/report_key.rs` | Per-boot `ReportKey`; `bind_to_attestation` (dstack `emit_event`) |
| `src/replica_state/redis_sink.rs` | Concrete `RedisSink`: connect (rustls provider, timeouts), `publish` pipeline |
| `src/replica_state/publisher.rs` | `Publisher` with `Vec<ReplicaSlot>`; `tick` |
| `src/lib.rs`, `src/main.rs` | One `pub mod` line; one `match` block to spawn |
| `docs/replica-state.md` | Operator doc |

---

### Task 1: Configuration (inside the new module)

**Files:**
- Create: `src/replica_state/mod.rs` (submodule declarations), `src/replica_state/config.rs`
- Modify: `src/lib.rs` (`pub mod replica_state;`)

**Interfaces:**
- Produces:
```rust
pub const DEFAULT_INTERVAL_MS: u64 = 500;
pub struct ReplicaStateConfig {
    pub redis_url: String,           // REPLICA_STATE_REDIS_URL (enables the feature)
    pub host_id: String,             // REPLICA_STATE_HOST_ID (required)
    pub replica_ids: Vec<String>,    // REPLICA_STATE_REPLICA_IDS (required; one per VLLM_BACKEND_URLS entry, same order)
    pub interval: std::time::Duration, // REPLICA_STATE_INTERVAL_MS (optional, 200..=2000, default 500)
}
impl ReplicaStateConfig {
    /// Ok(None) when REPLICA_STATE_REDIS_URL is unset or blank.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>, backend_count: usize) -> anyhow::Result<Option<Self>>;
    pub fn from_env(backend_count: usize) -> anyhow::Result<Option<Self>> { Self::from_lookup(|k| std::env::var(k).ok(), backend_count) }
    /// Host part of the URL only, for logs (never credentials).
    pub fn redis_host_for_logs(&self) -> String;
}
impl std::fmt::Debug for ReplicaStateConfig { /* prints host_id, replica_ids, interval, redis_host_for_logs(); never redis_url */ }
```

- [ ] **Step 1: Write failing tests** in `config.rs`. They use a `HashMap` lookup, so no environment is mutated:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    fn look(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn off_without_redis_url() {
        assert!(ReplicaStateConfig::from_lookup(look(&[("REPLICA_STATE_HOST_ID", "h")]), 1).unwrap().is_none());
        assert!(ReplicaStateConfig::from_lookup(look(&[("REPLICA_STATE_REDIS_URL", "  ")]), 1).unwrap().is_none());
    }

    #[test]
    fn requires_host_id_and_replica_ids() {
        let e = ReplicaStateConfig::from_lookup(look(&[("REPLICA_STATE_REDIS_URL", "redis://r:6379"), ("REPLICA_STATE_REPLICA_IDS", "r1")]), 1).unwrap_err();
        assert!(e.to_string().contains("REPLICA_STATE_HOST_ID"));
        let e = ReplicaStateConfig::from_lookup(look(&[("REPLICA_STATE_REDIS_URL", "redis://r:6379"), ("REPLICA_STATE_HOST_ID", "h")]), 1).unwrap_err();
        assert!(e.to_string().contains("REPLICA_STATE_REPLICA_IDS"));
    }

    #[test]
    fn replica_ids_must_match_backend_count_and_be_unique() {
        let base = [("REPLICA_STATE_REDIS_URL", "redis://r:6379"), ("REPLICA_STATE_HOST_ID", "h")];
        let mk = |ids: &str| { let mut v = base.to_vec(); v.push(("REPLICA_STATE_REPLICA_IDS", ids)); v };
        assert!(ReplicaStateConfig::from_lookup(look(&mk("r1")), 2).is_err());
        assert!(ReplicaStateConfig::from_lookup(look(&mk("r1,r1")), 2).is_err());
        let ok = ReplicaStateConfig::from_lookup(look(&mk(" r1 , r2 ")), 2).unwrap().unwrap();
        assert_eq!(ok.replica_ids, vec!["r1", "r2"]);
        assert_eq!(ok.interval, std::time::Duration::from_millis(DEFAULT_INTERVAL_MS));
    }

    #[test]
    fn interval_bounds() {
        let mk = |ms: &str| look(&[("REPLICA_STATE_REDIS_URL", "redis://r:6379"), ("REPLICA_STATE_HOST_ID", "h"), ("REPLICA_STATE_REPLICA_IDS", "r1"), ("REPLICA_STATE_INTERVAL_MS", ms)]);
        assert!(ReplicaStateConfig::from_lookup(mk("100"), 1).is_err());
        assert!(ReplicaStateConfig::from_lookup(mk("abc"), 1).is_err());
        assert_eq!(ReplicaStateConfig::from_lookup(mk("250"), 1).unwrap().unwrap().interval.as_millis(), 250);
    }

    #[test]
    fn debug_and_log_host_never_show_credentials() {
        let c = ReplicaStateConfig::from_lookup(look(&[("REPLICA_STATE_REDIS_URL", "rediss://user:s3cret@redis.internal:6380/0"), ("REPLICA_STATE_HOST_ID", "h"), ("REPLICA_STATE_REPLICA_IDS", "r1")]), 1).unwrap().unwrap();
        let d = format!("{c:?}");
        assert!(!d.contains("s3cret") && !d.contains("user:"));
        assert_eq!(c.redis_host_for_logs(), "redis.internal:6380");
    }
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::config` → FAIL.
- [ ] **Step 3: Implement** `from_lookup` with the checks the tests imply. Trim values, and treat blank as unset. Parse `redis_host_for_logs` with `url::Url` if `url` is already a dependency; otherwise take the substring after the last `@` and up to the next `/`. Add `pub mod replica_state;` to `lib.rs`, and in `mod.rs` declare only `pub mod config;` for now.
- [ ] **Step 4: Run** `cargo test --lib replica_state::config && cargo test` → PASS (full suite unchanged).
- [ ] **Step 5: Commit** `feat(replica-state): opt-in config for per-replica state publishing`

---

### Task 2: ReplicaReport v1 and the signing envelope

**Files:**
- Create: `src/replica_state/report.rs`

**Interfaces:**
- Produces:
```rust
pub const SIGNING_DOMAIN: &[u8] = b"nearai-replica-report-v1\n";
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)] #[serde(rename_all = "lowercase")]
pub enum Lifecycle { Warming, Ready, Degraded, Unhealthy, Draining, Drained } // v1 writer emits Warming/Ready/Unhealthy
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)] #[serde(rename_all = "lowercase")]
pub enum Engine { Sglang }
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Limits { pub max_running: Option<u32> }
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Load {
    pub running: Option<u32>, pub queued: Option<u32>, pub prefill_backlog_tokens: Option<u64>,
    pub kv_usage: Option<f64>, pub gen_tps: Option<f64>, pub cached_token_ratio: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReplicaReport {
    pub schema: u8, pub host_id: String, pub replica_id: String, pub boot_id: String, pub seq: u64,
    pub engine_sampled_at_ms: u64, pub reported_at_ms: u64, pub lifecycle_state: Lifecycle,
    pub model: String, pub engine: Engine, pub engine_version: Option<String>,
    pub limits: Limits, pub load: Load, pub proxy_inflight: u32, pub report_key_id: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Envelope { pub frame: String, pub sig: String }
pub fn seal(report: &ReplicaReport, key: &ed25519_dalek::SigningKey) -> Envelope;
/// Verify, then parse. None on a bad signature or bad JSON.
pub fn open(env: &Envelope, key: &ed25519_dalek::VerifyingKey) -> Option<ReplicaReport>;
```

- [ ] **Step 1: Write failing tests:**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use ed25519_dalek::{Signer, SigningKey};

    fn report() -> ReplicaReport {
        ReplicaReport {
            schema: 1, host_id: "glm53-gpu03".into(), replica_id: "r1".into(),
            boot_id: "00000000-0000-4000-8000-000000000001".into(), seq: 7,
            engine_sampled_at_ms: 1_790_000_000_011, reported_at_ms: 1_790_000_000_123,
            lifecycle_state: Lifecycle::Ready, model: "z-ai/glm-5.3-flash".into(),
            engine: Engine::Sglang, engine_version: None,
            limits: Limits { max_running: Some(32) },
            load: Load { running: Some(14), queued: Some(0), prefill_backlog_tokens: Some(51200), ..Default::default() },
            proxy_inflight: 16, report_key_id: "0123456789abcdef".into(),
        }
    }

    #[test]
    fn seal_open_roundtrip_through_json() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let wire = serde_json::to_string(&seal(&report(), &sk)).unwrap();
        let env: Envelope = serde_json::from_str(&wire).unwrap();
        assert_eq!(open(&env, &sk.verifying_key()), Some(report()));
    }

    #[test]
    fn frame_keeps_nulls_and_is_compact() {
        let env = seal(&report(), &SigningKey::from_bytes(&[7u8; 32]));
        assert!(env.frame.starts_with(r#"{"schema":1,"host_id":"glm53-gpu03""#));
        assert!(env.frame.contains(r#""kv_usage":null"#));
        assert!(!env.frame.contains(": "));
    }

    #[test]
    fn tampered_frame_or_wrong_key_fails() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let mut env = seal(&report(), &sk);
        env.frame = env.frame.replace(r#""running":14"#, r#""running":0"#);
        assert!(open(&env, &sk.verifying_key()).is_none());
        let env = seal(&report(), &sk);
        assert!(open(&env, &SigningKey::from_bytes(&[8u8; 32]).verifying_key()).is_none());
    }

    #[test]
    fn signature_without_domain_prefix_is_rejected() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let frame = serde_json::to_string(&report()).unwrap();
        let sig = base64::engine::general_purpose::STANDARD.encode(sk.sign(frame.as_bytes()).to_bytes());
        assert!(open(&Envelope { frame, sig }, &sk.verifying_key()).is_none());
    }
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::report` → FAIL.
- [ ] **Step 3: Implement:**

```rust
use base64::Engine as _;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

fn message(frame: &str) -> Vec<u8> { let mut m = SIGNING_DOMAIN.to_vec(); m.extend_from_slice(frame.as_bytes()); m }

pub fn seal(report: &ReplicaReport, key: &SigningKey) -> Envelope {
    let frame = serde_json::to_string(report).expect("ReplicaReport always serializes");
    let sig = base64::engine::general_purpose::STANDARD.encode(key.sign(&message(&frame)).to_bytes());
    Envelope { frame, sig }
}

pub fn open(env: &Envelope, key: &VerifyingKey) -> Option<ReplicaReport> {
    let raw = base64::engine::general_purpose::STANDARD.decode(&env.sig).ok()?;
    let sig = Signature::from_bytes(&<[u8; 64]>::try_from(raw.as_slice()).ok()?);
    key.verify(&message(&env.frame), &sig).ok()?;
    serde_json::from_str(&env.frame).ok()
}
```
Register `pub mod report;` in `mod.rs`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(replica-state): ReplicaReport v1 and signed envelope (frame as signed string)`

---

### Task 3: SGLang adapter

**Files:**
- Create: `src/replica_state/sglang.rs`

**Interfaces:**
- Consumes: `report::{Load, Limits}`.
- Produces:
```rust
#[derive(Clone, Debug, PartialEq)]
pub struct ReplicaLoad { pub load: Load, pub limits: Limits, pub engine_version: Option<String>, pub sampled_at_ms: u64 }
/// Pure normalization of a `/v1/loads?include=core` body. None if it has no `loads` array or no rank with a timestamp.
pub fn parse_sglang_loads(body: &serde_json::Value) -> Option<ReplicaLoad>;
/// GET {base}/v1/loads?include=core with a timeout, then parse.
pub async fn read_replica(client: &reqwest::Client, base_url: &str, timeout: std::time::Duration) -> Option<ReplicaLoad>;
```

Mapping, from SGLang `LoadSnapshot.to_dict()`. Every core key is present in the JSON; a missing key is treated as unknown (`null`):
- **Summed across `loads[]` (DP ranks):** `running` = Σ`num_running_reqs`, `queued` = Σ`num_waiting_reqs`, `prefill_backlog_tokens` = Σ`num_waiting_uncached_tokens`, `gen_tps` = Σ`gen_throughput`, `limits.max_running` = Σ`max_running_requests` (null if the sum is 0).
- **Computed ratio:** `kv_usage` = Σ`num_used_tokens` / Σ`max_total_num_tokens`. Null if the denominator is 0 or either field is missing. Clamped to 0..=1.
- **Cached tokens:** `cached_token_ratio` = `cache_hit_rate` with exactly one rank; null with more than one rank.
- **Sample time:** `sampled_at_ms` = min over ranks of `timestamp` (float seconds) × 1000, truncated. This is the scheduler's own clock.
- **Version:** `engine_version` = top-level `version` (string), if present.
- If any rank lacks a field used in a sum, that output field is `null`.

- [ ] **Step 1: Write failing tests:**

```rust
#[test]
fn single_rank_maps_fields() {
    let v = serde_json::json!({"version":"0.5.9","loads":[{"timestamp":1790000000.5,"num_running_reqs":14,"num_waiting_reqs":2,
        "num_waiting_uncached_tokens":51200,"num_used_tokens":630,"max_total_num_tokens":1000,"gen_throughput":910.0,
        "cache_hit_rate":0.71,"max_running_requests":32}]});
    let r = parse_sglang_loads(&v).unwrap();
    assert_eq!((r.load.running, r.load.queued, r.load.prefill_backlog_tokens), (Some(14), Some(2), Some(51200)));
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
    assert_eq!((r.load.running, r.load.queued, r.load.prefill_backlog_tokens), (Some(8), Some(1), Some(150)));
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
fn malformed_is_none() {
    assert!(parse_sglang_loads(&serde_json::json!({"nope":1})).is_none());
    assert!(parse_sglang_loads(&serde_json::json!({"loads":[]})).is_none());
    assert!(parse_sglang_loads(&serde_json::json!({"loads":[{"num_running_reqs":1}]})).is_none()); // no timestamp
}

#[tokio::test]
async fn read_replica_uses_timeout_and_parses() {
    // wiremock GET /v1/loads?include=core -> single-rank body => Some
    // wiremock with 2 s delay, timeout 200 ms => None within ~300 ms
    // 500 status => None
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::sglang` → FAIL.
- [ ] **Step 3: Implement** with small helpers: `sum_u64(ranks, key) -> Option<u64>` (None if any rank lacks the key or it isn't a non-negative integer), `sum_f64`, and `ts_ms`. `read_replica` does `client.get(format!("{}/v1/loads?include=core", base_url.trim_end_matches('/'))).timeout(timeout).send()`, requires a success status, then `json::<Value>()`. Any error → `None`, logged at `debug` with the replica index only (never the body). Register `pub mod sglang;`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(replica-state): SGLang /v1/loads adapter to a neutral load view`

---

### Task 4: Report key, attestation binding and the Redis sink

**Files:**
- Create: `src/replica_state/report_key.rs`, `src/replica_state/redis_sink.rs`
- Modify: `Cargo.toml` (`redis = { version = "1", default-features = false, features = ["tokio-comp", "tokio-rustls-comp", "connection-manager"] }`; add `rustls` as a direct dependency with the version the lock already resolves, default features off, feature `ring`)

**Interfaces:**
- Produces:
```rust
// report_key.rs
pub const REPORT_KEY_EVENT: &str = "nearai-replica-report-key-v1";
pub struct ReportKey { signing: ed25519_dalek::SigningKey, pub key_id: String, pub boot_id: String }
impl ReportKey {
    pub fn generate() -> Self;                              // random key; key_id = hex(sha256(pub))[..16]; boot_id = uuid v4
    pub fn signing_key(&self) -> &ed25519_dalek::SigningKey;
    pub fn public_hex(&self) -> String;
    pub fn event_payload(&self) -> Vec<u8>;                 // JSON {"key_id","public_key_hex","boot_id"}
}
impl std::fmt::Debug for ReportKey { /* key_id and boot_id only */ }
/// Ok(true) if recorded in the dstack event log; Ok(false) if skipped (dev mode or non-TEE).
pub async fn bind_to_attestation(key: &ReportKey, skip: bool) -> anyhow::Result<bool>;

// redis_sink.rs
pub const KEY_TTL_SECS: u64 = 5;
pub const STREAM_MAXLEN: usize = 20_000;                    // ~80 min at 2 replicas x 2 Hz; a buffer, not the recorder
pub fn state_key(host_id: &str, replica_id: &str) -> String;    // "replica:{host_id}:{replica_id}"
pub fn stream_key(host_id: &str) -> String;                    // "replica:{host_id}:frames"
pub struct RedisSink { conn: redis::aio::ConnectionManager, stream: String }
impl RedisSink {
    pub async fn connect(url: &str, host_id: &str) -> anyhow::Result<Self>;
    /// One pipeline: SET state_key json EX 5 for each frame, then XADD stream MAXLEN ~ 20000 * env json.
    pub async fn publish(&mut self, frames: &[(String /*replica_id*/, super::report::Envelope)]) -> anyhow::Result<()>;
}
```

- [ ] **Step 1: Write failing tests:**

```rust
// report_key.rs
#[test]
fn keys_and_ids_differ_per_boot_and_derive_correctly() {
    let a = ReportKey::generate(); let b = ReportKey::generate();
    assert_ne!(a.public_hex(), b.public_hex()); assert_ne!(a.boot_id, b.boot_id);
    use sha2::Digest;
    assert_eq!(a.key_id, hex::encode(sha2::Sha256::digest(a.signing_key().verifying_key().to_bytes()))[..16].to_string());
}
#[test]
fn event_payload_and_debug_expose_no_secret() {
    let k = ReportKey::generate();
    let v: serde_json::Value = serde_json::from_slice(&k.event_payload()).unwrap();
    assert_eq!(v.as_object().unwrap().len(), 3);
    assert_eq!(v["public_key_hex"], k.public_hex());
    assert!(!format!("{k:?}").contains(&hex::encode(k.signing_key().to_bytes())));
}
#[tokio::test]
async fn binding_is_skipped_when_asked() { assert!(!bind_to_attestation(&ReportKey::generate(), true).await.unwrap()); }

// redis_sink.rs
#[test]
fn key_layout_is_per_host() {
    assert_eq!(state_key("glm53-gpu03", "r1"), "replica:glm53-gpu03:r1");
    assert_eq!(stream_key("glm53-gpu03"), "replica:glm53-gpu03:frames");
}
#[tokio::test]
async fn unreachable_redis_fails_fast() {
    let t = std::time::Instant::now();
    assert!(RedisSink::connect("redis://127.0.0.1:1", "h").await.is_err());
    assert!(t.elapsed() < std::time::Duration::from_secs(5));
}
#[tokio::test]
async fn tls_url_does_not_panic_with_two_rustls_providers() {
    // Must return Err (nothing listening), not panic inside rustls ClientConfig::builder().
    assert!(RedisSink::connect("rediss://127.0.0.1:1", "h").await.is_err());
}
/// Real Redis, only when REPLICA_STATE_TEST_REDIS_URL is set.
#[tokio::test]
async fn publish_sets_ttl_keys_and_appends_stream() {
    let Ok(url) = std::env::var("REPLICA_STATE_TEST_REDIS_URL") else { return };
    // connect with host "test-host"; publish 2 frames; with a plain redis client assert:
    // GET replica:test-host:r1 == envelope json; TTL in 1..=5; XLEN replica:test-host:frames >= 2
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::report_key replica_state::redis_sink` → FAIL.
- [ ] **Step 3: Implement.**
  - **`bind_to_attestation`:** if `skip`, log `info!(key_id=%key.key_id, "Replica report key not bound: not in a TEE")` and return `Ok(false)`. Otherwise `dstack_sdk::dstack_client::DstackClient::new(None).emit_event(REPORT_KEY_EVENT.to_string(), key.event_payload()).await?` and return `Ok(true)`.
  - **`RedisSink::connect`:**
    - First `let _ = rustls::crypto::ring::default_provider().install_default();`. It's idempotent: an `Err` means already installed.
    - Then `redis::Client::open(url)` and `ConnectionManager::new_with_config(client, ConnectionManagerConfig::new().set_connection_timeout(Some(Duration::from_secs(2))).set_response_timeout(Some(Duration::from_secs(1))).set_number_of_retries(1))`.
    - Map errors to `anyhow!("redis connect failed: {kind}")` using `err.kind()`, never `err.to_string()`, which can echo the URL.
  - **`publish`:** `redis::pipe()`, with `.cmd("SET").arg(state_key).arg(&json).arg("EX").arg(KEY_TTL_SECS).ignore()` for each frame, then `.cmd("XADD").arg(&self.stream).arg("MAXLEN").arg("~").arg(STREAM_MAXLEN).arg("*").arg("env").arg(&json).ignore()`. Finish with `.query_async::<()>(&mut self.conn)`.
  - Register both modules.
- [ ] **Step 4: Run** tests, `cargo deny check licenses bans`, `cargo build --release` → PASS. Only warnings allowed are multiple versions.
- [ ] **Step 5: Commit** `feat(replica-state): per-boot report key bound via dstack event log; Redis sink`

---

### Task 5: Publisher and wiring

**Files:**
- Create: `src/replica_state/publisher.rs`
- Modify: `src/replica_state/mod.rs` (`spawn_replica_state_publisher`), `src/main.rs` (one `match` block after `backend_pool` is created)

**Interfaces:**
- Consumes: all of the above; `BackendPool::backends()` (index `i` = `VLLM_BACKEND_URLS[i]`; base backends first).
- Produces:
```rust
pub const UNHEALTHY_AFTER_FAILURES: u32 = 3;
struct ReplicaSlot { id: String, base_url: String, last: Option<ReplicaLoad>, failures: u32, lifecycle: Lifecycle }
pub struct Publisher { host_id: String, model: String, key: ReportKey, slots: Vec<ReplicaSlot>, seq: u64, client: reqwest::Client, read_timeout: Duration }
impl Publisher {
    pub fn new(host_id: String, model: String, key: ReportKey, replicas: Vec<(String, String)>, client: reqwest::Client, interval: Duration) -> Self; // read_timeout = interval * 4/5
    /// Read every replica concurrently, then build one frame per replica sharing one seq.
    pub async fn tick(&mut self, inflight: impl Fn(usize) -> u32, now_ms: u64) -> Vec<(String, Envelope)>;
}
pub fn spawn_replica_state_publisher(cfg: ReplicaStateConfig, model: String, pool: Arc<BackendPool>, client: reqwest::Client, skip_binding: bool);
```
Per-replica rules in `tick`:
- **Successful read:** set `last = Some(read)`, `failures = 0`, `lifecycle = Ready`. The frame gets `load`, `limits` and `engine_version` from the read, and `engine_sampled_at_ms = read.sampled_at_ms`.
- **Failed read:** `failures += 1`. The frame gets `load = Load::default()` (all null), `limits` and `engine_version` from `last` if any, and `engine_sampled_at_ms = last.sampled_at_ms` (or 0). Lifecycle: if never succeeded, `Warming`; else if `failures >= 3`, `Unhealthy`; else unchanged.
- **Every frame:** `schema = 1`, `seq = self.seq` (incremented once per tick, before building), `reported_at_ms = now_ms`, `boot_id` and `report_key_id` from the key, `proxy_inflight = inflight(i)`.

- [ ] **Step 1: Write failing tests** (tokio + wiremock). Verify with `report::open` and `key.signing_key().verifying_key()`:

```rust
#[tokio::test]
async fn one_signed_frame_per_replica_sharing_one_seq_per_tick() {
    // two mocks serving single-rank /v1/loads; tick twice
    // tick 1: 2 frames, replica ids r1,r2, both seq == 1, both open() == Some, lifecycle Ready, same boot_id
    // tick 2: both seq == 2
}
#[tokio::test]
async fn engine_timestamp_passes_through_even_when_stale() {
    // mock returns timestamp = 1000.0 (ancient); frame.engine_sampled_at_ms == 1_000_000 while reported_at_ms == now_ms
}
#[tokio::test]
async fn failed_read_keeps_last_sample_time_and_nulls_load() {
    // tick OK (sampled 1790000000.0) -> switch mock to 500 -> tick
    // frame: load all None, engine_sampled_at_ms == 1_790_000_000_000, limits kept, lifecycle Ready (1 failure)
    // two more failing ticks -> lifecycle Unhealthy; then success -> Ready
}
#[tokio::test]
async fn never_reachable_replica_is_warming_with_zero_sample_time() {
    // base_url http://127.0.0.1:1 -> lifecycle Warming after 5 ticks, engine_sampled_at_ms == 0
}
#[tokio::test]
async fn tick_is_bounded_by_interval() {
    // mock delays 5 s; interval 200 ms; tick returns in < 400 ms; that replica's load is null
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::publisher` → FAIL.
- [ ] **Step 3: Implement.**
  - **`tick`:** `futures_util::future::join_all` over the slots, calling `sglang::read_replica(&client, &slot.base_url, read_timeout)`.
  - **`spawn_replica_state_publisher`** spawns one task that:
    1. Builds `ReportKey::generate()` and runs `bind_to_attestation(&key, skip_binding)`. On `Err`, it logs `warn!` and continues: readers will reject frames whose key isn't attested, which is visible.
    2. Loops on `RedisSink::connect(&cfg.redis_url, &cfg.host_id)` with a 5 s sleep between attempts, logging `warn!(redis = %cfg.redis_host_for_logs(), "…")`.
    3. Then runs `tokio::time::interval(cfg.interval)` with `MissedTickBehavior::Skip`:
       - `frames = publisher.tick(|i| pool.backends()[i].active_conns.load(Relaxed), now_ms())`, where `now_ms()` is `SystemTime::now()` since `UNIX_EPOCH`, in ms.
       - `sink.publish(&frames)`; on error, increment `replica_state_publish_failures_total` and `warn!` with the error kind only.
       - Record `replica_state_frames_total` and a `replica_state_tick_seconds` histogram via the `metrics` crate, as `engine_load.rs` does.
    4. The Redis connection manager reconnects by itself after the first connect.
  - **`main.rs`**, right after `backend_pool` is created, using the backend client and the existing non-TEE flag (search `NON_TEE_DEPLOYMENT` for the field name):

```rust
match replica_state::config::ReplicaStateConfig::from_env(config.backend_urls.len()) {
    Ok(Some(rs)) => {
        info!(host_id = %rs.host_id, replicas = rs.replica_ids.len(), interval_ms = rs.interval.as_millis() as u64, redis = %rs.redis_host_for_logs(), "Publishing replica state to Redis");
        replica_state::spawn_replica_state_publisher(rs, config.model_name.clone(), backend_pool.clone(), backend_client.clone(), config.dev_mode || config.non_tee_deployment);
    }
    Ok(None) => {}
    Err(e) => error!(error = %e, "Replica state publishing disabled: invalid REPLICA_STATE_* configuration"),
}
```
  If the backend client is created after `backend_pool` in `main.rs`, place this block right after the backend client instead.
- [ ] **Step 4: Run** `cargo test`, `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` → PASS.
- [ ] **Step 5: Commit** `feat(replica-state): publisher tick and opt-in wiring`

---

### Task 6: Operator documentation

**Files:**
- Create: `docs/replica-state.md`
- Modify: `README.md` (env vars), `CLAUDE.md` (one line in the module map)

- [ ] **Step 1:** Write `docs/replica-state.md` with the following sections:
  - **Purpose:** stage 1 of the placement plan, with a link to the design doc.
  - **Env vars:** `REPLICA_STATE_REDIS_URL`, `REPLICA_STATE_HOST_ID`, `REPLICA_STATE_REPLICA_IDS`, `REPLICA_STATE_INTERVAL_MS`.
  - **Frame schema:** the Task 2 example and the signing rule.
  - **SGLang mapping** (Task 3).
  - **Redis layout**, per-host keys and stream, and the ACL pattern `~replica:{host_id}:*`.
  - **Lifecycle rules** (Task 5).
  - **How readers verify:**
    1. Find the `nearai-replica-report-key-v1` event in the attested event log and match `key_id`.
    2. Verify the signature over the frame string, then parse it.
    3. Require `seq` and `boot_id` to be monotonic, and `engine_sampled_at_ms` to be fresh by the reader's clock.
  - **Caveats:**
    - The capped stream is a ~80-minute buffer. The week-long recorder in the stage 1 exit needs a separate drain to object storage, not in this change.
    - `boot_id` is the proxy's boot, not the engine's.
    - Each proxy restart appends another event to RTMR3.
    - Nonce-less attestation reports are cached for up to `ATTESTATION_CACHE_TTL` (default 300 s), so the key event may be missing from cached reports for up to one TTL after boot.
  - **Privacy:** IDs and numbers only.
- [ ] **Step 2:** Update the README env table and the CLAUDE.md module map.
- [ ] **Step 3: Commit** `docs(replica-state): operator guide and schema`

---

## Verification before hand-off

- `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test && cargo deny check`
- Local: run Redis locally, run the proxy against a mock SGLang with `REPLICA_STATE_*` set and `DEV=1`, then check `redis-cli GET replica:local:r1`, `TTL`, and `XLEN replica:local:frames`.
- On a real CVM, before merge (per `docs/testing-on-cvm.md`):
  - `emit_event` succeeds on the deployed dstack version, and the key event appears in `/v1/attestation/report`'s `event_log`.
  - cloud-api's attestation verifier and a customer verifier still pass with the extra event.
  - Frames arrive with `engine_sampled_at_ms` close to wall-clock.

## Review changes (principal review, 25 Sep 2026)

- **Correctness fixes:**
  - the scheduler timestamp is used as the sample time;
  - a missing field is `null`, not 0;
  - DP ranks: counts summed, ratios computed;
  - a rustls provider is installed, fixing a panic on `rediss://`;
  - redis is pinned to 1.x with explicit timeouts and 1 retry;
  - the envelope carries the frame as the signed string;
  - `seq` is per tick;
  - a bad config never aborts startup;
  - the backend client is used;
  - the Redis URL is never logged.
- **Cut for stage 1:**
  - the vLLM adapter, engine detection and the `ENGINE` knob;
  - `prefill_tps` and cumulative-counter state;
  - `async-trait` and the sink/engine traits;
  - `MemorySink`;
  - the TTL, stream, maxlen, pool and tier knobs, plus `pool`/`tier` in the frame (the registry owns them).
- **Changed:**
  - replica IDs are required config (stable registry names);
  - Redis keys and the stream are per host, so one ACL pattern covers them;
  - the stream is documented as a buffer, not the recorder.
