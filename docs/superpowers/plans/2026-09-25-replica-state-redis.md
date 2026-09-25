# Replica State → Redis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each in-CVM host proxy samples its engine replicas every 250–500 ms and publishes one signed, normalized `ReplicaReport` v1 frame per replica to Redis (a TTL key for the latest state, plus a capped stream as a recorder). Opt-in, off by default, no change to request handling.

**Architecture:** A new `replica_state` module owns the whole feature. An `EngineAdapter` trait (Adapter pattern) turns SGLang `/v1/loads` or vLLM `/metrics` into one engine-neutral `ReplicaLoad`. A `ReplicaStatePublisher` runs one tick per interval: sample every replica concurrently, build frames (`boot_id`, monotonic `seq`, lifecycle), sign them with a per-boot Ed25519 report key (domain-separated), and hand them to a `StateSink` (Strategy: `RedisSink` in production, `MemorySink` in tests). The report key's public half is recorded in the dstack event log (`emit_event`), which the TDX quote covers, so `report_data` stays unchanged for existing verifiers.

**Tech Stack:** Rust (tokio, axum, reqwest, serde_json, ed25519-dalek), new dependency `redis` (BSD-3-Clause) with `tokio-comp`, `tokio-rustls-comp`, `connection-manager`, `streams`; tests with `wiremock`.

**Spec:** Design doc "Inference Placement Map" (https://claude.ai/artifact/T3WkANwFetyUYsHhMBNheo), sections "Latest recommendation" (stage 1), "Replica lifecycle and state reporting" §2 (engine signals), §4 (state-report contract, Redis transport, who can trust a frame, staleness), and the gated stage 1 exit criteria in "Baseline".

## Global Constraints

- Feature is **off unless `REPLICA_STATE_REDIS_URL` is set**. With it unset, behaviour, startup and tests are byte-for-byte unchanged.
- Frames carry **IDs and numbers only**: no prompts, no org IDs, no affinity keys, no content-derived hashes. Never log frame contents above `debug`.
- **An unknown value is `null`, never `0`.** A value SGLang omits from `/v1/loads` because it is zero (`omit_defaults`) is a known `0`; a failed read makes every load field `null`.
- Signature: `ed25519(report_key, b"nearai-replica-report-v1\n" ++ canonical_frame_bytes)`, base64 (standard, padded). `canonical_frame_bytes` = `serde_json::to_vec(&frame)` with struct field order as declared below.
- Today's attestation `report_data` layout (`attestation.rs::build_report_data`) must not change.
- Reproducible build: no new required build env vars or args (see `CLAUDE.md` "Deployment"). `cargo deny check` must pass.
- **Non-intrusive:** existing code changes are limited to `src/lib.rs` (one `pub mod replica_state;` line) and `src/main.rs` (one guarded spawn block). No changes to `Config`, `AppState`, request routes, `proxy.rs`, `attestation.rs`, `backend_pool.rs` or the integration-test helpers. The feature reads `BackendPool` through its existing public `backends()` accessor only.
- Run `cargo fmt` before each commit.

## Review Focus

1. **Engine unreachable or slow** → the replica's frame still goes out, with `lifecycle_state: "unhealthy"` after 3 consecutive failed reads (`"warming"` before the first success) and every load field `null`; the tick never blocks longer than one interval. (Task 6 test.)
2. **Redis down or slow at boot or mid-run** → the proxy starts and serves normally; the publisher logs, counts `replica_state_publish_failures_total`, and retries next tick. (Task 5 and Task 6 tests.)
3. **SGLang omits zero fields** (`omit_defaults`) → absent `num_waiting_reqs` parses as `0`, not `null`. (Task 3 test.)
4. **Proxy restart** → new `boot_id`, `seq` restarts at 1, new report key and a new event-log entry; readers can tell this apart from a regression. (Task 4 and Task 6 tests.)
5. **Metrics double-counting** (priority buckets, TP ranks) → the vLLM path reuses `engine_load::metric_sum`, which already prefers the empty-`priority` aggregate. (Task 3 test.)


## Design patterns (refactoring.guru catalog)

Chosen for what this feature actually varies on; nothing added for its own sake.

| Pattern | Where | Why |
|---|---|---|
| **Adapter** | `EngineAdapter` with `SglangAdapter` / `VllmAdapter` | Two incompatible engine interfaces (`/v1/loads` JSON vs Prometheus text) become one `ReplicaLoad`. New engines add an adapter; nothing else changes. |
| **Strategy** | `StateSink` with `RedisSink` / `MemorySink` | Where frames go is interchangeable: Redis in production, memory in tests. The publisher doesn't know which. |
| **Template Method** (as a fixed pipeline in `tick`) | `ReplicaStatePublisher::tick` | The invariant steps (read → normalize → lifecycle → frame → seal → publish) are fixed in one place; the variable steps are the adapter and the sink. |
| **Facade** | `replica_state::spawn_replica_state_publisher` | The only entry point `main.rs` calls. Keys, detection, retries and metrics stay hidden in the module. |

Not used on purpose: Observer (nothing subscribes to frames inside the proxy), Decorator (no layered behaviour needed), and Singleton (state lives in the spawned task).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/replica_state/mod.rs` | Module root; `spawn_replica_state_publisher(...)` wiring entry point |
| `src/replica_state/report.rs` | `ReplicaReport` v1 types, `Envelope`, canonical bytes, `sign_frame`/`verify_envelope` |
| `src/replica_state/engine.rs` | `EngineAdapter` trait, `SglangAdapter`, `VllmAdapter`, `ReplicaLoad`, engine auto-detection |
| `src/replica_state/report_key.rs` | Per-boot `ReportKey` (Ed25519), `key_id`, dstack event-log binding |
| `src/replica_state/sink.rs` | `StateSink` trait, `RedisSink`, `MemorySink` |
| `src/replica_state/publisher.rs` | `ReplicaStatePublisher` tick loop: sample → frame → sign → sink |
| `src/replica_state/config.rs` | `ReplicaStateConfig::from_env(backend_urls)`: parsed only by this module; `None` when disabled |
| `src/main.rs`, `src/lib.rs` | One `pub mod` line; one guarded spawn block (the only edits to existing code) |
| `docs/replica-state.md` | Operator doc: env vars, frame schema, Redis layout |

Out of scope for this plan (follow-up plan "stage 1b"): `x-nearai-replica` strict hint and served-replica echo, queue-full → 429 conversion, per-token trust levels. They touch the request path; this plan does not.

---

### Task 1: Configuration (inside the new module)

**Files:**
- Create: `src/replica_state/config.rs`, `src/replica_state/mod.rs` (declares submodules)
- Modify: `src/lib.rs` (`pub mod replica_state;`)

`Config` is not modified. The module parses its own environment, so existing tests and helpers are untouched.

**Interfaces:**
- Produces:
```rust
#[derive(Clone, Debug, PartialEq)]
pub enum EngineKind { Auto, Sglang, Vllm }

#[derive(Clone, Debug)]
pub struct ReplicaStateConfig {
    pub redis_url: String,            // REPLICA_STATE_REDIS_URL (required to enable)
    pub host_id: String,              // REPLICA_STATE_HOST_ID (required when enabled)
    pub replica_ids: Vec<String>,     // VLLM_BACKEND_REPLICA_IDS, default r1..rN
    pub interval: std::time::Duration,// REPLICA_STATE_INTERVAL_MS, default 500, range 100..=5000
    pub key_ttl_secs: u64,            // REPLICA_STATE_KEY_TTL_SECS, default 5
    pub stream: String,               // REPLICA_STATE_STREAM, default "replica-frames"
    pub stream_maxlen: usize,         // REPLICA_STATE_STREAM_MAXLEN, default 200000
    pub engine: EngineKind,           // REPLICA_STATE_ENGINE, default auto
    pub pool: Option<String>,         // REPLICA_STATE_POOL
    pub tier: Option<String>,         // REPLICA_STATE_TIER
}
impl ReplicaStateConfig {
    /// `Ok(None)` when `REPLICA_STATE_REDIS_URL` is unset. `backend_urls` = `Config::backend_urls` (for replica-ID defaults and count checks).
    pub fn from_env(backend_urls: &[String]) -> anyhow::Result<Option<Self>>;
}
```

- [ ] **Step 1: Write failing tests** in `replica_state/config.rs` `mod tests`. Serialize env-mutating tests with a module-local `static ENV_LOCK: std::sync::Mutex<()>` and a small guard that sets and restores variables:

```rust
#[test]
fn replica_state_is_off_without_redis_url() {
    let _g = EnvGuard::new(&[("REPLICA_STATE_REDIS_URL", None), ("REPLICA_STATE_HOST_ID", Some("h1"))]);
    assert!(ReplicaStateConfig::from_env(&["http://a:8000".into()]).unwrap().is_none());
}

#[test]
fn replica_state_requires_host_id() {
    let _g = EnvGuard::new(&[("REPLICA_STATE_REDIS_URL", Some("redis://r:6379")), ("REPLICA_STATE_HOST_ID", None)]);
    assert!(ReplicaStateConfig::from_env(&["http://a:8000".into()]).unwrap_err().to_string().contains("REPLICA_STATE_HOST_ID"));
}

#[test]
fn replica_ids_default_and_must_match_backends() {
    let _g = EnvGuard::new(&[
        ("REPLICA_STATE_REDIS_URL", Some("redis://r:6379")),
        ("REPLICA_STATE_HOST_ID", Some("glm53-gpu03")),
        ("VLLM_BACKEND_REPLICA_IDS", None),
    ]);
    let rs = ReplicaStateConfig::from_env(&["http://a:8000".into(), "http://b:8000".into()]).unwrap().unwrap();
    assert_eq!(rs.replica_ids, vec!["r1", "r2"]);
    assert_eq!(rs.interval, std::time::Duration::from_millis(500));
    assert_eq!(rs.engine, EngineKind::Auto);
}

#[test]
fn replica_ids_count_mismatch_is_an_error() {
    let _g = EnvGuard::new(&[
        ("REPLICA_STATE_REDIS_URL", Some("redis://r:6379")),
        ("REPLICA_STATE_HOST_ID", Some("h")),
        ("VLLM_BACKEND_REPLICA_IDS", Some("r1")),
    ]);
    assert!(ReplicaStateConfig::from_env(&["http://a:8000".into(), "http://b:8000".into()]).is_err());
}

#[test]
fn interval_out_of_range_is_an_error() {
    let _g = EnvGuard::new(&[
        ("REPLICA_STATE_REDIS_URL", Some("redis://r:6379")),
        ("REPLICA_STATE_HOST_ID", Some("h")),
        ("REPLICA_STATE_INTERVAL_MS", Some("50")),
    ]);
    assert!(ReplicaStateConfig::from_env(&["http://a:8000".into()]).is_err());
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::config` → FAIL (types don't exist).

- [ ] **Step 3: Implement** `ReplicaStateConfig::from_env` with small local helpers (`env_parse`, `env_or` equivalents; the ones in `config.rs` are private and stay untouched):

```rust
let replica_state = match env::var("REPLICA_STATE_REDIS_URL").ok().filter(|v| !v.trim().is_empty()) {
    None => None,
    Some(redis_url) => {
        let host_id = env::var("REPLICA_STATE_HOST_ID").ok().filter(|v| !v.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("REPLICA_STATE_HOST_ID is required when REPLICA_STATE_REDIS_URL is set"))?;
        let replica_ids: Vec<String> = match env::var("VLLM_BACKEND_REPLICA_IDS") {
            Ok(v) if !v.trim().is_empty() => v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            _ => (1..=backend_urls.len()).map(|i| format!("r{i}")).collect(),
        };
        if replica_ids.len() != backend_urls.len() {
            anyhow::bail!("VLLM_BACKEND_REPLICA_IDS must list one ID per VLLM_BACKEND_URLS entry, in the same order");
        }
        let interval_ms: u64 = env_parse("REPLICA_STATE_INTERVAL_MS", 500)?;
        if !(100..=5000).contains(&interval_ms) {
            anyhow::bail!("REPLICA_STATE_INTERVAL_MS must be between 100 and 5000");
        }
        let engine = match env_or("REPLICA_STATE_ENGINE", "auto").as_str() {
            "auto" => EngineKind::Auto, "sglang" => EngineKind::Sglang, "vllm" => EngineKind::Vllm,
            other => anyhow::bail!("REPLICA_STATE_ENGINE must be auto, sglang or vllm (got {other})"),
        };
        let opt = |n: &str| env::var(n).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        Some(ReplicaStateConfig {
            redis_url, host_id: host_id.trim().to_string(), replica_ids,
            interval: std::time::Duration::from_millis(interval_ms),
            key_ttl_secs: env_parse("REPLICA_STATE_KEY_TTL_SECS", 5)?,
            stream: env_or("REPLICA_STATE_STREAM", "replica-frames"),
            stream_maxlen: env_parse("REPLICA_STATE_STREAM_MAXLEN", 200_000)?,
            engine, pool: opt("REPLICA_STATE_POOL"), tier: opt("REPLICA_STATE_TIER"),
        })
    }
};
```

Return `Ok(replica_state)`. Nothing else in the crate changes in this task except the `pub mod replica_state;` line in `lib.rs`.

- [ ] **Step 4: Run** `cargo test --lib replica_state::config` and `cargo test` (full suite unchanged) → PASS.

- [ ] **Step 5: Commit** `feat(replica-state): opt-in config for per-replica state publishing`

---

### Task 2: ReplicaReport v1 and signing envelope

**Files:**
- Create: `src/replica_state/report.rs`

**Interfaces:**
- Produces:
```rust
pub const SIGNING_DOMAIN: &[u8] = b"nearai-replica-report-v1\n";
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Lifecycle { Warming, Ready, Degraded, Unhealthy, Draining, Drained }
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Engine { Sglang, Vllm }
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Limits { pub max_running: Option<u32>, pub max_queued: Option<u32>, pub max_context: Option<u64> }
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Load {
    pub running: Option<u32>, pub queued: Option<u32>,
    pub prefill_backlog_tokens: Option<u64>, pub prefill_running_tokens: Option<u64>,
    pub kv_usage: Option<f64>, pub gen_tps: Option<f64>, pub prefill_tps: Option<f64>,
    pub itl_p50_ms: Option<f64>, pub cached_token_ratio: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProxyLoad { pub inflight: u32 }
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReplicaReport {
    pub schema: u8, pub host_id: String, pub replica_id: String, pub boot_id: String, pub seq: u64,
    pub engine_sampled_at_ms: u64, pub reported_at_ms: u64, pub lifecycle_state: Lifecycle,
    pub model: String, pub pool: Option<String>, pub tier: Option<String>,
    pub engine: Option<Engine>, pub engine_version: Option<String>,
    pub limits: Limits, pub load: Load, pub proxy: ProxyLoad, pub report_key_id: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Envelope { pub frame: ReplicaReport, pub sig: String }

pub fn canonical_bytes(frame: &ReplicaReport) -> Vec<u8>;
pub fn signing_message(frame: &ReplicaReport) -> Vec<u8>;             // SIGNING_DOMAIN ++ canonical_bytes
pub fn seal(frame: ReplicaReport, key: &ed25519_dalek::SigningKey) -> Envelope;
pub fn verify(env: &Envelope, key: &ed25519_dalek::VerifyingKey) -> bool;
```
`prefill_tps` is an addition to the design doc's v1 draft (derived from SGLang's cumulative prefill counters). It is additive and nullable.

- [ ] **Step 1: Write failing tests** in `report.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn frame() -> ReplicaReport {
        ReplicaReport {
            schema: 1, host_id: "glm53-gpu03".into(), replica_id: "r1".into(),
            boot_id: "00000000-0000-4000-8000-000000000001".into(), seq: 7,
            engine_sampled_at_ms: 1_790_000_000_011, reported_at_ms: 1_790_000_000_123,
            lifecycle_state: Lifecycle::Ready, model: "z-ai/glm-5.3-flash".into(),
            pool: Some("glm53-base".into()), tier: None, engine: Some(Engine::Sglang), engine_version: None,
            limits: Limits { max_running: Some(32), max_queued: None, max_context: None },
            load: Load { running: Some(14), queued: Some(0), prefill_backlog_tokens: Some(51200), ..Default::default() },
            proxy: ProxyLoad { inflight: 16 }, report_key_id: "k1".into(),
        }
    }

    #[test]
    fn canonical_bytes_are_stable_and_keep_nulls() {
        let s = String::from_utf8(canonical_bytes(&frame())).unwrap();
        assert!(s.starts_with(r#"{"schema":1,"host_id":"glm53-gpu03","replica_id":"r1""#));
        assert!(s.contains(r#""tier":null"#));
        assert!(s.contains(r#""kv_usage":null"#));
        assert!(!s.contains(' '));
    }

    #[test]
    fn seal_then_verify_roundtrips_and_detects_tampering() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let env = seal(frame(), &sk);
        assert!(verify(&env, &sk.verifying_key()));
        let mut tampered = env.clone();
        tampered.frame.load.queued = Some(99);
        assert!(!verify(&tampered, &sk.verifying_key()));
    }

    #[test]
    fn signature_is_domain_separated() {
        use ed25519_dalek::Signer;
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let bare = sk.sign(&canonical_bytes(&frame()));
        let env = Envelope { frame: frame(), sig: base64::engine::general_purpose::STANDARD.encode(bare.to_bytes()) };
        assert!(!verify(&env, &sk.verifying_key()));
    }

    #[test]
    fn envelope_json_roundtrip() {
        let sk = SigningKey::from_bytes(&[7u8; 32]);
        let env = seal(frame(), &sk);
        let back: Envelope = serde_json::from_str(&serde_json::to_string(&env).unwrap()).unwrap();
        assert!(verify(&back, &sk.verifying_key()));
    }
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::report` → FAIL.

- [ ] **Step 3: Implement.** Types exactly as above (no `skip_serializing_if`: nulls must be present), then:

```rust
use base64::Engine as _;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

pub fn canonical_bytes(frame: &ReplicaReport) -> Vec<u8> {
    serde_json::to_vec(frame).expect("ReplicaReport always serializes")
}
pub fn signing_message(frame: &ReplicaReport) -> Vec<u8> {
    let mut m = SIGNING_DOMAIN.to_vec();
    m.extend_from_slice(&canonical_bytes(frame));
    m
}
pub fn seal(frame: ReplicaReport, key: &SigningKey) -> Envelope {
    let sig = key.sign(&signing_message(&frame));
    Envelope { frame, sig: base64::engine::general_purpose::STANDARD.encode(sig.to_bytes()) }
}
pub fn verify(env: &Envelope, key: &VerifyingKey) -> bool {
    let Ok(raw) = base64::engine::general_purpose::STANDARD.decode(&env.sig) else { return false };
    let Ok(bytes) = <[u8; 64]>::try_from(raw.as_slice()) else { return false };
    key.verify(&signing_message(&env.frame), &Signature::from_bytes(&bytes)).is_ok()
}
```

- [ ] **Step 4: Run** `cargo test --lib replica_state::report` → PASS.

- [ ] **Step 5: Commit** `feat(replica-state): ReplicaReport v1 schema and domain-separated signing envelope`

---

### Task 3: Engine adapters (SGLang, vLLM)

**Files:**
- Create: `src/replica_state/engine.rs`

**Interfaces:**
- Consumes: `report::{Engine, Load, Limits}`; `crate::engine_load::metric_sum`.
- Produces:
```rust
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReplicaLoad { pub load: Load, pub limits: Limits, pub engine_version: Option<String> }

/// Per-replica state an adapter keeps between reads (cumulative counters).
#[derive(Clone, Debug, Default)]
pub struct AdapterMemory { pub prev_prefill: Option<(u64 /*uncached tokens*/, u64 /*busy us*/)>, pub prev_gen: Option<(f64 /*tokens*/, std::time::Instant)> }

#[async_trait::async_trait]
pub trait EngineAdapter: Send + Sync {
    fn engine(&self) -> Engine;
    async fn read(&self, client: &reqwest::Client, base_url: &str, timeout: std::time::Duration, mem: &mut AdapterMemory) -> anyhow::Result<ReplicaLoad>;
}
pub struct SglangAdapter; pub struct VllmAdapter;
pub fn parse_sglang_loads(json: &serde_json::Value, mem: &mut AdapterMemory) -> Option<ReplicaLoad>;
pub fn parse_vllm_metrics(body: &str, mem: &mut AdapterMemory, now: std::time::Instant) -> Option<ReplicaLoad>;
pub async fn detect(client: &reqwest::Client, base_url: &str, timeout: std::time::Duration) -> Option<Engine>;
```
If `async-trait` is not already a dependency, use `fn read(...) -> Pin<Box<dyn Future<Output = ...> + Send + 'a>>` instead of adding it. Check `Cargo.toml` first.

Mapping (from design doc §2 and SGLang `load_snapshot.py`):
- **SGLang** `GET {base}/v1/loads?include=core,queues` → `loads[0]` (sum across `loads[]` if there are several DP ranks). A field missing from the object is `0` (`omit_defaults`).
  - `running` = `num_running_reqs`
  - `queued` = `num_waiting_reqs`
  - `prefill_backlog_tokens` = `num_waiting_uncached_tokens`
  - `kv_usage` = `token_usage`
  - `gen_tps` = `gen_throughput`
  - `cached_token_ratio` = `cache_hit_rate`
  - `max_running` = `max_running_requests` (null if 0)
  - `prefill_tps` = Δ`total_prefill_uncached_tokens` ÷ (Δ`total_prefill_busy_us` / 1e6) against the previous read. Null on the first read, and when Δbusy = 0.
  - `engine_version` = top-level `version`.
  - `prefill_running_tokens` and `itl_p50_ms`: `null` in v1.
- **vLLM** `GET {base}/metrics`:
  - `running` = `metric_sum(["vllm:num_requests_running"])`
  - `queued` = `metric_sum(["vllm:num_requests_waiting"])`
  - `kv_usage` = the value of `vllm:kv_cache_usage_perc` or `vllm:gpu_cache_usage_perc`. It is already 0–1: parse it as f64 directly, not via `metric_sum`, which rounds to u32.
  - `gen_tps` = Δ`vllm:generation_tokens_total` / Δt against the previous read.
  - `cached_token_ratio` = `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` (cumulative ratio; null if queries is 0).
  - `prefill_backlog_tokens`, `prefill_tps`: `null`. vLLM has no equivalent; the design doc lists this as a known gap.
- **detect**: `GET /v1/loads` 200 with a JSON `loads` array → Sglang. Otherwise `GET /metrics` containing `vllm:` → Vllm. Otherwise `None`.

- [ ] **Step 1: Write failing tests** with fixed inputs:

```rust
#[test]
fn sglang_loads_maps_fields_and_treats_omitted_as_zero() {
    let json = serde_json::json!({"version":"0.5.x","loads":[{"num_running_reqs":14,"num_waiting_uncached_tokens":51200,
        "token_usage":0.63,"gen_throughput":910.0,"cache_hit_rate":0.71,"max_running_requests":32,
        "total_prefill_uncached_tokens":1000,"total_prefill_busy_us":500000}]});
    let mut mem = AdapterMemory::default();
    let r = parse_sglang_loads(&json, &mut mem).unwrap();
    assert_eq!(r.load.running, Some(14));
    assert_eq!(r.load.queued, Some(0)); // omitted => 0
    assert_eq!(r.load.prefill_backlog_tokens, Some(51200));
    assert_eq!(r.load.kv_usage, Some(0.63));
    assert_eq!(r.limits.max_running, Some(32));
    assert_eq!(r.load.prefill_tps, None); // first read
    assert_eq!(r.engine_version.as_deref(), Some("0.5.x"));
}

#[test]
fn sglang_prefill_tps_from_counter_deltas() {
    let mut mem = AdapterMemory::default();
    let a = serde_json::json!({"loads":[{"total_prefill_uncached_tokens":1000,"total_prefill_busy_us":500000}]});
    let b = serde_json::json!({"loads":[{"total_prefill_uncached_tokens":8000,"total_prefill_busy_us":1500000}]});
    parse_sglang_loads(&a, &mut mem).unwrap();
    let r = parse_sglang_loads(&b, &mut mem).unwrap();
    assert_eq!(r.load.prefill_tps, Some(7000.0)); // 7000 tokens / 1.0 s busy
}

#[test]
fn sglang_malformed_is_none() {
    assert!(parse_sglang_loads(&serde_json::json!({"nope":1}), &mut AdapterMemory::default()).is_none());
}

#[test]
fn vllm_metrics_map_and_rates() {
    let t0 = std::time::Instant::now();
    let body0 = "vllm:num_requests_running{model_name=\"m\"} 5\nvllm:num_requests_waiting{model_name=\"m\"} 2\nvllm:kv_cache_usage_perc{model_name=\"m\"} 0.42\nvllm:generation_tokens_total{model_name=\"m\"} 1000\nvllm:prefix_cache_hits_total{model_name=\"m\"} 30\nvllm:prefix_cache_queries_total{model_name=\"m\"} 100\n";
    let body1 = body0.replace("generation_tokens_total{model_name=\"m\"} 1000", "generation_tokens_total{model_name=\"m\"} 1500");
    let mut mem = AdapterMemory::default();
    let r0 = parse_vllm_metrics(body0, &mut mem, t0).unwrap();
    assert_eq!((r0.load.running, r0.load.queued, r0.load.kv_usage), (Some(5), Some(2), Some(0.42)));
    assert_eq!(r0.load.gen_tps, None);
    assert_eq!(r0.load.cached_token_ratio, Some(0.3));
    assert_eq!(r0.load.prefill_backlog_tokens, None);
    let r1 = parse_vllm_metrics(&body1, &mut mem, t0 + std::time::Duration::from_secs(1)).unwrap();
    assert_eq!(r1.load.gen_tps, Some(500.0));
}

#[test]
fn vllm_without_running_gauge_is_none() {
    assert!(parse_vllm_metrics("foo 1\n", &mut AdapterMemory::default(), std::time::Instant::now()).is_none());
}
```

Add one `wiremock` test for `detect` (a `/v1/loads` route → Sglang; only `/metrics` with `vllm:` → Vllm; neither → None), and one for `SglangAdapter::read` against a mocked `/v1/loads`.

- [ ] **Step 2: Run** `cargo test --lib replica_state::engine` → FAIL.
- [ ] **Step 3: Implement** the parsers and adapters as specified. Round f64 fields with no special handling; clamp `kv_usage` and `cached_token_ratio` to 0.0..=1.0.
- [ ] **Step 4: Run** `cargo test --lib replica_state::engine` → PASS.
- [ ] **Step 5: Commit** `feat(replica-state): SGLang and vLLM engine adapters to a neutral load view`

---

### Task 4: Per-boot report key and attestation binding

**Files:**
- Create: `src/replica_state/report_key.rs`

**Interfaces:**
- Produces:
```rust
pub const REPORT_KEY_EVENT: &str = "nearai-replica-report-key-v1";
pub struct ReportKey { pub signing: ed25519_dalek::SigningKey, pub key_id: String, pub boot_id: String }
impl ReportKey {
    pub fn generate() -> Self;                         // random key, key_id = hex(sha256(pub))[..16], boot_id = uuid v4
    pub fn public_hex(&self) -> String;
    pub fn event_payload(&self) -> Vec<u8>;           // UTF-8 JSON {"key_id","public_key_hex","boot_id"}
}
/// Record the public key in the dstack event log (RTMR3). Returns Ok(false) when skipped (dev or non-TEE).
pub async fn bind_to_attestation(key: &ReportKey, dev_mode: bool, non_tee: bool) -> anyhow::Result<bool>;
```

- [ ] **Step 1: Write failing tests:**

```rust
#[test]
fn generated_keys_differ_per_boot_and_ids_are_derived() {
    let a = ReportKey::generate(); let b = ReportKey::generate();
    assert_ne!(a.public_hex(), b.public_hex());
    assert_ne!(a.boot_id, b.boot_id);
    assert_eq!(a.key_id.len(), 16);
    let expect = &hex::encode(sha2::Sha256::digest(a.signing.verifying_key().to_bytes()))[..16];
    assert_eq!(a.key_id, expect);
}

#[test]
fn event_payload_carries_ids_and_public_key_only() {
    let k = ReportKey::generate();
    let v: serde_json::Value = serde_json::from_slice(&k.event_payload()).unwrap();
    assert_eq!(v["key_id"], k.key_id);
    assert_eq!(v["public_key_hex"], k.public_hex());
    assert_eq!(v["boot_id"], k.boot_id);
    assert_eq!(v.as_object().unwrap().len(), 3);
}

#[tokio::test]
async fn binding_is_skipped_in_dev_mode() {
    assert!(!bind_to_attestation(&ReportKey::generate(), true, false).await.unwrap());
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::report_key` → FAIL.
- [ ] **Step 3: Implement.** Use `rand` (already a dependency) for 32 random bytes, `uuid::Uuid::new_v4()`, `sha2`. In `bind_to_attestation`: if `dev_mode || non_tee`, log `info!(key_id, "Replica report key not bound: not in a TEE")` and return `Ok(false)`. Otherwise call `dstack_sdk::dstack_client::DstackClient::new(None).emit_event(REPORT_KEY_EVENT.into(), key.event_payload()).await?` and return `Ok(true)`. Log only `key_id` and `boot_id`; never the private key.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(replica-state): per-boot report key bound via dstack event log`

---

### Task 5: State sinks (Redis, memory)

**Files:**
- Create: `src/replica_state/sink.rs`
- Modify: `Cargo.toml` (add `redis = { version = "0.32", default-features = false, features = ["tokio-comp", "tokio-rustls-comp", "connection-manager", "streams"] }`; use the latest 0.x that resolves with the existing tokio/rustls versions)

**Interfaces:**
- Consumes: `report::Envelope`.
- Produces:
```rust
#[async_trait::async_trait]   // or boxed futures, matching Task 3's choice
pub trait StateSink: Send + Sync {
    async fn publish(&self, frames: &[Envelope]) -> anyhow::Result<()>;
}
pub struct RedisSink { /* ConnectionManager, key_ttl_secs, stream, stream_maxlen */ }
impl RedisSink { pub async fn connect(url: &str, key_ttl_secs: u64, stream: String, stream_maxlen: usize) -> anyhow::Result<Self>; }
pub fn replica_key(host_id: &str, replica_id: &str) -> String;   // "replica:{host_id}:{replica_id}"
#[derive(Default)] pub struct MemorySink { pub published: std::sync::Mutex<Vec<Envelope>> }
```

Redis layout per publish (one pipeline, one round trip):
- `SET replica:{host_id}:{replica_id} <envelope json> EX key_ttl_secs` for each frame.
- `XADD {stream} MAXLEN ~ {stream_maxlen} * env <envelope json>` for each frame.
- `PUBLISH replica-frames <envelope json>` is **not** included in v1: readers poll keys, and the design says not to rely on pub/sub.

- [ ] **Step 1: Write failing tests:**

```rust
#[test]
fn key_layout() { assert_eq!(replica_key("glm53-gpu03", "r1"), "replica:glm53-gpu03:r1"); }

#[tokio::test]
async fn memory_sink_records_frames_in_order() { /* publish 2 envelopes, assert published == them */ }

/// Real Redis: runs only when REPLICA_STATE_TEST_REDIS_URL is set (CI/dev), otherwise returns early.
#[tokio::test]
async fn redis_sink_sets_ttl_key_and_appends_stream() {
    let Ok(url) = std::env::var("REPLICA_STATE_TEST_REDIS_URL") else { return };
    let sink = RedisSink::connect(&url, 5, "replica-frames-test".into(), 1000).await.unwrap();
    // publish one envelope; then with a plain client: GET key == json, TTL in 1..=5, XLEN >= 1
}

#[tokio::test]
async fn redis_connect_to_unreachable_host_errors_quickly() {
    let started = std::time::Instant::now();
    assert!(RedisSink::connect("redis://127.0.0.1:1", 5, "s".into(), 10).await.is_err());
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::sink` → FAIL.
- [ ] **Step 3: Implement.** `RedisSink::connect` builds a `redis::Client` and a `ConnectionManager` with a 2 s connection timeout and 1 s response timeout (`ConnectionManagerConfig`). `publish` builds one `redis::pipe()` with the commands above, and on error returns it (the caller counts it). Never log envelope JSON above `debug`.
- [ ] **Step 4: Run** tests, `cargo deny check licenses bans`, `cargo build --release` → PASS.
- [ ] **Step 5: Commit** `feat(replica-state): Redis sink (TTL key + capped stream) and memory sink`

---

### Task 6: Publisher loop and wiring

**Files:**
- Create: `src/replica_state/publisher.rs`
- Modify: `src/replica_state/mod.rs` (`spawn_replica_state_publisher`), `src/main.rs` (spawn after `backend_pool` is built, ~line 200)

**Interfaces:**
- Consumes: everything above; `backend_pool::BackendPool::backends()` (`Arc<Backend>` with `base_url` and `active_conns`).
- Produces:
```rust
pub struct ReplicaStatePublisher {
    host_id: String, model: String, pool: Option<String>, tier: Option<String>,
    replicas: Vec<(String /*replica_id*/, String /*base_url*/)>,
    adapters: Vec<Option<Arc<dyn EngineAdapter>>>,  // None until detected (Auto)
    memory: Vec<AdapterMemory>, failures: Vec<u32>, ever_ok: Vec<bool>,
    key: ReportKey, seq: u64, client: reqwest::Client, interval: Duration,
}
impl ReplicaStatePublisher {
    pub fn new(cfg: &ReplicaStateConfig, model: &str, backend_urls: &[String], key: ReportKey, client: reqwest::Client, forced: Option<Engine>) -> Self;
    /// One tick: read every replica concurrently (timeout = interval * 0.8), build and sign one frame per replica.
    pub async fn tick(&mut self, inflight: &dyn Fn(usize) -> u32, now_ms: u64) -> Vec<Envelope>;
}
pub fn spawn_replica_state_publisher(cfg: ReplicaStateConfig, model: String, pool: Arc<BackendPool>, client: reqwest::Client, dev_mode: bool, non_tee: bool);
```
Lifecycle rules for v1: no successful read yet → `warming`; a successful read → `ready` and reset failures; 3 or more consecutive failures after any success → `unhealthy`; otherwise keep the last state. On a failed read, `load = Load::default()` (all null) and `limits` keep their last known values. `engine_sampled_at_ms` = time the read completed; `reported_at_ms` = time the frame was built. `seq` increments once per frame, starting at 1.

- [ ] **Step 1: Write failing tests** (tokio + wiremock + `MemorySink`):

```rust
#[tokio::test]
async fn tick_emits_one_signed_frame_per_replica_with_monotonic_seq() {
    // two wiremock servers serving /v1/loads JSON; forced engine = Sglang
    // tick twice; assert 2 frames per tick, replica ids r1/r2, seq 1..=4, all verify with key.verifying_key(),
    // lifecycle Ready, same boot_id, report_key_id == key.key_id
}

#[tokio::test]
async fn unreachable_engine_goes_warming_then_unhealthy_with_null_load() {
    // replica r1 base_url = "http://127.0.0.1:1"; tick 4 times
    // frames for r1: warming (never succeeded) and load all None
    // then point at a working mock, tick => Ready; then break it, tick x3 => Unhealthy
}

#[tokio::test]
async fn tick_is_bounded_by_interval() {
    // mock that delays 5 s; interval 200 ms; assert tick() returns in < 400 ms with that replica's load null
}

#[tokio::test]
async fn frames_contain_no_request_content() {
    // serialize a tick's envelopes; assert keys are exactly the ReplicaReport field set (no extra fields)
}
```

- [ ] **Step 2: Run** `cargo test --lib replica_state::publisher` → FAIL.
- [ ] **Step 3: Implement** `tick` with `futures_util::future::join_all` over replicas, each read wrapped in `tokio::time::timeout(interval.mul_f32(0.8), …)`. Detect the engine lazily for `Auto`. `spawn_replica_state_publisher`:
  1. `let key = ReportKey::generate()`, then `bind_to_attestation(&key, dev_mode, non_tee).await` (log and continue on error: reports are still produced, but readers won't find an attested key and will reject them, so log at `warn!`).
  2. `RedisSink::connect` with retry every 5 s until it succeeds; don't block startup (spawned task).
  3. Loop on `tokio::time::interval(cfg.interval)` with `MissedTickBehavior::Skip`: `frames = publisher.tick(...)`, then `sink.publish(&frames)`. Emit `replica_state_frames_total` (counter), `replica_state_publish_failures_total` (counter) and `replica_state_tick_seconds` (histogram) via the `metrics` crate used in `engine_load.rs`.

  `inflight(i)` = `pool.backends()[i].active_conns.load(Relaxed)`.

  In `main.rs`, after `backend_pool` is created:
```rust
if let Some(rs) = replica_state::ReplicaStateConfig::from_env(&config.backend_urls)? {
    info!(host_id = %rs.host_id, replicas = rs.replica_ids.len(), interval_ms = rs.interval.as_millis() as u64, "Publishing replica state to Redis");
    replica_state::spawn_replica_state_publisher(rs, config.model_name.clone(), backend_pool.clone(), http_client.clone(), config.dev_mode, config.non_tee_deployment);
}
```
  Use the actual field name for non-TEE in `Config` (search for `NON_TEE_DEPLOYMENT`).
- [ ] **Step 4: Run** `cargo test` (full suite) → PASS; `cargo fmt --check`.
- [ ] **Step 5: Commit** `feat(replica-state): publisher loop sampling replicas and writing signed frames`

---

### Task 7: Operator documentation

**Files:**
- Create: `docs/replica-state.md`
- Modify: `README.md` (env var table), `CLAUDE.md` (module map: one line for `replica_state/`)

- [ ] **Step 1:** Write `docs/replica-state.md`:
  - purpose (stage 1 of the placement plan, a link to the design doc);
  - env vars with defaults (Task 1);
  - frame schema (Task 2 JSON example) and its signing rule;
  - the SGLang/vLLM field mapping (Task 3);
  - Redis layout (Task 5);
  - lifecycle rules (Task 6);
  - how readers verify: find the `nearai-replica-report-key-v1` entry in the attested event log, match `key_id`, verify the signature, check `seq`/`boot_id` monotonic and `engine_sampled_at_ms` freshness;
  - privacy: IDs and numbers only.
- [ ] **Step 2:** Add the env vars to README, and the module line to CLAUDE.md.
- [ ] **Step 3: Commit** `docs(replica-state): operator guide and schema`

---

## Verification before hand-off

- `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test && cargo deny check`
- Manual smoke test (local, no CVM): run a local Redis, a mock SGLang (`python -m http.server` is not enough; use the wiremock-based test or a tiny script serving `/v1/loads`), and the proxy with `REPLICA_STATE_REDIS_URL=redis://127.0.0.1:6379 REPLICA_STATE_HOST_ID=local DEV=1`. Then check `redis-cli GET replica:local:r1`, `TTL`, and `XLEN replica-frames`.
- A real-CVM check (stage 1 exit criteria) is a separate step, per `docs/testing-on-cvm.md`.
