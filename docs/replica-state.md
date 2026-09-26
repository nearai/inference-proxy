# Replica state: signed per-replica load to Redis

Opt-in, off by default. Each in-CVM proxy polls its own SGLang replicas'
`/v1/loads` and publishes one signed, normalized `ReplicaReport` frame per
replica to Redis every tick. It changes nothing on the request path: with
`REPLICA_STATE_REDIS_URL` unset, behavior, startup and tests are unchanged.

## Purpose

This is stage 1 ("Latest recommendation") of the placement plan described in
the design doc, [Inference Placement
Map](https://claude.ai/artifact/T3WkANwFetyUYsHhMBNheo) — specifically
"Replica lifecycle and state reporting" §2 and §4. A future placement layer
reads these frames to route new work to the least-loaded healthy replica
instead of relying on this proxy's own least-connections view. This change
only publishes the frames; nothing reads them yet.

## Env vars

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REPLICA_STATE_REDIS_URL` | No | unset (feature off) | Redis connection URL (`redis://` or `rediss://`). Presence (non-blank) is what turns the feature on. Never logged; only the host:port is (see Privacy below). |
| `REPLICA_STATE_HOST_ID` | Yes, if the URL is set | — | This proxy's host identifier. Used as the Redis key namespace (`replica:{host_id}:*`) and in every frame's `host_id` field. |
| `REPLICA_STATE_REPLICA_IDS` | Yes, if the URL is set | — | Comma-separated, one stable replica ID per backend, in pool order (see below). Must be unique and match the backend count exactly, or the feature disables itself. |
| `REPLICA_STATE_INTERVAL_MS` | No | `500` | Publish interval in ms; must be within `200..=2000` (otherwise the feature is disabled with an error). Reads for a tick are bounded to 4/5 of the interval so a slow replica never overruns its slot. |

### Pool order for `REPLICA_STATE_REPLICA_IDS`

The proxy assigns replica `i` of `REPLICA_STATE_REPLICA_IDS` to backend `i` of
the pool, where the pool is every `VLLM_BACKEND_URLS` entry followed by every
`VLLM_BACKEND_LONG_CONTEXT_URLS` entry (long-context backends are appended
after the base backends, so their indexes never move — see `docs/gateway-mode.md`).
The list's length must equal `base backends + long-context backends`; a
mismatch is treated as invalid configuration.

Example, two base backends and one long-context backend:

```bash
VLLM_BACKEND_URLS=https://host-a:8000,https://host-b:8000
VLLM_BACKEND_LONG_CONTEXT_URLS=https://host-a-long:8000
REPLICA_STATE_REPLICA_IDS=r1,r2,r3   # r1=host-a, r2=host-b, r3=host-a-long
```

### Bad configuration never stops the proxy

Any invalid `REPLICA_STATE_*` value (a Redis URL that doesn't parse or isn't
`redis://`/`rediss://`, missing `REPLICA_STATE_HOST_ID`, a replica-count
mismatch, a duplicate ID, an out-of-range interval) is logged as
an error and the feature is disabled for that boot. Startup, routing and every
other proxy behavior continue unaffected.

## Frame schema

Each tick produces one `ReplicaReport` per replica, JSON-serialized compactly
(no spaces) as the `frame` string of a signed envelope:

```json
{
  "schema": 1,
  "host_id": "gpu01",
  "replica_id": "r1",
  "boot_id": "00000000-0000-4000-8000-000000000001",
  "seq": 7,
  "engine_sampled_at_ms": 1790000000011,
  "reported_at_ms": 1790000000312,
  "lifecycle_state": "ready",
  "model": "z-ai/glm-5.3-flash",
  "engine": "sglang",
  "engine_version": "0.5.9",
  "limits": { "max_running": 32 },
  "load": {
    "running": 14,
    "queued": 0,
    "prefill_backlog_tokens": 51200,
    "kv_usage": 0.63,
    "gen_tps": 910.0,
    "cached_token_ratio": 0.71
  },
  "proxy_inflight": 16,
  "report_key_id": "0123456789abcdef"
}
```

Notes:

- `schema` is `1`. `engine` is currently always `"sglang"` (a vLLM adapter is
  future work, cut from stage 1).
- `lifecycle_state` is one of `warming`, `ready`, `unhealthy` in this writer
  (the full `Lifecycle` enum also has `degraded`, `draining`, `drained` for
  future writers/readers). A replica starts `warming`, becomes `ready` after
  its first successful `/v1/loads` read, and becomes `unhealthy` after 3
  consecutive failed reads (`publisher::UNHEALTHY_AFTER_FAILURES`). On
  failure the frame keeps the last known `limits`/`engine_version` and
  `engine_sampled_at_ms`, but every `load` field is `null` — a field the
  engine didn't report, or that couldn't be read, is always `null`, never
  `0`.
- `seq` is per tick, shared by every replica's frame in that tick, starting
  at 1 for this boot.
- `boot_id` is a random UUID v4 generated once per proxy process start — it
  identifies this proxy's boot, not the engine's (see Caveats).
- `engine_sampled_at_ms` is the engine's own scheduler timestamp (SGLang's
  `timestamp`, the oldest rank's when the engine has more than one), or
  `null` when unknown. A never-reachable replica reports `null`; a failed read
  after a success repeats the last known value, so it can be stale. Readers
  should treat `null` or a too-old value as unknown load.
- `reported_at_ms` is the proxy's wall-clock time when the frame was sealed,
  taken once per tick after every replica's read has completed (all frames of
  a tick share it). It is what readers use for frame freshness.
- `report_key_id` identifies the per-boot signing key (see Signing below).

### Signing

```
sig = base64(ed25519(report_key, b"nearai-replica-report-v1\n" ++ frame_bytes))
```

`frame_bytes` is the exact UTF-8 bytes of the `frame` string above. The
envelope on the wire is:

```json
{
  "frame": "{\"schema\":1,\"host_id\":\"gpu01\",...,\"report_key_id\":\"0123456789abcdef\"}",
  "sig": "<base64 ed25519 signature>",
  "key_id": "0123456789abcdef"
}
```

`key_id` is **not** signed; it is only a hint telling the reader which key to
verify with. The signed `report_key_id` inside `frame` must equal it, or the
frame is rejected. Readers must verify the signature over the received
`frame` bytes first, and only then parse them; nothing on the writing or
reading side re-serializes the JSON before verifying.

The report key is a fresh random Ed25519 key generated once per proxy boot
(never persisted). Its public half and `key_id` (first 16 hex chars of
`sha256(public_key)`) are recorded in the dstack attested event log under the
event name `nearai-replica-report-key-v1`, together with the host and the
replicas the key may report for:

```json
{
  "key_id": "0123456789abcdef",
  "public_key_hex": "<64 hex chars>",
  "boot_id": "00000000-0000-4000-8000-000000000001",
  "host_id": "gpu01",
  "model": "z-ai/glm-5.3-flash",
  "replica_ids": ["r1", "r2"]
}
```

That binds the key to the CVM's TDX quote without changing `report_data`.

## SGLang mapping

Frames are built from a `GET {base}/v1/loads?include=core` read of each
replica, normalized by `parse_sglang_loads`:

| Frame field | Derived from SGLang `/v1/loads` (summed across ranks unless noted) |
|---|---|
| `load.running` | `num_running_reqs` (`null` if the sum doesn't fit a `u32`) |
| `load.queued` | `num_waiting_reqs` (`null` if the sum doesn't fit a `u32`) |
| `load.prefill_backlog_tokens` | `num_waiting_uncached_tokens` |
| `load.gen_tps` | `gen_throughput` |
| `load.kv_usage` | `num_used_tokens / max_total_num_tokens`, clamped to `[0, 1]`; `null` if either is missing or the total is `0` |
| `load.cached_token_ratio` | `cache_hit_rate`, single-rank replicas only (`null` for multi-rank, since a hit ratio isn't meaningfully summed) |
| `limits.max_running` | `max_running_requests`, summed; `null` if the sum is `0` |
| `engine_version` | top-level `version` (`null` if absent) |
| `engine_sampled_at_ms` | the oldest rank's `timestamp` (float seconds), converted to milliseconds |

A read is discarded entirely (treated as a failure) if the response is
unreachable, times out, returns a non-success status, isn't JSON, has no
`loads` array, has an empty `loads` array, or any rank is missing a numeric
`timestamp`. A field missing from one rank, or a sum that overflows, makes
that summed field `null` for the whole replica, not `0`.

## Redis layout

Keys and the stream are namespaced per host, so one ACL pattern
(`~replica:{host_id}:*`) covers everything a given proxy writes:

- `replica:{host_id}:{replica_id}` — `SET ... EX 5` of the replica's latest
  signed envelope (JSON). 5 s TTL, refreshed every publish; a replica that
  stops publishing (crash, feature disabled, proxy down) disappears from
  reads after at most 5 s.
- `replica:{host_id}:frames` — a capped stream (`XADD ... MAXLEN ~ 20000 *
  env <json>`) of every envelope ever published by this host, most recent
  last.

Both are written in one Redis pipeline per tick (one `SET` per replica, then
one `XADD` per replica). The client is `redis` 0.32.x with `tokio-comp`,
`tokio-rustls-comp` and `connection-manager`; a 2 s connect timeout, 1 s
response timeout and one retry. Note: `redis` 1.x was ruled out — it pulls in
a BSL-1.0 dependency that `cargo deny` rejects — so this stays pinned to
0.32.x.

## Lifecycle rules

- **`warming`**: initial state, and the state while a replica has never had a
  successful read.
- **`ready`**: set on every successful read.
- **`unhealthy`**: set after `UNHEALTHY_AFTER_FAILURES` (3) consecutive
  failed reads, once the replica has had at least one prior success. A
  single blip does not flip lifecycle; three in a row does.
- A failed read never resets `failures` to zero implicitly — only a
  successful read does.
- The Redis TTL (5 s) is a second, independent signal: a reader can lose a
  replica's key even while the proxy still reports it as `ready` in the
  stream, if the proxy itself stopped publishing.

## How readers verify a frame

1. Verify the TDX quote and replay the event log against RTMR3 (the existing
   attestation verification).
2. Collect this host's `nearai-replica-report-key-v1` events; check
   `hex(sha256(public_key))[..16] == key_id` and that the event's `host_id`
   matches. Only the **last** such event in the log is current; frames signed
   by earlier keys are rejected.
3. Use the envelope `key_id` hint to pick the key; verify the sig over the
   frame string, then parse; require signed `report_key_id == key_id`, signed
   `host_id`/`replica_id` == the Redis key they came from, and
   `replica_id ∈` the event's `replica_ids`.
4. Within one key, `seq` strictly increases. `boot_id` is a random
   per-process id (not ordered) — ordering across restarts comes from the
   event log.
5. Freshness: reject if `now − reported_at_ms > 3 × interval`; treat
   `engine_sampled_at_ms` null or older than the reader's staleness bound as
   unknown load.

## Caveats

- The capped stream (`replica:{host_id}:frames`) is roughly an 80-minute
  buffer at 2 replicas and 2 Hz (`STREAM_MAXLEN` = 20,000 entries). It is not
  the week-long recorder called for in the stage 1 exit criteria; a separate
  drain to object storage is required for that and is not part of this
  change.
- `boot_id` identifies the proxy's boot, not the engine's. An engine restart
  behind an unrestarted proxy does not change `boot_id`.
- In-CVM mode only — in gateway mode VLLM_BACKEND_URLS point at other
  proxies, so replicas stay `warming`.
- Each proxy restart appends another `nearai-replica-report-key-v1` event to
  the attested event log (RTMR3 grows monotonically); this is expected and
  by design, not a leak or a bug to clean up. Pre-merge gate: confirm
  cloud-api's attestation verifier and the public verifier accept an unknown
  runtime event before enabling on a production host.
- Nonce-less attestation reports are cached for up to `ATTESTATION_CACHE_TTL`
  (default 300 s). A reader that fetches `/v1/attestation/report` right after
  a proxy restart may get a cached report from before the new key event was
  recorded, and so not find `key_id` yet — retry within one TTL.

## Metrics

- `replica_state_key_bound` (gauge): `1` once the report key is recorded in
  the event log; `0` if binding was skipped (dev/non-TEE), failed, or timed
  out. At `0`, readers will reject every frame from this boot.
- `replica_state_frames_total` (counter): frames written to Redis.
- `replica_state_publish_failures_total` (counter): ticks whose Redis
  pipeline failed.
- `replica_state_tick_seconds` (histogram): wall time of each tick (reads,
  signing and publish).

## Privacy

Frames carry IDs and numbers only: host/replica/boot identifiers, sequence
and timestamps, lifecycle state, model name, and load counters. No prompts,
completions, org IDs, affinity keys, or content-derived hashes ever appear in
a frame. The Redis URL (which may carry credentials) is never logged; only
its host:port is, and only a redis error's `kind()` is logged, never its
`Display` (which can echo the URL). Frame contents are logged, if at all,
only at `debug`.

Example of what an operator should *not* do — never write a real credentialed
URL into config files, shell history, or a runbook:

```bash
# Don't: REPLICA_STATE_REDIS_URL=rediss://user:hunter2@10.0.4.12:6379/0
# Do:    REPLICA_STATE_REDIS_URL=rediss://redis.internal:6379/0
```
