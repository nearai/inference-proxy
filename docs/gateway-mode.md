# Gateway mode: one inference-proxy in front of a model fleet

The same binary that runs inside every CVM (in front of one host's engines)
can run *outside* the TEE as a fleet-wide, model-scoped gateway: customer
`sk-` keys validated against cloud-api, usage reported to cloud-api, requests
forwarded to the CVM proxies — without cloud-api's inference pipeline in the
request path. Authentication, billing, streaming, cancellation and no-content
logging are the existing, deployed behaviors of this proxy. Gateway mode only
adds a backend bearer and a few opt-in policies.

```text
client ──sk-key──▶ inference-proxy (gateway, non-TEE) ──backend token──▶ model-proxy (L4/SNI)
                        │  check_api_key / internal usage                      │
                        ▼                                                      ▼
                    cloud-api                                CVM: nginx ▶ inference-proxy ▶ SGLang
```

## What the gateway does per request

Two compatibility repairs run on every lane (gateway and CVM alike), after
decryption and before dispatch, because the engine's `400` is the same
whichever lane sent the request: the tool-call `arguments` repair
(`src/tool_calls.rs`, nearai/inference-proxy#239) and the
`response_format.json_schema` repair (`src/response_format.rs`,
nearai/inference-proxy#279). The latter inserts the required
`json_schema.name` (`response_schema`) when a wrapper object has none, and
wraps a bare JSON Schema sent as `json_schema` (recognised by a JSON Schema
keyword at its root, such as `type` or `properties`) into `{"name", "schema"}`.
Ambiguous objects such as `{}` only get the name, so the engine reports the
missing schema rather than serving an accept-all grammar. An explicit `name`
of any value is preserved for native backend validation; the schema,
strictness and other fields are untouched.
Repairs are counted by `json_schema_response_format_repaired_total{repair}`.

A third rewrite is not for every lane: a model of a
[model list](#several-models-in-one-gateway) whose chat template takes one
`system` message, as the first message, can have a request's system messages
merged into that before dispatch (`merge_system_messages`,
`src/system_messages.rs`, see
[System messages a model takes only first](#system-messages-a-model-takes-only-first)).
It runs after the two repairs above. A process without a list never does it.

1. `Authorization: Bearer sk-…` → `POST {CLOUD_API_URL}/v1/check_api_key`
   (retries on transport/5xx; 401/402/429 pass through). With
   `VLLM_PROXY_ALLOWED_ORG_IDS` set, a valid key from any other organization
   gets a 403 here.
2. Policy: any content part whose `type` is in
   `VLLM_PROXY_REJECTED_CONTENT_PART_TYPES` (e.g. `video_url`) → `400` before
   anything is fetched or dispatched; image inputs are validated as today.
3. Backend selection over `VLLM_BACKEND_URLS`: least-connections with
   conversation affinity (`VLLM_BACKEND_CONVERSATION_AFFINITY=1`), so later
   turns of a long conversation stay on the host whose prefix cache holds them.
4. Forward with `Authorization: Bearer $VLLM_BACKEND_TOKEN` on the dedicated
   backend client. The CVM proxy treats it as a trusted config token: it does
   **not** re-validate the customer key and does **not** report usage, so
   exactly one component bills. The body is forwarded verbatim apart from the
   compatibility repairs above; for streams the gateway forces
   `stream_options.include_usage` and `continuous_usage_stats`.
5. Response streamed back. On client disconnect the upstream connection is
   dropped (the CVM proxy drops its engine connection, the engine aborts) and
   the usage observed so far is reported.
6. `POST {CLOUD_API_URL}/v1/internal/usage` with the shared usage token,
   idempotent on the provider completion id. The report is handed over when
   the request completes and sent off the request path: once, with a 5 s
   timeout, unless `VLLM_PROXY_USAGE_REPORT_*` says otherwise (see
   [Usage report delivery](#usage-report-delivery)). With
   `VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH` it is written to a file first and
   kept there until cloud-api has accepted it
   ([The outbox on disk](#the-outbox-on-disk)).

Nothing about request or response content is logged or stored at any hop.
(A gateway with an outbox keeps its usage reports in a file until cloud-api has
accepted them: ids and token counts, no content.)

## Backend membership: static handle URLs

model-proxy routes `<label>-b<handle>.<base>` to exactly one registered
backend (`Router::select_backend_by_handle`). A handle is a salted digest of
the backend's `ip:port` (`registry::backend_handle`), so it is stable across
redeploys and restarts of that host; it changes only if the host's address or
model-proxy's admin token changes. List the handles once, with the admin
token, from an operator machine:

```bash
curl -H "Authorization: Bearer $MODEL_PROXY_TOKEN" \
  "https://completions.near.ai/backends/list?domain=glm-5-3-flash.completions.near.ai"
```

and set `VLLM_BACKEND_URLS` to the
`https://glm-5-3-flash-b<handle>.completions.near.ai` URLs, one per host. A
host that goes down is taken out by the health checker
(`VLLM_BACKEND_HEALTH_PATH=/healthz`, the CVM proxy's readiness route) and put
back when it recovers; adding or replacing a host is an env change and a
restart. The admin token is not needed at runtime and must not live on the
gateway host.

## Configuration

Everything is an environment variable; the new ones are all opt-in and default
to the current in-CVM behavior.

| Variable | Gateway value | Purpose |
| --- | --- | --- |
| `MODEL_NAME` | `z-ai/glm-5.3-flash` | Must match cloud-api's model name exactly (usage rows are keyed on it). |
| `VLLM_PROXY_MODEL_LIST_FILE` | unset | Path of a JSON file listing several models for this one process to serve, in place of `MODEL_NAME` and the backend lists (see [Several models in one gateway](#several-models-in-one-gateway)). Unset = one model from the variables in this table. |
| `TOKEN` | a random secret | Trusted config token for *this* gateway (operators/tests). Customers use `sk-` keys. |
| `CLOUD_API_URL` / `CLOUD_API_USAGE_TOKEN` | prod values | Key validation and usage reporting. |
| `VLLM_BACKEND_URLS` | the `-b<handle>` URLs | Fleet membership (see above). |
| `VLLM_BACKEND_TOKEN` | a token from the CVM proxies' `TOKEN` list | Outbound bearer for backend requests only (dedicated HTTP client; never sent to cloud-api). Mint one for the gateway so it can be revoked on its own. |
| `VLLM_BACKEND_PRIORITY` | `-1` | Sent as `X-NearAI-Priority` on every backend request; the CVM proxies put it in the engine's `priority` (see below). |
| `VLLM_BACKEND_HEALTH_PATH` | `/healthz` | The CVM proxy's unauthenticated readiness route (dstack + engine). |
| `VLLM_BACKEND_CONVERSATION_AFFINITY` | `1` | Fleet-level conversation affinity (the CVM proxy still does its own across its replicas). |
| `HEALTH_CHECK_INTERVAL_SECS` / `_TIMEOUT_SECS` / `_MAX_FAILURES` | `5` / `4` / `3` | The timeout must exceed the CVM proxy's own 3 s backend probe. |
| `VLLM_PROXY_ALLOWED_ORG_IDS` | the partner's organization id | Only that organization's keys are served; any other valid key gets 403. Without it any cloud-api key works here, same as the direct `*.completions.near.ai` endpoints. |
| `VLLM_PROXY_REJECTED_CONTENT_PART_TYPES` | `video_url,input_audio,file` | Modalities this deployment does not serve → deterministic `400`. |
| `VLLM_PROXY_MODELS_DOCUMENT_URL` / `VLLM_PROXY_CAPACITY_REQUESTS_PER_MINUTE` | `https://cloud-api.near.ai/v1/models` / `150` | `GET /v1/models` serves cloud-api's entry for `MODEL_NAME` (pricing, modalities, `is_ready`, `openrouter.slug`) with `capacity` added: concurrency = `VLLM_PROXY_ADMISSION_MAX_INFLIGHT`, requests per minute = this value. One URL for inference and the listing; `is_ready` stays under cloud-api's catalog control (the kill switch). Source unreadable → the engine's list, as without the variable. |
| `VLLM_PROXY_DISCOUNT_TO_USER` | unset (list price) | The lane's discount off the list price, a fraction in `[0, 1)` with at most four decimal places (`0.2` = 20 % off): published as `discount_to_user` on the models document entry (in place of any the source carries) and sent with every usage report, so cloud-api bills the price the aggregator shows; empty or `0` = none, an invalid value fails startup, and it requires `VLLM_PROXY_MODELS_DOCUMENT_URL`. While that source is unreadable, the engine list served instead carries the discount on every entry, and an engine answer that is not a model list becomes a 502 rather than going out without it. cloud-api refuses a report whose discount exceeds its `INTERNAL_USAGE_MAX_DISCOUNT` (default `0.5`), leaving that usage unbilled, so keep the value within it. |
| `VLLM_PROXY_USAGE_REPORT_TIMEOUT_SECS` / `_MAX_ATTEMPTS` / `_INITIAL_BACKOFF_MS` / `_DEADLINE_SECS` | `30` / `5` / `5000` / `300` | How long one attempt at a usage report may take, how many attempts a report gets, the backoff before the first retry, and how long after its request a report may still be sent (see [Usage report delivery](#usage-report-delivery)). Unset = one attempt of 5 s and no deadline, as in a CVM. |
| `VLLM_PROXY_USAGE_REPORT_MAX_IN_FLIGHT` | `8` | Usage reports in flight at once; the others wait in a bounded in-memory queue (`_MAX_QUEUED`, default `10000`). Unset = no cap. Required for a timeout above 5 s or more than one attempt. |
| `VLLM_PROXY_USAGE_REPORT_SHUTDOWN_DRAIN_SECS` | below the service's stop timeout | On SIGTERM, once every open request has ended, how long to wait for usage reports still queued or in flight before exiting. Unset = no wait. |
| `VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH` | a file on a volume that outlives the process | Keep every usage report in this SQLite file from the completion of its request until cloud-api has accepted it, across restarts and cloud-api outages (see [The outbox on disk](#the-outbox-on-disk)). `_OUTBOX_MAX_PENDING` (default `1000000`) and `_OUTBOX_MAX_REJECTED` (default `100000`) bound it. Unset = no file, reports live in memory only. |
| `VLLM_PROXY_REASONING_OFF_EFFORT` | `low` | What "no reasoning" means for GLM-5.3 Flash (see below). |
| `VLLM_PROXY_SSE_KEEPALIVE_SECS` | `15` | `: keep-alive` SSE comments while the upstream is silent (long prefill/queueing), so intermediaries with read timeouts do not cancel. Off in CVMs: comments are not part of the signed bytes. |
| `VLLM_PROXY_MAP_QUEUE_FULL_TO_429` | `1` | The engine's admission rejection (queue full, or a queued request displaced by a higher-priority one) becomes 429 with `Retry-After: 2` and type `overloaded`, the same shape as the gateway's own refusals: back-pressure, not an outage. Off in CVMs: cloud-api's peer fallback keys on the 503. |
| `VLLM_PROXY_STREAM_ERROR_PEEK_MS` | `1000` | Streams wait up to 1 s for the first upstream event; an admission-time `data: {"error":…}` becomes a real 429/5xx instead of a 200 that fails mid-stream. A slow first token just times the peek out. |
| `VLLM_PROXY_STREAM_COMMIT_MS` | `5000` | Commit the stream's `200 text/event-stream` after 5 s even if the engine has not answered, so the keep-alives above actually reach the client during a long prefill (see below). |
| `VLLM_PROXY_FIRST_TOKEN_DEADLINE_MS` | `8500` | Refuse a streaming request with 429 + `Retry-After` when the engine has not produced its first event this long after the request arrived, instead of committing a 200 the caller is about to cancel (see below). `0`/unset = off. |
| `VLLM_PROXY_FIRST_TOKEN_DEADLINE_PER_1K_TOKENS_MS` | `800` | Added to that deadline per 1,000 estimated prompt tokens, because the caller's own deadline grows with the prompt too. Same estimate as the long-context tier. |
| `VLLM_PROXY_FIRST_TOKEN_DEADLINE_MAX_MS` | `30000` | Requests whose computed deadline is above this are exempt: they keep the commit window and the keep-alives. `0`/unset = no cap, every streaming request gets a deadline. |
| `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` / `_START_INFLIGHT` | `48` / `48` | The lane's in-flight budget: refuse with 429 + `Retry-After` before dispatch instead of queueing (see below). Set start equal to maximum for separately reviewed rollout stages; optional ramp settings remain available. |
| `VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS` | `30000` | Refuse new work while, over the last minute, at least 20 lane requests reached the engine and 5 % of them (at least two) waited longer than this for their first generation event. |
| `VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS` | `10` | A backend that rejected at engine admission within this window is steered around; when every healthy backend did, new work is refused. |
| `VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT` | `1` | Engine-reported queue depth at or above which a backend counts as saturated for placement and the fleet-wide queue refusal below; the OpenRouter lane runs `4` for engines that run chunked prefill, where a shallow queue is normal while slots are still free. |
| `VLLM_BACKEND_CONNECT_FAILOVER` | `1` | A backend that refuses the connection (host down, proxy restarting) costs the request nothing: it is re-sent once to another healthy backend, the dead one leaves the rotation until a probe succeeds, and a pinned conversation follows. Never on an HTTP error. |
| `VLLM_BACKEND_PROBE_URLS` / `_INTERVAL_SECS` | `http://<host-ip>:8000,…` / `2` | The engines' live running/queued counts, read from each host's plain metrics port (the same route model-proxy samples; reachable from the model-proxy hosts, no token). One reading covers the replica the host would route to, so a queue in it means no replica is free. Drives placement and the fleet-wide queue refusal below. |
| `VLLM_BACKEND_LONG_CONTEXT_URLS` | the `-long-b<handle>` URLs | The hosts of the long-context tier, listed as their handle URLs under the model's `-long` model-proxy domain (see below). Appended to the pool after `VLLM_BACKEND_URLS`, so the base backends keep their indexes. Empty = one flat pool, as today. |
| `VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS` | `http://<host-ip>:8000,…` | One engine-load probe per long-context backend, same order. Required when `VLLM_BACKEND_PROBE_URLS` is set, and empty when it is not; internally the two lists are concatenated in pool order. |
| `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS` | `100000` | Estimated input tokens above which a request is placed on that tier. `0`/unset switches the whole feature off, and nothing is even estimated. |
| `VLLM_BACKEND_TIER_STRICT` | `1` | Isolate the tiers in both directions: a request whose tier has no healthy backend is refused (429 + `Retry-After`, or a 503 with `error_type: "tier_unavailable"` when admission is off) instead of placed on the other tier. Off by default (see below). Only meaningful with `VLLM_BACKEND_LONG_CONTEXT_URLS` and a nonzero `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS`; without either it is ignored (a startup warning says so). |
| `NON_TEE_DEPLOYMENT` | `1` | No dstack socket outside a CVM: `/healthz` reports `"dstack":"skipped"`, no attestation refresh, and `/v1/attestation/report`, `/v1/signature/{id}`, `/internal/gpu_evidence` answer 404 so nothing unverifiable is advertised. |
| `DEV` / `GPU_NO_HW_MODE` | `1` / `1` | Non-TEE: random signing keys, no hardware evidence. |
| `LISTEN_ADDR` / `LISTEN_PORT` | `127.0.0.1` / `31700` | Bind behind the local TLS terminator. |
| `RATE_LIMIT_PER_SECOND` / `RATE_LIMIT_BURST_SIZE` | raised | Per-IP limiter; an aggregator arrives from a handful of IPs. |

In gateway mode the proxy also maps an aggregator's reasoning controls onto
the engine's switch (`reasoning.rs`). `{"reasoning": {"enabled": false}}`, an
effort of `none` or `minimal`, and those values sent as `reasoning_effort`
become `VLLM_PROXY_REASONING_OFF_EFFORT`; other efforts in the `reasoning`
object are copied to `reasoning_effort`; a caller's own `reasoning_effort` is
otherwise respected. The mapped value is also written back into
`reasoning.effort`, because the engine reads the object's field first when both
are present. Without this the model keeps thinking and the caller pays
for tokens it asked not to have. The off value is per model: GLM-5.3 Flash's
template only honours `low` and `high` (anything else means max), and with
thinking switched off outright it writes its reasoning as visible content, so
the lane runs with `low` (about 1-9 reasoning tokens, clean content). `exclude`
and `max_tokens` have no engine equivalent and are left to the aggregator.
Applied counts are in `reasoning_switch_applied_total{kind}`. A model of a
[model list](#several-models-in-one-gateway) can also replace efforts its
engine does not accept (`reasoning_effort_map`, see
[Reasoning efforts a model does not accept](#reasoning-efforts-a-model-does-not-accept)).

The router fails closed: only declared routes exist. The TLS terminator in
front of the gateway should additionally expose only `/v1/chat/completions`,
`/v1/completions`, `/v1/models` and `/healthz`; `/metrics` and `/v1/metrics`
are operator-only.

## Priority

The CVM proxy sets `priority` on every chat/completions body it forwards and
discards whatever the client sent: the `X-NearAI-Priority` value when the
caller authenticated with the proxy's own `TOKEN` (cloud-api, or this gateway),
0 otherwise. No CVM configuration is involved. cloud-api never forwards
customer headers and sends none of its own, so its requests are 0; this gateway
sends `-1`; a direct customer's header is ignored because they use an `sk-`
key. With SGLang's `--enable-priority-scheduling --disable-priority-preemption`
a cloud-api or direct request then skips ahead of queued OpenRouter requests,
and when the waiting queue is full it displaces the newest queued OpenRouter
request, which the engine aborts with "The request is aborted by a higher
priority request." (a 429 here, see below). The engine ignores `priority` until
its flag is on, and 0 is vLLM's default for the field, so the proxies roll
first, everywhere.

## Overload

The engine caps its waiting queue (`--max-queued-requests`) and rejects at
admission with "The request queue is full." (or, with priority scheduling,
"The request is aborted by a higher priority request."): HTTP 503 for JSON
requests, and for streams a first SSE event `data: {"error": …, "code": 503}`
on an HTTP 200. With `VLLM_PROXY_STREAM_ERROR_PEEK_MS` the gateway holds the
stream's status line for up to that long, so the rejection surfaces as a
status; with `VLLM_PROXY_MAP_QUEUE_FULL_TO_429` that status is 429.

### Silence during a long prefill

An engine sends the SSE response headers only together with its first event —
SGLang kick-starts the generator before returning the stream — so from the
client's side a long prefill is not a slow stream, it is *no bytes at all*: a
192k-token prompt measured 23.6 s of silence, and the keep-alive comments above
could not help because they only start once the upstream has answered.
Aggregators cancel a provider that goes quiet (OpenRouter: "send SSE comments as
keep-alives so we know you're still working on the request. Otherwise we may
cancel with a fetch timeout and fallback to another provider"), so the silence
costs the request and the fail-over is counted against us.

`VLLM_PROXY_STREAM_COMMIT_MS` bounds it: when the upstream has not answered
within the window, the gateway commits `200 text/event-stream` on its own and
the keep-alive ticks start immediately, while the same task keeps waiting for
the engine and then pumps the real stream through untouched.

The cost is explicit. After the commit the status line is spent, so an upstream
failure can only be delivered as a terminal `data: {"error": …}` event followed
by `[DONE]` — the same shape the engine itself uses when it rejects on a
committed 200, and the sanitized body the status response would have carried
(counter `stream_late_upstream_errors_total`). Engine rejections arrive in
milliseconds and are unaffected, but a genuine validation error can be slow when
it follows the tokenization of a very large prompt: a context-length 400 on a
1.2M-token body took 13.9 s. Size the window above the errors the deployment
actually produces, and keep it at `0` (the default, and every in-CVM
deployment) where the status must always come from the upstream. Counter
`stream_committed_before_upstream_total` says how often the window fires.

### First-token deadline

Committing the 200 keeps the connection, but it does not buy unlimited time: an
aggregator in front of the gateway runs its own deadline on the provider's first
token and cancels when it passes. Measured on the production lane on 2026-09-21
(header-only capture on the gateway's loopback port, 67 cancelled requests,
robust fit, 62 of them within ±1.5 s of the line):

```text
cancelled after ≈ 9.75 s + 0.242 s per KiB of request body   (≈ 10 s + 1 ms per prompt token)
```

Three things came out of the same capture. The deadline follows the prompt's
*token* count, not its bytes: bodies that were large only because of a base64
image were cancelled at 12-13 s, not at the minutes their size would imply. The
keep-alive comments do not extend it — every cancelled request that had already
received `: keep-alive` comments was cancelled on the same line (8 of 8). And
the cancel is expensive for us: on our side it is a `200` with a zero-byte body
(committed on the window, client gone before the first upstream event, logged as
"Client disconnected before the upstream answered"), while the aggregator books
it as a gateway timeout against the provider. Those cancels were ~97 % of what
it reported as our 5xx over 24 h (3,735 of 3,838), a period in which the gateway
and the TLS terminator in front of it returned no 502 or 504 at all.

A 429 is not counted against a provider the way a timeout is, and the
aggregator's own guidance is to "return early 429s if under load, rather than
queueing". So `VLLM_PROXY_FIRST_TOKEN_DEADLINE_MS` (plus
`_PER_1K_TOKENS_MS` × the estimated prompt) sets a deadline just under the
caller's, and a streaming request whose engine has not said anything by then
gets `429` + `Retry-After` and type `overloaded` instead of a committed 200.
The end user loses nothing: the aggregator fails over to another provider
immediately rather than one or two seconds later.

Details worth knowing:

- The clock starts when the request *arrives*, not when it is dispatched:
  authentication (a cloud-api round trip, sometimes retried) and pre-dispatch
  validation count against it. A request whose budget is already spent when it
  reaches dispatch is refused without asking an engine.
- A request under a deadline is never committed early — `VLLM_PROXY_STREAM_COMMIT_MS`
  does not apply to it, and no keep-alives are sent before the engine answers,
  since they would not move the caller's deadline anyway. Requests exempted by
  `_MAX_MS` keep the commit window and the keep-alives exactly as above.
- On a refusal the in-flight upstream request is dropped, so the CVM proxy and
  the engine abort it instead of generating for a client that is gone; the
  backend connection count and the admission slot are released with it, and the
  wait is recorded as a censored time-to-first-token sample. A client that
  disconnects on its own before the deadline is still handled as a disconnect.
- That censored sample is by construction shorter than the request's deadline,
  so with `_MAX_MS` at or below `VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS` a
  request covered by a deadline can never be one of the slow samples that trip
  the admission breaker (it still counts as a sample): past that point the
  breaker only reacts to the requests the cap exempts.
- What is timed is the upstream's first SSE *event*, not its status line. A
  request under a deadline waits for that event whatever
  `VLLM_PROXY_STREAM_ERROR_PEEK_MS` says, including when it is unset: an engine
  can send its `200` before it has generated anything (vLLM does), and the
  headers alone would then satisfy the deadline with silence. A side effect is
  that these requests always get the first-event error check, peek or no peek.
  What can still satisfy a deadline without a token is a hop that writes
  something of its own first — an in-CVM proxy that both commits early and
  sends `: keep-alive` comments, since a comment is an event on the wire. Leave
  `VLLM_PROXY_SSE_KEEPALIVE_SECS` off on the hosts behind a gateway, as CVMs do
  by default.
- The deadline reads the same input estimate as the long-context tier
  (`context_tier.rs`), and that estimate is computed even when the tier is off.
  Only prompt tokens size the deadline: the output reserve used for tier
  placement, including its 32,768-token cap, does not extend or exempt it.
- Counter `first_token_deadline_refusals_total`, plus one info line per refusal
  ("First-token deadline passed, refusing") with the deadline, the wait and the
  estimated prompt size — numbers only.

Why first tokens are late in the first place is an engine-side scheduling
question (short requests queued behind a long chunked prefill on the same
replica); the deadline only decides what the caller is told meanwhile.

Queue saturation checks and strict context-tier isolation still apply before
dispatch. They reduce overload and keep long prefills on the intended hosts;
this deadline also bounds the wait after a request has been admitted and
placed. Enabling it does not change tier selection or permit cross-tier
failover.

## Admission budget

Tier borrowing is opt-in: `VLLM_PROXY_ADMISSION_TIER_BORROWING=1` requires
admission, both backend tiers, and a positive
`VLLM_PROXY_ADMISSION_LONG_MAX_INFLIGHT_PER_HOST` (use 12 for the initial lane
rollout). Defaults preserve the legacy uniform-share policy.

With borrowing, each base host may hold `ceil(current_budget / configured_base_hosts)`;
each long host may hold `min(ceil(current_budget / configured_total_hosts), long_host_limit)`.
Configured counts prevent surviving hosts receiving larger limits during an outage.
The atomic global budget still bounds their sum, and by default no slots are reserved
for long traffic (see the reserve below). At 48 with three base hosts and one long host, base can use all 48 slots
(16 per host); the long host remains capped at 12. At budgets 56 and 64 the base
bounds become 19 and 22, while long stays at 12. A full shared budget refuses
both tiers. These bounds follow the destination backend during fallback and
connection failover; the context threshold and engine back-pressure policy are unchanged.

### Reserving budget for the long tier

Without a reserve, a flood of short requests can hold the whole budget. The
budget check does not look at the tier, so long-context requests are refused
with the rest while the long hosts sit under their ceiling.
`VLLM_PROXY_ADMISSION_LONG_RESERVED_INFLIGHT=N` keeps `N` slots of the budget
for requests bound for the long tier:

- A request that is not bound for the long tier is admitted only while fewer
  than `current_budget - N` such requests are in flight. Past that it gets the
  usual 429 with `Retry-After`, counted as
  `admission_rejections_total{reason="long_reserve"}`. `reason="budget"` keeps
  meaning that the whole budget is in use.
- The reserve is a floor and adds no ceiling of its own. The admission budget
  check lets long-bound requests past the reserve, but placement still applies
  the per-long-host bound. The long tier holds at most `long hosts x
  min(ceil(current_budget / configured_total_hosts), long-host ceiling)`.
  Keep `N` at or below that aggregate: any larger reserve cannot be used by
  the long tier and only reduces the base tier's allowance.
- Each base host may hold `ceil((current_budget - N) / configured_base_hosts)`,
  so the base bounds add up to what the base tier may hold. The long-host bound
  is unchanged.
- "Bound for the long tier" means placed there. If a long request falls back
  to a base host during placement or connection fail-over, it moves to the base
  count then. When the base allowance is full, that fallback is refused with
  `long_reserve`.

At a budget of 48 with a reserve of 12, three base hosts and one long host: the
base tier holds at most 36 (12 per host) and the long host its 12, whatever the
base demand. Raising the budget and the reserve together leaves the base bounds
where they were: 64 with a reserve of 16 gives the base hosts 16 each, as 48
did without one.

The setting requires `VLLM_PROXY_ADMISSION_TIER_BORROWING` and must be below
`VLLM_PROXY_ADMISSION_START_INFLIGHT`, the lowest the budget ever is. `0`, the
default, changes nothing. `admission_inflight_base` is the in-flight count the
reserve applies to and `admission_long_reserve` the configured value.

`admission_backend_inflight{backend,tier}` and `admission_backend_limit{backend,tier}`
are snapshots at metrics scrape time. `admission_selection_failures_total{requested_tier,tier,reason}`
separates host limits, back-pressure, mixed blockage, no healthy host, and reservation
contention. It counts failed **selection attempts**, including attempts recovered
by fallback; use existing `admission_rejections_total` for terminal request refusals.
The `fallback` destination label means selection was unrestricted.


Mapping the engine's rejection to 429 only helps once the engine's queue is
full; by then the lane's earlier requests are already waiting behind large
prefills and their time to first token is minutes. The gateway therefore
bounds the lane itself (`admission.rs`), in this order, before anything is sent
upstream:

1. **Observed overload.** The engines' own queues first: with
   `VLLM_BACKEND_PROBE_URLS` each host's running and queued request counts are
   polled every two seconds; a host whose queue is at or above
   `VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT` (default 1) is steered around,
   and once every healthy host queues, new work is refused (reason
   `backend_queue`). Then two signals the gateway measures on its own traffic.
   Time to first generation, measured from the moment the request is sent
   upstream (the engine sends its SSE headers only once it has something to
   say, so measuring from the response would skip the queueing and prefill
   wait): over the last minute, at least 20 lane requests
   reached the engine and 5 % of them (at least two) waited longer than
   `VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS` for their first generation event — a
   request that ends (client gone, idle timeout) before generating counts with
   the time it waited, so slow requests clients give up on are not lost; the
   verdict is re-evaluated at most once a second. Engine back-pressure: a
   backend that answered an admission rejection within
   `VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS` is steered around while other
   hosts have room, and once every healthy backend did, new work is refused.
   Either refusal lasts until the signal ages out (reasons `ttft` and
   `backend_queue`).
2. **Global in-flight budget.** At most `budget` lane requests in flight across
   the fleet (reason `budget`). The budget starts at
   `VLLM_PROXY_ADMISSION_START_INFLIGHT` and grows by
   `VLLM_PROXY_ADMISSION_RAMP_STEP` every `VLLM_PROXY_ADMISSION_RAMP_INTERVAL_SECS`
   up to `VLLM_PROXY_ADMISSION_MAX_INFLIGHT`, but only after an interval
   without any overload signal; a restart goes back to the start value.
3. **Per-host share and placement.** By default, `ceil(budget / healthy backends)` lane
   requests in flight per backend (other traffic on the pool does not count),
   reserved atomically at selection so concurrent requests cannot overshoot
   it, so a conversation-affinity pin cannot pile the whole budget onto one
   host. New conversations go to the host with the lowest engine load (running
   + queued) when the engines are polled, the gateway's own connection count
   otherwise. A pinned conversation stays on its host regardless of running
   counts — a free batch slot serves the cached prefix at once, a move costs a
   full re-prefill — and moves only when that host is queueing, at its share
   or steered around, to the least-loaded host with room; it is then re-pinned
   there. Only when no host has room is the request refused (reason
   `host_share`).

A refusal is `429` with `Retry-After: VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS`
and an error of type `overloaded`; the slot is released when the response —
the whole stream, for SSE — is complete. Nothing is retried on the engine's
behalf: one upstream attempt per request, with the single exception of a
connection that cannot be established at all (`VLLM_BACKEND_CONNECT_FAILOVER`),
where nothing reached the engine yet (a connection failure without fail-over
is a typed `502 upstream_unreachable`, a timeout a `504`). An engine rejection
that arrives after
the peek window is already on a committed 200 stream; it still counts as
back-pressure for the next admission decision
(`upstream_stream_error_events_total{phase="after_headers"}`).

`/metrics` exposes `backend_pool_size`, `backend_pool_healthy`,
`backend_affinity_*`, `rejected_content_parts_total{part_type}`,
`sse_keepalive_comments_total`, `upstream_stream_first_event_errors_total`,
`stream_client_disconnects_total`, `admission_inflight`, `admission_budget`,
`admission_rejections_total{reason}`, `admission_ttft_seconds`,
`admission_backpressure_total{backend}`, `backend_failover_total{outcome}`,
`upstream_stream_error_events_total{phase}`, `backend_engine_running{backend}`,
`backend_engine_queued{backend}`, `backend_engine_probe_failures_total{backend}`,
`backend_tier_requests_total{tier,outcome=routed|fallback|fallback_late|refused|refused_late}`
(the `refused*` outcomes only occur with `VLLM_BACKEND_TIER_STRICT`),
`request_estimated_prompt_tokens`,
`backend_tier_output_reserve_capped_total{tier}` (requests whose `max_tokens` counted only up to the reserve cap),
`request_model_match_total{result}` (how the body's `model` compares with what
the gateway serves, see [below](#before-switching-it-on)),
plus the existing usage-report and upstream metrics, and the delivery series
of [Usage report delivery](#usage-report-delivery) once a setting there
differs from its default.

## Long-context tier

A model can serve oversized prompts from dedicated hosts: the same CVMs
registered a second time under a `…-long.completions.near.ai` domain, so a
200k-token prefill does not sit in front of the short requests on the base
fleet. `VLLM_BACKEND_LONG_CONTEXT_URLS` lists those hosts the same way
`VLLM_BACKEND_URLS` does — `https://<model>-long-b<handle>.completions.near.ai`,
the handle being the same salted digest of the host's `ip:port` under the other
domain — and they are appended to the pool, so every backend index, engine
probe and affinity assignment of the base fleet is unchanged. The forwarded
body is not touched either: `model` stays the canonical id on both domains.

cloud-api routes to the tier from the model row's `long_context`
providerConfig; the gateway bypasses cloud-api, so it makes the same decision
itself and mirrors the estimate cloud-api routes with
(`inference_provider_pool::context_routing::estimate_input` plus the `required`
computation next to it):

```text
countable = (message text + serialized tool_calls + serialized tools) / 4
uncounted = media parts × 1024 + messages × 4
required  = ceil(countable × 1.2) + uncounted + min(max_tokens, 32768)
```

`required` strictly above `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS` means the
long tier. The 1.2 safety factor covers the byte estimate only — media parts,
template overhead, the reserved output window and `/v1/completions` token ids
are already token counts. Tool definitions and tool-call arguments are counted
because the lane's traffic is agentic, where they are most of the prompt.
The `max_tokens` reserve is capped at 32,768 (cloud-api's
`CONTEXT_ROUTE_OUTPUT_RESERVE_CAP`): it is the output a caller allows, not what it
will produce, and aggregator clients often send the advertised maximum on
one-line requests — counted in full, every such request would land on the
long-context hosts.
The trade-off is accepted deliberately: a capped request that really does generate
hundreds of thousands of tokens runs on the tier its prompt picked (normally base),
holding one engine slot and its growing KV there — both tiers run the same engine
with the same context length, so it completes. `backend_tier_output_reserve_capped_total`
counts these requests; read it beside `backend_engine_running`/`backend_engine_queued`.
cloud-api additionally refines the decision near the boundary with an exact
`POST /v1/tokenize`; the gateway deliberately does not — a tokenizer dependency
and an extra upstream round trip are not worth it for a placement that is a
preference rather than a correctness rule. Both tiers run the same engine with
the same context length, so a request on the "wrong" tier still succeeds.

Everything downstream of the decision is restricted to the request's tier:
placement, the connection fail-over, and the fleet-wide "every backend is
queueing" refusal. The consequences are deliberate:

- **Per-host share.** Legacy mode divides the budget across every healthy host.
  With three base hosts, one long host and budget 48, that limits base to 36.
  Enable the borrowing policy above to let base use idle shared capacity without
  increasing the long-host ceiling. A host cap is not a reserved allocation.
- **Untiered traffic stays on the base fleet.** `/tokenize`, media, `/v1/models`
  and the health probe carry no size of their own; they would all land on the
  idle long host under least-connections, so they are restricted to the base
  backends (cloud-api keeps its tokenize traffic off that host for the same
  reason). The pool health checker still probes every backend.
- **Full is a refusal, not a spill.** A long-tier host at its share or steered
  around (engine queue, recent engine rejection) means `429` + `Retry-After`
  for the next oversized request — keeping those prefills off the base fleet is
  the whole point, and a fast refusal lets the aggregator route elsewhere.
- **A tier with no healthy backend falls back — by default.** If the wanted
  tier is down entirely the request is placed in the other one rather than
  refused (`backend_tier_requests_total{outcome="fallback"}`), including when
  its last host goes unreachable mid-request: both the placement and the
  connection fail-over re-resolve the restriction after taking that host out
  of the rotation and cross over, counted `fallback_late` — so each request
  adds exactly one `routed` or `fallback`, plus a `fallback_late` if its tier
  died under it. `VLLM_BACKEND_TIER_STRICT` turns this off for deployments
  that isolate the tiers on purpose: the restriction never lifts, so an empty
  tier is refused instead of spilling onto the other one — `429` +
  `Retry-After` when admission is enabled
  (`admission_rejections_total{reason="tier_unavailable"}`), a plain `503`
  with `error_type: "tier_unavailable"` when it is not — counted
  `refused`/`refused_late` in place of `fallback`/`fallback_late`. A live
  request's connect failure marks its backend unreachable immediately, with
  no debounce; the pool health checker's own probe needs
  `HEALTH_CHECK_MAX_FAILURES` (default `3`) consecutive failures on its
  `HEALTH_CHECK_INTERVAL_SECS` (default `5` s) cadence to mark one down, but
  only one success to bring it back — so in strict mode a single failed
  connect can leave a one-host tier refusing everything for up to one
  interval before the next successful probe recovers it.
- **Conversation affinity crosses tiers.** A conversation pinned on a base host
  that grows past the threshold is placed fresh in the long tier and re-pinned
  there — the same re-prefill cloud-api pays at the boundary.
- **The TTFT breaker ignores long-context requests.** A 100k-token prefill
  takes tens of seconds by nature — on either tier — so a request estimated
  above the threshold contributes no time-to-first-generation sample, and that
  is fixed at admission: a request that falls back onto the base fleet does not
  start counting. Back-pressure marks and engine-queue avoidance still apply.
- **Not for fusion or the agent loop.** `VLLM_BACKEND_LONG_CONTEXT_URLS`
  together with `FUSION_ENABLED` or `WEB_CONTEXT_SEARCH_URL` is refused at
  startup: those modes place their own backend requests, which no tier
  restriction reaches.

## Usage report delivery

Every request made with a customer key ends with one usage report to cloud-api
(step 6 above). The report is handed over when the request completes and sent
off the request path: nothing a client sees waits for it.

With none of the settings below a report gets what it has always got: one
attempt with a 5 s timeout, as many reports in flight as requests complete, and
nothing awaited at shutdown. That is too little for a busy lane. cloud-api
writes the usage of one organization one report at a time, about 17 a second,
and a lane completes requests in waves. When 200 of them finish within a
second, the last report waits about 12 s for its write. With one 5 s attempt
the tail of every wave times out, and cloud-api abandons a write whose caller
hung up, so that usage is never recorded. The reports of a wave are also all
open at once, and each holds a database connection that cloud-api's key check
then has to wait for.

| Variable | Default | Lane value | Meaning |
| --- | --- | --- | --- |
| `VLLM_PROXY_USAGE_REPORT_TIMEOUT_SECS` | `5` | `30` | Timeout of one attempt (1 to 300). Above 5 requires `_MAX_IN_FLIGHT`. |
| `VLLM_PROXY_USAGE_REPORT_MAX_ATTEMPTS` | `1` | `5` | Attempts per report, the first one included; `1` = never retried, at most `10`. More than one requires `_MAX_IN_FLIGHT`. |
| `VLLM_PROXY_USAGE_REPORT_INITIAL_BACKOFF_MS` | `500` | `5000` | Backoff before the first retry (at most 30000); doubled for each later one, at most 30 s. |
| `VLLM_PROXY_USAGE_REPORT_DEADLINE_SECS` | `0` (none) | `300` | How long after its request completed a report may still be sent. Above the timeout: the difference is the time a report has to wait and to back off. |
| `VLLM_PROXY_USAGE_REPORT_MAX_IN_FLIGHT` | `0` (no cap) | `8` | Reports in flight at once, for the whole process (at most 1000). |
| `VLLM_PROXY_USAGE_REPORT_MAX_QUEUED` | `10000` | default | Reports that may wait for a place when the cap is reached (1 to 100000). Unused without a cap. |
| `VLLM_PROXY_USAGE_REPORT_SHUTDOWN_DRAIN_SECS` | `0` (no wait) | below the service's stop timeout | How long shutdown waits for pending reports once every open request has ended (at most 600). |

A value that cannot work fails startup. A report only gets more than the one
5 s attempt it has always had under a cap, so whatever is configured, what the
process holds while cloud-api is down stays bounded.

Why the lane values. cloud-api drains about 17 reports a second however many
are in flight, so a cap of 8 costs no throughput. A wave of 200 reports still
clears in about 12 s (200 / 17), each request waits about half a second for
its write (8 / 17), far inside the 30 s timeout, and this lane never holds
more than 8 of cloud-api's connections, so it cannot starve the pool.

The backoff is 5000 ms because a failing report runs out of attempts long
before it runs out of deadline. When cloud-api fails fast (a refused
connection, an immediate 503) an attempt takes no time, and only the backoff
spreads the attempts out. With the default 500 ms the five attempts are used
up 4 to 8 s after the first one, with minutes of the deadline still unused,
and during an outage every place gives up a report that often. With 5000 ms
the same five attempts span 33 to 65 s (2.5 to 5 s, 5 to 10 s, 10 to 20 s and
15 to 30 s between them), which covers a restart of cloud-api of about half a
minute; the reports behind them wait in the queue, for up to 270 s. An outage
longer than that still costs the reports whose attempts run out during it
(`attempts_exhausted`). More attempts buy more time: ten of them span about 2
to 3.5 minutes. Past the deadline a report is dropped rather than sent late
for ever.

**Retries.** A timeout, a connection or transport error, a 5xx and a 429 are
retried. Any other 4xx, and any other answer that is not a success, is final:
cloud-api refused the report itself, and the same bytes would get the same
answer. A retry sends the body of the first attempt byte for byte, with the
same `x-request-id`, and cloud-api deduplicates on the completion id, so a
report whose timed-out attempt was written after all is not billed twice. A
report is sent by one task, one attempt at a time, never two at once. The
backoff after attempt `n` is `INITIAL_BACKOFF_MS × 2^(n-1)`, at most 30 s: half
of it fixed, the other half random, so the reports that failed together do not
come back together. `Retry-After` is not read.

**Cap and queue.** At most `MAX_IN_FLIGHT` reports hold a place at once. A
report keeps its place while it backs off, so a struggling cloud-api gets
fewer requests, not the same number a little later. The others wait in memory,
oldest first, at most `MAX_QUEUED` of them. When the queue is full, a new
report takes the place of the one that has waited longest: that one is
dropped, logged with its ids and counted as `queue_full`. The oldest goes
because it has the least time left before its deadline. The process never
holds more than `MAX_IN_FLIGHT + MAX_QUEUED` reports, a kilobyte or two each,
however long cloud-api is down.

**Deadline.** Counted from the completion of the request. An attempt is only
started while its whole timeout still fits before the deadline, so no attempt
is cut short. A report that can no longer get one, because it waited too long
for a place or because its next backoff would end too late, is dropped,
logged and counted as `deadline`. Cutting attempts short instead would, with
a backlog as old as the deadline, send every report with almost no time left
and get none of them accepted. With the lane values a report can wait up to
270 s for a place.

**Shutdown.** On SIGTERM the server finishes the open requests as before, then
waits up to `SHUTDOWN_DRAIN_SECS` for the reports still queued or in flight
and logs `Usage reports drained before shutdown` or `Usage reports left
undelivered at shutdown` with how many were pending and how many were left
(the count, not the ids of the reports). The queue is in memory only: what is
left at exit, and whatever a process held when it was killed, is lost, unless
the reports are kept in [the outbox on disk](#the-outbox-on-disk).

The wait starts only when every open request has ended. One stream that is
still open keeps the server serving, and when the service's stop timeout runs
out first the process is killed before any drain and without either line. So
the setting covers a process that has no open stream left, and it is not what
makes a planned retirement safe. When a process is taken out of service on
purpose (a blue/green switch), stop it later than `DEADLINE_SECS` plus one
`TIMEOUT_SECS` after traffic moved away from it, 330 s with the lane values,
and count that from the end of its last stream if one was still open then. By
that time every report it took has been accepted or dropped, and its queue is
empty whatever the drain does.

**Metrics.** `inference_proxy_usage_reports_total{outcome}` is still the final
outcome of a report, one count per report: with retries the outcome of its
last attempt, plus `queue_full` and `deadline_exceeded` for a report dropped
before cloud-api answered it. `inference_proxy_usage_report_duration_seconds`
is the duration of that last attempt (a report dropped before any attempt has
none). A process whose settings differ from the defaults also has:

- `inference_proxy_usage_report_attempts_total{outcome}`: every attempt, by
  what it met;
- `inference_proxy_usage_report_retries_total{reason}`: retries by cause
  (`timeout`, `connect_error`, `transport_error`, `http_5xx`, `http_429`);
- `inference_proxy_usage_reports_dropped_total{reason}`: reports given up on,
  that is usage not billed: `queue_full`, `deadline`, `attempts_exhausted`,
  `rejected` (a final answer that is not a success);
- `inference_proxy_usage_report_queue_depth` and
  `inference_proxy_usage_reports_in_flight`: reports waiting for a place and
  reports holding one;
- `inference_proxy_usage_report_time_to_accepted_seconds`: from the completion
  of the request to cloud-api's acceptance, waiting, backoff and every attempt
  included.

The counters and the histogram carry `auth_path` and `ingress_route`, like the
two series that were there before. A process with none of the settings, or
with every one at its default, has none of these series and none of the extra
log fields (`attempts`, `since_completion_ms`): its `/metrics` and its log
lines are unchanged. Log
lines carry ids only (request, organization, workspace, key, model), never a
report's content.

### The outbox on disk

Everything above keeps a report in memory. A report dropped from a full queue
or at its deadline, one whose attempts ran out while cloud-api was down, and
whatever a process held when it was killed are usage that was served and is
never billed. `VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH` names a SQLite file to keep
reports in instead (`src/usage_outbox.rs`). A report is then written to the
file first and sent from it, and it stays there until cloud-api accepts it or
refuses it for good: across a restart, a crash, and an outage of any length.

It is for a gateway; a CVM proxy never sets it. Without the variable there is
no file, no extra series, no extra log line and no change to any request.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_PROXY_USAGE_REPORT_OUTBOX_PATH` | unset (no outbox) | The SQLite file. It is created, with mode `0600`, when it is not there. Its directory must exist, be writable by the proxy and outlive the process and its container (a mounted volume). SQLite keeps two more files beside it, `-wal` and `-shm`, with the same mode. |
| `VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_PENDING` | `1000000` | Reports the file may hold while they wait to be accepted (1 to 10000000; a row is about half a kilobyte). Past it the report that has waited longest is dropped, logged with its ids and counted as `queue_full`. |
| `VLLM_PROXY_USAGE_REPORT_OUTBOX_MAX_REJECTED` | `100000` | Rows kept in the `rejected` table (1 to 1000000, about half a kilobyte each). Past it the oldest is removed and counted. |

**What the other settings mean with an outbox.** The settings of the table
further up still apply, and what is not set starts from different values:

- `MAX_ATTEMPTS` not set: no cap, a report is sent until it is accepted. Set
  (1 to 10), it is the cap it always was, and a report that uses it up moves
  to the `rejected` table instead of being dropped.
- `MAX_IN_FLIGHT` not set, or `0`: `8`. An outbox is always delivered under a
  cap. "No cap" would send the whole backlog of an outage at once when
  cloud-api came back, so it is not offered, and the rule that a longer
  timeout or retries need a cap is met without setting one.
- `DEADLINE_SECS` not set: none, as always. Set, it is the maximum age of a
  report, counted from the completion of its request whichever process served
  it. A report past it is dropped, logged and counted as `deadline`.
- `TIMEOUT_SECS` keeps its default of 5 s. A lane whose reports wait for
  cloud-api's write wants the 30 s of the table further up here too: a report
  is sent again for as long as it takes, but an attempt cloud-api cannot
  answer in time is never accepted however often it is made.
- `INITIAL_BACKOFF_MS` is the backoff of a report and of a place, see below.
  `0` cannot work and is replaced by the default.
- `MAX_QUEUED` bounds what waits in memory: the reports not written to the
  file yet, and the queue reports fall back to while the file cannot be used.
- `SHUTDOWN_DRAIN_SECS` is the time the attempts in flight get at shutdown.

The effective values are logged at startup (`Usage report outbox enabled`,
`Usage report delivery configured`). Nothing about the outbox fails startup: an
outbox size that cannot be read is replaced by its default with an error
line, and a path that cannot be used costs the durability of the reports and
nothing else (see "When the file cannot be used"). The delivery settings of
the table further up still fail startup on a value that cannot work, with or
without an outbox, as they did.

**Written first.** Handing a report over is still a lock and a push: the
request path never touches the file and never waits for it. One thread, which
owns the SQLite connection, writes what was handed over, and it gathers the
reports that complete within 50 ms of each other into one transaction (at most
2000 per transaction). So a report is in the file about 50 ms after its
request completed; later only while the other process on the file holds the
write lock or the disk is slow. Until then it waits in memory, at most
`MAX_QUEUED` of them, and a report beyond that is delivered from memory
without being kept. A report is not sent before it is written.

That is the durability window: **a report handed over less than 50 ms before
the process is killed (`kill -9`, an out-of-memory kill, a crash) may be
lost.** A process that is stopped in good order writes everything first.

The file is in WAL mode with `synchronous=NORMAL`. A commit is then one append
to the write-ahead log that is not flushed to the disk by itself. It survives
the process, whatever ends it, because the operating system has the bytes, and
the file cannot be corrupted by a crash of either. It does not necessarily
survive the host: after a host crash or a power loss, the commits since the
log was last flushed can be gone. The log is flushed at every checkpoint, and
the proxy runs one every 5 s while something was written, so that is a few
seconds of reports. `synchronous=FULL` would close that gap with a disk flush
on every commit. What the outbox is there for is restarts, deploys and
cloud-api outages, where `NORMAL` loses nothing, so the flush per commit was
not taken.

**What is in the file.** One row per report, in `pending`:

| Column | What it holds |
| --- | --- |
| `body` | The report as it is sent, the JSON of `POST /v1/internal/usage`: organization, workspace and key ids, model, token counts, completion id, discount. |
| `request_id` | The request id, sent again as `x-request-id` and logged. |
| `model_label`, `auth_path`, `ingress_route` | The labels of the report's metric series and log lines. |
| `completed_at_ms` | When its request completed (Unix time, ms). |
| `attempts`, `last_outcome` | Attempts made so far, by any process, and why the last one failed (`timeout`, `connect_error`, `transport_error`, `http_5xx`, `http_429`, `http_401`, `http_403`). |
| `next_attempt_at_ms` | When it is due (again). |
| `lease_until_ms`, `lease_owner` | While a process is sending it: until when it is that process's, and which process. |

The bearer token is never written: it is read from `CLOUD_API_USAGE_TOKEN`
when a report is sent, and so is the URL, from `CLOUD_API_URL`. A token that
was rotated, or a cloud-api that moved, applies to the whole backlog at the
next start. Nothing of a request or a response is in the file either. What is
in it is who used how many tokens of which model, which is billing data:
hence mode `0600`, and it belongs on a volume only the gateway reads.

**Delivery.** At most `MAX_IN_FLIGHT` places send at once. A free place takes
the report that has been due longest, by marking it with a lease: the process
holding it and a time, two attempt timeouts and 30 s ahead, until which nobody
else takes it. Then, on the answer:

- accepted: the row is deleted;
- a failure that can pass (timeout, connection or transport error, 5xx, 429,
  and here also 401 and 403, see below): `attempts` goes up, the report is
  due again after the backoff (`INITIAL_BACKOFF_MS × 2^(attempts-1)`, at most
  30 s, half of it random) and its lease is given back;
- any other answer: the row moves to `rejected` (next paragraph) and the place
  goes on with the next report. A report cloud-api will never take holds
  nothing up.

After a failure that can pass, the place that met it also stays taken for a
while before it takes another report: the backoff of the number of such
failures in a row across the process, so it doubles while nothing gets
through, up to 30 s, and starts over with the first report that gets an
answer of its own. A cloud-api that is down therefore gets `MAX_IN_FLIGHT`
attempts every half minute, not the backlog in a loop, and each of them is on
a different report, so an outage uses up nobody's attempts. When it is back
the places resume as their pauses end, within 30 s.

**Reports cloud-api refuses.** A 401 or a 403 is not a refusal of a report:
it says that cloud-api does not accept this gateway (a usage token that is
wrong, or was rotated under a running process), and every report gets it
until that is put right. With an outbox those reports wait like any other
that failed, counted as retries with reason `http_401` or `http_403`, and
are sent once the token is right; nothing has to be put back by hand. (From
memory both still end a report, as they always did.)

Any other 4xx except 429, and any other answer that is neither a success nor
a failure that can pass, means cloud-api refused the report itself (an
unknown model, a discount above its maximum, a body it cannot read). The row
moves to the `rejected` table with `reason`
(`rejected`, or `attempts_exhausted` under an explicit `MAX_ATTEMPTS`), the
HTTP `status`, the number of `attempts`, the time and everything the row
held. It is counted once, as the outcome it ended with and as dropped, and it
is logged with its ids. Nothing sends it again by itself. Look at the table
with the `sqlite3` CLI, on the host that has the volume (the image has none):

```sql
-- sqlite3 /path/to/usage-outbox.db
-- the backlog
SELECT COUNT(*) FROM pending;
-- the reports that were refused, newest first
SELECT id, datetime(rejected_at_ms / 1000, 'unixepoch') AS rejected_at,
       reason, status, attempts,
       json_extract(body, '$.id') AS completion,
       json_extract(body, '$.organization_id') AS organization,
       json_extract(body, '$.model') AS model
FROM rejected ORDER BY id DESC LIMIT 20;
```

Once what was wrong is fixed on cloud-api's side, a report is sent again by
moving its row back (here the one with `id` 42; drop the `WHERE` clauses to
move them all):

```sql
BEGIN IMMEDIATE;
INSERT INTO pending (body, request_id, model_label, auth_path, ingress_route, completed_at_ms)
  SELECT body, request_id, model_label, auth_path, ingress_route, completed_at_ms
  FROM rejected WHERE id = 42;
DELETE FROM rejected WHERE id = 42;
COMMIT;
```

This is safe while the proxy runs: it finds the row within two seconds and
sends it like any other, and cloud-api deduplicates on the completion id.
With `DEADLINE_SECS` set, a row older than the deadline is dropped again;
give it a later `completed_at_ms` in that case. Do not leave a transaction
open in the CLI: the proxy waits 5 s for the write lock and then treats the
file as unusable until it gets it.

**Two processes on one file.** A blue/green switch overlaps the old process
and the new one, and both may have the file open: WAL mode lets one write
while the other reads, and a write waits up to 5 s for the other's lock. The
leases keep them apart. A report is sent by whichever process holds its lease,
and a report written by one may well be sent by the other. A process that
stops gives its leases back; one that dies leaves them to run out, after
which its reports are anybody's. So the reports a killed process was sending
or about to send, at most twice `MAX_IN_FLIGHT` of them, wait for two attempt
timeouts and 30 s (40 s by default, 90 s with a 30 s timeout) before the next
process sends them. A report can be sent twice when a lease runs out under an
attempt that is still in flight, or when a process is stopped without a drain
while it is sending; cloud-api deduplicates those. Nothing is lost either way.

Run the two processes of a switch with the same file. A file of its own for
each would leave the backlog of the old one where no process reads it.

**Shutdown and restart.** On SIGTERM, once every open request has ended, the
reports still in memory are written to the file, the attempts in flight get
`SHUTDOWN_DRAIN_SECS`, and the leases of the process are given back, so the
next process can send what is left at once. The process logs `Usage report
outbox closed for shutdown` with how many reports the file holds. Nothing has
to be delivered before it exits, so with an outbox a process can be retired
without waiting out its queue (the rule of "Shutdown" above). A new process
creates the file or brings its schema up to date (`PRAGMA user_version`),
reads the backlog and delivers it, oldest first, at the pace of
`MAX_IN_FLIGHT`. A file written by a newer version of the proxy than the one
that opens it is refused and left alone (a rollback): reports are then
delivered from memory, and the newer version finds its backlog when it is
back.

**When the file cannot be used.** The outbox never keeps the gateway from
starting or serving, and never slows a request. If the file cannot be opened
at startup (a directory that is not there, permissions, a file that is not a
database, a file from a newer version), or a transaction fails later (a full
disk, an I/O error, a write lock that is not free within 5 s), then:

- the reports of the failed transaction, and every report after it, are
  delivered from memory, as without an outbox and under the same settings.
  They are sent; they are just not kept across a restart;
- `inference_proxy_usage_report_outbox_available` reads 0,
  `inference_proxy_usage_report_outbox_errors_total{op}` goes up and
  `inference_proxy_usage_report_outbox_bypassed_total{reason}` counts the
  reports that went around the file;
- the log says `Usage report outbox unavailable`, once, at error level, with
  the step that failed and the reason;
- opening is tried again every 5 s. When it works the log says `Usage report
  outbox available again`, the reports still waiting in memory are written to
  the file after all, and what the file already held is delivered.

The outcome of an attempt that could not be written (an accepted report whose
row is still there) is written when the file is back. If the process ends
before that, the report is sent once more by the next one.

With the path set and `CLOUD_API_URL` or `CLOUD_API_USAGE_TOKEN` missing,
nothing can be reported, so the file is not opened (an error line says so and
the gauge reads 0).

**Metrics.** A process with an outbox has every series of "Metrics" above,
whatever its other settings, and they keep their meaning:

- `inference_proxy_usage_reports_total{outcome}` counts a report once, at its
  final outcome, in the process that got it: `accepted`, the answer it was
  refused with, `queue_full` for a report dropped at `MAX_PENDING`,
  `deadline_exceeded` under a deadline. A report that is being sent again is
  not counted yet.
- `inference_proxy_usage_report_attempts_total`, `..._retries_total` and
  `inference_proxy_usage_reports_dropped_total{reason}` as above. `rejected`
  and `attempts_exhausted` are the rows that moved to the `rejected` table.
- `inference_proxy_usage_report_time_to_accepted_seconds` counts from the
  completion of the request as the file has it, so it is right for a report
  that an earlier process served.
- `inference_proxy_usage_report_queue_depth` is what waits: the rows of the
  file nobody in this process is sending, and what waits in memory.
  `inference_proxy_usage_reports_in_flight` is the places taken.

And these, which exist only with an outbox:

- `inference_proxy_usage_report_outbox_available`: 1 while the last thing
  tried on the file worked;
- `inference_proxy_usage_report_outbox_pending`: rows of `pending`, waiting or
  being sent (with `model` in list mode);
- `inference_proxy_usage_report_outbox_oldest_pending_age_seconds`: how long
  ago the request of the oldest of them completed, 0 when there is none;
- `inference_proxy_usage_report_outbox_rejected`: rows of `rejected`;
- `inference_proxy_usage_report_outbox_rejected_evicted_total`: rows removed
  from `rejected` to keep it within `MAX_REJECTED`;
- `inference_proxy_usage_report_outbox_errors_total{op}`: failures of the
  file, by the step that failed (`open`, `begin`, `insert`, `settle`, `evict`,
  `claim`, `release`, `stats`, `commit`, `checkpoint`);
- `inference_proxy_usage_report_outbox_bypassed_total{reason}`: reports
  delivered from memory because the file could not take them (`unavailable`,
  `write_failed`, `buffer_full`).

The gauges describe the file, not the process: two processes on one file each
report the same backlog, so take the maximum over them, not the sum. Worth an
alert: `available` at 0, the age of the oldest report above what an outage of
cloud-api is allowed to last, and anything in `rejected`.

## Several models in one gateway

A gateway is model-scoped: `MODEL_NAME` is its billing key, it has one backend
pool and one admission budget, and the `model` in a request body is not read.
`VLLM_PROXY_MODEL_LIST_FILE` makes the same process serve a list of models
instead (`src/model_list.rs`), picking the model from the body.

Without the variable nothing changes. That is every CVM proxy and the
single-model gateway described above: the same routes, responses, metric
series and labels, and log lines.

### What is per model and what is per process

Each model of the list has its own bundle, and nothing in it is shared with
another model:

- the backend pool and its health checker;
- conversation affinity;
- the admission controller: budget and ramp, per-host share, the
  time-to-first-token breaker, back-pressure marks, the queue refusal;
- the engine-load poller;
- the long-context tier;
- the HTTP client that carries the model's backend bearer and priority header.

So a model at its budget, with its breaker tripped or with every backend down
refuses and slows nothing for another model, and a model's backend token is
only ever sent to that model's backends: not to another model's, and not to
cloud-api.

One per process, as before: the key check against cloud-api, the organization
allowlist, the usage-report endpoint and token and the delivery of the reports
(`VLLM_PROXY_USAGE_REPORT_*`: one cap, one queue and one outbox for all models,
since they report to the same cloud-api), the content policy, the models
document source,
the stream timings (keep-alive, commit window, error peek, first-token
deadline) and the health-check timings. The remaining gateway
switches are process-level too and apply to every model:
`VLLM_BACKEND_CONVERSATION_AFFINITY`, `VLLM_BACKEND_CONNECT_FAILOVER`,
`VLLM_PROXY_MAP_QUEUE_FULL_TO_429`, `VLLM_BACKEND_HEALTH_PATH`,
`VLLM_BACKEND_PROBE_INTERVAL_SECS`, and the admission ramp
(`_RAMP_STEP`, `_RAMP_INTERVAL_SECS`), `_TTFT_P95_MAX_MS`,
`_BACKPRESSURE_SECS` and `_RETRY_AFTER_SECS`.

### The list file

A JSON file rendered by deploy tooling: `{"models": [ … ]}`, one object per
model. It holds no secrets: a model's backend token is referenced by the *name*
of the environment variable that carries it. Every key except `id` and
`backend_urls` is optional and falls back to the process-level variable, so
the variables act as defaults for the whole list (`reasoning_effort_map` and
`merge_system_messages` have no variable: a model has either only when its
entry writes it). A key that is not in the tables below is refused, so a typo
cannot silently become a default.

| Key | When omitted | Meaning |
| --- | --- | --- |
| `id` | required | The exact cloud-api model name. Requests select the model by it and usage is billed under it. |
| `backend_urls` | required | The model's backends, as `VLLM_BACKEND_URLS` lists them. |
| `backend_probe_urls` | no engine view | One engine-load probe per backend, same order, as `VLLM_BACKEND_PROBE_URLS`. |
| `long_context` | no tier | The model's long-context tier, see the next table. |
| `admission_max_inflight` | `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` | The model's in-flight budget; `0` = no admission for it. |
| `admission_start_inflight` | `VLLM_PROXY_ADMISSION_START_INFLIGHT` when that is set, else the model's own maximum | Budget at start-up. |
| `admission_queue_saturated_at` | `VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT` | Engine queue depth at which a backend counts as saturated. |
| `capacity_requests_per_minute` | `VLLM_PROXY_CAPACITY_REQUESTS_PER_MINUTE` | Requests per minute declared on the model's `/v1/models` entry. |
| `discount_to_user` | `VLLM_PROXY_DISCOUNT_TO_USER` | A JSON number under the same rules as the variable; `0` = list price. Published on the model's entry and sent on its usage reports. |
| `reasoning_off_effort` | `VLLM_PROXY_REASONING_OFF_EFFORT` | What "no reasoning" means for this model. |
| `reasoning_effort_map` | no mapping | An object of effort to effort, for example `{"high": "xhigh"}`: a request's reasoning effort that is one of the keys is sent to this model's engine as the value. For the model of the entry only, with no process-level variable. See [Reasoning efforts a model does not accept](#reasoning-efforts-a-model-does-not-accept). |
| `merge_system_messages` | `false` | `true` for a model whose chat template refuses a `system` message that is not the first message: a request with one anywhere else is sent to this model's engine with a single `system` message, first, holding the text of all of them. For the model of the entry only, with no process-level variable. See [System messages a model takes only first](#system-messages-a-model-takes-only-first). |
| `backend_token_env` | the value of `VLLM_BACKEND_TOKEN` | Name of the variable that holds this model's backend bearer: upper-case letters, digits and underscores, starting with `VLLM_BACKEND_TOKEN` (for example `VLLM_BACKEND_TOKEN_ALPHA`), and set. |
| `backend_priority` | `VLLM_BACKEND_PRIORITY` | Sent as `X-NearAI-Priority` to this model's backends. |

`long_context`, for a model that has a tier (the tier variables are defaults
for such a block; a model without one has no tier, whatever they say):

| Key | When omitted | Meaning |
| --- | --- | --- |
| `backend_urls` | required | The tier's backends, as `VLLM_BACKEND_LONG_CONTEXT_URLS`. |
| `backend_probe_urls` | none | One probe per tier backend; required exactly when the model has `backend_probe_urls`. |
| `above_tokens` | `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS` | Estimated input tokens above which a request goes to the tier. |
| `strict` | `VLLM_BACKEND_TIER_STRICT` | Refuse instead of placing on the other tier. |
| `borrowing` | `VLLM_PROXY_ADMISSION_TIER_BORROWING` | Base hosts may use the idle shared budget. |
| `max_inflight_per_host` | `VLLM_PROXY_ADMISSION_LONG_MAX_INFLIGHT_PER_HOST` | Long-host ceiling while borrowing. |
| `reserved_inflight` | `VLLM_PROXY_ADMISSION_LONG_RESERVED_INFLIGHT` | Budget slots kept for the tier. |

Three models, one with a long-context tier and two plain ones:

```json
{
  "models": [
    {
      "id": "example/alpha",
      "backend_urls": [
        "https://alpha-b1.inference.example",
        "https://alpha-b2.inference.example",
        "https://alpha-b3.inference.example"
      ],
      "backend_probe_urls": [
        "http://alpha-1.internal.example:8000",
        "http://alpha-2.internal.example:8000",
        "http://alpha-3.internal.example:8000"
      ],
      "long_context": {
        "backend_urls": ["https://alpha-long-b4.inference.example"],
        "backend_probe_urls": ["http://alpha-4.internal.example:8000"],
        "above_tokens": 100000,
        "strict": true,
        "borrowing": true,
        "max_inflight_per_host": 12,
        "reserved_inflight": 12
      },
      "admission_max_inflight": 48,
      "admission_start_inflight": 48,
      "admission_queue_saturated_at": 4,
      "capacity_requests_per_minute": 150,
      "discount_to_user": 0.3,
      "reasoning_off_effort": "low",
      "reasoning_effort_map": {"high": "xhigh"},
      "merge_system_messages": true,
      "backend_token_env": "VLLM_BACKEND_TOKEN_ALPHA",
      "backend_priority": -1
    },
    {
      "id": "example/beta",
      "backend_urls": [
        "https://beta-b1.inference.example",
        "https://beta-b2.inference.example"
      ],
      "admission_max_inflight": 16,
      "backend_token_env": "VLLM_BACKEND_TOKEN_BETA"
    },
    {
      "id": "example/gamma",
      "backend_urls": ["https://gamma-b1.inference.example"]
    }
  ]
}
```

`example/gamma` is a complete entry: everything it leaves out comes from the
process-level variables, or from their built-in defaults when those are unset
too.

The rest of the environment in list mode:

- `MODEL_NAME`, `VLLM_BACKEND_URLS`, `VLLM_BACKEND_PROBE_URLS`,
  `VLLM_BACKEND_LONG_CONTEXT_URLS` and `VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS`
  must not be set: they name the one model of a single-model process and its
  backends, for which a list entry has no fallback. `VLLM_BASE_URL` is not
  read.
- `NON_TEE_DEPLOYMENT=1` and `VLLM_PROXY_MODELS_DOCUMENT_URL` are required.
- `FUSION_ENABLED`, `WEB_CONTEXT_SEARCH_URL`, `OHTTP_ENABLED`,
  `VLLM_DATA_PARALLEL_SIZE`, `OPENAI_CHAT_COMPATIBILITY_CHECK` and
  `REPLICA_STATE_REDIS_URL` cannot be combined with a list: each assumes one
  model per process.
- A model that ends up with a backend token, its own or the process-level one,
  requires `CLOUD_API_URL` and `CLOUD_API_USAGE_TOKEN`, as `VLLM_BACKEND_TOKEN`
  does.
- The image-validation defaults that depend on the model name do not apply;
  set the `VLLM_PROXY_IMAGE_VALIDATION_*` variables explicitly where needed.

Anything wrong fails startup rather than serving a partial list: a file that
cannot be read or is not valid, an empty list, an id listed twice, a model
without backends, a URL that is not a plain `http(s)` base URL (no query, no
fragment, no credentials), a probe count that differs from the backend count,
a `backend_token_env` that is not set, a `reasoning_effort_map` that breaks
one of its rules below, a `merge_system_messages` that is not `true` or
`false`, a backend or probe listed under two
models (a backend serves one model), and every rule the variables are held to
for one model — the admission and long-context tier rules above. Those are
checked on what each entry resolves to, and their messages name the variable a
key is named after, prefixed with the model.

Whether two URLs are the same backend is decided on where they point, not on
how they are written: scheme, host (case-insensitive, with or without a
trailing dot), effective port and normalised path. `https://HOST.example`,
`https://host.example:443` and `https://host.example/v1/..` are one backend,
under two models as well as in both tiers of one. Two names that resolve to
the same host are still two backends as far as this check can tell.

### Reasoning efforts a model does not accept

Which efforts a model takes is decided by its chat template, and some engines
answer `400` to any other. A template that knows `low`, `medium` and `xhigh`
refuses `high`, which callers of an aggregator send routinely, so a valid
request fails for that model only. `reasoning_effort_map` on the model's entry
names such efforts and what to send instead:

```json
"reasoning_effort_map": {"high": "xhigh"}
```

On `/v1/chat/completions`, after the handling of "no reasoning" described
[above](#configuration) (`reasoning_off_effort`), a string that is a key of
the map is replaced by its value in the two places an engine reads an effort
from:

- the top-level `reasoning_effort`, whether the caller sent it or the gateway
  copied it there from the `reasoning` object;
- `reasoning.effort`, when the body has a `reasoning` object with one.

Each of the two is looked up on its own, so an effort the map names reaches
the engine in neither field, whichever one the engine reads first. A match is
exact and case-sensitive. Everything else is forwarded as before: an effort
that is not a key, a value that is not a string, the rest of the `reasoning`
object, and a body without an effort, to which none is added. Only these two
fields are read: an effort a caller passes to the template by another route
(inside `chat_template_kwargs`, say) is not looked at, with a map as without.
`/v1/completions` is not touched, like the rest of the reasoning handling, and
neither is another model of the same process.

"Off" stays with `reasoning_off_effort`. `reasoning.enabled: false` and the
efforts `none` and `minimal` become the model's off effort first, and the map
is not allowed to interfere with that in either direction. These are refused at
startup, with the model named:

| Refused | Why |
| --- | --- |
| A key or a value that is not 1 to 32 characters of `a-z`, `0-9`, `_` and `-`; an empty one; a value that is not a string; the same key twice; more than 16 entries | The file is rendered, and a value is sent to the engine and is a metric label. |
| A value that is also a key (`{"high": "xhigh", "xhigh": "max"}`, or `{"high": "high"}`) | Every effort is looked up once. With a chain the result would depend on an order. |
| `none` or `minimal` as a key | Those mean "off", which is `reasoning_off_effort`. |
| The model's `reasoning_off_effort` as a key (`{"low": "medium"}` on a model whose off effort is `low`) | The off effort is sent as configured. Mapping it would make "off" something else than that key says. |
| `none` or `minimal` as a value, unless it is the model's `reasoning_off_effort` | The off handling replaces those values with the model's off effort, and the map runs after it: it must not write one back. |
| A map on a model without a backend token (`backend_token_env` or `VLLM_BACKEND_TOKEN`) | The reasoning handling runs for a model that has one, and the map is part of it. Without the token the map would never be applied. |

The off effort these rules use is the one the model ends up with: its own
`reasoning_off_effort`, or the process-level variable when the entry leaves it
out. An empty object is a model without a map.

A request the map changed is counted once in
`inference_proxy_model_reasoning_effort_mapped_total{effort, model}`, where
`effort` is the value that was sent (a value of the map; the `reasoning`
object's when the two fields were mapped to different ones) and `model` the
configured id. What the caller sent is not a label, and nothing is logged per
request. The startup line of each model prints its map.

### System messages a model takes only first

Some chat templates accept a `system` message only as the first message of
the conversation, and the engine answers `400` ("System message must be at
the beginning.") to anything else: two system messages in a row, a second one
later in the history, or one that follows a user message. Agent frameworks
send all of these and other providers accept them, so a valid request fails
for that model only. `merge_system_messages` on the model's entry makes the
gateway put such a request in the shape the template takes:

```json
"merge_system_messages": true
```

On `/v1/chat/completions`, a request whose `messages` hold more than one
message with the role `system`, or a single one that is not the first
message, is sent to the engine with exactly one `system` message, first:

- its content is one string: the text of every system message, in the order
  they were sent, with a blank line between two of them. An empty content
  adds nothing, not even the blank line;
- every other message keeps its place among the others, and is not changed;
- only the role `system`, spelled exactly so, is read as a system message. A
  `developer` message is like any other: never merged, and after the merged
  system message when one is moved in front of it.

| Sent | Reaches the engine as |
| --- | --- |
| `system` `"A"`, `system` `"B"`, `user` | `system` `"A\n\nB"`, `user` |
| `system` `"A"`, `user`, `system` `"B"` | `system` `"A\n\nB"`, `user` |
| `user`, `system` `"B"` | `system` `"B"`, `user` |

A system message's content may be a string or an array of text parts
(`{"type": "text", "text": "…"}`); the parts of one message are concatenated
in order, with nothing between them. The gateway never drops part of a
request to make it fit. When it cannot keep everything, it does not rewrite
at all: the request goes out as the caller sent it, and the engine answers.
That is the case when

- a system message has a part that is not a text part (an image, any other
  type, a text part without a string `text`), or a content that is neither a
  string nor an array (`null`, or none at all);
- a text part has a key other than `type` and `text` (a `cache_control`
  breakpoint, for instance). A string cannot carry it, and whether the engine
  reads it is not this gateway's to judge;
- two system messages give a field other than `role` and `content` different
  values (two different `name`s, say). Such a field is otherwise kept on the
  merged message, whichever system message carried it, and then covers the
  whole of it.

A request the template already takes, with no system message or with one as
its first message, is not looked at further and goes out byte for byte as it
would without the key, whatever that first message holds. So do
`/v1/completions`, which has no messages, and every request of another model
of the same process.

The merge runs at a fixed place among the things a request goes through:

1. the model is selected, and the reasoning handling (`reasoning_off_effort`,
   `reasoning_effort_map`) is applied. It reads the top-level `reasoning` and
   `reasoning_effort` only, never the messages;
2. an encrypted request is decrypted; then the tool-call `arguments` and
   `response_format` repairs and the content part policy
   (`VLLM_PROXY_REJECTED_CONTENT_PART_TYPES`), on the messages as the caller
   sent them;
3. the merge;
4. the input estimate (long-context tier, first-token deadline) and
   conversation affinity, on the messages as they are dispatched, and image
   validation, which is the same either way: a system message with an image
   is never merged.

A system message that a caller adds or changes late in a conversation ends up
in the first message, so that turn's prompt differs from the previous one's
inside its first message: the engine's prefix cache covers what comes before
the change and nothing after it, and the conversation may be placed on another
backend. That is the price of a template that takes system messages nowhere
else.

A request that was rewritten is counted in
`inference_proxy_model_system_messages_merged_total{model}`, where `model` is
the configured id, the only label. Requests that were left alone are not
counted, nothing about the messages is a label, and nothing is logged per
request. The startup line of each model prints its `merge_system_messages`.

### Requests

`/v1/chat/completions` and `/v1/completions` authenticate as before and then
read the body's `model`. An exact, case-sensitive match against the configured
ids selects that model's bundle: its pool, affinity, admission, engine view,
tier, backend token and priority, reasoning-off effort, effort map, system
message handling and discount. The body is forwarded as it would be for a
single model, `model` included; for a model with a `reasoning_effort_map`,
with the efforts it names replaced, and for a model with
`merge_system_messages`, with its system messages merged into the first
message.

Anything else — another model, a configured id in a different case, no `model`,
a `model` that is not a string — is OpenAI's `404`:

```json
{"error": {"message": "The model `example/delta` does not exist or you do not have access to it.",
           "type": "invalid_request_error", "param": null, "code": "model_not_found"}}
```

Without a `model` the message says that one is required; status, type and code
are the same. Nothing is dispatched, no budget slot is taken and nothing is
billed. The check comes after authentication, so a caller without a valid key
gets the same `401` whatever it names and cannot use the answer to enumerate
models.

Usage reports carry the selected model's `id` and its discount.

Only what is defined for several models is served: `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/healthz`, `/metrics` and `/version`. Every
other route of the binary (tokenize, embeddings, rerank, score, images, audio,
privacy, `/v1/metrics`, attestation, signatures, OHTTP, `/`) answers the `404`
of an undeclared route.

### `/v1/models`

One read of the models document per request. The entries of the configured
models are kept, in the document's order, each completed with its own model's
`capacity` (its `admission_max_inflight` and `capacity_requests_per_minute`)
and `discount_to_user`. A configured model the document does not list is left
out, so the catalog stays the switch for what is advertised. Every read it is
missing from is counted in `models_document_missing_models_total{model}`; the
log has one line when it drops out ("Configured model is not in the models
document") and one when it is back ("… is in the models document again"), not
one per read of a route that is polled.

When the document cannot be read the answer is `502` with error type
`models_document_unavailable`. There is no engine-list fallback in list mode
and never a partial list: no single engine speaks for the others, and a list
without the document would advertise models at prices nothing vouches for.

### `/healthz`

`200` while at least one model has a healthy backend, `503` when none has. One
model's outage does not take the process off its load balancer.

The route is unauthenticated, and which models are configured is not for
everyone: an unknown `model` is only answered after authentication so that the
ids cannot be enumerated. So the body has the single-model shape and nothing
else:

```json
{"status": "ok", "checks": {"dstack": "skipped", "backend": "ok"}}
```

`checks.backend` is `"ok"` exactly when the status is, `"unhealthy"` otherwise.

A caller that presents the gateway's own config `TOKEN` as its bearer (the
operator, deploy tooling) also gets one entry per model, in list order:

```json
{
  "status": "ok",
  "checks": {"dstack": "skipped", "backend": "ok"},
  "models": [
    {"id": "example/alpha", "backend": "ok"},
    {"id": "example/beta", "backend": "unreachable"},
    {"id": "example/gamma", "backend": "ok"}
  ]
}
```

Status code, `status` and `checks` are the same with and without the token; a
customer key is not the token and gets the short body. A model's `backend` is
the token the single-model probe would report (`ok`, `unreachable`, `timeout`,
`http_5xx`, …) for one backend picked from its pool, all models probed at once.
A model whose pool already has no healthy backend reads `"unhealthy"` without
being probed, so its dead hosts cannot make the answer slow for everyone.
Deploy tooling that waits for every model should send the token and check each
entry, not the status.

For that reason, and because a host taken out after a failed connect only comes
back through a probe, list mode runs the pool health checker for every model,
including one with a single backend (a single-model process starts it only for
two or more).

### Metrics and logs

In list mode the per-model series carry a `model` label, always the configured
id, as their last label:

- `admission_*` (budget, in-flight, rejections, TTFT, back-pressure, per-backend
  limits, selection failures);
- `backend_pool_size`, `backend_pool_healthy`, `backend_failover_total`,
  `backend_affinity_*`, `placement_hint_*`;
- `backend_engine_running`, `backend_engine_queued`,
  `backend_engine_probe_failures_total`;
- `backend_tier_requests_total`, `backend_tier_output_reserve_capped_total`,
  `request_estimated_prompt_tokens`;
- `first_token_deadline_refusals_total`;
- `inference_proxy_usage_reports_total`,
  `inference_proxy_usage_report_duration_seconds` and the series of
  [Usage report delivery](#usage-report-delivery)
  (`inference_proxy_usage_report_attempts_total`,
  `inference_proxy_usage_report_retries_total`,
  `inference_proxy_usage_reports_dropped_total`,
  `inference_proxy_usage_report_queue_depth`,
  `inference_proxy_usage_reports_in_flight`,
  `inference_proxy_usage_report_time_to_accepted_seconds`,
  `inference_proxy_usage_report_outbox_pending`,
  `inference_proxy_usage_report_outbox_bypassed_total`),
  `inference_proxy_completed_requests_total`,
  `inference_proxy_input_tokens_total`, `inference_proxy_input_tokens`,
  `inference_proxy_request_duration_seconds`.

`backend` in these series is an index into that model's own pool, so
`{backend="0"}` means a different host for each `model`. Everything else is
process-wide and unlabelled, as before: HTTP and error counters, the key
check, stream mechanics, content policy and repairs.

With one model there is no `model` label on any series: existing dashboards
and alerts match on exactly those label sets. `src/model_metrics.rs` is where
the difference lives.

Four series exist in list mode only. Two of them count responses:
`http_requests_total` and `http_errors_total` stay process-wide, so without
these an error rate could not be read per model:

- `inference_proxy_model_requests_total{endpoint, status, model}`: one per
  `/v1/chat/completions` or `/v1/completions` response of a request whose
  model was resolved. `endpoint` is the route and `status` the HTTP status the
  client was sent, whoever decided it: the engine (a `200`, or its own `4xx`
  or `5xx` as the gateway forwards it, so an engine `500` is a `500` here and
  not a `502`), the gateway refusing (`429` at the admission budget or on the
  first-token deadline, `503` for a strict tier with no healthy backend) or
  the gateway answering for an upstream that failed (`502` unreachable or cut
  short, `504` timed out). A connection fail-over is still one response.
- `inference_proxy_model_stream_errors_total{endpoint, model}`: one per
  stream that failed after its `200` was sent, which the series above can
  only show as a `200`. That is an engine error event on the open stream, a
  failure that arrives after the early commit (`VLLM_PROXY_STREAM_COMMIT_MS`),
  a cut or unreadable upstream connection, the idle watchdog
  (`VLLM_PROXY_STREAM_IDLE_TIMEOUT_SECS`) and a stream that ends without
  `[DONE]`.

A response is counted when its status goes out, the moment
`http_requests_total` counts it. A stream is therefore counted when its `200`
is committed, whatever happens to it afterwards, and a client that
disconnects before any status was sent is not counted at all. A client that
disconnects from an open stream is not a stream error either
(`stream_client_disconnects_total`, process-wide, has those).

A request without a resolved model is in neither series: an unknown or missing
`model` (the `404` above, counted by `request_model_match_total`) and
everything refused before the body's `model` is read (the key check's `401`,
`402`, `403` and `429`, the per-IP rate limit, a `413`, a body that is not a
JSON object).

The share of one model's responses that are server errors, and of its streams
that failed after their `200`:

```promql
sum by (model) (rate(inference_proxy_model_requests_total{status=~"5.."}[5m]))
  / sum by (model) (rate(inference_proxy_model_requests_total[5m]))

sum by (model) (rate(inference_proxy_model_stream_errors_total[5m]))
  / sum by (model) (rate(inference_proxy_model_requests_total{status="200"}[5m]))
```

(The second ratio's denominator also holds the non-streaming `200`s: the
status series does not tell streams apart.)

The third,
`inference_proxy_model_reasoning_effort_mapped_total{effort, model}`, exists
for a model with a `reasoning_effort_map`: one per request whose reasoning
effort the map replaced, under the effort that was sent instead (see
[Reasoning efforts a model does not accept](#reasoning-efforts-a-model-does-not-accept)).

The fourth, `inference_proxy_model_system_messages_merged_total{model}`,
exists for a model with `merge_system_messages`: one per request whose system
messages were merged into the first message (see
[System messages a model takes only first](#system-messages-a-model-takes-only-first)).

Startup logs one "Serving model" line per model with its effective settings
(the backend token only as `backend_token=true|false`) and a "Model list
enabled" line with the settings every lane shares, the first-token deadline
and connection fail-over included. The single-model startup lines about one
model's token, tier, admission, health checker and attestation are not
written. The lane log lines of
`admission.rs`, the first-token refusal and the engine probe carry a `model`
field in list mode and are unchanged without one.

### Before switching it on

Exact matching refuses requests that a single-model gateway serves today, since
that gateway does not look at `model` at all. To read the effect from
production first, a single-model gateway counts how each chat/completions
request's `model` compares with `MODEL_NAME`:

```text
request_model_match_total{result="exact" | "case_differs" | "other" | "missing"}
```

`missing` is an absent `model` or one that is not a string. Nothing else
changes: every request is served and billed as `MODEL_NAME`, as before. The
counter exists only in a gateway lane, a process with `NON_TEE_DEPLOYMENT=1`
and `VLLM_BACKEND_TOKEN`; a proxy inside a CVM does not emit it, with or
without a backend token. In list mode the same counter compares against the
configured ids, and everything but `exact` is the `404` above. The requested name itself is the caller's string: it is never a
metric label and never in a log line.

## What is deliberately not offered here

Attestation (`/v1/attestation/report`), response signatures
(`/v1/signature/{id}`) and GPU-evidence delegation (`/internal/gpu_evidence`)
are meaningful only inside the CVM; with `NON_TEE_DEPLOYMENT=1` they answer
404. Customers who need TEE guarantees keep using the per-model
`*.completions.near.ai` domains or the Cloud API.
