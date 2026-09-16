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
   exactly one component bills. The body is forwarded verbatim; for streams the
   gateway forces `stream_options.include_usage` and `continuous_usage_stats`.
5. Response streamed back. On client disconnect the upstream connection is
   dropped (the CVM proxy drops its engine connection, the engine aborts) and
   the usage observed so far is reported.
6. `POST {CLOUD_API_URL}/v1/internal/usage` with the shared usage token,
   idempotent on the provider completion id.

Nothing about request or response content is logged or stored at any hop.

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
| `VLLM_PROXY_REASONING_OFF_EFFORT` | `low` | What "no reasoning" means for GLM-5.3 Flash (see below). |
| `VLLM_PROXY_SSE_KEEPALIVE_SECS` | `15` | `: keep-alive` SSE comments while the upstream is silent (long prefill/queueing), so intermediaries with read timeouts do not cancel. Off in CVMs: comments are not part of the signed bytes. |
| `VLLM_PROXY_MAP_QUEUE_FULL_TO_429` | `1` | The engine's admission rejection (queue full, or a queued request displaced by a higher-priority one) becomes 429 with `Retry-After: 2` and type `overloaded`, the same shape as the gateway's own refusals: back-pressure, not an outage. Off in CVMs: cloud-api's peer fallback keys on the 503. |
| `VLLM_PROXY_STREAM_ERROR_PEEK_MS` | `1000` | Streams wait up to 1 s for the first upstream event; an admission-time `data: {"error":…}` becomes a real 429/5xx instead of a 200 that fails mid-stream. A slow first token just times the peek out. |
| `VLLM_PROXY_STREAM_COMMIT_MS` | `5000` | Commit the stream's `200 text/event-stream` after 5 s even if the engine has not answered, so the keep-alives above actually reach the client during a long prefill (see below). |
| `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` / `_START_INFLIGHT` | `48` / `32` | The lane's in-flight budget: refuse with 429 + `Retry-After` before dispatch instead of queueing (see below). Starts at 32 and ramps by 8 every 30 min while the lane stays healthy. |
| `VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS` | `30000` | Refuse new work while, over the last minute, at least 20 lane requests reached the engine and 5 % of them (at least two) waited longer than this for their first generation event. |
| `VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS` | `10` | A backend that rejected at engine admission within this window is steered around; when every healthy backend did, new work is refused. |
| `VLLM_BACKEND_CONNECT_FAILOVER` | `1` | A backend that refuses the connection (host down, proxy restarting) costs the request nothing: it is re-sent once to another healthy backend, the dead one leaves the rotation until a probe succeeds, and a pinned conversation follows. Never on an HTTP error. |
| `VLLM_BACKEND_PROBE_URLS` / `_INTERVAL_SECS` | `http://<host-ip>:8000,…` / `2` | The engines' live running/queued counts, read from each host's plain metrics port (the same route model-proxy samples; reachable from the model-proxy hosts, no token). One reading covers the replica the host would route to, so a queue in it means no replica is free. Drives placement and the fleet-wide queue refusal below. |
| `VLLM_BACKEND_LONG_CONTEXT_URLS` | the `-long-b<handle>` URLs | The hosts of the long-context tier, listed as their handle URLs under the model's `-long` model-proxy domain (see below). Appended to the pool after `VLLM_BACKEND_URLS`, so the base backends keep their indexes. Empty = one flat pool, as today. |
| `VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS` | `http://<host-ip>:8000,…` | One engine-load probe per long-context backend, same order. Required when `VLLM_BACKEND_PROBE_URLS` is set, and empty when it is not; internally the two lists are concatenated in pool order. |
| `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS` | `100000` | Estimated input tokens above which a request is placed on that tier. `0`/unset switches the whole feature off, and nothing is even estimated. |
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
Applied counts are in `reasoning_switch_applied_total{kind}`.

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

## Admission budget

Mapping the engine's rejection to 429 only helps once the engine's queue is
full; by then the lane's earlier requests are already waiting behind large
prefills and their time to first token is minutes. The gateway therefore
bounds the lane itself (`admission.rs`), in this order, before anything is sent
upstream:

1. **Observed overload.** The engines' own queues first: with
   `VLLM_BACKEND_PROBE_URLS` each host's running and queued request counts are
   polled every two seconds; a host with a non-empty queue is steered around,
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
3. **Per-host share and placement.** `ceil(budget / healthy backends)` lane
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
`backend_tier_requests_total{tier,outcome}`, `request_estimated_prompt_tokens`,
plus the existing usage-report and upstream metrics.

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
required  = ceil(countable × 1.2) + uncounted + max_tokens reserve
```

`required` strictly above `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS` means the
long tier. The 1.2 safety factor covers the byte estimate only — media parts,
template overhead, the reserved output window and `/v1/completions` token ids
are already token counts. Tool definitions and tool-call arguments are counted
because the lane's traffic is agentic, where they are most of the prompt.
cloud-api additionally refines the decision near the boundary with an exact
`POST /v1/tokenize`; the gateway deliberately does not — a tokenizer dependency
and an extra upstream round trip are not worth it for a placement that is a
preference rather than a correctness rule. Both tiers run the same engine with
the same context length, so a request on the "wrong" tier still succeeds.

Everything downstream of the decision is restricted to the request's tier:
placement, the connection fail-over, and the fleet-wide "every backend is
queueing" refusal. The consequences are deliberate:

- **Per-host share.** Unchanged: `ceil(budget / healthy backends)` over the
  whole pool. With 3 base hosts, 1 long host and a budget of 48 the long host
  holds at most 12 lane requests, which is about what its KV cache fits for
  200k-token prompts. The budget is the lever; there is no separate knob.
  Note that adding the tier also lowers each base host's share
  (`ceil(48/3) = 16` becomes `ceil(48/4) = 12`), so the base fleet's ceiling
  drops from 48 to 36 with the remaining 12 reserved for oversized prompts:
  raise `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` if it needs its old headroom.
- **Untiered traffic stays on the base fleet.** `/tokenize`, media, `/v1/models`
  and the health probe carry no size of their own; they would all land on the
  idle long host under least-connections, so they are restricted to the base
  backends (cloud-api keeps its tokenize traffic off that host for the same
  reason). The pool health checker still probes every backend.
- **Full is a refusal, not a spill.** A long-tier host at its share or steered
  around (engine queue, recent engine rejection) means `429` + `Retry-After`
  for the next oversized request — keeping those prefills off the base fleet is
  the whole point, and a fast refusal lets the aggregator route elsewhere.
- **A tier with no healthy backend falls back.** If the wanted tier is down
  entirely the request is placed in the other one rather than refused
  (`backend_tier_requests_total{outcome="fallback"}`), including when its last
  host goes unreachable mid-request: the connection fail-over re-resolves the
  restriction after taking that host out of the rotation, and crosses over.
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

## What is deliberately not offered here

Attestation (`/v1/attestation/report`), response signatures
(`/v1/signature/{id}`) and GPU-evidence delegation (`/internal/gpu_evidence`)
are meaningful only inside the CVM; with `NON_TEE_DEPLOYMENT=1` they answer
404. Customers who need TEE guarantees keep using the per-model
`*.completions.near.ai` domains or the Cloud API.
