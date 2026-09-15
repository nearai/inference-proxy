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
| `VLLM_PROXY_SSE_KEEPALIVE_SECS` | `15` | `: keep-alive` SSE comments while the upstream is silent (long prefill/queueing), so intermediaries with read timeouts do not cancel. Off in CVMs: comments are not part of the signed bytes. |
| `VLLM_PROXY_MAP_QUEUE_FULL_TO_429` | `1` | The engine's admission rejection (queue full, or a queued request displaced by a higher-priority one) becomes 429: back-pressure, not an outage. Off in CVMs: cloud-api's peer fallback keys on the 503. |
| `VLLM_PROXY_STREAM_ERROR_PEEK_MS` | `1000` | Streams wait up to 1 s for the first upstream event; an admission-time `data: {"error":…}` becomes a real 429/5xx instead of a 200 that fails mid-stream. A slow first token just times the peek out. |
| `NON_TEE_DEPLOYMENT` | `1` | No dstack socket outside a CVM: `/healthz` reports `"dstack":"skipped"`, no attestation refresh, and `/v1/attestation/report`, `/v1/signature/{id}`, `/internal/gpu_evidence` answer 404 so nothing unverifiable is advertised. |
| `DEV` / `GPU_NO_HW_MODE` | `1` / `1` | Non-TEE: random signing keys, no hardware evidence. |
| `LISTEN_ADDR` / `LISTEN_PORT` | `127.0.0.1` / `31700` | Bind behind the local TLS terminator. |
| `RATE_LIMIT_PER_SECOND` / `RATE_LIMIT_BURST_SIZE` | raised | Per-IP limiter; an aggregator arrives from a handful of IPs. |

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

`/metrics` exposes `backend_pool_size`, `backend_pool_healthy`,
`backend_affinity_*`, `rejected_content_parts_total{part_type}`,
`sse_keepalive_comments_total`, `upstream_stream_first_event_errors_total`,
`stream_client_disconnects_total`, plus the existing usage-report and upstream
metrics.

## What is deliberately not offered here

Attestation (`/v1/attestation/report`), response signatures
(`/v1/signature/{id}`) and GPU-evidence delegation (`/internal/gpu_evidence`)
are meaningful only inside the CVM; with `NON_TEE_DEPLOYMENT=1` they answer
404. Customers who need TEE guarantees keep using the per-model
`*.completions.near.ai` domains or the Cloud API.
