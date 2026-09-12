# Gateway mode: one inference-proxy in front of a model fleet

The same binary that runs inside every CVM (in front of one host's engines)
can run *outside* the TEE as a fleet-wide, model-scoped gateway. It is the
"direct completions" path — customer `sk-` keys validated against cloud-api,
usage reported to cloud-api, requests forwarded to the inference engines —
without cloud-api's inference pipeline in the request path. Nothing new is
introduced for authentication, billing, streaming, cancellation, or privacy:
those are the existing, deployed behaviors of this proxy. Gateway mode only
adds dynamic fleet membership and a few opt-in policies.

```text
client ──sk-key──▶ inference-proxy (gateway, non-TEE) ──PROXY_TOKEN──▶ model-proxy (L4/SNI)
                        │  check_api_key / internal usage                    │
                        ▼                                                    ▼
                    cloud-api                              CVM: nginx ▶ inference-proxy ▶ SGLang
```

## What the gateway does per request

1. `Authorization: Bearer sk-…` → `POST {CLOUD_API_URL}/v1/check_api_key`
   (retries on transport/5xx; 401/402/429 pass through). The response carries
   the authoritative organization / workspace / API-key identity.
2. Policy: `video_url` (or any configured content part type) → `400` before
   anything is fetched or dispatched; image inputs are validated as today.
3. Backend selection over the *discovered* fleet: least-connections with
   conversation affinity (`VLLM_BACKEND_CONVERSATION_AFFINITY=1`), so later
   turns of a long agent conversation stay on the host whose prefix cache
   already holds them.
4. Forward with `Authorization: Bearer $VLLM_BACKEND_TOKEN`. The CVM proxy
   treats this as its trusted config token: it does **not** re-validate the
   customer key and does **not** report usage, so exactly one component bills.
   The request body is forwarded verbatim (`reasoning_content` on prior
   assistant messages, tool history, etc. all reach the engine unchanged); for
   streams the gateway forces `stream_options.include_usage` and
   `continuous_usage_stats` so every chunk carries running token counts.
5. Response streamed back. On client disconnect the upstream connection is
   dropped (the CVM proxy drops its engine connection, and the engine aborts
   generation) and usage observed so far is reported — never the full
   generation.
6. `POST {CLOUD_API_URL}/v1/internal/usage` with the shared usage token and
   the identity from step 1, idempotent on the provider completion id.

Nothing about the request or response content is logged or stored at any hop:
this proxy logs request ids, tenant ids, model, token counts, and statuses only;
the signature cache holds hashes; the engines run payload-free logging.

## Configuration

Everything is an environment variable; the new ones are all opt-in and default
to the current in-CVM behavior.

| Variable | Gateway value | Purpose |
| --- | --- | --- |
| `MODEL_NAME` | `z-ai/glm-5.3-flash` | Must match cloud-api's model name exactly (usage rows are keyed on it). |
| `TOKEN` | a random secret | Trusted config token for *this* gateway (operators/tests). Customers use `sk-` keys. |
| `CLOUD_API_URL` / `CLOUD_API_USAGE_TOKEN` | prod values | Key validation and usage reporting. |
| `VLLM_BACKEND_DISCOVERY_URL` | `https://completions.near.ai/backends/list?domain=glm-5-3-flash.completions.near.ai` | model-proxy listing of registered backends (stable handles + health). |
| `VLLM_BACKEND_DISCOVERY_TOKEN` | model-proxy admin token | Bearer for the listing (handles are bearer capabilities, so the listing is auth-gated). |
| `VLLM_BACKEND_DISCOVERY_URL_TEMPLATE` | `https://glm-5-3-flash-b{handle}.completions.near.ai` | Per-backend base URL; `{handle}` is substituted. The SNI pins model-proxy to that one backend. |
| `VLLM_BACKEND_DISCOVERY_INTERVAL_SECS` | `5` | Poll interval. Errors/empty listings keep the last membership; the health checker removes dead backends. |
| `VLLM_BACKEND_TOKEN` | the CVM proxies' `PROXY_TOKEN` | Outbound bearer for backend requests only (dedicated HTTP client; never sent to cloud-api or the registry). |
| `VLLM_BACKEND_HEALTH_PATH` | `/healthz` | The CVM proxy's unauthenticated readiness route (dstack + engine). |
| `VLLM_BACKEND_CONVERSATION_AFFINITY` | `1` | Fleet-level conversation affinity (the CVM proxy still does its own across its two replicas). |
| `VLLM_PROXY_REJECTED_CONTENT_PART_TYPES` | `video_url,input_audio,file` | Modalities this deployment does not serve → deterministic `400`. |
| `VLLM_PROXY_CATCH_ALL_DISABLED` | `1` | Only the declared inference routes exist; `/v1/responses`, `/v1/conversations`, and anything undeclared is `404`. Stateless by construction. |
| `VLLM_PROXY_SSE_KEEPALIVE_SECS` | `15` | `: keep-alive` SSE comments while the upstream is silent (long prefill/queueing), so intermediaries with read timeouts do not cancel. Off in CVMs: comments are not part of the signed bytes. |
| `NON_TEE_DEPLOYMENT` | `1` | No dstack socket outside a CVM: `/healthz` reports `"dstack":"skipped"`, no attestation refresh, and the attestation, signature and GPU-evidence routes answer 404 so nothing unverifiable is advertised. |
| `VLLM_PROXY_MAP_QUEUE_FULL_TO_429` | `1` | The engine's admission rejection ("The request queue is full.", 503) becomes 429: back-pressure, not an outage. |
| `VLLM_PROXY_STREAM_ERROR_PEEK_MS` | `1000` | Streams wait up to 1 s for the first upstream event; an admission-time `data: {"error":…}` becomes a real 429/5xx instead of a 200 that fails mid-stream. Normal streams are unaffected (a slow first token just times the peek out). |
| `DEV` / `GPU_NO_HW_MODE` | `1` / `1` | Non-TEE: random signing keys, no hardware evidence. The gateway offers no attestation. |
| `LISTEN_ADDR` / `LISTEN_PORT` | `127.0.0.1` / `31700` | Bind behind the local TLS terminator. |
| `RATE_LIMIT_PER_SECOND` / `RATE_LIMIT_BURST_SIZE` | raise for aggregator traffic | Per-IP limiter; an aggregator arrives from a handful of IPs. |

Mutually exclusive with discovery: `VLLM_BACKEND_URLS` (static membership)
and `VLLM_DATA_PARALLEL_SIZE` (single-engine DP affinity).

## Behavior under fleet changes

- A backend that appears in the registry is added on the next poll and starts
  receiving new conversations; a backend that disappears stops receiving new
  requests immediately, while in-flight requests on it finish.
- Conversation pins are keyed by backend URL, not by position, so membership
  changes never re-route unrelated conversations; a pin to a departed backend
  is re-homed on that conversation's next turn.
- If the registry is unreachable the gateway serves from the last known set;
  if it has never answered, inference returns `503 service_unavailable` and
  `/healthz` reports `"backend":"no_backends"`.
- Overload: the engine caps its waiting queue (`--max-queued-requests`) and
  rejects at admission with "The request queue is full." — HTTP 503 for JSON
  requests, and for streams a first SSE event `data: {"error": …, "code": 503}`
  on an HTTP 200. With `VLLM_PROXY_STREAM_ERROR_PEEK_MS` the gateway holds the
  stream's status line for up to that long, so the rejection surfaces as a
  status; with `VLLM_PROXY_MAP_QUEUE_FULL_TO_429` that status is 429.
- `/metrics` exposes `backend_pool_size`, `backend_pool_healthy`,
  `backend_discovery_polls_total{outcome}`,
  `backend_discovery_consecutive_failures`, `backend_affinity_*`,
  `rejected_content_parts_total{part_type}`, `sse_keepalive_comments_total`,
  plus the existing usage-report and upstream metrics.

## What is deliberately not offered here

Attestation (`/v1/attestation/report`), response signatures
(`/v1/signature/{id}`) and GPU-evidence delegation (`/internal/gpu_evidence`)
are meaningful only inside the CVM; with `NON_TEE_DEPLOYMENT=1` they answer
404, so a non-TEE gateway never advertises something a verifier cannot check.
Customers who need TEE guarantees keep using the per-model
`*.completions.near.ai` domains or the Cloud API.
