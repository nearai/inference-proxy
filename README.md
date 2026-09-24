# vllm-proxy-rs

Rust proxy for vLLM/sglang inference engines running in GPU TEE environments. Adds Intel TDX + NVIDIA GPU attestation and cryptographic signing (ECDSA secp256k1 + Ed25519) to standard OpenAI-compatible API endpoints.

Rewrite of [nearai/vllm-proxy](https://github.com/nearai/vllm-proxy) (Python).

## Features

- **Dual signing** — every response is signed with both ECDSA (EIP-191, secp256k1) and Ed25519. Signatures are cached and retrievable per chat ID.
- **TEE attestation** — generates Intel TDX quotes via [dstack-sdk](https://github.com/Dstack-TEE/dstack) and NVIDIA GPU evidence via Python subprocess.
- **Backend-agnostic** — works with any OpenAI-compatible backend (vLLM, sglang, etc.).
- **Streaming support** — SSE streams are hashed incrementally and signed on completion.
- **In-memory cache** — moka-based TTL cache for signatures (no Redis dependency).
- **Fusion orchestration** — optional server-side multi-model deliberation for `/v1/chat/completions`, gated by `FUSION_ENABLED`.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Health check |
| GET | `/version` | No | Proxy version |
| GET | `/v1/metrics` | No | Backend metrics passthrough |
| GET | `/v1/models` | No | Backend models passthrough |
| POST | `/v1/chat/completions` | Yes | Chat completions (streaming + non-streaming) |
| POST | `/v1/completions` | Yes | Text completions |
| POST | `/v1/embeddings` | Yes | Embeddings |
| POST | `/v1/tokenize` | Yes | Tokenization (no signing) |
| POST | `/v1/rerank` | Yes | Reranking |
| POST | `/v1/score` | Yes | Scoring |
| POST | `/v1/privacy/classify` | Yes | Privacy classification |
| POST | `/v1/images/generations` | Yes | Image generation |
| POST | `/v1/images/edits` | Yes | Image editing (multipart) |
| POST | `/v1/audio/transcriptions` | Yes | Audio transcription (multipart) |
| GET | `/v1/signature/{chat_id}` | Yes | Retrieve cached signature |
| GET | `/v1/attestation/report` | No | TEE attestation report |
| POST | `/internal/gpu_evidence` | Trusted token | Sibling-proxy GPU evidence |

All other paths return `404` locally. They are never forwarded to the backend
inference engine.

## Error Handling

All error responses use the OpenAI-compatible JSON format:

```json
{"error": {"message": "...", "type": "...", "param": null, "code": null}}
```

### Proxy-generated errors

| Status | Type | When |
|--------|------|------|
| 400 | `bad_request` | Invalid JSON, bad parameters |
| 401 | `unauthorized` | Invalid or missing Bearer token |
| 404 | `not_found` | Signature chat ID not found |
| 413 | `payload_too_large` | Request body exceeds size limit |
| 429 | `rate_limited` | Per-IP rate limit exceeded |
| 500 | `server_error` | Internal proxy error (details hidden from client) |

### Upstream errors (vLLM/sglang)

Named routes (`/v1/chat/completions`, `/v1/completions`, etc.) pass through the backend error body verbatim, preserving the original status code.

Common upstream errors:

| Status | Type | Example message |
|--------|------|-----------------|
| 400 | `BadRequestError` | `"This model's maximum context length is 2048 tokens. However, you requested 4374 tokens"` |
| 400 | `BadRequestError` | `"temperature must be non-negative, got -0.5"` |
| 400 | `BadRequestError` | `"Stream options can only be defined when 'stream=True'"` |
| 400 | `BadRequestError` | `"please provide at least one prompt"` |
| 400 | `BadRequestError` | `"auto tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set"` |
| 404 | `Not Found` | `"The model 'gpt-5' does not exist"` |
| 422 | `Bad Request` | Pydantic validation details (field type mismatches) |
| 500 | `InternalServerError` | `"Internal server error"` (GPU OOM, engine crash) |
| 501 | `NotImplementedError` | `"Tool usage is only supported for Chat Completions API"` |

### Logging and privacy

All upstream errors are logged with structured fields for diagnostics:

```
WARN request{request_id=abc-123 method=POST path=/v1/chat/completions}:
  Backend returned non-success status
  upstream_status=400 upstream_url=http://vllm:8000/v1/chat/completions
  error_message="This model's maximum context length is 2048 tokens..."
  error_type=BadRequestError
```

**What is logged**: HTTP status codes, backend URLs, error messages (token counts, parameter names), error types, request IDs.

**What is never logged**: Request bodies, response bodies, prompt content, user messages, completion text.

## Configuration

All configuration is via environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_NAME` | Yes | — | Model name for cache key namespacing |
| `TOKEN` | Yes | — | Bearer token for API authentication |
| `VLLM_BASE_URL` | No | `http://localhost:8000` | Backend base URL |
| `VLLM_DATA_PARALLEL_SIZE` | No | unset | Number of independent vLLM DP engines in one backend. When set, assigns new chat conversations round-robin and keeps append-only turns on the same rank via `X-data-parallel-rank`, improving local prefix-cache reuse without trusting a client-supplied rank. Requires exactly one backend URL. |
| `VLLM_BACKEND_CONVERSATION_AFFINITY` | No | `false` | With several `VLLM_BACKEND_URLS`, keep append-only chat conversations on the backend that served their first turn (same salted conversation digest as the DP affinity above) so multi-turn prompts reuse that engine's prefix cache instead of being re-prefilled after every least-connections switch. New conversations still go to the least-loaded healthy backend; an unhealthy pinned backend is skipped. Applies to the exact and alias `/v1/chat/completions` routes; other routes keep least-connections. Mutually exclusive with `VLLM_DATA_PARALLEL_SIZE`. |
| `VLLM_BACKEND_AFFINITY_MAX_IMBALANCE` | No | `8` | Load-balance guard for the affinity above: when the pinned backend has more than this many extra in-flight requests over the least-loaded healthy backend, the turn goes to the least-loaded backend and the conversation is re-pinned there. |
| `DEV` | No | `false` | Dev mode (random signing keys instead of KMS) |
| `GPU_NO_HW_MODE` | No | `false` | Use canned GPU evidence |
| `CHAT_CACHE_EXPIRATION` | No | `1200` | Signature cache TTL in seconds |
| `VLLM_PROXY_MAX_REQUEST_SIZE` | No | `10485760` | Max JSON request body (bytes) |
| `VLLM_PROXY_MAX_IMAGE_REQUEST_SIZE` | No | `52428800` | Max image request body (bytes) |
| `VLLM_PROXY_MAX_AUDIO_REQUEST_SIZE` | No | `104857600` | Max audio request body (bytes) |
| `VLLM_PROXY_IMAGE_VALIDATION_DISABLED` | No | `false` | Disable pre-dispatch image URL/data validation |
| `VLLM_PROXY_IMAGE_VALIDATION_TIMEOUT_SECS` | No | `5` | Per-fetch timeout and semaphore acquire deadline for validation |
| `VLLM_PROXY_IMAGE_VALIDATION_MAX_BYTES` | No | `8192` | Max fetched/decode-head bytes used for image sniffing |
| `VLLM_PROXY_IMAGE_VALIDATION_MAX_CONCURRENCY` | No | `8` | Global concurrent outbound image-validation fetches |
| `VLLM_PROXY_IMAGE_VALIDATION_ALLOW_PRIVATE_HOSTS` | No | `false` | Permit private/loopback image hosts for trusted deployments/tests |
| `VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS` | No | empty; Gemma-4 defaults to `prod-files-secure.s3.us-west-2.amazonaws.com` | Exact remote `image_url` host allowlist enforced before fetch and on every redirect hop. When unset, falls back to `VLLM_ALLOWED_MEDIA_DOMAINS`; set explicitly to an empty string to disable the proxy-side domain restriction |
| `VLLM_ALLOWED_MEDIA_DOMAINS` | No | empty | vLLM-compatible media-domain allowlist used by the proxy only when `VLLM_PROXY_IMAGE_VALIDATION_ALLOWED_DOMAINS` is unset |
| `VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB` | No | `false` (`1` forces strict mode) | Gemma-4 defaults to rejecting observed one-channel PNG/JPEG crash inputs; set `1` to reject broader non-RGB PNG/JPEG classes |
| `VLLM_PROXY_MAX_KEEPALIVE` | No | `100` | Connection pool max idle per host |
| `VLLM_PROXY_STREAM_IDLE_TIMEOUT_SECS` | No | `0` (disabled) | Maximum idle time between upstream SSE chunks after the first client-visible generation-progress event. It does not cap queueing, prefill, or a metadata-only assistant-role event (vLLM may emit that before hidden reasoning). When enabled, internally reassembled JSON fails with 504; native streams terminate with a body error. EOF without `[DONE]` is also treated as incomplete |
| `LISTEN_PORT` | No | `8000` | Server listen port |
| `LISTEN_ADDR` | No | `0.0.0.0` | Interface to bind |
| `VLLM_BACKEND_TOKEN` | No | unset | Bearer attached to backend requests only (gateway mode: the backends are CVM inference-proxies and accept it as a trusted config token). Never sent to cloud-api |
| `VLLM_BACKEND_PRIORITY` | No | unset | Gateway mode: engine priority for this proxy's traffic, sent as `X-NearAI-Priority` on every backend request (e.g. `-1` for the OpenRouter lane). The CVM proxy honors the header only from callers using its `TOKEN`; it sets `priority` on every chat/completions body itself (header value or 0), discarding client values |
| `VLLM_BACKEND_HEALTH_PATH` | No | `/health` | Path probed by the pool health checker and `/healthz` |
| `NON_TEE_DEPLOYMENT` | No | `false` | This proxy runs outside a TEE: `/healthz` skips the dstack probe, the attestation cache refresh is not started, and `/v1/attestation/report`, `/v1/signature/{id}`, `/internal/gpu_evidence` return 404 |
| `VLLM_PROXY_MAP_QUEUE_FULL_TO_429` | No | `false` | Rewrite the engine's admission rejection (503 "The request queue is full." / "aborted by a higher priority request") to 429 with `Retry-After: 2` and type `overloaded`. Off in CVMs: cloud-api's peer fallback keys on the 503 |
| `VLLM_PROXY_STREAM_ERROR_PEEK_MS` | No | `0` (off) | Streaming: wait up to this long for the first upstream SSE chunk before committing a 200; an admission-time `data: {"error":…}` first event becomes a real error status instead of a 200 that fails mid-stream |
| `VLLM_PROXY_STREAM_COMMIT_MS` | No | `0` (off) | Streaming: commit `200 text/event-stream` after this long even when the upstream has not answered, so keep-alives can start during a long prefill. Past the window an upstream failure arrives as a terminal SSE `error` event, not a status — set it above the slowest error the deployment produces |
| `VLLM_PROXY_ALLOWED_ORG_IDS` | No | empty | Organizations whose cloud-api keys may use this deployment (comma-separated ids from `/v1/check_api_key`); other valid keys get 403 (counted in `cloud_api_org_allowlist_rejections_total`). Empty = everyone. Config-token callers are not gated |
| `VLLM_PROXY_REJECTED_CONTENT_PART_TYPES` | No | empty | Chat content part `type`s refused with 400 before dispatch, e.g. `video_url,input_audio,file` |
| `VLLM_PROXY_MODELS_DOCUMENT_URL` | No | empty | Gateway mode: serve `GET /v1/models` from this URL (cloud-api's `/v1/models`: pricing, modalities, `is_ready`, `openrouter.slug`) reduced to `MODEL_NAME` and completed with `capacity`; when the source cannot be read the engine's own list is passed through (counted in `models_document_source_failures_total`) |
| `VLLM_PROXY_CAPACITY_REQUESTS_PER_MINUTE` | No | `0` (not declared) | Requests per minute declared in the models document's `capacity`; the concurrency entry is `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` |
| `VLLM_PROXY_DISCOUNT_TO_USER` | No | empty (list price) | Gateway mode: the lane's discount off the list price, a fraction in `[0, 1)` with at most four decimal places (`0.2` = 20 % off; empty or `0` = none; an invalid value fails startup). Requires `VLLM_PROXY_MODELS_DOCUMENT_URL`. Published as `discount_to_user` on the models document entry (and on every entry of the engine list served while that source is unreadable) and sent as `discount_to_user` on every usage report, so the advertised and the billed price come from one setting |
| `VLLM_PROXY_REASONING_OFF_EFFORT` | No | `none` | Gateway mode: the `reasoning_effort` sent for "as little reasoning as possible" when an aggregator asks for `reasoning.enabled: false`, an effort of `none`/`minimal`, or sends those as `reasoning_effort`. Other efforts in the `reasoning` object are copied. GLM-5.3 Flash needs `low` (its template only knows `low`/`high`; switched off outright it writes its reasoning as content) |
| `VLLM_PROXY_SSE_KEEPALIVE_SECS` | No | `0` (off) | Emit `: keep-alive` SSE comments to the client whenever the upstream stream is silent this long. Not hashed into signatures — keep off where clients verify raw stream bytes |
| `VLLM_PROXY_ADMISSION_TIER_BORROWING` | No | off | Let base hosts share the global admission budget using configured tier counts; requires admission, both tiers, and a positive long-host ceiling |
| `VLLM_PROXY_ADMISSION_LONG_MAX_INFLIGHT_PER_HOST` | No | `0` | Long-host ceiling when borrowing is enabled; does not reserve global slots |
| `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` | No | `0` (off) | Gateway mode: ceiling on chat/completions requests in flight across the fleet. Beyond it, and while the lane looks overloaded (see below), new requests get `429` + `Retry-After` before anything is sent upstream (`admission.rs`) |
| `VLLM_PROXY_ADMISSION_START_INFLIGHT` | No | = max | Budget at start-up; it ramps by `VLLM_PROXY_ADMISSION_RAMP_STEP` (default `8`) every `VLLM_PROXY_ADMISSION_RAMP_INTERVAL_SECS` (default `1800`) up to the maximum, but only after an interval without an overload signal |
| `VLLM_PROXY_ADMISSION_TTFT_P95_MAX_MS` | No | `30000` (`0` = off) | Refuse new lane work while, over the last minute, at least 20 lane requests reached the engine and 5 % of them (at least two) waited longer than this for their first generation event. A request that ends before generating counts with the time it waited |
| `VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS` | No | `10` | A backend that rejected at engine admission (queue full / priority abort) within this many seconds is steered around while other hosts have room; when every healthy backend did, new lane work is refused |
| `VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT` | No | `1` | Engine-reported queue depth (from `VLLM_BACKEND_PROBE_URLS`) at or above which a backend counts as saturated for placement and the fleet-wide queue refusal |
| `VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS` | No | `2` | `Retry-After` value on admission refusals |
| `VLLM_BACKEND_CONNECT_FAILOVER` | No | `false` | Retry a chat/completions request once on another healthy backend when the connection to the chosen one fails before anything was sent; the unreachable backend leaves the rotation until the health checker sees it again and a pinned conversation follows the request. HTTP errors, queue-full included, are never retried |
| `VLLM_BACKEND_PROBE_URLS` | No | empty | Gateway mode: one plain-HTTP base URL per backend (same order as `VLLM_BACKEND_URLS`) whose `/v1/metrics` is polled for the engine's running and queued requests. New conversations go to the least-loaded engine, a host whose queue is at or above `VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT` (default 1) is steered around, and when every host queues new lane work gets 429. Empty = the gateway's own counts only |
| `VLLM_BACKEND_PROBE_INTERVAL_SECS` | No | `2` | Poll interval for the probes; a sample older than three intervals counts as unknown |
| `VLLM_BACKEND_LONG_CONTEXT_URLS` | No | empty | Gateway mode: backends of the long-context tier — the same hosts' handle URLs under the model's `-long` model-proxy domain — appended to the pool after `VLLM_BACKEND_URLS` so their indexes do not move. Requests estimated above `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS` are placed there, everything else on the base backends. Mutually exclusive with `VLLM_DATA_PARALLEL_SIZE` |
| `VLLM_BACKEND_LONG_CONTEXT_PROBE_URLS` | No | empty | One engine-load probe base URL per long-context backend, same order. Required when `VLLM_BACKEND_PROBE_URLS` is set, empty when it is not |
| `VLLM_BACKEND_LONG_CONTEXT_ABOVE_TOKENS` | No | `0` (off) | Estimated input tokens above which a request goes to the long-context tier, using cloud-api's own routing estimate (text, tool calls and tool definitions at bytes/4 with its 1.2 safety factor, plus a flat cost per media part, per-message template overhead and the reserved output window; `prompt` token ids counted exactly). By default a tier with no healthy backend falls back to the other one (`VLLM_BACKEND_TIER_STRICT` turns that off); a full one refuses with 429. Not combinable with `FUSION_ENABLED` or `WEB_CONTEXT_SEARCH_URL` |
| `VLLM_BACKEND_TIER_STRICT` | No | `false` | Isolate the context tiers in both directions instead of falling back: a request whose tier has no healthy backend is refused rather than placed on the other one — 429 with `Retry-After` when admission is enabled (`admission_rejections_total{reason="tier_unavailable"}`), 503 (`error_type: "tier_unavailable"`) when it is not. Only meaningful with `VLLM_BACKEND_LONG_CONTEXT_URLS` set; without one it is ignored (a startup warning is logged) |
| `VLLM_IMAGES_URL` | No | `{base}/v1/images/generations` | Override images endpoint |
| `VLLM_IMAGES_EDITS_URL` | No | `{base}/v1/images/edits` | Override image edits endpoint |
| `VLLM_TRANSCRIPTIONS_URL` | No | `{base}/v1/audio/transcriptions` | Override transcriptions endpoint |
| `VLLM_RERANK_URL` | No | `{base}/v1/rerank` | Override rerank endpoint |
| `VLLM_SCORE_URL` | No | `{base}/v1/score` | Override score endpoint |

### Fusion

Fusion is disabled by default. When enabled, `/v1/chat/completions` intercepts
OpenRouter-compatible `openrouter:fusion` server tools, NEAR `nearai:fusion`
tools, and OpenRouter plugin entries with `{"id":"fusion"}`; all other routes
and non-Fusion chat requests keep the normal proxy behavior. Cloud API remains a
pass-through: billing observes the single final response, whose `usage` contains
the aggregate token usage from panel, judge, and synthesis calls.

Supported request shapes:

```json
{"tools":[{"type":"openrouter:fusion","parameters":{"analysis_models":["model-a"],"model":"judge-model"}}]}
```

```json
{"plugins":[{"id":"fusion","analysis_models":["model-a"],"model":"judge-model"}]}
```

Legacy flat tool fields continue to work for `nearai:fusion` and existing
clients. `plugins[].enabled=false` is not treated as a Fusion invocation. The
`openrouter/fusion` model alias is not resolved in inference-proxy because
cloud-api routes by model name before pass-through.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FUSION_ENABLED` | No | `false` | Enables server-side Fusion orchestration |
| `FUSION_INTERNAL_BEARER_TOKEN` | Yes, when enabled | — | Bearer token used for internal direct completions calls |
| `FUSION_ENDPOINTS_URL` | No | `https://completions.near.ai/endpoints` | Endpoint discovery source |
| `FUSION_ENDPOINTS_TTL_SECS` | No | `300` | Discovery cache TTL |
| `FUSION_DEFAULT_ANALYSIS_MODELS` | No | — | Comma-separated fallback panel models |
| `FUSION_MAX_PANEL_MODELS` | No | `8` | Hard cap on panel fan-out |
| `FUSION_MAX_DEPTH` | No | `1` | Recursion guard for Fusion-to-Fusion calls |
| `FUSION_PANEL_TIMEOUT_SECS` | No | `120` | Timeout for Fusion panel, judge, and synthesis chat calls |
| `FUSION_MAX_RESPONSE_BYTES` | No | `10485760` | Max bytes buffered from Fusion endpoint discovery and internal model responses |
| `FUSION_INTERNAL_MAX_ATTEMPTS` | No | `2` | Attempts for transient Fusion direct model HTTP calls; `1` disables retries; max `5` |
| `FUSION_INTERNAL_RETRY_INITIAL_BACKOFF_MS` | No | `250` | Initial backoff for Fusion direct model retries; doubles per attempt with full jitter |
| `AGENT_LOOP_MAX_ITERATIONS` | No | `5` | Also caps Fusion `web_context_search` tool calls |
| `WEB_CONTEXT_SEARCH_URL` | If Fusion web search is used | — | Brave LLM Context endpoint |
| `WEB_CONTEXT_SEARCH_API_KEY` | If Fusion web search is used | — | Brave LLM Context API key |
| `BRAVE_LLM_CONTEXT_API_KEY` | No | — | Alias for `WEB_CONTEXT_SEARCH_API_KEY` |

Production launch checklist:

- Keep `FUSION_ENABLED=false` until the direct completions token and Brave key
  are provisioned in the deployment secret store.
- Set `FUSION_INTERNAL_BEARER_TOKEN` to a token accepted by the direct model
  proxies listed by `FUSION_ENDPOINTS_URL`.
- Treat `FUSION_ENDPOINTS_URL` as a trust anchor. Every returned panel or judge
  domain receives `FUSION_INTERNAL_BEARER_TOKEN`; do not point it at
  user-controlled endpoint lists or allow per-request endpoint overrides.
- V1 does not SSRF-filter discovered panel domains beyond trusting
  `FUSION_ENDPOINTS_URL`. Keep the endpoint list operator-controlled and use
  network policy if a deployment needs additional egress restrictions.
- V1 attestation covers the synthesis proxy response. Panel attestation is an
  informational `/attestation/report` liveness check and is not
  cryptographically bound into the final response signature.
- For Fusion requests that include `{"type":"web_context_search"}`, set
  `WEB_CONTEXT_SEARCH_URL=https://api.search.brave.com/res/v1/llm/context`
  and either `WEB_CONTEXT_SEARCH_API_KEY` or `BRAVE_LLM_CONTEXT_API_KEY`.
- Run the local/live smoke test before flipping the feature flag:

```bash
FUSION_INTERNAL_BEARER_TOKEN=... \
WEB_CONTEXT_SEARCH_API_KEY=... \
scripts/fusion_e2e.py --real-brave
```

The smoke test starts only local helper processes, calls live direct model
proxies, verifies non-streaming and streaming Fusion, checks aggregate usage,
and retrieves the final response signature. It does not deploy production.

## Running

```bash
# Dev mode (random signing keys, no TEE required)
DEV=1 MODEL_NAME=my-model TOKEN=secret cargo run

# Production (requires dstack TEE environment)
MODEL_NAME=my-model TOKEN=secret cargo run --release
```

The server listens on `0.0.0.0:8000` by default (configurable via `LISTEN_PORT`).

## Building

The Docker image (`./build-image.sh`) is a slim Ubuntu 22.04 runtime with no vLLM or CUDA; its apt and pip inputs are pinned in `pinned-packages-builder.txt`, `pinned-packages-runtime.txt` and `attestation-constraints.txt` (see the Dockerfile header). To build only the binary:

```bash
cargo build --release
```

The Docker image is built with `./build-image.sh` (or `make build`, which also
loads it into the local Docker daemon as `vllm-proxy-rs:latest`); see
[Reproducible build & verification](#reproducible-build--verification).

## Reproducible build & verification

The published image, `nearaidev/vllm-proxy-rs`, is reproducible: rebuilding a
commit gives the same image digest, bit for bit. External verifiers rely on this
to check that a deployed proxy runs the source it claims to run.

### What external verifiers do

Every deployment pins the proxy image by digest, so each served model runs one
image built from one commit of this repository. For each model, a verifier
records that source commit and then:

1. Checks the image's build provenance. `build.yml` publishes a GitHub artifact
   attestation and a keyless cosign signature for every image it pushes, and
   both name the source commit.
2. Rebuilds the commit from a fresh clone. The only inputs it sets are two
   environment variables, and each image rebuild gets 60 minutes:

   ```bash
   git clone https://github.com/nearai/inference-proxy && cd inference-proxy
   git checkout <commit>
   ENABLE_NV_ATTESTATION_SDK=1 SOURCE_DATE_EPOCH=0 bash build-image.sh
   tar -xOf oci.tar index.json | jq -r '.manifests[0].digest'   # must equal the deployed digest
   ```

To find the commit behind a published digest:

```bash
gh api /repos/nearai/inference-proxy/attestations/sha256:<hex> \
  --jq '.attestations[0].bundle.dsseEnvelope.payload' | base64 -d \
  | jq -r '.predicate.buildDefinition.resolvedDependencies[0].digest.gitCommit'
```

### The contract

- The command above, run on a fresh clone of a published commit, must produce
  the published digest: today, and when the commit is rebuilt months later.
- **Never add a required build environment variable or argument without
  coordinating with external verifiers first.** Their build environment is
  fixed. A new Dockerfile build arg must default to the value CI builds with.
- `build-image.sh` writes `./oci.tar`, which holds exactly one image manifest,
  and prints its digest. Its default path needs `docker` with buildx, `git`,
  `jq` and `tar`. It must not depend on `skopeo`, which only `--push` uses
  (some verifier environments replace it with a stub). `LOAD_IMAGE=1` also
  loads the image into the local Docker daemon; the digest is the same.
- Check out with umask 022, the default for CI runners and root. A file the
  Dockerfile copies without `--chmod` keeps its checkout mode in the image, so
  a clone made with umask 0002 (the default for many interactive Ubuntu
  accounts) can give a different digest.

### Expected rebuild time

Two things set the rebuild time: the builder stage (Ubuntu packages, Rust
toolchain, cargo build) and the size of the runtime base image, which is
downloaded, unpacked and exported in full. With the vLLM-based runtime image
(September 2026), a rebuild with no cached layers took 4 to 5 minutes on a
24-thread x86_64 machine with a 10 Gbit/s link, and about 7 minutes on our CI
hosts. Downloading and unpacking the 12.5 GB base took about 1.5 minutes, in
parallel with the builder stage, and writing the 12.5 GB archive 1.3 to 2
minutes. Slow package mirrors add minutes, and shared, busy hosts have needed
close to 30 minutes. The verifier-parity job below reports the time of every
rebuild in its run summary and warns above 30 minutes.

### How CI checks it

`reproducible-build.yml` runs on every merge to `main` that touches a build
input, weekly on `main` (Mondays 03:00 UTC, to catch drift in the archives and
registries the build reads), and on manual dispatch:

- two builds in parallel, normally on two different hosts, and two in a row on
  one host, whose archives must be identical byte for byte;
- a verifier-parity rebuild: a fresh `git clone` of the commit under umask 022,
  `env -i PATH=… HOME=… ENABLE_NV_ATTESTATION_SDK=1 SOURCE_DATE_EPOCH=0 bash build-image.sh`
  with a 60-minute limit, digest read from `index.json`;
- every digest must match the others and the digest `build.yml` published for
  the same commit (read from the `build.yml` run, and for `main` also from the
  `staging-<date>-<sha>` tag on Docker Hub);
- a pins guard: one build runs with `LOAD_IMAGE=1`, and every committed
  `pinned-packages-*.txt` must equal the package list that build installed.

Failures on `main` post an alert to Slack. To check a branch before merging,
dispatch `build.yml` on it first (so there is a published digest to compare
with), then `reproducible-build.yml`:

```bash
gh workflow run build.yml --ref <branch> -R nearai/inference-proxy
gh workflow run reproducible-build.yml --ref <branch> -R nearai/inference-proxy
```

### Hermetic inputs and how to bump them

Everything the build downloads is pinned; the Dockerfile header describes each
pin. Every bump changes the image digest, which is expected: bump in a pull
request and dispatch `reproducible-build.yml` on its branch.

| Input | Pinned in | How to bump |
|---|---|---|
| Base images | `FROM …@sha256:` lines in `Dockerfile` | Replace the digest (`docker buildx imagetools inspect <image>:<tag>`), then regenerate the pin files. |
| Ubuntu packages | `UBUNTU_SNAPSHOT` in `Dockerfile`, plus the pin files | Pick a snapshot timestamp in the past (`YYYYMMDDTHHMMSSZ`; the service also answers for future timestamps, whose content is not frozen yet), check that `https://snapshot.ubuntu.com/ubuntu/<timestamp>/dists/<suite>/InRelease` exists for every suite the Dockerfile uses, then regenerate the pin files. |
| Pin files | `pinned-packages-*.txt`, one per stage that installs Ubuntu packages | Empty them (`: > pinned-packages-builder.txt`), run `LOAD_IMAGE=1 ENABLE_NV_ATTESTATION_SDK=1 ./build-image.sh`, copy each `pinned-packages-*.resolved.txt` over its committed file, rebuild, and check that `diff -u` between each pair is empty (the CI pins guard runs the same diff). Always use `ENABLE_NV_ATTESTATION_SDK=1`: its package set includes the other one. |
| libnvat | `LIBNVAT_VERSION` in `Dockerfile` | Pick a version listed in NVIDIA's `ubuntu2204` repository index, then regenerate the pin files. |
| rustup | `RUSTUP_VERSION`, `RUSTUP_INIT_SHA256` in `Dockerfile` | Take the version from `https://static.rust-lang.org/rustup/release-stable.toml`, download `rustup-init` for it, hash it yourself and compare with the published `rustup-init.sha256`. |
| Rust toolchain | `--default-toolchain` in `Dockerfile` | Change the version. Published toolchain releases never change. |
| NVIDIA apt keyring | `CUDA_KEYRING_SHA256` in `Dockerfile` | `sha256sum` of the new `cuda-keyring` .deb. |
| Python packages | `NV_ATTESTATION_SDK_VERSION`, `NV_PPCIE_VERIFIER_VERSION` in `Dockerfile`, `attestation-constraints.txt` | Regenerate the constraints with the recipe in the file header. |
| Rust crates | `Cargo.lock` (with checksums) | `cargo update`. |
| BuildKit | `BUILDKIT_IMAGE` in `build-image.sh` (tag and digest) | BuildKit versions serialize layers differently, so a new version also changes the digest. |

### What the Ubuntu snapshot pin guarantees

- The snapshot service keeps each dated archive immutable:
  `https://snapshot.ubuntu.com/ubuntu/<timestamp>` serves the Ubuntu archive
  as it was at that time, and apt checks every package it downloads against
  that snapshot's indexes, which are signed with Ubuntu's archive key.
  Installing from the same timestamp gives the same versions and the same files.
- The committed exact versions are the second check. apt holds every package at
  the version listed in `pinned-packages-*.txt` (pin priority 1001): it installs
  that version or fails, never another one, and the CI pins guard compares what
  was installed with the committed list.

What it does not guarantee:

- Availability. If the snapshot service stops serving a timestamp, commits that
  use it can no longer be rebuilt. The build fails; it does not silently
  produce a different image. The same holds for the base images on Docker Hub,
  crates.io, the Rust release server, NVIDIA's repository and PyPI.
- Byte identity beyond the archive signatures. The pin files record versions,
  not package hashes: we trust the snapshot service to keep serving the same
  signed indexes for a timestamp, and the rebuilt digest is the final check.
- Inputs outside the snapshot. NVIDIA's apt repository has no snapshot service
  (libnvat is pinned to an exact version; if NVIDIA dropped it, older commits
  would stop building). Python packages are pinned by version, not by hash, and
  a project can add files to an existing version. The throwaway `ca-bootstrap`
  stage installs `ca-certificates` from the live Ubuntu archive, but only its CA
  bundle is used, to reach the snapshot over TLS, and nothing from that stage
  reaches the image. Any stage that still installs from the live archive is
  called out in the Dockerfile header.

## Testing

```bash
cargo test
```

The suite includes unit tests for signing, cache, config, errors,
attestation helpers, SSE parsing, Fusion orchestration, and integration tests
with wiremock mock backends for cryptographic signature verification,
multipart endpoints, streaming completions, E2EE, web tools, and Fusion.

## Project Structure

```
src/
  lib.rs              # Public module exports, AppState, request ID middleware
  main.rs             # Entry point, server startup, graceful shutdown
  config.rs           # Env var configuration
  error.rs            # AppError -> OpenAI-style JSON error responses
  types.rs            # SignedChat, AttestationReport, SignatureResponse
  signing.rs          # ECDSA (secp256k1) + Ed25519 signing, key derivation
  attestation.rs      # TDX + GPU attestation report generation
  cache.rs            # moka in-memory cache with TTL
  proxy.rs            # Generic proxy helpers (JSON, streaming SSE, multipart)
  auth.rs             # Bearer token auth extractor
  routes/
    mod.rs            # Router assembly
    health.rs         # GET /, GET /version
    chat.rs           # POST /v1/chat/completions
    completions.rs    # POST /v1/completions
    passthrough.rs    # embeddings, rerank, score, images, audio, tokenize
    signature.rs      # GET /v1/signature/{chat_id}
    attestation.rs    # GET /v1/attestation/report
    metrics.rs        # GET /v1/metrics, GET /v1/models
tests/
  integration.rs      # Integration tests with wiremock
```

## Signing

Every signed response produces a `SignedChat` cached by response ID:

```json
{
  "text": "{sha256_request}:{sha256_response}",
  "signature_ecdsa": "0x{r}{s}{v}",
  "signing_address_ecdsa": "0x{ethereum_address}",
  "signature_ed25519": "{hex_signature}",
  "signing_address_ed25519": "{hex_public_key}"
}
```

- **ECDSA**: EIP-191 `personal_sign` format, recoverable secp256k1 signature
- **Ed25519**: Direct message signing, 64-byte signature

In production, signing keys are derived from dstack KMS (`DstackClient::get_key`). In dev mode (`DEV=1`), random keys are generated at startup.
