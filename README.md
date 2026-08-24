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
| POST | `/v1/images/generations` | Yes | Image generation |
| POST | `/v1/images/edits` | Yes | Image editing (multipart) |
| POST | `/v1/audio/transcriptions` | Yes | Audio transcription (multipart) |
| GET | `/v1/signature/{chat_id}` | Yes | Retrieve cached signature |
| GET | `/v1/attestation/report` | Yes | TEE attestation report |

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

Named routes (`/v1/chat/completions`, `/v1/completions`, etc.) pass through the backend error body verbatim, preserving the original status code. The catch-all route (arbitrary paths) parses the backend error and re-wraps it in the OpenAI format above.

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
| `VLLM_PROXY_STREAM_IDLE_TIMEOUT_SECS` | No | `0` (disabled) | Maximum idle time between upstream SSE chunks after the first chunk. It does not cap queueing/prefill time to first chunk. When enabled, internally reassembled JSON fails with 504; native streams terminate with a body error. EOF without `[DONE]` is also treated as incomplete |
| `LISTEN_PORT` | No | `8000` | Server listen port |
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

```bash
cargo build --release
```

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
