# vllm-proxy-rs

Rust proxy for vLLM/sglang inference engines running in GPU TEE environments. Adds Intel TDX + NVIDIA GPU attestation and cryptographic signing (ECDSA secp256k1 + Ed25519) to standard OpenAI-compatible API endpoints.

Rewrite of [nearai/vllm-proxy](https://github.com/nearai/vllm-proxy) (Python).

## Features

- **Dual signing** — every response is signed with both ECDSA (EIP-191, secp256k1) and Ed25519. Signatures are cached and retrievable per chat ID.
- **TEE attestation** — generates Intel TDX quotes via [dstack-sdk](https://github.com/Dstack-TEE/dstack) and NVIDIA GPU evidence via Python subprocess.
- **Backend-agnostic** — works with any OpenAI-compatible backend (vLLM, sglang, etc.).
- **Streaming support** — SSE streams are hashed incrementally and signed on completion.
- **In-memory cache** — moka-based TTL cache for signatures (no Redis dependency).

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
| `DEV` | No | `false` | Dev mode (random signing keys instead of KMS) |
| `GPU_NO_HW_MODE` | No | `false` | Use canned GPU evidence |
| `CHAT_CACHE_EXPIRATION` | No | `1200` | Signature cache TTL in seconds |
| `VLLM_PROXY_MAX_REQUEST_SIZE` | No | `10485760` | Max JSON request body (bytes) |
| `VLLM_PROXY_MAX_IMAGE_REQUEST_SIZE` | No | `52428800` | Max image request body (bytes) |
| `VLLM_PROXY_MAX_AUDIO_REQUEST_SIZE` | No | `104857600` | Max audio request body (bytes) |
| `VLLM_PROXY_MAX_KEEPALIVE` | No | `100` | Connection pool max idle per host |
| `LISTEN_PORT` | No | `8000` | Server listen port |
| `VLLM_IMAGES_URL` | No | `{base}/v1/images/generations` | Override images endpoint |
| `VLLM_IMAGES_EDITS_URL` | No | `{base}/v1/images/edits` | Override image edits endpoint |
| `VLLM_TRANSCRIPTIONS_URL` | No | `{base}/v1/audio/transcriptions` | Override transcriptions endpoint |
| `VLLM_RERANK_URL` | No | `{base}/v1/rerank` | Override rerank endpoint |
| `VLLM_SCORE_URL` | No | `{base}/v1/score` | Override score endpoint |

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

89 tests: 50 unit tests (signing, cache, config, error, attestation helpers, SSE parser) and 39 integration tests (full app with wiremock mock backend, including cryptographic signature verification, multipart endpoints, streaming completions).

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
