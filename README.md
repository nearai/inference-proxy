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
