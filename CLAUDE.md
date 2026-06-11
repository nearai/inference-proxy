# CLAUDE.md

## Build & Test

```bash
cargo build --release    # Release build
cargo check              # Fast type-check
cargo test               # Run all tests (116 unit + 77 integration)
cargo test <test_name>   # Run a specific test
cargo fmt                # Format code
```

No special env vars needed for tests — integration tests use wiremock and fixed signing keys.

For changes that touch NVML, the libnvat SDK FFI, dstack TDX, or
proxy-to-proxy contracts (e.g. `/internal/gpu_evidence`), `cargo test`
isn't enough — see [docs/testing-on-cvm.md](docs/testing-on-cvm.md)
for the real-CVM smoke-test recipe (build a branch image with
`gh workflow run build.yml --ref <branch>`, deploy a 2-proxy stack
inside a gpu0X CVM, probe both happy and leader-down paths).

## Architecture

This is a Rust rewrite of [nearai/vllm-proxy](https://github.com/nearai/vllm-proxy). It proxies OpenAI-compatible API requests to a vLLM/sglang backend, adding cryptographic signing and TEE attestation.

**Key flow**: Request arrives -> auth check -> read body -> forward to backend -> hash request+response -> sign with ECDSA+Ed25519 -> cache signature -> return response.

### Module map

- `lib.rs` — `AppState` (shared state via Arc), `request_id_middleware`, public module exports
- `config.rs` — `Config::from_env()`, all env vars loaded at startup
- `signing.rs` — `EcdsaContext` (secp256k1 EIP-191), `Ed25519Context`, `SigningPair`
- `proxy.rs` — `proxy_json_request()`, `proxy_streaming_request()`, `proxy_multipart_request()`, `proxy_simple()` — the core proxy+sign+cache helpers that routes delegate to
- `cache.rs` — `ChatCache` wrapping moka with TTL, key format `{model}:chat:{id}`
- `attestation.rs` — `AttestationCache`, `generate_attestation()`, GPU evidence collection with retry/serialization, dstack TDX quotes
- `auth.rs` — `RequireAuth` axum extractor (validates Bearer token)
- `routes/` — thin handlers that parse request, call proxy helpers

### Important patterns

- `dstack-sdk` is on crates.io now (was previously a git dependency). Path: `dstack_sdk::dstack_client::DstackClient`
- `generate_attestation()` takes `AttestationParams` struct + optional `&AttestationCache`
- Attestation nonces are cryptographically bound to GPU evidence and TDX quotes — cannot cache across different nonces
- Nonce-less attestation requests can be cached (caller accepts whatever nonce we generate)
- GPU evidence collection spawns a Python subprocess calling NVIDIA's `cc_admin.collect_gpu_evidence_remote()` — serialized behind a semaphore to avoid NVML driver contention
- `reqwest::multipart::Part::mime_str()` consumes self — use `.expect()` not `?` in chains
- Streaming uses `tokio::spawn` + `mpsc` channel: background task hashes chunks and signs on stream completion
- `strip_empty_tool_calls` in `routes/chat.rs` is a vLLM bug workaround (still needed as of vLLM v0.15.1)
- Signed text format: `"{model_name}:{sha256_request}:{sha256_response}"` signed by both algos
- `serde_json::to_string` matches Python's `json.dumps(separators=(",",":"))`

### Dependencies to know about

- `futures-util` — needed for `StreamExt` on reqwest byte streams
- `http-body-util` — needed for `BodyExt::frame()` / `BodyExt::collect()` on axum Body
- `k256` — secp256k1 ECDSA with `PrehashSigner` trait from `signature::hazmat`
- `sha3` — Keccak256 for Ethereum address derivation and EIP-191

## Code style

- Run `cargo fmt` before committing
- Error types in `error.rs` produce OpenAI-compatible JSON: `{"error": {"message": ..., "type": ..., "param": null, "code": null}}`
- Internal errors hide details from clients (returns generic "Internal server error")
- Named route upstream errors are passed through verbatim (status code + body)
- Catch-all upstream errors are parsed and re-wrapped via `AppError::UpstreamParsed` (never leaks raw body)
- `parse_upstream_error()` in `proxy.rs` handles both vLLM flat (`{"message":"..."}`) and nested (`{"error":{"message":"..."}}`) formats
- All upstream errors logged with `warn!` including `error_message` and `error_type` (never body content)
- `request_id_middleware` wraps requests in a tracing span — all log lines automatically include `request_id`, `method`, `path`
- Unit tests live in `#[cfg(test)] mod tests` within each source file
- Integration tests in `tests/integration.rs` use `tower::ServiceExt::oneshot` with wiremock
- When adding fields to `Config` or `AppState`, also update both `build_test_app_*` helpers in `tests/integration.rs`

## Deployment

- Docker image: `nearaidev/vllm-proxy-rs` (published with digest-pinned refs in cvm-conf)
- Deployed via compose files in [nearai/cvm-compose-files](https://github.com/nearai/cvm-compose-files)
- Each proxy instance needs: `MODEL_NAME`, `TOKEN`, `VLLM_BASE_URL`, `TLS_CERT_PATH`
- Optional: `CLOUD_API_URL` (enables usage reporting + `sk-` API key auth via cloud-api), `LOG_FORMAT=json` (structured JSON logs)
- Pre-dispatch image validation is enabled by default for chat-completions image inputs. `VLLM_PROXY_IMAGE_VALIDATION_DISABLED=1` disables it. Tunables: `VLLM_PROXY_IMAGE_VALIDATION_TIMEOUT_SECS` (default 5), `VLLM_PROXY_IMAGE_VALIDATION_MAX_BYTES` (8192), `VLLM_PROXY_IMAGE_VALIDATION_MAX_CONCURRENCY` (8), `VLLM_PROXY_IMAGE_VALIDATION_ALLOW_PRIVATE_HOSTS` (default off), `VLLM_PROXY_IMAGE_VALIDATION_REJECT_NON_RGB` (default off; `1` forces strict non-RGB PNG/JPEG rejection. Gemma-4 model names still auto-reject observed one-channel PNG/JPEG crash inputs).
- `ATTESTATION_CACHE_TTL` (default 300s) — TTL for cached nonce-less attestation reports; background refresh runs at half-TTL
- `DSTACK_SOCKET_PATH` (default `/var/run/dstack.sock`) — probed by `GET /healthz` so upstream load balancers (e.g. model-proxy) can detach this instance when the dstack guest-agent socket is unreachable. `/v1/models` alone won't catch this failure mode — sglang/vLLM keep serving while `/v1/attestation/report` silently 500s. The backend leg of `/healthz` probes `/health` (not `/v1/models`) since `/v1/models` serializes against the OpenAI request loop and can stall for >1s during prefill, producing spurious 503s on otherwise-healthy hosts.

### Cloud API integration

- `CLOUD_API_URL` enables two features: (1) `sk-live-`/`sk-test-` API key validation via `POST /v1/check_api_key`, (2) fire-and-forget usage reporting via `POST /v1/internal/usage`
- Usage reporting is in `proxy.rs`: `spawn_usage_report()` posts to `/v1/internal/usage` with the shared `CLOUD_API_USAGE_TOKEN` as Bearer and the subject identity (org/workspace/api_key_id, from the `/v1/check_api_key` response) in the body. The legacy `sk-`-authenticated `POST /v1/usage` endpoint was removed from cloud-api; if `CLOUD_API_USAGE_TOKEN` (or any identity field) is missing, reporting is **skipped with an error log** — there is no fallback. Failures are logged as warnings.
- **MODEL_NAME must exactly match `model_name` in cloud-api's model table** — cloud-api does NOT check model aliases, so a mismatch causes silent 404s on usage reporting
