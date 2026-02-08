# CLAUDE.md

## Build & Test

```bash
cargo build --release    # Release build
cargo check              # Fast type-check
cargo test               # Run all tests (42 unit + 30 integration)
cargo test <test_name>   # Run a specific test
cargo fmt                # Format code
```

No special env vars needed for tests — integration tests use wiremock and fixed signing keys.

## Architecture

This is a Rust rewrite of [nearai/vllm-proxy](https://github.com/nearai/vllm-proxy). It proxies OpenAI-compatible API requests to a vLLM/sglang backend, adding cryptographic signing and TEE attestation.

**Key flow**: Request arrives -> auth check -> read body -> forward to backend -> hash request+response -> sign with ECDSA+Ed25519 -> cache signature -> return response.

### Module map

- `lib.rs` — `AppState` (shared state via Arc), `request_id_middleware`, public module exports
- `config.rs` — `Config::from_env()`, all env vars loaded at startup
- `signing.rs` — `EcdsaContext` (secp256k1 EIP-191), `Ed25519Context`, `SigningPair`
- `proxy.rs` — `proxy_json_request()`, `proxy_streaming_request()`, `proxy_multipart_request()`, `proxy_simple()` — the core proxy+sign+cache helpers that routes delegate to
- `cache.rs` — `ChatCache` wrapping moka with TTL, key format `{model}:chat:{id}`
- `attestation.rs` — `generate_attestation()` calls dstack for TDX quotes and Python subprocess for GPU evidence
- `auth.rs` — `RequireAuth` axum extractor (validates Bearer token)
- `routes/` — thin handlers that parse request, call proxy helpers

### Important patterns

- `dstack-sdk` is a **git dependency** (not on crates.io). Path: `dstack_sdk::dstack_client::DstackClient`
- `reqwest::multipart::Part::mime_str()` consumes self — use `.expect()` not `?` in chains
- Streaming uses `tokio::spawn` + `mpsc` channel: background task hashes chunks and signs on stream completion
- `strip_empty_tool_calls` in `routes/chat.rs` is a vLLM bug workaround (still needed as of vLLM v0.15.1)
- Signed text format: `"{sha256_request}:{sha256_response}"` signed by both algos
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
- Upstream errors are passed through verbatim (status code + body)
- Unit tests live in `#[cfg(test)] mod tests` within each source file
- Integration tests in `tests/integration.rs` use `tower::ServiceExt::oneshot` with wiremock
