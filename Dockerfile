# Stage 1: Build the Rust binary
FROM rust:1.93.0-bookworm AS builder

RUN apt-get update && apt-get install -y git pkg-config && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependencies: copy manifests first, then do a dummy build
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs \
    && cargo build --release 2>/dev/null || true \
    && rm -rf src \
    && rm -f target/release/deps/*vllm_proxy_rs* \
    && rm -f target/release/vllm-proxy-rs* \
    && rm -rf target/release/.fingerprint/vllm-proxy-rs-*

# Copy real source and build — touch to ensure cargo detects changes
COPY src/ src/
RUN find src -name '*.rs' -exec touch {} + && cargo build --release

# Stage 2: Runtime image
# GPU attestation requires pynvml (needs CUDA), so use vllm base image
FROM vllm/vllm-openai@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c

ENV PYTHONUNBUFFERED=1

# Install the verifier packages needed for GPU attestation evidence
# nv-attestation-sdk provides the `verifier` module for GPU evidence collection
# nv-ppcie-verifier is additionally needed for PPCIE multi-GPU systems
RUN pip install --no-cache-dir nv-attestation-sdk nv-ppcie-verifier

WORKDIR /app

# Copy compiled binary from builder
COPY --from=builder /build/target/release/vllm-proxy-rs /app/vllm-proxy-rs

# Bake in git revision for version tracking
COPY --chmod=664 .GIT_REV /etc/

ENV LISTEN_PORT=8000
EXPOSE 8000

ENTRYPOINT ["/app/vllm-proxy-rs"]
