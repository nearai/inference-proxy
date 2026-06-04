# Build args:
#   ENABLE_NV_ATTESTATION_SDK=1  → build with the nv-attestation-sdk Cargo
#       feature, link against libnvat.so for direct-FFI GPU evidence
#       collection (no Python subprocess). The runtime image is also
#       provisioned with libnvat.so. Default off until staging validates.
#       Even at "1" the runtime path stays Python-backed unless the env
#       var USE_NV_ATTESTATION_SDK=true is set on the container.
#   LIBNVAT_VERSION  → exact apt-pinned version of NVIDIA's libnvat package
#       (see https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/).
#       The trailing ".<timestamp>-1" suffix is part of the upstream version
#       string and changes per build; pin it so rebuilds stay reproducible.
#   NV_ATTESTATION_SDK_VERSION / NV_PPCIE_VERIFIER_VERSION → exact PyPI
#       versions of the GPU-attestation Python packages installed in the
#       runtime image. Left unpinned, pip resolves whatever is current at
#       build time — a drift source; pin them for the same reason as
#       LIBNVAT_VERSION. (Their direct deps are mostly == pinned upstream;
#       a few transitive ranges — nvidia-ml-py, requests — can still float.)
ARG ENABLE_NV_ATTESTATION_SDK=0
ARG LIBNVAT_VERSION=1.2.1.1777487608-1
ARG NV_ATTESTATION_SDK_VERSION=2.7.3
ARG NV_PPCIE_VERIFIER_VERSION=2.0.0

# ─────────────────────────────────────────────────────────────────────
# Stage 1: Build the Rust binary
#
# Switched from rust:1.93.0-bookworm (Debian 12) to ubuntu:22.04 +
# rustup so the libnvat we link against is the same .deb the runtime
# image installs (NVIDIA only publishes libnvat for Ubuntu 22.04/24.04;
# no Debian 12 build). Matching distributions on both sides eliminates
# any libssl3/libcurl4/libxml2 ABI risk.
# ─────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04@sha256:4f838adc7181d9039ac795a7d0aba05a9bd9ecd480d294483169c5def983b64d AS builder
ARG ENABLE_NV_ATTESTATION_SDK
ARG LIBNVAT_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git pkg-config build-essential gcc \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.93.0 (matching the previous rust:1.93.0-bookworm base).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain 1.93.0 --profile minimal --no-modify-path
ENV PATH=/root/.cargo/bin:$PATH

# Install libnvat-dev (headers + .so symlink) and libclang for bindgen
# only when the SDK feature is on. -dev pulls in libnvat (the runtime
# .so) as a versioned dependency, plus libcurl4/libxml2/libxmlsec1-openssl
# which libnvat dynamically links against.
RUN if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        set -e && \
        apt-get update && apt-get install -y --no-install-recommends wget gnupg && \
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
        apt-get update && apt-get install -y --no-install-recommends \
            clang libclang-dev \
            "libnvat-dev=${LIBNVAT_VERSION}" "libnvat=${LIBNVAT_VERSION}" && \
        ldconfig && \
        rm -rf /var/lib/apt/lists/* ; \
    fi

# Tell nv-attestation-sdk-sys's build.rs to look for the system-installed
# libnvat (/usr/include/nvat.h + /usr/lib/.../libnvat.so) rather than
# trying to build the C++ SDK from a sibling source directory. No-op when
# the Cargo feature is disabled.
ENV NVAT_USE_SYSTEM_LIB=1

WORKDIR /build

ARG SOURCE_DATE_EPOCH=0
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

# Resolve the cargo feature flag once so dependency-cache and real builds
# agree on the feature set.
RUN if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        echo "--features nv-attestation-sdk" > /tmp/cargo-features; \
    else \
        : > /tmp/cargo-features; \
    fi

# Cache dependencies: copy manifests first, then do a dummy build
COPY Cargo.toml Cargo.lock ./
RUN FEATURES=$(cat /tmp/cargo-features) && \
    mkdir src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs \
    && mkdir -p benches && echo "fn main() {}" > benches/hot_path.rs && echo "fn main() {}" > benches/e2e.rs \
    && cargo build --release --locked $FEATURES 2>/dev/null || true \
    && rm -rf src benches \
    && rm -f target/release/deps/*vllm_proxy_rs* \
    && rm -f target/release/vllm-proxy-rs* \
    && rm -rf target/release/.fingerprint/vllm-proxy-rs-*

# Copy real source and build — touch to ensure cargo detects changes
COPY src/ src/
COPY benches/ benches/
RUN FEATURES=$(cat /tmp/cargo-features) && \
    find src -name '*.rs' -exec touch {} + && cargo build --release --locked $FEATURES

# ─────────────────────────────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────────────
FROM vllm/vllm-openai@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c
ARG ENABLE_NV_ATTESTATION_SDK
ARG LIBNVAT_VERSION
ARG NV_ATTESTATION_SDK_VERSION
ARG NV_PPCIE_VERIFIER_VERSION

ENV PYTHONUNBUFFERED=1

# Install the verifier packages needed for GPU attestation evidence
# nv-attestation-sdk provides the `verifier` module for GPU evidence collection
# nv-ppcie-verifier is additionally needed for PPCIE multi-GPU systems.
# When ENABLE_NV_ATTESTATION_SDK=1 these are still installed for the
# Python fallback path (USE_NV_ATTESTATION_SDK=false at runtime); a
# follow-up will drop them once the SDK path proves out.
# --no-compile: do NOT byte-compile .pyc at build time. CPython stamps each
# timestamp-invalidated .pyc with the source file's mtime, and pip writes the
# .py files with the current wall-clock mtime — so the .pyc embed the build
# time and this layer (hence the image digest) changes on every build.
# rewrite-timestamp normalizes tar mtimes but not the bytes inside a .pyc.
# Skipping compilation keeps the layer deterministic; CPython compiles the
# modules in memory on first import at runtime (negligible for this service).
RUN pip install --no-cache-dir --no-compile \
        "nv-attestation-sdk==${NV_ATTESTATION_SDK_VERSION}" \
        "nv-ppcie-verifier==${NV_PPCIE_VERIFIER_VERSION}"

# Install libnvat (runtime) when the feature is built. vllm/vllm-openai
# already has the CUDA apt repo configured, so cuda-keyring isn't needed
# here. apt pulls in libcurl4/libxml2/libxmlsec1-openssl as deps.
#
# Reproducibility: apt/dpkg write per-line timestamps to their logs and
# ldconfig writes a non-deterministic aux-cache, all of which get baked into
# this layer and make the image digest change on every build. Remove them in
# the same RUN (rewrite-timestamp only normalizes tar mtimes, not file bytes).
# Clean the whole /var/log/apt dir, not just *.log: apt also writes
# eipp.log.xz (the EIPP solver log), whose embedded APT-IDs drift as the
# package indices change between build days — a *.log glob misses the .xz and
# leaves a cross-day-non-reproducible layer that the same-day reproducible-build
# double-build cannot catch.
RUN if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        set -e && \
        apt-get update && apt-get install -y --no-install-recommends \
            "libnvat=${LIBNVAT_VERSION}" && \
        ldconfig && \
        rm -rf /var/lib/apt/lists/* \
               /var/log/apt/* /var/log/dpkg.log /var/log/alternatives.log \
               /var/cache/ldconfig/aux-cache ; \
    fi

WORKDIR /app

# Copy compiled binary and GPU evidence worker from builder
COPY --from=builder /build/target/release/vllm-proxy-rs /app/vllm-proxy-rs
COPY gpu_evidence_worker.py /app/gpu_evidence_worker.py

# Bake in git revision for version tracking
COPY --chmod=664 .GIT_REV /etc/

ENV LISTEN_PORT=8000
EXPOSE 8000

ENTRYPOINT ["/app/vllm-proxy-rs"]
