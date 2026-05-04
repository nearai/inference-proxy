# Build args:
#   ENABLE_NV_ATTESTATION_SDK=1  → build with the nv-attestation-sdk Cargo
#       feature, link against libnvat.so for direct-FFI GPU evidence
#       collection (no Python subprocess). The runtime image is also
#       provisioned with libnvat.so. Default off until staging validates.
#       Even at "1" the runtime path stays Python-backed unless the env
#       var USE_NV_ATTESTATION_SDK=true is set on the container.
ARG ENABLE_NV_ATTESTATION_SDK=0
ARG NVAT_TAG=2026.04.29

# ─────────────────────────────────────────────────────────────────────
# Stage 0: build libnvat.so + nvat.h from NVIDIA/attestation-sdk source.
# When the feature is disabled this stage is still emitted (cheap noop)
# so subsequent COPY --from=nvat-builder steps don't need conditional
# escapes. Output paths /opt/nvat/lib /opt/nvat/include always exist.
# ─────────────────────────────────────────────────────────────────────
# rust:1.93.0-bookworm gives us Debian + rustc/cargo in one image — the
# SDK's CMake build uses Corrosion to compile a Rust dependency (regorus,
# the policy engine), so cargo must be on PATH.
FROM rust:1.93.0-bookworm AS nvat-builder
ARG ENABLE_NV_ATTESTATION_SDK
ARG NVAT_TAG
RUN mkdir -p /opt/nvat/lib /opt/nvat/include && touch /opt/nvat/.empty
RUN if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        set -e && \
        apt-get update && apt-get install -y --no-install-recommends \
            git ca-certificates cmake ninja-build g++ pkg-config make \
            curl perl python3 zlib1g-dev && \
        git clone --depth 1 --branch ${NVAT_TAG} https://github.com/NVIDIA/attestation-sdk.git /src/attestation-sdk && \
        cmake -S /src/attestation-sdk/nv-attestation-sdk-cpp -B /src/attestation-sdk/build \
              -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON && \
        # The SDK's CMakeLists adds OpenSSL/xmlsec/curl as ExternalProjects
        # without BUILD_BYPRODUCTS, so ninja can't sequence libnvat.so's
        # link step against the .a files those externals produce. Build
        # them explicitly first, then link nvat in a second pass.
        cmake --build /src/attestation-sdk/build \
              --target openssl_external --target xmlsec_external --target curl_external && \
        cmake --build /src/attestation-sdk/build --target nvat && \
        # Stage just the runtime artifacts we need (libnvat.so* + header).
        cp /src/attestation-sdk/build/include/nvat.h /opt/nvat/include/ && \
        cp -P /src/attestation-sdk/build/libnvat.so* /opt/nvat/lib/ && \
        rm -rf /src /var/lib/apt/lists/* ; \
    fi

# ─────────────────────────────────────────────────────────────────────
# Stage 1: Build the Rust binary
# ─────────────────────────────────────────────────────────────────────
FROM rust:1.93.0-bookworm AS builder
ARG ENABLE_NV_ATTESTATION_SDK

RUN apt-get update && apt-get install -y --no-install-recommends git pkg-config \
    && if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        # bindgen needs libclang to parse nvat.h.
        apt-get install -y --no-install-recommends clang libclang-dev; \
    fi \
    && rm -rf /var/lib/apt/lists/* /var/log/* /var/cache/ldconfig/aux-cache

# Stage in libnvat header + .so so bindgen can link. The COPY always
# succeeds (paths always exist, even when feature is off).
COPY --from=nvat-builder /opt/nvat/ /opt/nvat/
RUN if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        cp /opt/nvat/include/nvat.h /usr/include/ && \
        cp -P /opt/nvat/lib/libnvat.so* /usr/lib/ && \
        ldconfig; \
    fi

# Tell the SDK's build.rs to use the system-installed header/lib we
# just placed (vs. trying to build the C++ SDK from source again).
ENV NVAT_USE_SYSTEM_LIB=1

WORKDIR /build

ARG SOURCE_DATE_EPOCH=0
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}

# Resolve the cargo feature flag once and reuse below.
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

ENV PYTHONUNBUFFERED=1

# Install the verifier packages needed for GPU attestation evidence
# nv-attestation-sdk provides the `verifier` module for GPU evidence collection
# nv-ppcie-verifier is additionally needed for PPCIE multi-GPU systems.
# When ENABLE_NV_ATTESTATION_SDK=1 these are still installed for the
# Python fallback path (USE_NV_ATTESTATION_SDK=false at runtime); a
# follow-up will drop them once the SDK path proves out.
RUN pip install --no-cache-dir nv-attestation-sdk nv-ppcie-verifier

# Stage in libnvat.so for the runtime image when the feature was built.
COPY --from=nvat-builder /opt/nvat/lib/ /opt/nvat-runtime/lib/
RUN if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        cp -P /opt/nvat-runtime/lib/libnvat.so* /usr/lib/ && ldconfig; \
    fi && rm -rf /opt/nvat-runtime

WORKDIR /app

# Copy compiled binary and GPU evidence worker from builder
COPY --from=builder /build/target/release/vllm-proxy-rs /app/vllm-proxy-rs
COPY gpu_evidence_worker.py /app/gpu_evidence_worker.py

# Bake in git revision for version tracking
COPY --chmod=664 .GIT_REV /etc/

ENV LISTEN_PORT=8000
EXPOSE 8000

ENTRYPOINT ["/app/vllm-proxy-rs"]
