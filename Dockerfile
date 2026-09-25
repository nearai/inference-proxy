# Reproducible build contract
# ───────────────────────────
# `ENABLE_NV_ATTESTATION_SDK=1 SOURCE_DATE_EPOCH=0 ./build-image.sh` on a fresh
# clone of a commit must produce the image digest CI published for that
# commit, both today and when the commit is rebuilt months later: external
# verifiers check deployed digests exactly that way. build-image.sh passes
# ENABLE_NV_ATTESTATION_SDK and SOURCE_DATE_EPOCH through; every other ARG
# below defaults to the value CI builds with. Never make another build arg or
# environment variable required for that command.
#
# Runtime image
# ─────────────
# The final stage is ubuntu:22.04 (the builder's base digest) plus exactly
# what the proxy uses at runtime, about 95 MB compressed:
#   * /app/vllm-proxy-rs, the Rust binary.
#   * python3 (3.10) and /app/gpu_evidence_worker.py: the Python GPU-evidence
#     path (USE_NV_ATTESTATION_SDK unset or false) runs `python3
#     /app/gpu_evidence_worker.py`, which imports verifier.cc_admin from
#     nv-local-gpu-verifier and reads the GPUs through NVML via nvidia-ml-py.
#     pip (python3-pip) is only used at build time to install those packages.
#   * libnvat (ENABLE_NV_ATTESTATION_SDK=1): NVIDIA's attestation SDK, linked
#     by the binary for the in-process evidence path
#     (USE_NV_ATTESTATION_SDK=true). libnvat.so.1 links libxml2 and carries
#     its own copies of curl, OpenSSL and xmlsec; its package still depends
#     on libcurl4 and libxmlsec1-openssl, so all three are listed explicitly.
#   * ca-certificates: the CA bundle (/etc/ssl/certs/ca-certificates.crt)
#     that outbound TLS verifies against (cloud-api, image-URL fetches).
# Nothing in the image uses CUDA. NVML (libnvidia-ml.so.1) and nvidia-smi
# belong to the host driver: the NVIDIA container runtime mounts them into
# the container (the proxy runs with `runtime: nvidia`), as selected by
# NVIDIA_VISIBLE_DEVICES and NVIDIA_DRIVER_CAPABILITIES in the runtime stage.
#
# NVIDIA inputs: cuda-keyring 1.1-1 (CUDA_KEYRING_SHA256), libnvat
# LIBNVAT_VERSION from NVIDIA's ubuntu2204 apt repo; nv-attestation-sdk
# NV_ATTESTATION_SDK_VERSION and nv-ppcie-verifier NV_PPCIE_VERIFIER_VERSION
# from PyPI, which pull in nv-local-gpu-verifier and nvidia-ml-py at the
# versions pinned in attestation-constraints.txt.
#
# Hermetic inputs
# ───────────────
# The release binary contains natively compiled code (aws-lc-sys, ring,
# secp256k1-sys, bindgen output via libclang, OpenSSL linkage), so a newer
# gcc, binutils, glibc or libclang in the builder changes its bytes, and any
# package that moves in the runtime stage changes the image. Neither stage
# installs anything from a moving source:
#   * Base image: ubuntu:22.04, pinned by digest, for both stages.
#   * Ubuntu packages come from the Ubuntu snapshot archive frozen at
#     UBUNTU_SNAPSHOT (https://snapshot.ubuntu.com/ubuntu/<timestamp>), and
#     each installed package is held at the exact version listed in
#     pinned-packages-builder.txt (builder) or pinned-packages-runtime.txt
#     (runtime), with apt Pin-Priority 1001. The lists the build actually
#     resolved ship in the image at /app/pinned-packages-builder.txt and
#     /app/pinned-packages-runtime.txt and must match the committed files.
#     `LOAD_IMAGE=1 bash build-image.sh` also writes them to
#     pinned-packages-{builder,runtime}.resolved.txt.
#   * The snapshot archive is https-only and ubuntu:22.04 ships no CA
#     certificates, so the throwaway `ca-bootstrap` stage installs
#     ca-certificates from the regular archive. Both stages use only that CA
#     bundle, and only for TLS to the snapshot archive (and, in the runtime
#     stage, NVIDIA's repo); they install their own ca-certificates from the
#     snapshot like every other package.
#   * Rust: rustup-init RUSTUP_VERSION, checked against RUSTUP_INIT_SHA256,
#     installs toolchain 1.93.0. rustup verifies the toolchain against the
#     release's channel manifest, and published releases never change.
#   * NVIDIA (ENABLE_NV_ATTESTATION_SDK=1 only): the cuda-keyring .deb is
#     checked against CUDA_KEYRING_SHA256 (by the builder, and by BuildKit
#     in the throwaway `cuda-keyring` stage the runtime stage uses), and
#     libnvat/libnvat-dev are pinned to LIBNVAT_VERSION. Residual risk:
#     NVIDIA's apt repo has no snapshot service. It keeps old versions today;
#     if it ever dropped LIBNVAT_VERSION, older commits would stop building
#     (loudly: apt cannot silently pick another version).
#   * Python packages: pip installs wheels only (--only-binary=:all:), at the
#     versions in attestation-constraints.txt, and the build fails unless
#     that file lists exactly the installed set. The one exception is
#     timeout-decorator, a pure-Python dependency of nv-ppcie-verifier that
#     has never published a wheel: pip installs it from its sdist with
#     Ubuntu's (pinned) setuptools, no build isolation and no compiler.
# Network access needed at build time: the snapshot and regular Ubuntu
# archives, static.rust-lang.org, crates.io, NVIDIA's repo and PyPI.
#
# How to bump (each bump changes the image digest, which is expected):
#   * UBUNTU_SNAPSHOT: pick a timestamp in the past (YYYYMMDDTHHMMSSZ). The
#     service also answers for future timestamps, whose content is not frozen
#     yet. Check that dists/{jammy,jammy-updates,jammy-security}/InRelease
#     exist under it, then regenerate both pin files.
#   * pinned-packages-builder.txt / pinned-packages-runtime.txt: empty the
#     file(s) to regenerate (`: > pinned-packages-runtime.txt`), run
#     `LOAD_IMAGE=1 ENABLE_NV_ATTESTATION_SDK=1 bash build-image.sh`, copy
#     pinned-packages-{builder,runtime}.resolved.txt over them, rebuild the
#     same way, and check that each .resolved.txt equals its committed file.
#     (The same lists ship in the image: `docker load -i oci.tar`, then
#     `docker run --rm --entrypoint cat <image id> /app/pinned-packages-runtime.txt`.)
#     Always generate them with ENABLE_NV_ATTESTATION_SDK=1: that package set
#     is a superset of the ENABLE_NV_ATTESTATION_SDK=0 one. Regenerate after
#     any change to UBUNTU_SNAPSHOT, LIBNVAT_VERSION, the base image or the
#     apt install lists.
#   * RUSTUP_VERSION / RUSTUP_INIT_SHA256: the version is in
#     https://static.rust-lang.org/rustup/release-stable.toml. Download
#     https://static.rust-lang.org/rustup/archive/<version>/x86_64-unknown-linux-gnu/rustup-init,
#     hash it yourself and compare with the published rustup-init.sha256
#     next to it.
#   * CUDA_KEYRING_SHA256: sha256sum of the cuda-keyring .deb.
#   * LIBNVAT_VERSION: a version listed in the NVIDIA repo index
#     (https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz);
#     the builder installs the matching libnvat-dev. Then regenerate both pin
#     files and run the CVM smoke test in docs/testing-on-cvm.md.
#   * NV_ATTESTATION_SDK_VERSION / NV_PPCIE_VERIFIER_VERSION: regenerate
#     attestation-constraints.txt (recipe in its header) and run the CVM
#     smoke test.
#
# Build args:
#   ENABLE_NV_ATTESTATION_SDK=1  → build with the nv-attestation-sdk Cargo
#       feature, link against libnvat.so for direct-FFI GPU evidence
#       collection (no Python subprocess). The runtime image is also
#       provisioned with libnvat.so. Default off for local builds; CI and
#       external verifiers build with 1. Even at "1" the runtime path stays
#       Python-backed unless the env var USE_NV_ATTESTATION_SDK=true is set
#       on the container.
#   LIBNVAT_VERSION  → exact apt-pinned version of NVIDIA's libnvat package.
#       The trailing ".<timestamp>-1" suffix is part of the upstream version
#       string and changes per build; pin it so rebuilds stay reproducible.
#   NV_ATTESTATION_SDK_VERSION / NV_PPCIE_VERIFIER_VERSION → exact PyPI
#       versions of the GPU-attestation Python packages installed in the
#       runtime image; attestation-constraints.txt pins their whole closure.
#   UBUNTU_SNAPSHOT / RUSTUP_VERSION / RUSTUP_INIT_SHA256 /
#   CUDA_KEYRING_SHA256 → hermetic inputs, see above.
ARG ENABLE_NV_ATTESTATION_SDK=0
ARG LIBNVAT_VERSION=1.2.1.1777487608-1
ARG NV_ATTESTATION_SDK_VERSION=2.7.3
ARG NV_PPCIE_VERIFIER_VERSION=2.0.0
ARG UBUNTU_SNAPSHOT=20260924T000000Z
ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_INIT_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71
ARG CUDA_KEYRING_SHA256=d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df

# ─────────────────────────────────────────────────────────────────────
# Stage 0: CA bundle for reaching the snapshot archive (throwaway)
#
# Only /etc/ssl/certs/ca-certificates.crt is used, bind-mounted into the
# first apt step of the builder and of the runtime stage. Whatever versions
# the regular archive serves here never reach the builder or the image.
# ─────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04@sha256:4f838adc7181d9039ac795a7d0aba05a9bd9ecd480d294483169c5def983b64d AS ca-bootstrap
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────
# Stage 0b: NVIDIA cuda-keyring .deb for the runtime stage (throwaway)
#
# BuildKit downloads it and fails the build unless its sha256 equals
# CUDA_KEYRING_SHA256, the same value the builder checks. The runtime stage
# bind-mounts the verified file for `dpkg -i`, so the image needs neither
# wget nor gnupg.
# ─────────────────────────────────────────────────────────────────────
FROM scratch AS cuda-keyring
ARG CUDA_KEYRING_SHA256
ADD --checksum=sha256:${CUDA_KEYRING_SHA256} \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    /cuda-keyring_1.1-1_all.deb

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
ARG UBUNTU_SNAPSHOT
ARG RUSTUP_VERSION
ARG RUSTUP_INIT_SHA256
ARG CUDA_KEYRING_SHA256

ENV DEBIAN_FRONTEND=noninteractive

# Point apt at the Ubuntu snapshot archive and hold every package at the
# version committed in pinned-packages-builder.txt (see the header). Both
# apt calls here reach the https snapshot with the bootstrap CA bundle; the
# ca-certificates installed below (from the snapshot) serves later steps.
RUN --mount=type=bind,source=pinned-packages-builder.txt,target=/run/pinned-packages-builder.txt \
    --mount=type=bind,from=ca-bootstrap,source=/etc/ssl/certs/ca-certificates.crt,target=/run/bootstrap-ca.crt \
    set -e; \
    echo "deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT} jammy main restricted universe multiverse" > /etc/apt/sources.list; \
    echo "deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT} jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list; \
    echo "deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT} jammy-security main restricted universe multiverse" >> /etc/apt/sources.list; \
    rm -f /etc/apt/sources.list.d/*; \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/10no-check-valid-until; \
    while read -r line || [ -n "$line" ]; do \
        pkg="${line%%=*}"; ver="${line#*=}"; \
        if [ -n "$pkg" ] && [ -n "$ver" ] && [ "$pkg" != "$line" ]; then \
            printf 'Package: %s\nPin: version %s\nPin-Priority: 1001\n\n' "$pkg" "$ver"; \
        fi; \
    done < /run/pinned-packages-builder.txt > /etc/apt/preferences.d/pinned-packages; \
    apt-get -o Acquire::https::CAInfo=/run/bootstrap-ca.crt update; \
    apt-get -o Acquire::https::CAInfo=/run/bootstrap-ca.crt install -y --no-install-recommends \
        ca-certificates curl git pkg-config build-essential gcc \
        libssl-dev; \
    rm -rf /var/lib/apt/lists/*

# Install Rust 1.93.0 (matching the previous rust:1.93.0-bookworm base)
# with a pinned, checksum-verified rustup-init rather than the moving
# https://sh.rustup.rs script. x86_64 only: the image is built for
# linux/amd64.
RUN curl --proto '=https' --tlsv1.2 -sSf -o /tmp/rustup-init \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
    && echo "${RUSTUP_INIT_SHA256}  /tmp/rustup-init" | sha256sum -c - \
    && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --default-toolchain 1.93.0 --profile minimal --no-modify-path \
    && rm /tmp/rustup-init
ENV PATH=/root/.cargo/bin:$PATH

# Install libnvat-dev (headers + .so symlink) and libclang for bindgen
# only when the SDK feature is on. -dev pulls in libnvat (the runtime
# .so) as a versioned dependency, plus libcurl4/libxml2/libxmlsec1-openssl
# which libnvat dynamically links against. Ubuntu packages still come from
# the snapshot, and the pins above also cover these packages.
RUN if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        set -e && \
        apt-get update && apt-get install -y --no-install-recommends wget gnupg && \
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
        echo "${CUDA_KEYRING_SHA256}  cuda-keyring_1.1-1_all.deb" | sha256sum -c - && \
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

# Record the exact package set this stage resolved; the runtime stage
# copies it to /app/pinned-packages-builder.txt.
RUN dpkg -l | awk '/^ii/{print $2"="$3}' | sort > /build/pinned-packages-builder.txt

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
# Stage 2: Runtime image (see "Runtime image" in the header)
# ─────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04@sha256:4f838adc7181d9039ac795a7d0aba05a9bd9ecd480d294483169c5def983b64d AS runtime
ARG ENABLE_NV_ATTESTATION_SDK
ARG LIBNVAT_VERSION
ARG NV_ATTESTATION_SDK_VERSION
ARG NV_PPCIE_VERIFIER_VERSION
ARG UBUNTU_SNAPSHOT

ENV PYTHONUNBUFFERED=1
# Ask the NVIDIA container runtime for every GPU and for the driver's
# compute + utility libraries (NVML, nvidia-smi) — the values the image had
# when it derived from a CUDA base image. The compose files also set
# NVIDIA_VISIBLE_DEVICES=all.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Ubuntu packages come from the snapshot archive, held at the versions in
# pinned-packages-runtime.txt (same sources, CA bootstrap and pin mechanism
# as the builder, see the header). With the SDK feature on, the verified
# cuda-keyring adds NVIDIA's repo for libnvat.
#
# Reproducibility: apt/dpkg write per-line timestamps to their logs and
# ldconfig writes a non-deterministic aux-cache, all of which get baked into
# this layer and make the image digest change on every build. Remove them in
# the same RUN (rewrite-timestamp only normalizes tar mtimes, not file bytes).
# Clean the whole /var/log/apt dir, not just *.log: apt also writes
# eipp.log.xz (the EIPP solver log), whose embedded APT-IDs drift as the
# package indices change between build days — a *.log glob misses the .xz and
# leaves a cross-day-non-reproducible layer that the same-day reproducible-build
# double-build cannot catch. Also remove every __pycache__: the python3
# packages byte-compile themselves at install time, and marshal's output
# depends on interpreter state (reference counts), not just on the source, so
# the same code can yield different .pyc bytes. Python compiles on import.
RUN --mount=type=bind,source=pinned-packages-runtime.txt,target=/run/pinned-packages-runtime.txt \
    --mount=type=bind,from=ca-bootstrap,source=/etc/ssl/certs/ca-certificates.crt,target=/run/bootstrap-ca.crt \
    --mount=type=bind,from=cuda-keyring,source=/cuda-keyring_1.1-1_all.deb,target=/run/cuda-keyring.deb \
    set -e; \
    echo "deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT} jammy main restricted universe multiverse" > /etc/apt/sources.list; \
    echo "deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT} jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list; \
    echo "deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT} jammy-security main restricted universe multiverse" >> /etc/apt/sources.list; \
    rm -f /etc/apt/sources.list.d/*; \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/10no-check-valid-until; \
    while read -r line || [ -n "$line" ]; do \
        pkg="${line%%=*}"; ver="${line#*=}"; \
        if [ -n "$pkg" ] && [ -n "$ver" ] && [ "$pkg" != "$line" ]; then \
            printf 'Package: %s\nPin: version %s\nPin-Priority: 1001\n\n' "$pkg" "$ver"; \
        fi; \
    done < /run/pinned-packages-runtime.txt > /etc/apt/preferences.d/pinned-packages; \
    nvat=""; \
    if [ "$ENABLE_NV_ATTESTATION_SDK" = "1" ]; then \
        dpkg -i /run/cuda-keyring.deb; \
        nvat="libnvat=${LIBNVAT_VERSION}"; \
    fi; \
    apt-get -o Acquire::https::CAInfo=/run/bootstrap-ca.crt update; \
    DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::https::CAInfo=/run/bootstrap-ca.crt \
        install -y --no-install-recommends \
        python3 python3-pip ca-certificates \
        libcurl4 libxml2 libxmlsec1-openssl $nvat; \
    ldconfig; \
    rm -rf /var/lib/apt/lists/* \
           /var/log/apt/* /var/log/dpkg.log /var/log/alternatives.log \
           /var/cache/ldconfig/aux-cache; \
    find /usr -name __pycache__ -type d -prune -exec rm -rf {} +; \
    mkdir -p /app; \
    dpkg -l | awk '/^ii/{print $2"="$3}' | sort > /app/pinned-packages-runtime.txt

# GPU-attestation Python packages. nv-attestation-sdk provides the `verifier`
# module the evidence worker imports (via nv-local-gpu-verifier);
# nv-ppcie-verifier is additionally needed for PPCIE multi-GPU systems. They
# are installed even with ENABLE_NV_ATTESTATION_SDK=1 for the Python path
# (USE_NV_ATTESTATION_SDK=false at runtime).
#   --only-binary=:all: --no-binary=timeout-decorator: wheels only, so nothing
#     is compiled here; timeout-decorator has no wheel (see the header). Keep
#     this order: a later --only-binary=:all: would reset the exception.
#   --no-build-isolation: build timeout-decorator with Ubuntu's pinned
#     setuptools instead of whatever setuptools PyPI serves that day.
#   --no-compile / PYTHONDONTWRITEBYTECODE: no .pyc files. CPython stamps each
#     .pyc with the source mtime (the install time) and pip's .py files get
#     the wall-clock mtime, so .pyc bytes change on every build;
#     rewrite-timestamp normalizes tar mtimes but not the bytes inside a .pyc.
#   -c attestation-constraints.txt: pins the whole closure (the NVIDIA
#     packages pin their own dependencies only loosely). Bind-mounted, so it
#     adds no layer. The check after the install fails the build if pip
#     installed anything the file does not list, or a different version; it
#     is skipped while the file has no pins (regeneration, see its header).
RUN --mount=type=bind,source=attestation-constraints.txt,target=/run/attestation-constraints.txt \
    set -e; \
    export PYTHONDONTWRITEBYTECODE=1; \
    python3 -m pip install --no-cache-dir --no-compile --disable-pip-version-check \
        --no-build-isolation --only-binary=:all: --no-binary=timeout-decorator \
        -c /run/attestation-constraints.txt \
        "nv-attestation-sdk==${NV_ATTESTATION_SDK_VERSION}" \
        "nv-ppcie-verifier==${NV_PPCIE_VERIFIER_VERSION}"; \
    grep -vE '^[[:space:]]*(#|$)' /run/attestation-constraints.txt | sort -f > /tmp/pinned; \
    python3 -m pip list --disable-pip-version-check --format=freeze \
        --path /usr/local/lib/python3.10/dist-packages \
        | grep -viE '^(nv-attestation-sdk|nv-ppcie-verifier)==' | sort -f > /tmp/installed; \
    if [ -s /tmp/pinned ] && ! diff -u /tmp/pinned /tmp/installed; then \
        echo "attestation-constraints.txt does not match the installed packages:" >&2; \
        cat /tmp/installed >&2; \
        exit 1; \
    fi; \
    rm -f /tmp/pinned /tmp/installed

WORKDIR /app

# Copy compiled binary and GPU evidence worker from builder. Explicit modes
# keep the layers independent of the checkout's umask (a clone made with
# umask 0002 would otherwise ship gpu_evidence_worker.py as 0664 and change
# the digest).
COPY --from=builder --chmod=755 /build/target/release/vllm-proxy-rs /app/vllm-proxy-rs
COPY --from=builder --chmod=644 /build/pinned-packages-builder.txt /app/pinned-packages-builder.txt
COPY --chmod=644 gpu_evidence_worker.py /app/gpu_evidence_worker.py

# Bake in git revision for version tracking
COPY --chmod=664 .GIT_REV /etc/

ENV LISTEN_PORT=8000
EXPOSE 8000

ENTRYPOINT ["/app/vllm-proxy-rs"]
