#!/bin/bash

# Reproducible Docker image build script.
#
# Usage:
#   ./build-image.sh                            # build ./oci.tar and print its digest
#   ./build-image.sh --push <repo>[:<tag>]      # build, then push ./oci.tar with skopeo
#
# Environment (all optional):
#   ENABLE_NV_ATTESTATION_SDK=1  Build with the nv-attestation-sdk Cargo feature, link
#                                libnvat and ship it in the runtime image. Default 0;
#                                the published images are built with 1.
#   SOURCE_DATE_EPOCH            Accepted for external verifiers, which set it to 0.
#                                The script always passes SOURCE_DATE_EPOCH=0 to the
#                                build itself.
#   LOAD_IMAGE=1                 Also load the image into the local Docker daemon and
#                                write the package lists it was built with to
#                                pinned-packages-*.resolved.txt (to compare with, or
#                                regenerate, the committed pin files). This exports
#                                the whole image a second time, so it is off by
#                                default. The digest is the same either way.
#   LOAD_IMAGE_TAG=<name>:<tag>  With LOAD_IMAGE=1, keep the loaded image under this
#                                tag. Otherwise it is removed when the script exits.
#
# Output: ./oci.tar, an OCI archive holding one image manifest. Its digest,
#   tar -xOf oci.tar index.json | jq -r '.manifests[0].digest'
# is the digest of the published image.
#
# External verifiers rebuild published images from a fresh clone of the source
# commit with exactly
#   ENABLE_NV_ATTESTATION_SDK=1 SOURCE_DATE_EPOCH=0 bash build-image.sh
# and compare that digest with the deployed one. Keep that command working with no
# other input: never make a new environment variable or argument required. See
# "Reproducible build & verification" in README.md.

set -euo pipefail

usage() {
    echo "Usage: $0 [--push <repo>[:<tag>]]" >&2
    exit 1
}

PUSH=false
REPO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            if [ -z "${2:-}" ]; then
                echo "Error: --push requires a repository argument" >&2
                usage
            fi
            PUSH=true
            REPO="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

: "${ENABLE_NV_ATTESTATION_SDK:=0}"
LOAD_IMAGE="${LOAD_IMAGE:-0}"
LOAD_IMAGE_TAG="${LOAD_IMAGE_TAG:-}"

# skopeo is only needed to push. The default path must not depend on it: some
# verifier environments replace it with a stub.
REQUIRED_COMMANDS=(docker git jq tar)
if [ "$PUSH" = true ]; then
    REQUIRED_COMMANDS+=(skopeo)
fi
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: required command '$cmd' not found" >&2; exit 1; }
done

if [ -n "$LOAD_IMAGE_TAG" ] && [ "$LOAD_IMAGE" != "1" ]; then
    echo "Note: LOAD_IMAGE_TAG is ignored without LOAD_IMAGE=1" >&2
fi

# BuildKit: a fresh builder, created for this invocation and pinned to one
# BuildKit release (by tag and digest). OCI layer serialization and compression
# differ between BuildKit versions, and a long-lived shared builder only honours
# its version pin when it is first created, so a host whose builder predates the
# pin silently produces different digests. A unique name also keeps concurrent
# builds on one Docker daemon apart. The builder is not made the default, and it
# is removed on exit together with its state.
BUILDKIT_IMAGE="moby/buildkit:v0.20.2@sha256:c457984bd29f04d6acc90c8d9e717afe3922ae14665f3187e0096976fe37b1c8"
# The creation time leads the unique suffix so that leftovers of a killed run can
# be recognised as stale.
RUN_ID="$(date +%s)-$$-${RANDOM}"
BUILDER_NAME="vllm-proxy-rs-build-${RUN_ID}"
TEMP_TAG="vllm-proxy-rs-temp:${RUN_ID}"
BUILDER_CREATED=false
IMAGE_LOADED=false

cleanup() {
    rm -f .GIT_REV
    if [ "$IMAGE_LOADED" = true ]; then
        docker image rm "$TEMP_TAG" >/dev/null 2>&1 || true
    fi
    if [ "$BUILDER_CREATED" = true ]; then
        docker buildx rm "$BUILDER_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker buildx create --driver docker-container \
    --driver-opt "image=${BUILDKIT_IMAGE}" --name "$BUILDER_NAME" >/dev/null
BUILDER_CREATED=true

git rev-parse HEAD > .GIT_REV

BUILD_OUTPUTS=(--output "type=oci,dest=./oci.tar,rewrite-timestamp=true")
if [ "$LOAD_IMAGE" = "1" ]; then
    rm -f pinned-packages-*.resolved.txt
    BUILD_OUTPUTS+=(--output "type=docker,name=${TEMP_TAG},rewrite-timestamp=true")
    IMAGE_LOADED=true
fi

if ! docker buildx build --builder "$BUILDER_NAME" --no-cache --platform linux/amd64 \
    --build-arg SOURCE_DATE_EPOCH="0" \
    --build-arg "ENABLE_NV_ATTESTATION_SDK=${ENABLE_NV_ATTESTATION_SDK}" \
    "${BUILD_OUTPUTS[@]}" .; then
    echo "Build failed" >&2
    exit 1
fi

# Read the digest straight from the archive's index.json, the way external
# verifiers do. (skopeo inspect oci-archive: would unpack the whole archive into a
# temporary directory first.)
MANIFEST_COUNT=$(tar -xOf ./oci.tar index.json | jq '.manifests | length')
if [ "$MANIFEST_COUNT" != "1" ]; then
    echo "Error: ./oci.tar holds $MANIFEST_COUNT manifests, expected exactly 1" >&2
    exit 1
fi
DIGEST=$(tar -xOf ./oci.tar index.json | jq -r '.manifests[0].digest')

echo ""
echo "Build completed, manifest digest:"
echo ""
echo "$DIGEST"
echo ""

if [ "$PUSH" = true ]; then
    echo "Pushing image to $REPO..."
    SKOPEO_AUTH=()
    if [ -f "$HOME/.docker/config.json" ]; then
        SKOPEO_AUTH=(--authfile "$HOME/.docker/config.json")
    fi
    skopeo copy --insecure-policy "${SKOPEO_AUTH[@]}" oci-archive:./oci.tar docker://"$REPO"
    echo "Image pushed successfully to $REPO"
else
    echo "To push the image to a registry, run:"
    echo ""
    echo "  $0 --push <repo>[:<tag>]"
    echo ""
    echo "Or use skopeo directly:"
    echo ""
    echo "  skopeo copy --insecure-policy oci-archive:./oci.tar docker://<repo>[:<tag>]"
    echo ""
fi

if [ "$LOAD_IMAGE" = "1" ]; then
    # Package lists the image was built with, for comparison with the committed
    # pinned-packages-*.txt (or to regenerate them; see README.md):
    #   builder: recorded by the builder stage, shipped at /app/pinned-packages-builder.txt
    #   runtime: /app/pinned-packages-runtime.txt when the image ships one, otherwise
    #            what dpkg reports in the final image
    # Plain runc: GPU hosts may default to a runtime that injects devices.
    RUN_IMAGE=(docker run --rm --pull=never --network=none --runtime=runc)
    if "${RUN_IMAGE[@]}" --entrypoint cat "$TEMP_TAG" /app/pinned-packages-builder.txt > pinned-packages-builder.resolved.txt; then
        echo "Builder package list written to pinned-packages-builder.resolved.txt ($(wc -l < pinned-packages-builder.resolved.txt) packages)"
    else
        echo "Warning: could not read /app/pinned-packages-builder.txt from the image" >&2
        rm -f pinned-packages-builder.resolved.txt
    fi
    # shellcheck disable=SC2016  # the awk program is expanded by the container's shell
    if "${RUN_IMAGE[@]}" --entrypoint cat "$TEMP_TAG" /app/pinned-packages-runtime.txt > pinned-packages-runtime.resolved.txt 2>/dev/null; then
        echo "Runtime package list written to pinned-packages-runtime.resolved.txt ($(wc -l < pinned-packages-runtime.resolved.txt) packages, from /app/pinned-packages-runtime.txt)"
    elif "${RUN_IMAGE[@]}" --entrypoint sh "$TEMP_TAG" -c \
        'dpkg -l | awk '\''/^ii/{print $2"="$3}'\'' | LC_ALL=C sort' > pinned-packages-runtime.resolved.txt; then
        echo "Runtime package list written to pinned-packages-runtime.resolved.txt ($(wc -l < pinned-packages-runtime.resolved.txt) packages, from dpkg -l)"
    else
        echo "Warning: could not list the runtime packages of the image" >&2
        rm -f pinned-packages-runtime.resolved.txt
    fi
    for resolved in pinned-packages-*.resolved.txt; do
        [ -f "$resolved" ] || continue
        committed="${resolved%.resolved.txt}.txt"
        if [ -f "$committed" ] && ! cmp -s "$committed" "$resolved"; then
            echo "Warning: $resolved differs from $committed" >&2
        fi
    done

    if [ -n "$LOAD_IMAGE_TAG" ]; then
        docker tag "$TEMP_TAG" "$LOAD_IMAGE_TAG"
        echo "Image loaded into the Docker daemon as $LOAD_IMAGE_TAG"
    fi
    echo ""
fi
