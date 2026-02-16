#!/bin/bash

# Reproducible Docker image build script.
# Matching pattern from sibling repos (tee-attestation-server, cloud-api, dstack-ingress-vpc).
#
# Usage:
#   ./build-image.sh                         # Build only
#   ./build-image.sh --push registry/repo:tag  # Build and push

set -euo pipefail

PUSH=false
REPO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            REPO="$2"
            if [ -z "$REPO" ]; then
                echo "Error: --push requires a repository argument"
                echo "Usage: $0 [--push <repo>[:<tag>]]"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Usage: $0 [--push <repo>[:<tag>]]"
            exit 1
            ;;
    esac
done

for cmd in docker skopeo jq git; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "Error: required command '$cmd' not found"; exit 1; }
done

# Check if buildkit_20 already exists before creating it
if ! docker buildx inspect buildkit_20 &>/dev/null; then
    docker buildx create --use --driver-opt image=moby/buildkit:v0.20.2 --name buildkit_20
fi

git rev-parse HEAD > .GIT_REV
TEMP_TAG="vllm-proxy-rs-temp:$(date +%s)"

# When pushing, only produce the OCI archive (saves disk space in CI).
# For local builds, also load into Docker daemon for convenience.
BUILD_OUTPUTS=(--output "type=oci,dest=./oci.tar,rewrite-timestamp=true")
if [ "$PUSH" = false ]; then
    BUILD_OUTPUTS+=(--output "type=docker,name=$TEMP_TAG,rewrite-timestamp=true")
fi

docker buildx build --builder buildkit_20 --no-cache --platform linux/amd64 \
    --build-arg SOURCE_DATE_EPOCH="0" \
    "${BUILD_OUTPUTS[@]}" .

if [ "$?" -ne 0 ]; then
    echo "Build failed"
    rm -f .GIT_REV
    exit 1
fi

echo ""
echo "Build completed, manifest digest:"
echo ""
skopeo inspect oci-archive:./oci.tar | jq .Digest
echo ""

if [ "$PUSH" = true ]; then
    echo "Pushing image to $REPO..."
    skopeo copy --insecure-policy oci-archive:./oci.tar docker://"$REPO"
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

# Clean up the temporary image from Docker daemon
docker rmi "$TEMP_TAG" 2>/dev/null || true

rm -f .GIT_REV
