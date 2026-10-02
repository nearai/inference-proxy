IMAGE_NAME ?= vllm-proxy-rs

.PHONY: build push clean

# Both targets go through build-image.sh, the one build definition that CI and
# external verifiers use. Set ENABLE_NV_ATTESTATION_SDK=1 in the environment to
# build the published variant.

# Writes ./oci.tar, loads the image into the local Docker daemon as
# $(IMAGE_NAME):latest and writes pinned-packages-*.resolved.txt.
build:
	LOAD_IMAGE=1 LOAD_IMAGE_TAG="$(IMAGE_NAME):latest" ./build-image.sh

push:
ifndef REPO
	$(error REPO is required. Usage: make push REPO=myregistry/vllm-proxy-rs:tag)
endif
	./build-image.sh --push "$(REPO)"

clean:
	rm -f .GIT_REV oci.tar pinned-packages-*.resolved.txt
