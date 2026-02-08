IMAGE_NAME ?= vllm-proxy-rs
PLATFORM ?= linux/amd64
BUILDKIT_VERSION ?= v0.20.2
BUILDER_NAME ?= buildkit_20

.PHONY: build push clean

build:
	@for cmd in docker skopeo jq git; do \
		command -v $$cmd >/dev/null 2>&1 || { echo "Error: required command '$$cmd' not found"; exit 1; }; \
	done
	@if ! docker buildx inspect $(BUILDER_NAME) >/dev/null 2>&1; then \
		docker buildx create --use --driver-opt image=moby/buildkit:$(BUILDKIT_VERSION) --name $(BUILDER_NAME); \
	fi
	git rev-parse HEAD > .GIT_REV
	docker buildx build --builder $(BUILDER_NAME) --no-cache \
		--platform $(PLATFORM) \
		--build-arg SOURCE_DATE_EPOCH="0" \
		--output type=oci,dest=./oci.tar,rewrite-timestamp=true \
		--output type=docker,name="$(IMAGE_NAME):latest",rewrite-timestamp=true \
		. || { rm -f .GIT_REV; exit 1; }
	@echo ""
	@echo "Build completed, manifest digest:"
	@echo ""
	@skopeo inspect oci-archive:./oci.tar | jq .Digest
	@echo ""
	rm -f .GIT_REV

push:
ifndef REPO
	$(error REPO is required. Usage: make push REPO=myregistry/vllm-proxy-rs:tag)
endif
	@for cmd in docker skopeo jq git; do \
		command -v $$cmd >/dev/null 2>&1 || { echo "Error: required command '$$cmd' not found"; exit 1; }; \
	done
	@if ! docker buildx inspect $(BUILDER_NAME) >/dev/null 2>&1; then \
		docker buildx create --use --driver-opt image=moby/buildkit:$(BUILDKIT_VERSION) --name $(BUILDER_NAME); \
	fi
	git rev-parse HEAD > .GIT_REV
	docker buildx build --builder $(BUILDER_NAME) --no-cache \
		--platform $(PLATFORM) \
		--build-arg SOURCE_DATE_EPOCH="0" \
		--output type=oci,dest=./oci.tar,rewrite-timestamp=true \
		. || { rm -f .GIT_REV; exit 1; }
	@echo ""
	@echo "Build completed, manifest digest:"
	@echo ""
	@skopeo inspect oci-archive:./oci.tar | jq .Digest
	@echo ""
	skopeo copy --insecure-policy oci-archive:./oci.tar docker://$(REPO)
	@echo "Image pushed to $(REPO)"
	rm -f .GIT_REV

clean:
	rm -f .GIT_REV oci.tar
