# Testing inference-proxy on a real CVM

`cargo test` covers the unit + integration suites with wiremock-mocked
upstreams. Some changes — anything that touches NVML, dstack TDX, or
the SDK FFI — need a real CVM to validate. This doc is the recipe.

## When to do this

- Changes to `attestation.rs` GPU evidence dispatch
- Changes to the libnvat SDK call (`attestation_sdk.rs`) or its mutex
- New env vars that gate which evidence path is taken
- Anything that adds a new HTTP endpoint to a proxy-to-proxy contract
  (e.g. `/internal/gpu_evidence`)

If you're only changing pure-Rust logic with no FFI / dstack / NVIDIA
surface, `cargo test` is enough.

## Where to run

A spare GPU CVM with `USE_NV_ATTESTATION_SDK=true` and a working
`/var/run/dstack.sock`. As of 2026-05-08 that's any `gpu0X` host. Pick
one that isn't load-bearing — gpu07 is the usual canary. Tester is
responsible for not disturbing whatever production model is already
running on the host.

## Build a branch image

The `Build & Deploy` workflow only auto-fires on `main` and tags. For
branch testing, dispatch it manually:

```bash
gh workflow run build.yml --ref <branch> --repo nearai/inference-proxy
gh run list --workflow=build.yml --branch <branch> --repo nearai/inference-proxy --limit 1
```

It tags `:dev` (shared with all non-main branches — pin by digest in
your test compose, not by tag) and prints the digest in the run log
(`IMAGE_DIGEST: sha256:...`).

## 2-proxy delegate smoke test

Validates `GPU_EVIDENCE_DELEGATE_URL` end-to-end: one leader proxy
owns NVML, the other delegates. Created for [PR #122][pr122].

[pr122]: https://github.com/nearai/inference-proxy/pull/122

### Compose file

```yaml
# test-delegate.yaml
x-nvidia: &nvidia
  runtime: nvidia
  ipc: host
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

services:
  delegate-leader:
    <<: *nvidia
    image: ${PROXY_IMAGE}
    container_name: delegate-leader
    user: root
    privileged: true
    ports:
      - "127.0.0.1:18001:8000"   # CVM-loopback only, no host exposure
    volumes:
      - /var/run/dstack.sock:/var/run/dstack.sock
    environment:
      - MODEL_NAME=zai-org/GLM-5-FP8
      - TOKEN=${PROXY_TOKEN}
      - VLLM_BASE_URL=http://glm:8000
      - USE_NV_ATTESTATION_SDK=true
      - LOG_FORMAT=json
      - OPENAI_CHAT_COMPATIBILITY_CHECK=false   # don't gate on upstream
    restart: "no"

  delegate-follower:
    <<: *nvidia
    image: ${PROXY_IMAGE}
    container_name: delegate-follower
    user: root
    privileged: true
    ports:
      - "127.0.0.1:18002:8000"
    volumes:
      - /var/run/dstack.sock:/var/run/dstack.sock
    environment:
      - MODEL_NAME=zai-org/GLM-5-FP8
      - TOKEN=${PROXY_TOKEN}
      - VLLM_BASE_URL=http://glm:8000
      - USE_NV_ATTESTATION_SDK=true
      - GPU_EVIDENCE_DELEGATE_URL=http://delegate-leader:8000
      - LOG_FORMAT=json
      - OPENAI_CHAT_COMPATIBILITY_CHECK=false
    depends_on:
      - delegate-leader
    restart: "no"
```

`MODEL_NAME` is just a label here — neither proxy serves real
inference in this test, so set it to whatever the running model on
the host is so logs aren't confusing. `VLLM_BASE_URL` only matters if
you flip `OPENAI_CHAT_COMPATIBILITY_CHECK=true`.

### Deploy and probe

CVM access on gpu0X is via host jump: `ssh gpuNN` then
`ssh -p 10022 root@localhost`. The CVM's `/tmp` is writable; `/root`
is not.

```bash
# scp the file in (two-hop)
scp test-delegate.yaml gpu07:/tmp/
ssh gpu07 'scp -P 10022 /tmp/test-delegate.yaml root@localhost:/tmp/'

# run inside the CVM
ssh gpu07 'ssh -p 10022 root@localhost' <<'CVM'
mkdir -p /tmp/deltest && cd /tmp/deltest && mv /tmp/test-delegate.yaml .
PROXY_IMAGE='nearaidev/vllm-proxy-rs@sha256:<digest from build run>' \
PROXY_TOKEN=delegate-test-token-1234 \
docker compose -f test-delegate.yaml -p deltest up -d
CVM
```

### What to verify

```bash
# happy path — fresh nonce, leader up
NONCE=$(openssl rand -hex 32)
curl -w "code=%{http_code} t=%{time_total}\n" -o /tmp/r.json \
  "http://127.0.0.1:18002/v1/attestation/report?signing_algo=ed25519&nonce=$NONCE"
# expect: 200, ~290 KB body, request_nonce matches

# loop-guard / dependency proof — fresh nonce, leader DOWN
docker stop delegate-leader
NONCE=$(openssl rand -hex 32)
curl -w "code=%{http_code} t=%{time_total}\n" \
  "http://127.0.0.1:18002/v1/attestation/report?signing_algo=ed25519&nonce=$NONCE"
# expect: 500
# follower logs: "delegate request to http://delegate-leader:8000/internal/gpu_evidence failed"

# isolation check — leader's logs should have all libnvat output
docker logs delegate-leader 2>&1 | grep '\[nvat\]' | head    # many lines
docker logs delegate-follower 2>&1 | grep '\[nvat\]' | head  # zero lines
```

### Tear down

```bash
docker compose -f test-delegate.yaml -p deltest down -v
rm -rf /tmp/deltest
```

Then on the host: confirm `docker ps --filter name=delegate` is empty
and the production model (e.g. `glm51`, `qwen3-vl`) shows
`RestartCount=0` in `docker inspect`.

## CVM gotchas

- The dstack OS is busybox, not Ubuntu. `head -c`, `head -3` etc.
  don't work — use `dd if=… bs=N count=1` for byte-cap reads.
- `/root` is read-only at SSH level; use `/tmp/<subdir>` for any test
  artifacts.
- `python3` is absent — use `jq` for JSON inspection.
- `--gpus all` is fine for read-only NVML access; you don't need to
  unplug the running model. The whole point of PR #122 is that
  multiple proxies CAN share GPUs as long as only one talks to NVML.
- The `:dev` image tag is shared across all non-main branches. **Pin
  by digest** in test compose files so a parallel branch build can't
  swap the image under you.
