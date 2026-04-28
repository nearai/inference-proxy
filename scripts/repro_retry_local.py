#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "aiohttp"]
# ///
"""End-to-end local validation of the auth retry fix.

Spins up:
  1. A flaky mock cloud-api on localhost that returns 503 for the first N
     calls to /v1/check_api_key, then 200. (Toggle: --flake-503-count.)
  2. A trivial mock vLLM backend on localhost that returns a canned chat
     completion.
  3. The real `vllm-proxy-rs` binary, built from this checkout, configured
     to use both mocks.

Then hammers the proxy with `sk-...` requests and reports the success rate.

Run twice — once with retries disabled, once with retries enabled — to see
the fix in action:

    # Builds the binary if needed.
    uv run scripts/repro_retry_local.py --max-attempts 1   # expect 401s
    uv run scripts/repro_retry_local.py --max-attempts 3   # expect ~all 200s

Exits non-zero if observed behavior contradicts the expected outcome for the
chosen --max-attempts.
"""

import argparse
import asyncio
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from aiohttp import web

REPO_ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Mock cloud-api ────────────────────────────────────────────────────────────


class FlakyCloudApi:
    """Returns 503 for the first `flake_count` /v1/check_api_key calls, then 200."""

    def __init__(self, flake_count: int) -> None:
        self.flake_count = flake_count
        self.call_count = 0
        self.calls_503 = 0
        self.calls_200 = 0

    async def check_api_key(self, request: web.Request) -> web.Response:
        self.call_count += 1
        if self.call_count <= self.flake_count:
            self.calls_503 += 1
            return web.Response(status=503, text='{"error":"upstream blip"}')
        self.calls_200 += 1
        return web.json_response({"valid": True})

    async def usage(self, request: web.Request) -> web.Response:
        # Fire-and-forget usage reporting; just acknowledge.
        return web.Response(status=200)


def make_cloud_api_app(state: FlakyCloudApi) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/check_api_key", state.check_api_key)
    app.router.add_post("/v1/usage", state.usage)
    return app


# ── Mock vLLM backend ─────────────────────────────────────────────────────────


CANNED_COMPLETION = {
    "id": "chatcmpl-local",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


async def vllm_chat(_request: web.Request) -> web.Response:
    return web.json_response(CANNED_COMPLETION)


async def vllm_models(_request: web.Request) -> web.Response:
    return web.json_response(
        {"object": "list", "data": [{"id": "test-model", "object": "model"}]}
    )


async def vllm_health(_request: web.Request) -> web.Response:
    return web.Response(status=200)


def make_vllm_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", vllm_chat)
    app.router.add_get("/v1/models", vllm_models)
    app.router.add_get("/health", vllm_health)
    return app


# ── Inference-proxy lifecycle ─────────────────────────────────────────────────


async def build_binary() -> Path:
    """Always invoke cargo so source edits are picked up; incremental rebuilds
    are fast when nothing changed."""
    binary = REPO_ROOT / "target" / "release" / "vllm-proxy-rs"
    print("[*] Building vllm-proxy-rs (release, incremental) ...", flush=True)
    proc = await asyncio.create_subprocess_exec(
        "cargo",
        "build",
        "--release",
        "--bin",
        "vllm-proxy-rs",
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out_b, _ = await proc.communicate()
    if proc.returncode != 0:
        sys.stderr.write(out_b.decode(errors="replace"))
        raise SystemExit(f"cargo build failed (rc={proc.returncode})")
    if not binary.exists():
        raise SystemExit(f"binary not found at {binary} after build")
    return binary


def proxy_env(
    listen_port: int,
    vllm_port: int,
    cloud_api_port: int,
    max_attempts: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": "test-model",
            "TOKEN": "test-proxy-token",
            "VLLM_BASE_URL": f"http://127.0.0.1:{vllm_port}",
            "CLOUD_API_URL": f"http://127.0.0.1:{cloud_api_port}",
            "DEV": "1",
            "GPU_NO_HW_MODE": "1",
            "OHTTP_ENABLED": "0",
            "LISTEN_PORT": str(listen_port),
            "RUST_LOG": "vllm_proxy_rs=info,warn",
            "CLOUD_API_AUTH_MAX_ATTEMPTS": str(max_attempts),
            "CLOUD_API_AUTH_INITIAL_BACKOFF_MS": "5",
            "CLOUD_API_AUTH_TIMEOUT_SECS": "5",
        }
    )
    return env


async def wait_listening(port: int, deadline_s: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise SystemExit(f"port {port} never came up")


# ── Hammer ────────────────────────────────────────────────────────────────────


async def hammer(proxy_url: str, total: int, concurrency: int) -> tuple[int, int, int]:
    """Returns (n_200, n_proxy_401, n_other)."""
    sem = asyncio.Semaphore(concurrency)
    n_200 = n_401 = n_other = 0
    lock = asyncio.Lock()

    async with httpx.AsyncClient(http2=False) as client:

        async def one() -> None:
            nonlocal n_200, n_401, n_other
            async with sem:
                try:
                    r = await client.post(
                        f"{proxy_url}/v1/chat/completions",
                        headers={
                            "Authorization": "Bearer sk-local-test-key-0000000000",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 5,
                        },
                        timeout=30.0,
                    )
                except httpx.HTTPError as e:
                    async with lock:
                        n_other += 1
                    print(f"  transport: {type(e).__name__}: {e}", flush=True)
                    return
                async with lock:
                    if r.status_code == 200:
                        n_200 += 1
                    elif r.status_code == 401 and "Invalid or missing Authorization" in r.text:
                        n_401 += 1
                    else:
                        n_other += 1
                        if n_other <= 3:
                            print(f"  unexpected {r.status_code}: {r.text[:200]}", flush=True)

        await asyncio.gather(*[one() for _ in range(total)])

    return n_200, n_401, n_other


# ── Orchestration ─────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> int:
    cloud_state = FlakyCloudApi(flake_count=args.flake_503_count)

    cloud_port = free_port()
    vllm_port = free_port()
    proxy_port = free_port()

    print(f"[*] Mock cloud-api: 127.0.0.1:{cloud_port} (503 for first {args.flake_503_count} calls)")
    print(f"[*] Mock vLLM:      127.0.0.1:{vllm_port}")
    print(f"[*] Inference-proxy: 127.0.0.1:{proxy_port} "
          f"(CLOUD_API_AUTH_MAX_ATTEMPTS={args.max_attempts})")

    cloud_runner = web.AppRunner(make_cloud_api_app(cloud_state))
    vllm_runner = web.AppRunner(make_vllm_app())
    await cloud_runner.setup()
    await vllm_runner.setup()
    cloud_site = web.TCPSite(cloud_runner, "127.0.0.1", cloud_port)
    vllm_site = web.TCPSite(vllm_runner, "127.0.0.1", vllm_port)
    await cloud_site.start()
    await vllm_site.start()

    binary = await build_binary()

    env = proxy_env(proxy_port, vllm_port, cloud_port, args.max_attempts)
    proxy = subprocess.Popen(
        [str(binary)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        await wait_listening(proxy_port)
        # Brief warmup so initial logs flush.
        await asyncio.sleep(0.2)

        print(f"[*] Sending {args.total} requests at concurrency={args.concurrency} ...")
        n_200, n_401, n_other = await hammer(
            f"http://127.0.0.1:{proxy_port}", args.total, args.concurrency
        )
    finally:
        proxy.send_signal(signal.SIGTERM)
        try:
            proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill()
        await cloud_runner.cleanup()
        await vllm_runner.cleanup()

    print()
    print("=" * 70)
    print(f"Mock cloud-api stats: total calls={cloud_state.call_count}  "
          f"503s={cloud_state.calls_503}  200s={cloud_state.calls_200}")
    print(f"Proxy results: 200={n_200}  401(auth)={n_401}  other={n_other}")
    print()

    # Expected outcome for self-check.
    #
    # The mock cloud-api returns 503 for the first `flake_503_count` calls
    # globally, then 200 forever. Each client request makes up to
    # `max_attempts` calls before giving up, so at most `max_attempts` flakes
    # are absorbed by any single client request, but flakes are *consumed
    # across* requests too. With concurrency=1, the math is exact:
    #
    #   * If `flake <= max_attempts - 1`: every request retries until it
    #     succeeds. No client-visible 401s.
    #   * If `flake >= max_attempts`: the first request burns `max_attempts`
    #     flakes and 401s. The next request burns up to `max_attempts` more,
    #     etc., until the flake budget is depleted.
    if args.max_attempts == 1:
        # Each request gets one shot. First `flake` requests fail.
        expected_401 = min(args.flake_503_count, args.total)
        expected_200 = args.total - expected_401
        ok = n_401 == expected_401 and n_200 == expected_200 and n_other == 0
        if ok:
            print(
                f"PASS: with retries disabled (max_attempts=1), saw the "
                f"expected {expected_401} proxy-auth 401s and "
                f"{expected_200} successes. The bug reproduces."
            )
            return 0
        print(
            f"UNEXPECTED: expected 401={expected_401}, 200={expected_200}; "
            f"got 401={n_401}, 200={n_200}, other={n_other}"
        )
        return 1

    # max_attempts > 1
    if args.flake_503_count <= args.max_attempts - 1:
        # The first request alone absorbs all flakes and still succeeds.
        ok = n_401 == 0 and n_other == 0 and n_200 == args.total
        if ok:
            print(
                f"PASS: with retries enabled (max_attempts={args.max_attempts}), "
                f"all {args.total} requests succeeded despite "
                f"{cloud_state.calls_503} cloud-api 503s. Fix works."
            )
            return 0
        print(
            f"FAIL: expected 200={args.total}, 401=0; "
            f"got 200={n_200}, 401={n_401}, other={n_other}"
        )
        return 1

    # flake >= max_attempts: some early requests still fail because the
    # flake budget per request is smaller than total flakes. Compute exact
    # counts assuming concurrency=1 (sequential consumption).
    flakes_left = args.flake_503_count
    expected_401 = 0
    for _ in range(args.total):
        if flakes_left >= args.max_attempts:
            expected_401 += 1
            flakes_left -= args.max_attempts
        else:
            flakes_left = 0
            break
    expected_200 = args.total - expected_401
    ok = n_401 == expected_401 and n_200 == expected_200 and n_other == 0
    if ok:
        print(
            f"PASS: with max_attempts={args.max_attempts} and "
            f"flake_503_count={args.flake_503_count}, observed "
            f"{expected_401} 401s and {expected_200} successes — exactly the "
            f"deterministic outcome. Increase --max-attempts or lower "
            f"--flake-503-count to drive 401s to zero."
        )
        return 0
    print(
        f"UNEXPECTED: expected 401={expected_401}, 200={expected_200}; "
        f"got 401={n_401}, 200={n_200}, other={n_other}"
    )
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--max-attempts", type=int, default=3,
                   help="CLOUD_API_AUTH_MAX_ATTEMPTS for inference-proxy (1 = no retry)")
    p.add_argument("--flake-503-count", type=int, default=2,
                   help="how many initial cloud-api calls return 503 "
                        "(default 2; with default max_attempts=3 the first "
                        "request retries through them and succeeds)")
    p.add_argument("--total", type=int, default=10, help="total requests to send")
    p.add_argument("--concurrency", type=int, default=1,
                   help="concurrent in-flight requests; keep at 1 for deterministic counting")
    return p.parse_args()


def main() -> None:
    if shutil.which("cargo") is None:
        sys.exit("error: cargo not found in PATH")
    args = parse_args()
    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
