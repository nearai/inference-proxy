#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Reproduce intermittent 401 from inference-proxy `sk-` key validation.

Inference-proxy validates `sk-...` Bearer tokens by calling
`POST {CLOUD_API_URL}/v1/check_api_key` on every request. There is no cache.
When that call hits a transport error (typical from inside a CVM: QEMU SLIRP
NAT + dstack ingress flakes), the proxy returns 401 to the client with body:

    {"error":{"code":null,"message":"Invalid or missing Authorization header",
              "param":null,"type":"unauthorized"}}

This script hammers an inference-proxy endpoint with concurrent requests using
the same auth pattern as the Datadog synthetic test. It distinguishes:

  * 200          — request succeeded
  * 401 (auth)   — the bug we're after; specifically the "Invalid or missing
                   Authorization header" body that inference-proxy emits when
                   `check_api_key` fails (transport error or non-2xx)
  * other 4xx/5xx — model-proxy / nginx / vLLM / SGLang errors
  * transport    — connection refused / TLS error / read timeout from us → proxy

Use it to:
  * Confirm the bug reproduces against current prod (expect non-zero 401 rate)
  * Confirm the retry fix mitigates it once deployed (expect 0% 401)

Usage:
    # Default: GLM-5, the synthetic test's API key, 200 reqs, 10-way concurrent
    uv run scripts/repro_intermittent_auth_401.py

    # Sustained run against another model
    uv run scripts/repro_intermittent_auth_401.py \\
        --url https://qwen35-122b.completions.near.ai/v1/chat/completions \\
        --model Qwen/Qwen3.5-122B-A10B \\
        --total 1000 --concurrency 20

    # Use your own key
    API_KEY=sk-... uv run scripts/repro_intermittent_auth_401.py

Exits non-zero if any 401 with the inference-proxy auth error body is seen,
so it can be used as a one-shot probe in cron/CI.
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass

import httpx

# Synthetic test's API key (from Datadog synthetic mjg-788-p48).
DEFAULT_API_KEY = "sk-67acfa4d689b4e94a0edb0087e043a9c"
DEFAULT_URL = "https://glm-5.completions.near.ai/v1/chat/completions"
DEFAULT_MODEL = "zai-org/GLM-5-FP8"

# Body that inference-proxy returns when `check_api_key` fails or the header
# is missing/invalid. Distinct from cloud-api / nginx / vLLM 401 bodies.
INFERENCE_PROXY_AUTH_MESSAGE = "Invalid or missing Authorization header"


@dataclass
class Result:
    status: int  # HTTP status, or 0 for transport error
    latency_s: float
    body_snippet: str
    is_proxy_auth_failure: bool  # 401 with inference-proxy's auth body
    error: str | None  # transport-error string, if any


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> Result:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        return Result(
            status=0,
            latency_s=time.perf_counter() - t0,
            body_snippet="",
            is_proxy_auth_failure=False,
            error=f"{type(e).__name__}: {e}",
        )
    elapsed = time.perf_counter() - t0
    text = r.text or ""
    is_proxy_auth = r.status_code == 401 and INFERENCE_PROXY_AUTH_MESSAGE in text
    return Result(
        status=r.status_code,
        latency_s=elapsed,
        body_snippet=text[:300],
        is_proxy_auth_failure=is_proxy_auth,
        error=None,
    )


async def run(args: argparse.Namespace) -> int:
    sem = asyncio.Semaphore(args.concurrency)
    results: list[Result] = []
    started = 0
    completed = 0
    completed_lock = asyncio.Lock()
    progress_every = max(1, args.total // 20)

    # Fresh connections per request best mirrors the synthetic test (which
    # opens a new TCP connection per probe). Disabling http/2 keepalives also
    # exercises the proxy's per-request auth path more aggressively.
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=0,
    )
    async with httpx.AsyncClient(http2=False, limits=limits, verify=True) as client:

        async def task() -> None:
            nonlocal completed
            async with sem:
                r = await one_request(
                    client,
                    args.url,
                    args.api_key,
                    args.model,
                    args.prompt,
                    args.max_tokens,
                    args.request_timeout,
                )
                results.append(r)
                async with completed_lock:
                    completed += 1
                    if completed % progress_every == 0 or completed == args.total:
                        rate = completed / max(time.perf_counter() - t_start, 1e-6)
                        proxy401 = sum(1 for x in results if x.is_proxy_auth_failure)
                        print(
                            f"  [{completed:>5}/{args.total}]  "
                            f"{rate:5.1f} req/s  "
                            f"proxy-401 so far: {proxy401}",
                            flush=True,
                        )

        t_start = time.perf_counter()
        coros = []
        while started < args.total:
            coros.append(asyncio.create_task(task()))
            started += 1
            # Spread starts so we don't open all connections at the very
            # first instant (more realistic load shape).
            if args.start_delay_ms:
                await asyncio.sleep(args.start_delay_ms / 1000.0)
        await asyncio.gather(*coros)
        wall = time.perf_counter() - t_start

    return summarize(results, wall, args)


def summarize(results: list[Result], wall_s: float, args: argparse.Namespace) -> int:
    n = len(results)
    statuses: Counter[str] = Counter()
    latencies = [r.latency_s for r in results]
    proxy_401 = [r for r in results if r.is_proxy_auth_failure]
    transport_errors = [r for r in results if r.error]

    for r in results:
        if r.error:
            statuses[f"transport: {r.error.split(':',1)[0]}"] += 1
        else:
            tag = (
                "200"
                if r.status == 200
                else (
                    "401 (proxy auth bug)"
                    if r.is_proxy_auth_failure
                    else f"{r.status}"
                )
            )
            statuses[tag] += 1

    print()
    print("=" * 70)
    print(f"URL:          {args.url}")
    print(f"Model:        {args.model}")
    print(f"Total:        {n}    Concurrency: {args.concurrency}    Wall: {wall_s:.1f}s")
    print(f"Throughput:   {n / wall_s:.1f} req/s")
    print()
    print("Status distribution:")
    for label, count in statuses.most_common():
        print(f"  {label:<40} {count:>6}  ({100*count/n:5.1f}%)")
    print()

    if latencies:
        ls = sorted(latencies)

        def pct(p: float) -> float:
            return ls[min(int(p * len(ls)), len(ls) - 1)]

        print(
            f"Latency (s):  p50={pct(0.50):.3f}  p95={pct(0.95):.3f}  "
            f"p99={pct(0.99):.3f}  max={max(ls):.3f}  "
            f"mean={statistics.mean(ls):.3f}"
        )
        print()

    if proxy_401:
        print(
            f"REPRODUCED: {len(proxy_401)}/{n} ({100*len(proxy_401)/n:.2f}%) "
            f"requests got 401 with inference-proxy's auth body."
        )
        sample = proxy_401[0]
        print(f"  sample latency: {sample.latency_s:.3f}s")
        print(f"  sample body:    {sample.body_snippet}")
        print()

    if transport_errors:
        kinds = Counter(e.error.split(":", 1)[0] for e in transport_errors)
        print(f"Transport errors: {len(transport_errors)} total")
        for kind, count in kinds.most_common():
            print(f"  {kind:<40} {count:>6}")
        print()

    if proxy_401:
        print(
            "Interpretation: this is the inference-proxy bug — a transient "
            "transport error or 5xx from cloud-api on /v1/check_api_key was "
            "translated to a 401 to the client. The retry-with-backoff fix "
            "should drive this to ~0%."
        )
        return 1
    elif transport_errors:
        print(
            "No proxy-auth 401s, but transport errors from this client → proxy. "
            "Those are different (network between you and the proxy)."
        )
        return 2
    else:
        print("No 401s observed in this run. Either the fix is deployed, the rate "
              "is below the sample size, or the upstream blip didn't happen.")
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help=f"chat-completions URL (default: {DEFAULT_URL})")
    p.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", DEFAULT_API_KEY),
        help="Bearer token (or env API_KEY). Defaults to the Datadog synthetic's key.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"model id (default: {DEFAULT_MODEL})")
    p.add_argument("--prompt", default="Hi there", help="user prompt")
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--total", type=int, default=200, help="total requests to send")
    p.add_argument("--concurrency", type=int, default=10, help="max concurrent in-flight")
    p.add_argument("--start-delay-ms", type=float, default=10.0, help="stagger start (ms)")
    p.add_argument("--request-timeout", type=float, default=60.0, help="per-request timeout (s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Hammering {args.url} with {args.total} requests "
        f"(concurrency={args.concurrency}) ...",
        flush=True,
    )
    rc = asyncio.run(run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
