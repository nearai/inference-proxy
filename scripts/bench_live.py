#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp"]
# ///
"""Benchmark attestation and completion endpoints on a live inference-proxy.

Usage:
    uv run scripts/bench_live.py https://glm-5-fp8.completions.near.ai
    uv run scripts/bench_live.py http://160.72.54.171:8000 --token secret123
    uv run scripts/bench_live.py http://160.72.54.171:8000 --concurrency 20 --duration 60

Tests:
    1. Attestation (no nonce)  — should hit cache after first call
    2. Attestation (with nonce) — forces fresh generation every time
    3. Chat completion (non-streaming, short)
    4. Chat completion (streaming, short)
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class Stats:
    name: str
    latencies: list[float] = field(default_factory=list)
    errors: int = 0
    status_codes: dict[int, int] = field(default_factory=dict)

    def record(self, latency: float, status: int):
        self.latencies.append(latency)
        self.status_codes[status] = self.status_codes.get(status, 0) + 1
        if status >= 400:
            self.errors += 1

    def report(self) -> str:
        if not self.latencies:
            return f"  {self.name}: no completed requests"
        n = len(self.latencies)
        ok = n - self.errors
        s = sorted(self.latencies)
        lines = [
            f"  {self.name}:",
            f"    requests:  {n} ({ok} ok, {self.errors} errors)",
            f"    latency:   p50={s[n//2]*1000:.0f}ms  p90={s[int(n*0.9)]*1000:.0f}ms  p99={s[int(n*0.99)]*1000:.0f}ms",
            f"    min/avg/max: {s[0]*1000:.0f}/{statistics.mean(s)*1000:.0f}/{s[-1]*1000:.0f} ms",
            f"    throughput: {n / (s[-1] - s[0] + 0.001):.1f} req/s (wall-clock)" if n > 1 else "",
            f"    status codes: {dict(sorted(self.status_codes.items()))}",
        ]
        return "\n".join(l for l in lines if l)


async def run_bench(
    session: aiohttp.ClientSession,
    stats: Stats,
    make_request,
    concurrency: int,
    duration: float,
):
    """Run a benchmark: spawn `concurrency` workers hitting `make_request` for `duration` seconds."""
    stop = asyncio.Event()

    async def worker():
        while not stop.is_set():
            try:
                t0 = time.monotonic()
                method, url, kwargs = make_request()
                async with session.request(method, url, **kwargs) as resp:
                    # Consume body to measure full latency
                    await resp.read()
                    elapsed = time.monotonic() - t0
                    stats.record(elapsed, resp.status)
            except asyncio.CancelledError:
                break
            except Exception as e:
                elapsed = time.monotonic() - t0
                stats.record(elapsed, 0)
                stats.errors += 1

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]

    await asyncio.sleep(duration)
    stop.set()

    # Give workers a moment to finish in-flight requests
    await asyncio.sleep(0.5)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark live inference-proxy endpoints")
    parser.add_argument("endpoint", help="Base URL (e.g. https://glm-5-fp8.completions.near.ai)")
    parser.add_argument("--token", default="secret123", help="Bearer token (default: secret123)")
    parser.add_argument("--concurrency", "-c", type=int, default=20, help="Concurrent requests (default: 20)")
    parser.add_argument("--duration", "-d", type=float, default=60, help="Duration in seconds (default: 60)")
    parser.add_argument("--model", "-m", default=None, help="Model name for completions (auto-detected if not set)")
    parser.add_argument("--skip-completions", action="store_true", help="Skip completion benchmarks")
    parser.add_argument("--skip-attestation", action="store_true", help="Skip attestation benchmarks")
    args = parser.parse_args()

    base = args.endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"}

    connector = aiohttp.TCPConnector(limit=args.concurrency + 5, ssl=False)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
        # Detect model name
        model = args.model
        if not model:
            try:
                async with session.get(f"{base}/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            model = models[0].get("id", "unknown")
                            print(f"Detected model: {model}")
            except Exception:
                pass
        if not model:
            model = "unknown"

        # Warmup: single request to each endpoint
        print(f"\nEndpoint: {base}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Duration: {args.duration}s per test")
        print(f"Model: {model}")
        print()

        # ── 1. Attestation (no nonce) — should be cached ──
        if not args.skip_attestation:
            print("Warming up attestation (no nonce)...")
            try:
                async with session.get(f"{base}/v1/attestation/report") as resp:
                    body = await resp.read()
                    print(f"  warmup status: {resp.status} ({len(body)} bytes)")
            except Exception as e:
                print(f"  warmup failed: {e}")
                print("  (continuing anyway — benchmark will show errors)")
                print()

            stats_att_cached = Stats("attestation_cached (no nonce)")
            print(f"Running attestation (cached) benchmark ({args.concurrency}x for {args.duration}s)...")

            def make_att_cached():
                return ("GET", f"{base}/v1/attestation/report", {})

            await run_bench(session, stats_att_cached, make_att_cached, args.concurrency, args.duration)
            print(stats_att_cached.report())
            print()

            # ── 2. Attestation (with nonce) — forces fresh generation ──
            stats_att_nonce = Stats("attestation_fresh (with nonce)")
            print(f"Running attestation (fresh/nonce) benchmark ({args.concurrency}x for {args.duration}s)...")

            nonce_counter = 0
            def make_att_nonce():
                nonlocal nonce_counter
                nonce_counter += 1
                # Each request gets a unique nonce → forces fresh GPU evidence + TDX quote
                nonce = f"{nonce_counter:064x}"
                return ("GET", f"{base}/v1/attestation/report?nonce={nonce}", {})

            await run_bench(session, stats_att_nonce, make_att_nonce, args.concurrency, args.duration)
            print(stats_att_nonce.report())
            print()

        # ── 3. Chat completion (non-streaming) ──
        if not args.skip_completions:
            stats_chat = Stats("chat_completion (non-streaming)")
            print(f"Running chat completion (non-streaming) benchmark ({args.concurrency}x for {args.duration}s)...")

            def make_chat():
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                    "max_tokens": 5,
                    "stream": False,
                }
                return ("POST", f"{base}/v1/chat/completions", {"json": body})

            await run_bench(session, stats_chat, make_chat, args.concurrency, args.duration)
            print(stats_chat.report())
            print()

            # ── 4. Chat completion (streaming) ──
            stats_stream = Stats("chat_completion (streaming)")
            print(f"Running chat completion (streaming) benchmark ({args.concurrency}x for {args.duration}s)...")

            async def stream_worker_fn():
                """Custom worker that reads SSE stream to completion."""
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                    "max_tokens": 5,
                    "stream": True,
                }
                t0 = time.monotonic()
                try:
                    async with session.post(f"{base}/v1/chat/completions", json=body) as resp:
                        async for _ in resp.content:
                            pass
                        elapsed = time.monotonic() - t0
                        stats_stream.record(elapsed, resp.status)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    elapsed = time.monotonic() - t0
                    stats_stream.record(elapsed, 0)
                    stats_stream.errors += 1

            stop = asyncio.Event()

            async def streaming_loop():
                while not stop.is_set():
                    await stream_worker_fn()

            tasks = [asyncio.create_task(streaming_loop()) for _ in range(args.concurrency)]
            await asyncio.sleep(args.duration)
            stop.set()
            await asyncio.sleep(0.5)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            print(stats_stream.report())
            print()

    # ── Summary ──
    print("=" * 60)
    print(f"SUMMARY — {base}")
    print(f"  concurrency={args.concurrency}  duration={args.duration}s")
    print("=" * 60)
    if not args.skip_attestation:
        print(stats_att_cached.report())
        print(stats_att_nonce.report())
    if not args.skip_completions:
        print(stats_chat.report())
        print(stats_stream.report())


if __name__ == "__main__":
    asyncio.run(main())
