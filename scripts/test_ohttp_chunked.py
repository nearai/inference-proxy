#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Manual test for chunked OHTTP endpoints.

Tests that the /ohttp endpoint correctly dispatches on Content-Type and
returns the right response format. Does NOT do HPKE crypto (use the Rust
examples for full roundtrip verification).

Usage:
    uv run scripts/test_ohttp_chunked.py <base_url>
    uv run scripts/test_ohttp_chunked.py https://<model>.completions.near.ai

Tests:
    1. GET /.well-known/ohttp-gateway returns key config
    2. POST /ohttp with message/ohttp-req (empty) → 400
    3. POST /ohttp with message/ohttp-chunked-req (empty) → 400
    4. POST /ohttp with message/ohttp-chunked-req (garbage) → 400
    5. POST /ohttp with no content-type (garbage) → 400 (defaults to standard)
    6. Verify both content-type paths reject each other's payloads
"""

import sys
import requests

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <base_url>")
        sys.exit(1)

    base = sys.argv[1].rstrip("/")
    passed = 0
    failed = 0

    def check(desc: str, ok: bool, detail: str = ""):
        nonlocal passed, failed
        if ok:
            print(f"  PASS: {desc}")
            passed += 1
        else:
            print(f"  FAIL: {desc} — {detail}")
            failed += 1

    print(f"=== Chunked OHTTP Smoke Tests ===")
    print(f"Target: {base}\n")

    # 1. Key config
    r = requests.get(f"{base}/.well-known/ohttp-gateway", timeout=10)
    check(
        "Key config endpoint returns 200",
        r.status_code == 200 and "ohttp-keys" in r.headers.get("content-type", ""),
        f"status={r.status_code}, ct={r.headers.get('content-type', '')}",
    )
    config_bytes = r.content

    # 2. Standard OHTTP: empty body → 400
    r = requests.post(
        f"{base}/ohttp",
        headers={"Content-Type": "message/ohttp-req"},
        timeout=10,
    )
    check("Standard OHTTP empty body → 400", r.status_code == 400, f"status={r.status_code}")

    # 3. Chunked OHTTP: empty body → 400
    r = requests.post(
        f"{base}/ohttp",
        headers={"Content-Type": "message/ohttp-chunked-req"},
        timeout=10,
    )
    check("Chunked OHTTP empty body → 400", r.status_code == 400, f"status={r.status_code}")

    # 4. Chunked OHTTP: garbage → 400
    r = requests.post(
        f"{base}/ohttp",
        headers={"Content-Type": "message/ohttp-chunked-req"},
        data=b"\xde\xad\xbe\xef" * 25,
        timeout=10,
    )
    check("Chunked OHTTP garbage body → 400", r.status_code == 400, f"status={r.status_code}")

    # 5. No content-type: garbage → 400 (defaults to standard path)
    r = requests.post(
        f"{base}/ohttp",
        data=b"\xde\xad\xbe\xef" * 25,
        timeout=10,
    )
    check("No content-type garbage → 400", r.status_code == 400, f"status={r.status_code}")

    # 6. Standard content-type with chunked-format payload → 400
    # (We don't have a real chunked payload, but we can verify that standard
    # decapsulation rejects garbage and chunked decapsulation also rejects garbage)
    r = requests.post(
        f"{base}/ohttp",
        headers={"Content-Type": "message/ohttp-req"},
        data=b"\x01\x00\x20" + b"\x00" * 100,  # Looks like OHTTP header but is invalid
        timeout=10,
    )
    check("Standard OHTTP invalid payload → 400", r.status_code == 400, f"status={r.status_code}")

    r = requests.post(
        f"{base}/ohttp",
        headers={"Content-Type": "message/ohttp-chunked-req"},
        data=b"\x01\x00\x20" + b"\x00" * 100,
        timeout=10,
    )
    check("Chunked OHTTP invalid payload → 400", r.status_code == 400, f"status={r.status_code}")

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
