"""Host-side acceptance driver for the real AWS AppConfig Agent lane."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "test-application:test-environment:test-profile.json"
FIXTURES = ROOT / "fixtures"
PROXY = "http://127.0.0.1:18000"
ENGINE = "http://127.0.0.1:18001"
COMPOSE = ["docker", "compose", "-f", str(ROOT / "compose.yaml")]


def request(
    method: str,
    url: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return (
                response.status,
                {name.lower(): value for name, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            {name.lower(): value for name, value in error.headers.items()},
            error.read(),
        )


def json_request(
    method: str,
    url: str,
    payload: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], Any]:
    body = None if payload is None else json.dumps(payload).encode()
    request_headers = {} if body is None else {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    status, response_headers, response_body = request(method, url, body, request_headers)
    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError:
        decoded = response_body.decode(errors="replace")
    return status, response_headers, decoded


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wait_until(description: str, predicate: Any, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, AssertionError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.2)
    suffix = f" ({last_error})" if last_error else ""
    raise AssertionError(f"timed out waiting for {description}{suffix}")


def compose_exec_agent_ping() -> str:
    # The Agent has no published host port by design. The mock-engine container
    # is on the same private Compose network and has Python's stdlib available.
    code = (
        "import json,urllib.request; "
        "r=urllib.request.urlopen('http://appconfig-agent:2772/ping',timeout=3); "
        "b=json.load(r); "
        "assert r.status == 200 and b.get('version'), b; "
        "print(json.dumps(b,sort_keys=True))"
    )
    result = subprocess.run(
        [*COMPOSE, "exec", "-T", "mock-engine", "python3", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def engine_control(action: str) -> dict[str, Any]:
    status, _, body = json_request("POST", f"{ENGINE}/control/{action}")
    check(status == 200, f"engine control {action}: {status} {body}")
    return body


def engine_stats() -> dict[str, Any]:
    status, _, body = json_request("GET", f"{ENGINE}/control/stats")
    check(status == 200, f"engine stats: {status} {body}")
    return body


def chat_request() -> tuple[int, dict[str, str], Any]:
    return json_request(
        "POST",
        f"{PROXY}/v1/chat/completions",
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"Authorization": "Bearer test-token"},
    )


def start_chat() -> tuple[threading.Thread, dict[str, Any]]:
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result["value"] = chat_request()
        except Exception as error:  # surfaced below with the thread result
            result["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def wait_for_engine_active(expected: int) -> None:
    wait_until(
        f"mock engine active requests = {expected}",
        lambda: engine_stats().get("active") == expected,
    )


def finish_threads(threads: list[tuple[threading.Thread, dict[str, Any]]]) -> None:
    for thread, result in threads:
        thread.join(timeout=10)
        check(not thread.is_alive(), "held chat request did not finish after release")
        check("error" not in result, f"chat request failed: {result.get('error')}")
        status, _, body = result["value"]
        check(status == 200, f"held chat request returned {status}: {body}")


def write_fixture(name: str) -> None:
    # os.replace keeps the Agent from observing a partially-written document.
    temporary = CONFIG.with_name(CONFIG.name + ".next")
    temporary.write_bytes((FIXTURES / name).read_bytes())
    os.replace(temporary, CONFIG)


def model_capacity() -> int:
    status, _, body = json_request("GET", f"{PROXY}/v1/models")
    check(status == 200, f"/v1/models: {status} {body}")
    capacities = body["data"][0]["capacity"]
    concurrency = next(item for item in capacities if item["type"] == "concurrency")
    return concurrency["value"]


def proxy_logs() -> str:
    result = subprocess.run(
        [*COMPOSE, "logs", "--no-color", "--no-log-prefix", "proxy"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def wait_for_capacity(expected: int) -> None:
    wait_until(f"advertised capacity = {expected}", lambda: model_capacity() == expected)


def wait_for_log(fragment: str) -> None:
    wait_until(f"proxy log containing {fragment!r}", lambda: fragment in proxy_logs())


def container_identity() -> tuple[str, str]:
    container = subprocess.run(
        [*COMPOSE, "ps", "-q", "proxy"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    check(container != "", "Compose did not report a proxy container")
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.Id}} {{.State.StartedAt}}",
            container,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    identity = result.stdout.strip().split(maxsplit=1)
    check(len(identity) == 2, f"unexpected proxy identity: {result.stdout!r}")
    return identity[0], identity[1]


def assert_refused() -> None:
    status, headers, body = chat_request()
    check(status == 429, f"expected admission 429, got {status}: {body}")
    check(headers.get("retry-after") == "2", f"unexpected Retry-After: {headers}")
    check(isinstance(body, dict) and "error" in body, f"unexpected refusal body: {body}")


def main() -> None:
    print("Agent /ping:", compose_exec_agent_ping())
    wait_until("proxy /version", lambda: json_request("GET", f"{PROXY}/version")[0] == 200)
    wait_for_capacity(1)
    wait_for_log('"configuration_version":"1"')
    proxy_version = json_request("GET", f"{PROXY}/version")[2]
    identity_before = container_identity()
    print("baseline capacity=1, proxy version:", proxy_version)

    # Baseline: one held request consumes the only admission slot; the second
    # request is rejected before the deterministic engine sees it.
    engine_control("reset")
    engine_control("hold")
    baseline = [start_chat()]
    wait_for_engine_active(1)
    assert_refused()
    check(engine_stats()["total_requests"] == 1, "baseline refusal reached the engine")

    # Hot increase: the same proxy process adopts max=3 and two more requests
    # reach the held engine without reconstructing its router or controller.
    write_fixture("increase.json")
    wait_for_capacity(3)
    wait_for_log('"configuration_version":"2"')
    increased = baseline + [start_chat(), start_chat()]
    wait_for_engine_active(3)

    # Decrease under load: all three existing permits continue, while new work
    # is rejected against the new max=2 and keeps the new policy's Retry-After.
    write_fixture("decrease.json")
    wait_for_capacity(2)
    wait_for_log('"configuration_version":"3"')
    check(engine_stats()["active"] == 3, "decrease cancelled an existing request")
    assert_refused()
    check(engine_stats()["total_requests"] == 3, "decrease refusal reached the engine")
    engine_control("release")
    finish_threads(increased)
    wait_until("mock engine drain", lambda: engine_stats()["active"] == 0)

    # Invalid document retention: max=0 is rejected by the proxy and the last
    # good max=2 remains both enforced and advertised.
    write_fixture("invalid.json")
    wait_for_log("retaining last-known-good policy")
    check(model_capacity() == 2, "invalid document changed advertised capacity")
    engine_control("reset")
    engine_control("hold")
    retained = [start_chat(), start_chat()]
    wait_for_engine_active(2)
    retained_refusal = start_chat()
    retained_refusal[0].join(timeout=5)
    check("error" not in retained_refusal[1], f"retention request failed: {retained_refusal[1]}")
    check(retained_refusal[1]["value"][0] == 429, "invalid document disabled admission")
    engine_control("release")
    finish_threads(retained)
    wait_until("mock engine drain after retention", lambda: engine_stats()["active"] == 0)

    # Rollback: restore the baseline fixture through the same Agent cache and
    # prove the live ceiling moves back to one without a process restart.
    write_fixture("rollback.json")
    wait_for_capacity(1)
    wait_for_log('"configuration_version":"5"')
    engine_control("reset")
    engine_control("hold")
    rolled_back = [start_chat()]
    wait_for_engine_active(1)
    assert_refused()
    engine_control("release")
    finish_threads(rolled_back)
    wait_until("mock engine final drain", lambda: engine_stats()["active"] == 0)

    identity_after = container_identity()
    check(identity_after == identity_before, f"proxy restarted: {identity_before} -> {identity_after}")
    check(
        json_request("GET", f"{PROXY}/version")[2] == proxy_version,
        "proxy /version changed during policy updates",
    )
    logs = proxy_logs()
    for version in ("1", "2", "3", "5"):
        check(
            f'"configuration_version":"{version}"' in logs,
            f"proxy logs lack applied Configuration-Version {version}",
        )
    print("passed: health, baseline, increase, decrease-under-load, invalid retention, rollback")
    print("passed: /v1/models capacity, version/log evidence, unchanged proxy identity")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        print(proxy_logs(), file=sys.stderr)
        raise
