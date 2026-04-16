#!/usr/bin/env bash
# Quick smoke test for OHTTP gateway endpoints.
# Only tests that endpoints respond correctly — no crypto verification.
#
# Usage:
#   ./scripts/test_ohttp_smoke.sh https://<model>.completions.near.ai
#   ./scripts/test_ohttp_smoke.sh http://127.0.0.1:8000

set -euo pipefail

BASE_URL="${1:?Usage: $0 <base_url>}"
BASE_URL="${BASE_URL%/}"

pass=0
fail=0

check() {
    local desc="$1" expected_status="$2" expected_ct="$3"
    shift 3
    local resp
    resp=$(curl -sS -o /dev/null -w '%{http_code} %{content_type}' "$@" 2>&1) || true
    local status="${resp%% *}"
    local ct="${resp#* }"

    if [[ "$status" == "$expected_status" ]]; then
        if [[ -z "$expected_ct" || "$ct" == *"$expected_ct"* ]]; then
            echo "  PASS: $desc (status=$status)"
            ((pass++))
        else
            echo "  FAIL: $desc (status=$status, content-type='$ct', expected '$expected_ct')"
            ((fail++))
        fi
    else
        echo "  FAIL: $desc (status=$status, expected $expected_status)"
        ((fail++))
    fi
}

echo "=== OHTTP Smoke Tests ==="
echo "Target: $BASE_URL"
echo

echo "-- Key Config Endpoints --"
check "GET /.well-known/ohttp-gateway returns 200 + ohttp-keys" \
    200 "application/ohttp-keys" \
    "$BASE_URL/.well-known/ohttp-gateway"

check "GET /v1/ohttp/config returns 200 + ohttp-keys" \
    200 "application/ohttp-keys" \
    "$BASE_URL/v1/ohttp/config"

echo
echo "-- Config bytes match --"
A=$(curl -sS "$BASE_URL/.well-known/ohttp-gateway" | xxd -p | tr -d '\n')
B=$(curl -sS "$BASE_URL/v1/ohttp/config" | xxd -p | tr -d '\n')
if [[ "$A" == "$B" && -n "$A" ]]; then
    echo "  PASS: Both endpoints return identical config (${#A} hex chars)"
    ((pass++))
else
    echo "  FAIL: Config bytes differ or empty"
    ((fail++))
fi

echo
echo "-- Relay Endpoint --"
check "POST /ohttp with empty body returns 400" \
    400 "" \
    -X POST "$BASE_URL/ohttp" -H "Content-Type: message/ohttp-req"

check "POST /ohttp with garbage returns 400" \
    400 "" \
    -X POST "$BASE_URL/ohttp" -H "Content-Type: message/ohttp-req" -d "garbage_not_hpke"

echo
echo "-- Attestation --"
ATTESTATION=$(curl -sS "$BASE_URL/v1/attestation/report?signing_algo=ed25519" 2>/dev/null || echo "{}")
if echo "$ATTESTATION" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'ohttp_key_config' in d; print(f'  PASS: ohttp_key_config present ({len(d[\"ohttp_key_config\"])} hex chars)')" 2>/dev/null; then
    ((pass++))
elif echo "$ATTESTATION" | grep -q "error"; then
    echo "  SKIP: Attestation unavailable (no TDX hardware)"
else
    echo "  FAIL: ohttp_key_config missing from attestation response"
    ((fail++))
fi

echo
echo "=== Results: $pass passed, $fail failed ==="
[[ $fail -eq 0 ]] && exit 0 || exit 1
