#!/usr/bin/env bash
#
# Opt-in protected smoke test for the real AWS AppConfig Agent path.
#
# The AWS control-plane profile and the Agent's temporary credentials are
# deliberately separate. The proxy container receives only the Agent URL and
# AppConfig identity; its environment is checked before any request is sent.
#
# AWS API contracts used here:
#   https://docs.aws.amazon.com/cli/latest/reference/appconfig/create-configuration-profile.html
#   https://docs.aws.amazon.com/cli/latest/reference/appconfig/create-hosted-configuration-version.html
#   https://docs.aws.amazon.com/cli/latest/reference/appconfig/start-deployment.html
#   https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-integration-containers-agent-configuring.html

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FIXTURE_DIR="$SCRIPT_DIR/appconfig-smoke"

usage() {
    sed -n '1,48p' "$0"
    cat <<'USAGE'

Required environment:
  AWS_PROFILE                         named control-plane CLI profile (not default)
  AWS_REGION                          test Region
  AWS_ACCOUNT_ID                      expected 12-digit test account
  APPCONFIG_SMOKE_CONFIRM             I_UNDERSTAND_REAL_AWS_APPCONFIG_SMOKE:<account>
  AWS_APPCONFIG_AGENT_ACCESS_KEY_ID   temporary Agent credential
  AWS_APPCONFIG_AGENT_SECRET_ACCESS_KEY
  AWS_APPCONFIG_AGENT_SESSION_TOKEN
  PROXY_IMAGE                         immutable proxy image reference (@sha256:...)

Optional environment:
  APPCONFIG_SMOKE_TIMEOUT_SECS        bounded wait deadline, default 180 (max 600)
  APPCONFIG_SMOKE_PROXY_PORT          loopback proxy port, otherwise random 20000-39999
  APPCONFIG_AGENT_IMAGE               pinned Agent image override
  APPCONFIG_BACKEND_IMAGE             backend fixture image, default python:3.12-alpine
USAGE
}

die() {
    printf 'appconfig smoke: ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

for command_name in aws docker jq curl; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command not found: $command_name"
done

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
CONFIRMATION="${APPCONFIG_SMOKE_CONFIRM:-}"
AGENT_ACCESS_KEY_ID="${AWS_APPCONFIG_AGENT_ACCESS_KEY_ID:-}"
AGENT_SECRET_ACCESS_KEY="${AWS_APPCONFIG_AGENT_SECRET_ACCESS_KEY:-}"
AGENT_SESSION_TOKEN="${AWS_APPCONFIG_AGENT_SESSION_TOKEN:-}"
PROXY_IMAGE="${PROXY_IMAGE:-}"
AGENT_IMAGE="${APPCONFIG_AGENT_IMAGE:-public.ecr.aws/aws-appconfig/aws-appconfig-agent:2.x@sha256:5e23ec8b3eb883680cdf1e4224c8d0f99405e14eb67131a2def7316eaa418f1a}"
BACKEND_IMAGE="${APPCONFIG_BACKEND_IMAGE:-python:3.12-alpine}"
TIMEOUT_SECS="${APPCONFIG_SMOKE_TIMEOUT_SECS:-180}"

[[ -n "$AWS_PROFILE" ]] || die "AWS_PROFILE is required"
[[ "$AWS_PROFILE" != "default" ]] || die "AWS_PROFILE must name an explicit test profile, not default"
[[ -n "$AWS_REGION" ]] || die "AWS_REGION is required"
[[ "$AWS_REGION" =~ ^[a-z0-9-]+$ ]] || die "AWS_REGION contains invalid characters"
[[ "$AWS_ACCOUNT_ID" =~ ^[0-9]{12}$ ]] || die "AWS_ACCOUNT_ID must be a 12-digit account"
[[ "$CONFIRMATION" == "I_UNDERSTAND_REAL_AWS_APPCONFIG_SMOKE:${AWS_ACCOUNT_ID}" ]] || die "set APPCONFIG_SMOKE_CONFIRM=I_UNDERSTAND_REAL_AWS_APPCONFIG_SMOKE:${AWS_ACCOUNT_ID}"
[[ -n "$AGENT_ACCESS_KEY_ID" ]] || die "AWS_APPCONFIG_AGENT_ACCESS_KEY_ID is required"
[[ -n "$AGENT_SECRET_ACCESS_KEY" ]] || die "AWS_APPCONFIG_AGENT_SECRET_ACCESS_KEY is required"
[[ -n "$AGENT_SESSION_TOKEN" ]] || die "AWS_APPCONFIG_AGENT_SESSION_TOKEN is required; use temporary Agent credentials"
[[ "$PROXY_IMAGE" == *@sha256:* ]] || die "PROXY_IMAGE must be an immutable image reference containing @sha256:"
[[ "$AGENT_IMAGE" == *@sha256:* ]] || die "APPCONFIG_AGENT_IMAGE must be an immutable image reference containing @sha256:"
[[ "$TIMEOUT_SECS" =~ ^[0-9]+$ && "$TIMEOUT_SECS" -gt 0 && "$TIMEOUT_SECS" -le 600 ]] || die "APPCONFIG_SMOKE_TIMEOUT_SECS must be between 1 and 600"

AWS_CLI=(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" --no-cli-pager)
aws_call() {
    "${AWS_CLI[@]}" "$@"
}

utc_timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

CALLER_JSON="$(aws_call sts get-caller-identity --output json)" || die "AWS profile could not call sts:GetCallerIdentity"
CALLER_ACCOUNT="$(jq -r '.Account // empty' <<<"$CALLER_JSON")"
CALLER_ARN="$(jq -r '.Arn // empty' <<<"$CALLER_JSON")"
[[ "$CALLER_ACCOUNT" == "$AWS_ACCOUNT_ID" ]] || die "AWS profile resolved to account $CALLER_ACCOUNT, expected AWS_ACCOUNT_ID"
[[ -n "$CALLER_ARN" ]] || die "STS caller identity did not include an ARN"

RUN_ID="$(date -u +%Y%m%d%H%M%S)-$$-${RANDOM}"
RESOURCE_SUFFIX="smoke-${RUN_ID}"
APP_NAME="inference-proxy-${RESOURCE_SUFFIX}"
ENV_NAME="gateway-${RESOURCE_SUFFIX}"
PROFILE_NAME="admission-${RESOURCE_SUFFIX}"
STRATEGY_NAME="fast-${RESOURCE_SUFFIX}"
TARGET_NAME="gateway-${RUN_ID}"
NETWORK_NAME="appconfig-smoke-${RUN_ID}"
BACKUP_VOLUME="appconfig-smoke-backup-${RUN_ID}"
BACKEND_NAME="appconfig-smoke-backend-${RUN_ID}"
AGENT_NAME="appconfig-smoke-agent-${RUN_ID}"
PROXY_NAME="appconfig-smoke-proxy-${RUN_ID}"
PROXY_TOKEN="appconfig-smoke-token-${RUN_ID}"
PROXY_HOST_PORT="${APPCONFIG_SMOKE_PROXY_PORT:-$((20000 + RANDOM % 20000))}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/appconfig-smoke.XXXXXX")"
APP_ID=""
ENV_ID=""
PROFILE_ID=""
STRATEGY_ID=""
NETWORK_CREATED=0
BACKUP_VOLUME_CREATED=0
BACKEND_STARTED=0
AGENT_STARTED=0
PROXY_STARTED=0
HOSTED_VERSIONS=()
DEPLOYMENTS=()

cleanup_deployment() {
    local deployment_number="$1"
    local state
    state="$(aws_call appconfig get-deployment --application-id "$APP_ID" --environment-id "$ENV_ID" --deployment-number "$deployment_number" --query State --output text 2>/dev/null || true)"
    case "$state" in
        BAKING|VALIDATING|DEPLOYING|ROLLING_BACK)
            aws_call appconfig stop-deployment --application-id "$APP_ID" --environment-id "$ENV_ID" --deployment-number "$deployment_number" >/dev/null 2>&1 || true
            ;;
    esac
    for _ in $(seq 1 15); do
        state="$(aws_call appconfig get-deployment --application-id "$APP_ID" --environment-id "$ENV_ID" --deployment-number "$deployment_number" --query State --output text 2>/dev/null || true)"
        case "$state" in
            COMPLETE|ROLLED_BACK|REVERTED|""|None) return 0 ;;
        esac
        sleep 1
    done
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    set +e

    # Stop an active deployment before deleting the hosted versions and
    # control-plane resources it references.
    if [[ -n "$APP_ID" && -n "$ENV_ID" ]]; then
        for deployment_number in "${DEPLOYMENTS[@]}"; do
            cleanup_deployment "$deployment_number"
        done
    fi
    if [[ -n "$APP_ID" && -n "$PROFILE_ID" ]]; then
        for version_number in "${HOSTED_VERSIONS[@]}"; do
            aws_call appconfig delete-hosted-configuration-version \
                --application-id "$APP_ID" --configuration-profile-id "$PROFILE_ID" \
                --version-number "$version_number" >/dev/null 2>&1 || true
        done
    fi
    if [[ -n "$STRATEGY_ID" ]]; then
        aws_call appconfig delete-deployment-strategy --deployment-strategy-id "$STRATEGY_ID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$PROFILE_ID" && -n "$APP_ID" ]]; then
        aws_call appconfig delete-configuration-profile --application-id "$APP_ID" --configuration-profile-id "$PROFILE_ID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$ENV_ID" && -n "$APP_ID" ]]; then
        aws_call appconfig delete-environment --application-id "$APP_ID" --environment-id "$ENV_ID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$APP_ID" ]]; then
        aws_call appconfig delete-application --application-id "$APP_ID" >/dev/null 2>&1 || true
    fi

    if [[ "$PROXY_STARTED" == 1 ]]; then docker rm -f "$PROXY_NAME" >/dev/null 2>&1 || true; fi
    if [[ "$AGENT_STARTED" == 1 ]]; then docker rm -f "$AGENT_NAME" >/dev/null 2>&1 || true; fi
    if [[ "$BACKEND_STARTED" == 1 ]]; then docker rm -f "$BACKEND_NAME" >/dev/null 2>&1 || true; fi
    if [[ "$BACKUP_VOLUME_CREATED" == 1 ]]; then docker volume rm "$BACKUP_VOLUME" >/dev/null 2>&1 || true; fi
    if [[ "$NETWORK_CREATED" == 1 ]]; then docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true; fi
    rm -rf -- "$TMP_DIR"

    if [[ "$status" -eq 0 ]]; then
        printf 'appconfig smoke: cleanup complete\n'
    else
        printf 'appconfig smoke: cleanup complete after failure (status %s)\n' "$status" >&2
    fi
    exit "$status"
}

# Install this before the first AWS or Docker resource creation.
trap cleanup EXIT
trap 'exit 130' INT TERM

printf 'appconfig smoke: profile=%s region=%s account=%s caller=%s\n' "$AWS_PROFILE" "$AWS_REGION" "$AWS_ACCOUNT_ID" "$CALLER_ARN"
printf 'appconfig smoke: Agent image=%s\n' "$AGENT_IMAGE"
printf 'appconfig smoke: proxy image=%s\n' "$PROXY_IMAGE"
printf 'appconfig smoke: unique prefix=%s\n' "$RESOURCE_SUFFIX"

render_config() {
    local output="$1"
    local maximum="$2"
    local target="$3"
    jq -n --arg target "$target" --argjson maximum "$maximum" \
        '{schema_version: 1, target: $target, admission: {max_inflight: $maximum, backpressure_secs: 10, retry_after_secs: 2}}' \
        >"$output"
}

render_config "$TMP_DIR/v1.json" 1 "$TARGET_NAME"
render_config "$TMP_DIR/v2.json" 2 "$TARGET_NAME"
render_config "$TMP_DIR/invalid.json" 0 "$TARGET_NAME"
render_config "$TMP_DIR/wrong-target.json" 2 "wrong-target-$RUN_ID"

printf 'appconfig smoke: creating uniquely prefixed AppConfig resources\n'
APP_ID="$(aws_call appconfig create-application --name "$APP_NAME" --description "Protected inference-proxy AppConfig smoke ${RUN_ID}" --query Id --output text)"
ENV_ID="$(aws_call appconfig create-environment --application-id "$APP_ID" --name "$ENV_NAME" --description "Protected smoke environment ${RUN_ID}" --query Id --output text)"
SCHEMA_CONTENT="$(<"$FIXTURE_DIR/schema.json")"
VALIDATOR_JSON="$(jq -cn --arg content "$SCHEMA_CONTENT" '[{Type: "JSON_SCHEMA", Content: $content}]')"
PROFILE_ID="$(aws_call appconfig create-configuration-profile \
    --application-id "$APP_ID" --name "$PROFILE_NAME" --location-uri hosted --type AWS.Freeform \
    --validators "$VALIDATOR_JSON" --query Id --output text)"
STRATEGY_ID="$(aws_call appconfig create-deployment-strategy --name "$STRATEGY_NAME" \
    --description "Zero-minute protected smoke strategy ${RUN_ID}" \
    --deployment-duration-in-minutes 0 --final-bake-time-in-minutes 0 \
    --growth-factor 100 --growth-type LINEAR --query Id --output text)"

[[ "$APP_ID" != None && -n "$APP_ID" ]] || die "AppConfig application creation returned no ID"
[[ "$ENV_ID" != None && -n "$ENV_ID" ]] || die "AppConfig environment creation returned no ID"
[[ "$PROFILE_ID" != None && -n "$PROFILE_ID" ]] || die "AppConfig profile creation returned no ID"
[[ "$STRATEGY_ID" != None && -n "$STRATEGY_ID" ]] || die "AppConfig strategy creation returned no ID"
printf 'appconfig smoke: application=%s environment=%s profile=%s strategy=%s\n' "$APP_ID" "$ENV_ID" "$PROFILE_ID" "$STRATEGY_ID"

printf 'appconfig smoke: pulling test images\n'
docker pull "$AGENT_IMAGE" >/dev/null
docker pull "$BACKEND_IMAGE" >/dev/null
AGENT_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$AGENT_IMAGE" 2>/dev/null || true)"
BACKEND_DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$BACKEND_IMAGE" 2>/dev/null || true)"
[[ -n "$AGENT_DIGEST" ]] || AGENT_DIGEST="$AGENT_IMAGE"
[[ -n "$BACKEND_DIGEST" ]] || BACKEND_DIGEST="$BACKEND_IMAGE"
printf 'appconfig smoke: resolved Agent image=%s backend image=%s\n' "$AGENT_DIGEST" "$BACKEND_DIGEST"

docker network create "$NETWORK_NAME" >/dev/null
NETWORK_CREATED=1
docker volume create "$BACKUP_VOLUME" >/dev/null
BACKUP_VOLUME_CREATED=1
docker run --detach --name "$BACKEND_NAME" --network "$NETWORK_NAME" --network-alias backend \
    --volume "$FIXTURE_DIR/mock_backend.py:/mock_backend.py:ro" \
    --env PORT=8001 "$BACKEND_IMAGE" python3 /mock_backend.py >/dev/null
BACKEND_STARTED=1

# Credentials are written to a mode-0600 temporary env file consumed only by
# the Agent. The proxy has a separate explicit environment list below.
AGENT_ENV_FILE="$TMP_DIR/agent.env"
umask 077
printf 'AWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\nAWS_SESSION_TOKEN=%s\n' \
    "$AGENT_ACCESS_KEY_ID" "$AGENT_SECRET_ACCESS_KEY" "$AGENT_SESSION_TOKEN" >"$AGENT_ENV_FILE"
docker run --detach --name "$AGENT_NAME" --network "$NETWORK_NAME" --network-alias agent \
    --env-file "$AGENT_ENV_FILE" \
    --env AWS_REGION="$AWS_REGION" \
    --env SERVICE_REGION="$AWS_REGION" \
    --env POLL_INTERVAL=5s \
    --env PREFETCH_LIST="$APP_ID:$ENV_ID:$PROFILE_ID" \
    --env BACKUP_DIRECTORY=/tmp/appconfig-backups \
    --env HTTP_HOST=all \
    --env LOG_LEVEL=info \
    --mount "type=volume,src=${BACKUP_VOLUME},dst=/tmp/appconfig-backups" \
    "$AGENT_IMAGE" >/dev/null
AGENT_STARTED=1

docker run --detach --name "$PROXY_NAME" --network "$NETWORK_NAME" --network-alias proxy \
    --publish "127.0.0.1:${PROXY_HOST_PORT}:8000" \
    --env MODEL_NAME=appconfig-smoke-model \
    --env TOKEN="$PROXY_TOKEN" \
    --env DEV=1 \
    --env GPU_NO_HW_MODE=1 \
    --env NON_TEE_DEPLOYMENT=1 \
    --env LISTEN_ADDR=0.0.0.0 \
    --env LISTEN_PORT=8000 \
    --env VLLM_BASE_URL=http://backend:8001 \
    --env VLLM_PROXY_ADMISSION_MAX_INFLIGHT=1 \
    --env VLLM_PROXY_ADMISSION_START_INFLIGHT=1 \
    --env VLLM_PROXY_ADMISSION_RAMP_STEP=1 \
    --env VLLM_PROXY_ADMISSION_RAMP_INTERVAL_SECS=1 \
    --env VLLM_PROXY_ADMISSION_BACKPRESSURE_SECS=10 \
    --env VLLM_PROXY_ADMISSION_RETRY_AFTER_SECS=2 \
    --env AWS_APPCONFIG_APPLICATION="$APP_ID" \
    --env AWS_APPCONFIG_ENVIRONMENT="$ENV_ID" \
    --env AWS_APPCONFIG_PROFILE="$PROFILE_ID" \
    --env AWS_APPCONFIG_TARGET="$TARGET_NAME" \
    --env AWS_APPCONFIG_AGENT_URL=http://agent:2772 \
    --env AWS_APPCONFIG_REFRESH_SECS=5 \
    --env LOG_FORMAT=json \
    "$PROXY_IMAGE" >/dev/null
PROXY_STARTED=1

if docker inspect "$PROXY_NAME" --format '{{json .Config.Env}}' | jq -e 'any(.[]; test("^AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN|PROFILE|ROLE_ARN|WEB_IDENTITY_TOKEN_FILE)="))' >/dev/null; then
    die "proxy container received AWS credentials or AWS identity configuration"
fi
printf 'appconfig smoke: credential boundary verified (Agent only)\n'

proxy_base="http://127.0.0.1:${PROXY_HOST_PORT}"
wait_for_agent_ping() {
    local deadline=$((SECONDS + TIMEOUT_SECS))
    while (( SECONDS < deadline )); do
        if docker exec "$PROXY_NAME" python3 -c \
            'import urllib.request; urllib.request.urlopen("http://agent:2772/ping", timeout=4).read()' \
            >/dev/null 2>&1; then
            printf 'appconfig smoke: Agent /ping healthy\n'
            return 0
        fi
        sleep 2
    done
    die "AppConfig Agent /ping did not become healthy before the bounded deadline"
}

wait_for_proxy() {
    local deadline=$((SECONDS + TIMEOUT_SECS))
    while (( SECONDS < deadline )); do
        if curl --silent --show-error --fail --connect-timeout 2 --max-time 4 "$proxy_base/healthz" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    docker logs "$PROXY_NAME" >&2 || true
    die "proxy did not become healthy before the bounded deadline"
}

agent_version() {
    local endpoint="http://agent:2772/applications/${APP_ID}/environments/${ENV_ID}/configurations/${PROFILE_ID}"
    docker exec "$PROXY_NAME" python3 -c \
        'import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=4).headers.get("Configuration-Version", ""))' \
        "$endpoint" 2>/dev/null || true
}

metric_value() {
    curl --silent --show-error --fail --connect-timeout 2 --max-time 4 "$proxy_base/metrics" 2>/dev/null \
        | awk '$1 == "appconfig_active_max_inflight" {print $2; exit}' || true
}

wait_for_policy() {
    local version="$1"
    local maximum="$2"
    local deadline=$((SECONDS + TIMEOUT_SECS))
    local metric current_version logs
    while (( SECONDS < deadline )); do
        metric="$(metric_value)"
        current_version="$(agent_version)"
        logs="$(docker logs "$PROXY_NAME" 2>&1 || true)"
        if [[ "$current_version" == "$version" ]] \
            && [[ "$metric" == "$maximum" || "$metric" == "$maximum.0" ]] \
            && grep -Fq "\"configuration_version\":\"$version\"" <<<"$logs"; then
            printf 'appconfig smoke: version=%s active_max_inflight=%s\n' "$version" "$metric"
            return 0
        fi
        sleep 2
    done
    die "version $version did not reach Agent header, structured proxy log, and active gauge"
}

wait_for_wrong_target_retention() {
    local version="$1"
    local maximum="$2"
    local deadline=$((SECONDS + TIMEOUT_SECS))
    local metric current_version logs
    while (( SECONDS < deadline )); do
        metric="$(metric_value)"
        current_version="$(agent_version)"
        logs="$(docker logs "$PROXY_NAME" 2>&1 || true)"
        if [[ "$current_version" == "$version" ]] \
            && [[ "$metric" == "$maximum" || "$metric" == "$maximum.0" ]] \
            && grep -Fq 'AppConfig admission policy read failed; retaining last-known-good policy' <<<"$logs"; then
            printf 'appconfig smoke: wrong-target version=%s retained last-known-good active_max_inflight=%s\n' "$version" "$metric"
            return 0
        fi
        sleep 2
    done
    die "wrong-target version $version was not observed and retained as last-known-good"
}

wait_for_deployment() {
    local deployment_number="$1"
    local deadline=$((SECONDS + TIMEOUT_SECS))
    local state
    while (( SECONDS < deadline )); do
        state="$(aws_call appconfig get-deployment --application-id "$APP_ID" --environment-id "$ENV_ID" \
            --deployment-number "$deployment_number" --query State --output text)"
        case "$state" in
            COMPLETE) printf 'appconfig smoke: deployment=%s state=COMPLETE\n' "$deployment_number"; return 0 ;;
            ROLLED_BACK|REVERTED) die "valid deployment $deployment_number ended in $state" ;;
            BAKING|VALIDATING|DEPLOYING|ROLLING_BACK) sleep 2 ;;
            *) die "deployment $deployment_number returned unexpected state $state" ;;
        esac
    done
    die "deployment $deployment_number exceeded bounded deadline"
}

deploy_version() {
    local version="$1"
    local description="$2"
    local response deployment_number
    response="$(aws_call appconfig start-deployment --application-id "$APP_ID" --environment-id "$ENV_ID" \
        --deployment-strategy-id "$STRATEGY_ID" --configuration-profile-id "$PROFILE_ID" \
        --configuration-version "$version" --description "$description" --output json)"
    deployment_number="$(jq -r '.DeploymentNumber // empty' <<<"$response")"
    [[ -n "$deployment_number" ]] || die "AppConfig deployment returned no deployment number"
    DEPLOYMENTS+=("$deployment_number")
    printf 'appconfig smoke: timestamp=%s deployment=%s version=%s\n' "$(utc_timestamp)" "$deployment_number" "$version"
    wait_for_deployment "$deployment_number"
}

create_hosted_version() {
    local file="$1"
    local label="$2"
    local version
    if ! version="$(aws_call appconfig create-hosted-configuration-version --application-id "$APP_ID" \
        --configuration-profile-id "$PROFILE_ID" --content "fileb://$file" \
        --content-type application/json --description "Protected smoke ${label}" \
        --version-label "$label" --query VersionNumber --output text \
        2>"$TMP_DIR/hosted-${label}.error.log")"; then
        return 1
    fi
    [[ "$version" != None && -n "$version" ]] || return 1
    HOSTED_VERSIONS+=("$version")
    CREATED_VERSION="$version"
    printf 'appconfig smoke: hosted_version=%s label=%s\n' "$version" "$label" >&2
}

wait_for_proxy
wait_for_agent_ping
printf 'appconfig smoke: proxy container=%s started_at=%s\n' \
    "$(docker inspect "$PROXY_NAME" --format '{{.Id}}')" \
    "$(docker inspect "$PROXY_NAME" --format '{{.State.StartedAt}}')"

create_hosted_version "$TMP_DIR/v1.json" "smoke-v1-${RUN_ID}"
V1="$CREATED_VERSION"
deploy_version "$V1" "Protected smoke baseline v1"
wait_for_policy "$V1" 1

run_request() {
    local output="$1"
    local sleep_ms="$2"
    curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
        --connect-timeout 3 --max-time 30 \
        -H "Authorization: Bearer $PROXY_TOKEN" \
        -H 'Content-Type: application/json' \
        --data "{\"model\":\"appconfig-smoke-model\",\"messages\":[{\"role\":\"user\",\"content\":\"smoke\"}],\"stream\":false,\"smoke_sleep_ms\":$sleep_ms}" \
        "$proxy_base/v1/chat/completions" >"$output" || printf '000\n' >"$output"
}

parallel_requests() {
    start_parallel_requests "$1" "$2"
    wait_parallel_requests
}

start_parallel_requests() {
    local prefix="$1"
    local sleep_ms="$2"
    local first="$TMP_DIR/${prefix}-1.status"
    local second="$TMP_DIR/${prefix}-2.status"
    run_request "$first" "$sleep_ms" &
    PARALLEL_FIRST_PID=$!
    run_request "$second" "$sleep_ms" &
    PARALLEL_SECOND_PID=$!
    PARALLEL_FIRST_STATUS="$first"
    PARALLEL_SECOND_STATUS="$second"
}

wait_parallel_requests() {
    wait "$PARALLEL_FIRST_PID" || true
    wait "$PARALLEL_SECOND_PID" || true
    printf '%s %s\n' "$(<"$PARALLEL_FIRST_STATUS")" "$(<"$PARALLEL_SECOND_STATUS")"
}

baseline_status="$(parallel_requests baseline 3000)"
baseline_successes="$(tr ' ' '\n' <<<"$baseline_status" | awk '$1 == "200" {count++} END {print count + 0}')"
baseline_refusals="$(tr ' ' '\n' <<<"$baseline_status" | awk '$1 == "429" {count++} END {print count + 0}')"
[[ "$baseline_successes" == 1 && "$baseline_refusals" == 1 ]] || die "v1 baseline did not admit one request and refuse one (statuses: $baseline_status)"
printf 'appconfig smoke: baseline max=1 statuses=%s (one admitted, one 429)\n' "$baseline_status"

if create_hosted_version "$TMP_DIR/invalid.json" "smoke-invalid-${RUN_ID}"; then
    INVALID_VERSION="$CREATED_VERSION"
    if invalid_response="$(aws_call appconfig start-deployment --application-id "$APP_ID" --environment-id "$ENV_ID" \
        --deployment-strategy-id "$STRATEGY_ID" --configuration-profile-id "$PROFILE_ID" \
        --configuration-version "$INVALID_VERSION" \
        --description "Protected smoke invalid schema rejection" --output json 2>/dev/null)"; then
        invalid_number="$(jq -r '.DeploymentNumber // empty' <<<"$invalid_response")"
        [[ -n "$invalid_number" ]] || die "invalid deployment unexpectedly started without a deployment number"
        DEPLOYMENTS+=("$invalid_number")
        invalid_deadline=$((SECONDS + TIMEOUT_SECS))
        invalid_state=""
        while (( SECONDS < invalid_deadline )); do
            invalid_state="$(aws_call appconfig get-deployment --application-id "$APP_ID" --environment-id "$ENV_ID" \
                --deployment-number "$invalid_number" --query State --output text)"
            case "$invalid_state" in
                ROLLED_BACK|REVERTED) break ;;
                COMPLETE) die "invalid JSON Schema version completed deployment" ;;
                BAKING|VALIDATING|DEPLOYING|ROLLING_BACK) sleep 2 ;;
                *) die "invalid deployment returned unexpected state $invalid_state" ;;
            esac
        done
        [[ "$invalid_state" == ROLLED_BACK || "$invalid_state" == REVERTED ]] || die "invalid deployment was not rejected before deadline"
        printf 'appconfig smoke: timestamp=%s invalid hosted version rejected in deployment=%s state=%s\n' "$(utc_timestamp)" "$invalid_number" "$invalid_state"
    else
        printf 'appconfig smoke: timestamp=%s invalid hosted version rejected by start-deployment validation\n' "$(utc_timestamp)"
    fi
else
    printf 'appconfig smoke: timestamp=%s invalid hosted version rejected while creating hosted version\n' "$(utc_timestamp)"
fi
wait_for_policy "$V1" 1

create_hosted_version "$TMP_DIR/v2.json" "smoke-v2-${RUN_ID}"
V2="$CREATED_VERSION"
deploy_version "$V2" "Protected smoke hot increase to v2"
wait_for_policy "$V2" 2

start_parallel_requests increase 12000
sleep 1
deploy_version "$V1" "Protected smoke rollback to v1"
wait_for_policy "$V1" 1
rollback_status_file="$TMP_DIR/rollback-new.status"
run_request "$rollback_status_file" 0
rollback_new_status="$(<"$rollback_status_file")"
wait_parallel_requests
v2_status="$(printf '%s %s\n' "$(<"$PARALLEL_FIRST_STATUS")" "$(<"$PARALLEL_SECOND_STATUS")")"
v2_successes="$(tr ' ' '\n' <<<"$v2_status" | awk '$1 == "200" {count++} END {print count + 0}')"
[[ "$v2_successes" == 2 ]] || die "v2 hot increase did not admit two active requests (statuses: $v2_status)"
printf 'appconfig smoke: hot increase max=2 statuses=%s (two admitted)\n' "$v2_status"
[[ "$rollback_new_status" == 429 ]] || die "new request during v2-to-v1 decrease was not refused (status: $rollback_new_status)"
printf 'appconfig smoke: rollback v2->v1 retained active requests and refused new request (status=%s)\n' "$rollback_new_status"

create_hosted_version "$TMP_DIR/wrong-target.json" "smoke-wrong-target-${RUN_ID}"
WRONG="$CREATED_VERSION"
deploy_version "$WRONG" "Protected smoke semantically wrong target"
wait_for_wrong_target_retention "$WRONG" 1
printf 'appconfig smoke: wrong-target deployment retained v1 policy; reverting immediately\n'
deploy_version "$V1" "Protected smoke immediate revert after wrong target"
wait_for_policy "$V1" 1

proxy_started_at="$(docker inspect "$PROXY_NAME" --format '{{.State.StartedAt}}')"
proxy_container_id="$(docker inspect "$PROXY_NAME" --format '{{.Id}}')"
printf 'appconfig smoke: PASS account=%s region=%s app=%s env=%s profile=%s\n' "$AWS_ACCOUNT_ID" "$AWS_REGION" "$APP_ID" "$ENV_ID" "$PROFILE_ID"
printf 'appconfig smoke: evidence agent=%s proxy=%s proxy_started_at=%s versions=%s deployments=%s\n' \
    "$AGENT_DIGEST" "$PROXY_IMAGE" "$proxy_started_at" "${HOSTED_VERSIONS[*]}" "${DEPLOYMENTS[*]}"
printf 'appconfig smoke: proxy container id=%s (unchanged through all updates)\n' "$proxy_container_id"
