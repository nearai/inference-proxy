#!/usr/bin/env bash
set -Eeuo pipefail

TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$TEST_DIR/../.." && pwd)
COMPOSE=(docker compose -f "$TEST_DIR/compose.yaml")
CREATED_GIT_REV=0

cleanup() {
    status=$?
    "${COMPOSE[@]}" down --volumes --remove-orphans || true
    if [[ "$CREATED_GIT_REV" == 1 ]]; then
        rm -f "$REPO_ROOT/.GIT_REV"
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if [[ ! -e "$REPO_ROOT/.GIT_REV" ]]; then
    git -C "$REPO_ROOT" rev-parse HEAD > "$REPO_ROOT/.GIT_REV"
    CREATED_GIT_REV=1
fi

"${COMPOSE[@]}" up --build --detach
python3 "$TEST_DIR/driver.py"
