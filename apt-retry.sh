#!/bin/sh

# Retry archive failures without multiplying APT's own download retries.
attempt=1
while :; do
    if apt-get -o Acquire::Retries=0 \
        -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 \
        -o APT::Update::Error-Mode=any "$@"; then
        exit 0
    else
        status=$?
    fi
    if [ "$attempt" -ge 3 ]; then
        exit "$status"
    fi
    delay=$((attempt * 5))
    printf 'APT attempt %s/3 failed (exit %s); retrying in %ss\n' \
        "$attempt" "$status" "$delay" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
done
