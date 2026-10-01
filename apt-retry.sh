#!/bin/sh

# Retry whole APT commands on top of APT's own per-file retries, backing off
# long enough to ride out snapshot.ubuntu.com outages that last minutes.
attempt=1
delay=30
while :; do
    if apt-get \
        -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 \
        -o APT::Update::Error-Mode=any "$@"; then
        exit 0
    else
        status=$?
    fi
    if [ "$attempt" -ge 5 ]; then
        exit "$status"
    fi
    printf 'APT attempt %s/5 failed (exit %s); retrying in %ss\n' \
        "$attempt" "$status" "$delay" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
done
