#!/usr/bin/env bash
# Wait for a container's Docker HEALTHCHECK to report healthy. Fails if it
# reports unhealthy or never becomes healthy within the timeout.
set -euo pipefail
name="$1"
timeout="${2:-180}"
for _ in $(seq 1 "$timeout"); do
  status=$(docker inspect --format '{{.State.Health.Status}}' "$name")
  case "$status" in
    healthy)
      echo "$name is healthy"
      exit 0
      ;;
    unhealthy)
      echo "$name is unhealthy"
      docker inspect --format '{{json .State.Health.Log}}' "$name"
      exit 1
      ;;
  esac
  sleep 1
done
echo "$name did not become healthy within ${timeout}s (last status: $status)"
exit 1
