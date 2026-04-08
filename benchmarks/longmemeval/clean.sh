#!/usr/bin/env bash
# Clean up benchmark runs: kill running processes and clear results
set -euo pipefail

echo "=== LongMemEval Cleanup ==="

# Kill any running benchmark processes
pids=$(ps aux | grep '[r]un_benchmark.py' | awk '{print $2}' || true)
if [ -n "$pids" ]; then
    echo "Killing benchmark processes: $pids"
    echo "$pids" | xargs kill -9 2>/dev/null || true
else
    echo "No running benchmark processes found."
fi

# Kill any running evaluate processes
pids=$(ps aux | grep '[e]valuate.py' | awk '{print $2}' || true)
if [ -n "$pids" ]; then
    echo "Killing evaluate processes: $pids"
    echo "$pids" | xargs kill -9 2>/dev/null || true
else
    echo "No running evaluate processes found."
fi

# Clear results
RESULTS_DIR="$(dirname "$0")/results"
if [ -d "$RESULTS_DIR" ]; then
    count=$(find "$RESULTS_DIR" -type f | wc -l | tr -d ' ')
    rm -f "$RESULTS_DIR"/*
    echo "Cleared $count file(s) from results/"
else
    echo "No results directory found."
fi

echo "=== Done ==="
