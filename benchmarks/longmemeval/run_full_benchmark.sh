#!/bin/bash
# Run the full 500-question LongMemEval benchmark in batches of 10.
# Results accumulate in results/hypotheses_full.jsonl (single file).
# Each batch file is kept as backup — nothing is ever deleted.
# Usage: ./run_full_benchmark.sh [start_offset]
#
# RESUME: If crashed at batch 230, run: ./run_full_benchmark.sh 230
# All prior results in hypotheses_full.jsonl are preserved.

# Do NOT use set -e — we want to continue even if a batch fails
set -o pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate
export OPENAI_API_KEY=$MENTEDB_OPENAI_API_KEY
export ANTHROPIC_API_KEY=$MENTEDB_ANTHROPIC_API_KEY

TOTAL=500
BATCH=10
WORKERS=3
START=${1:-0}
RESULTS_DIR="benchmarks/longmemeval/results"
FULL_RESULTS="$RESULTS_DIR/hypotheses_full.jsonl"
FAILED_BATCHES="$RESULTS_DIR/failed_batches.txt"

mkdir -p "$RESULTS_DIR"

# Never delete hypotheses_full.jsonl — only append.
# If starting from 0, back up any existing file first.
if [ "$START" -eq 0 ] && [ -f "$FULL_RESULTS" ] && [ -s "$FULL_RESULTS" ]; then
    backup="$RESULTS_DIR/hypotheses_full_backup_$(date +%s).jsonl"
    cp "$FULL_RESULTS" "$backup"
    echo "Backed up existing results to $backup"
    > "$FULL_RESULTS"
fi
touch "$FULL_RESULTS"

echo "========================================"
echo "  LongMemEval Full Benchmark"
echo "  Total: $TOTAL | Batch: $BATCH | Workers: $WORKERS"
echo "  Start: $START | Results: $FULL_RESULTS"
echo "========================================"

total_failures=0

for (( offset=START; offset<TOTAL; offset+=BATCH )); do
    remaining=$((TOTAL - offset))
    limit=$((remaining < BATCH ? remaining : BATCH))
    end=$((offset + limit))
    batch_file="$RESULTS_DIR/hypotheses_q${offset}-${end}.jsonl"

    echo ""
    echo "========================================"
    echo "  BATCH: questions $offset..$end ($limit questions)"
    total_so_far=$(wc -l < "$FULL_RESULTS" | tr -d ' ')
    echo "  Progress: $total_so_far answers so far | $(( offset * 100 / TOTAL ))% batches started"
    echo "  Time: $(date '+%H:%M:%S')"
    echo "========================================"

    # Skip if batch already completed (resume support)
    if [ -f "$batch_file" ]; then
        existing=$(wc -l < "$batch_file" | tr -d ' ')
        if [ "$existing" -eq "$limit" ]; then
            echo "  ⏭ Batch already complete ($existing results), skipping"
            cat "$batch_file" >> "$FULL_RESULTS"
            continue
        fi
    fi

    # Run this batch — capture exit code but don't die
    if python benchmarks/longmemeval/run_benchmark.py \
        --offset "$offset" --limit "$limit" --workers "$WORKERS" 2>&1; then
        echo "  ✅ Batch $offset-$end completed successfully"
    else
        echo "  ⚠️ Batch $offset-$end FAILED (exit code $?) — continuing to next batch"
        echo "$offset" >> "$FAILED_BATCHES"
        total_failures=$((total_failures + 1))
    fi

    # Append whatever results we got (even partial) to full file
    if [ -f "$batch_file" ]; then
        cat "$batch_file" >> "$FULL_RESULTS"
        count=$(wc -l < "$batch_file" | tr -d ' ')
        total_so_far=$(wc -l < "$FULL_RESULTS" | tr -d ' ')
        echo "  📊 Batch: $count answers | Total: $total_so_far / $TOTAL"
    fi

    # Show summary every 50 questions
    if (( end % 50 == 0 )); then
        echo ""
        echo "╔══════════════════════════════════════╗"
        echo "║  CHECKPOINT: $end / $TOTAL questions  "
        echo "╚══════════════════════════════════════╝"
        total_so_far=$(wc -l < "$FULL_RESULTS" | tr -d ' ')
        echo "  Total answers:    $total_so_far"
        echo "  Failed batches:   $total_failures"
        echo "  Time:             $(date '+%H:%M:%S')"
    fi
done

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  BENCHMARK COMPLETE                  ║"
echo "╚══════════════════════════════════════╝"
total=$(wc -l < "$FULL_RESULTS" | tr -d ' ')
echo "  Total answers:    $total / $TOTAL"
echo "  Failed batches:   $total_failures"
echo "  Results:          $FULL_RESULTS"
echo "  Time:             $(date '+%H:%M:%S')"

if [ "$total_failures" -gt 0 ] && [ -f "$FAILED_BATCHES" ]; then
    echo ""
    echo "  Failed batch offsets (retry with ./run_full_benchmark.sh <offset>):"
    cat "$FAILED_BATCHES"
fi

echo ""
echo "  EVALUATE: python benchmarks/longmemeval/evaluate.py $FULL_RESULTS"
