# LongMemEval Benchmark for MenteDB

Run MenteDB against the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark,
the standard evaluation suite for long term memory in chat assistants (ICLR 2025).

500 questions across 5 memory abilities: information extraction, multi session reasoning,
knowledge updates, temporal reasoning, and abstention.

## Quick Start

```bash
# Install dependencies
pip install -r benchmarks/longmemeval/requirements.txt

# Build the Python SDK if not already installed
cd sdks/python && maturin develop --release && cd ../..

# Download the dataset
python benchmarks/longmemeval/download_data.py

# Raise file descriptor limit (needed for parallel extraction)
ulimit -n 4096

# Run the benchmark (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)
python benchmarks/longmemeval/run_benchmark.py --offset 0 --limit 50 --workers 3

# Evaluate results (requires OPENAI_API_KEY for GPT-4o judge)
python benchmarks/longmemeval/evaluate.py results/hypotheses_q0-50.jsonl

# Run all 500 questions in pages of 50, then combine
python benchmarks/longmemeval/run_benchmark.py --offset 0 --limit 50 --workers 3
python benchmarks/longmemeval/run_benchmark.py --offset 50 --limit 50 --workers 3
# ... repeat for offsets 100, 150, 200, 250, 300, 350, 400, 450
python benchmarks/longmemeval/evaluate_all.py
```

> **Note:** Before running, ensure `ulimit -n 4096` is set in your shell.
> The benchmark uses parallel extraction threads that can exceed the default
> macOS file descriptor limit (256), causing "Too many open files" errors.

## How It Works

1. **Ingest**: Each chat session from the dataset is stored into MenteDB with its timestamp
   as metadata. Sessions are ingested chronologically, simulating a real conversation history.
   Extraction runs in parallel (10 threads per worker) with GIL released during HTTP calls.

2. **Retrieve**: For each question, MenteDB's `search_expanded()` decomposes the query into
   sub-queries, then runs BM25 + vector hybrid search with RRF merging for broad recall.
   Time-aware filtering ensures only memories from before the question date are returned.

3. **Answer**: Retrieved memories plus the question are sent to an LLM (GPT-4o or Claude)
   using the official LongMemEval reader prompt (verbatim). No gold answers are accessible.

4. **Evaluate**: The official LongMemEval GPT-4o-2024-08-06 judge scores each answer using
   the exact evaluation prompts from the paper. Results are saved with per-question labels.

## Configuration

Environment variables:

- `OPENAI_API_KEY` — Required for evaluation, optional for answer generation
- `ANTHROPIC_API_KEY` — Optional, used for answer generation if set
- `EMBEDDING_PROVIDER` — Embedding provider (default: `openai` if key available, else `hash`)
- `EMBEDDING_MODEL` — Embedding model (default: `text-embedding-3-small`)
- `READER_MODEL` — Override reader LLM (default: `gpt-4o` for OpenAI, `claude-sonnet-4-20250514` for Anthropic)
- `TOP_K` — Number of memories to retrieve per question (default: 40)
- `DATASET` — Which dataset to use: `s`, `m`, or `oracle` (default: `s`)

## Output

Results are saved to `benchmarks/longmemeval/results/`:
- `hypotheses_q{start}-{end}.jsonl` — Model predictions per page
- `hypotheses_q{start}-{end}.jsonl.eval-results-gpt-4o-2024-08-06` — Per-question eval labels
- `hypotheses_q{start}-{end}_report.md` — Shareable markdown report per page
- `metrics.txt` — Summary metrics by category
- `combined_hypotheses.jsonl` — All pages combined (via `evaluate_all.py`)
- `combined_report.md` — Full 500-question report
