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
cd sdks/python && maturin develop && cd ../..

# Download the dataset
python benchmarks/longmemeval/download_data.py

# Run the benchmark (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)
python benchmarks/longmemeval/run_benchmark.py

# Evaluate results (requires OPENAI_API_KEY for GPT-4o judge)
python benchmarks/longmemeval/evaluate.py results/hypotheses.jsonl
```

## How It Works

1. **Ingest**: Each chat session from the dataset is stored into MenteDB with its timestamp
   as metadata. Sessions are ingested chronologically, simulating a real conversation history.

2. **Retrieve**: For each question, MenteDB's semantic search finds the most relevant memories.
   The question date is used for temporal filtering.

3. **Answer**: Retrieved memories plus the question are sent to an LLM (GPT-4o or Claude)
   which extracts relevant information and then reasons over it to produce a final answer.

4. **Evaluate**: The official LongMemEval GPT-4o judge scores each answer as correct or incorrect.

## Configuration

Environment variables:

- `OPENAI_API_KEY` — Required for evaluation, optional for answer generation
- `ANTHROPIC_API_KEY` — Optional, used for answer generation if set
- `EMBEDDING_PROVIDER` — Embedding provider (default: `openai`)
- `EMBEDDING_MODEL` — Embedding model (default: `text-embedding-3-small`)
- `READER_MODEL` — LLM for answer generation (default: auto detect from available keys)
- `TOP_K` — Number of memories to retrieve per question (default: 20)
- `DATASET` — Which dataset to use: `s`, `m`, or `oracle` (default: `s`)

## Output

Results are saved to `benchmarks/longmemeval/results/`:
- `hypotheses.jsonl` — Model predictions
- `hypotheses.jsonl.eval-results-gpt-4o` — Evaluation results with per question labels
- `metrics.txt` — Summary metrics by category
