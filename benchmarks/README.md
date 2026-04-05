# MenteDB Quality Benchmarks

This suite proves that MenteDB produces better AI agent outcomes than naive RAG
or flat vector stores. Each benchmark isolates a specific capability and measures
it quantitatively.

## Quick Start

```bash
# Install the Python SDK (from the workspace root)
cd sdks/python && maturin develop && cd ../..

# Run engine tests (no API key needed)
python benchmarks/run_all.py --no-llm

# Run all tests including LLM evaluations
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
python benchmarks/run_all.py
```

## Benchmarks

### Stale Belief Test (`stale_belief.py`)

The most important test in the suite. It proves MenteDB's belief propagation
handles contradictory information correctly, something no flat vector store can
do.

**Scenario:** A user says "I use PostgreSQL" in turn 1, then "I switched to
SQLite" in turn 5. At turn 10, we query "What database does the user prefer?"

**What competitors return:** Both results, often with PostgreSQL ranked first
because it has higher keyword similarity to "database."

**What MenteDB returns:** SQLite only. The `supersedes` and `contradicts` edges
tell the engine that the PostgreSQL belief is stale.

**No LLM required.** This tests the graph engine directly.

### Delta Savings Test (`delta_savings.py`)

Quantifies token efficiency over a 20 turn conversation. Compares two
strategies:

  A) Full retrieval every turn (what competitors do), where each turn sends all
     top K results to the LLM regardless of whether they changed.

  B) Delta only after the first turn (what MenteDB enables), where subsequent
     turns only send new or updated memories plus a compact marker for unchanged
     ones.

**Expected result:** 40%+ token savings over 20 turns. In production
conversations that run longer, savings compound further.

**No LLM required.** This is a pure measurement of retrieval overhead.

### Sustained Conversation Test (`sustained_conversation.py`)

The real-world stress test. Simulates a developer working with an AI assistant
over 100 turns across 3 projects (Python API, Rust CLI, React Native mobile app)
with 6 belief changes scattered throughout.

**What it measures:**

  - Insert throughput at 100 memories (avg 0.24ms per insert)
  - Search accuracy with growing memory (83%+ retrieval accuracy)
  - Belief update correctness (0% stale beliefs returned across 6 supersessions)
  - Delta token savings at scale (90%+ reduction)
  - Project isolation (cross-project leakage detection)

**Expected result:** 0% stale returns, 80%+ retrieval accuracy, 90%+ delta
savings, sub-millisecond search times.

**No LLM required.** Pure engine test.

### Attention Budget Test (`attention_budget.py`)

Proves that memory ordering affects LLM compliance. Hides a critical instruction
among 19 filler memories and measures whether the LLM follows it under three
ordering strategies:

  A) Random order (baseline).
  B) Chronological order (what most systems do).
  C) U curve order (what MenteDB does), placing high priority memories at the
     start and end of the context window where LLM attention is strongest.

**Expected result:** U curve ordering achieves equal or higher compliance than
chronological, and meaningfully higher than random.

**Requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.** Uses gpt-4o-mini or Claude for evaluation.

### Noise Ratio Test (`noise_ratio.py`)

Measures extraction quality by comparing two ingestion strategies:

  A) Naive: store every conversation line as a separate memory.
  B) Extracted: use an LLM to pull out actionable facts, then store those.

An LLM judge evaluates what percentage of stored memories are actually useful for
an AI assistant. The MenteDB approach (structured extraction into typed memories)
should produce a higher useful-to-noise ratio with fewer total memories.

**Requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.** Uses gpt-4o-mini or Claude for extraction and evaluation.

### Mem0 vs MenteDB Comparison (`mem0_comparison.py`)

Head-to-head on the stale belief scenario. Runs the same test on both engines
side by side.

**Actual results:**

```
--- MenteDB ---
  SQLite in results: True
  PostgreSQL (stale) returned: False
  Time: 4.7ms
  Verdict: PASS

--- Mem0 ---
  SQLite in results: True
  PostgreSQL (stale) returned: True
  Time: 21,164ms
  Verdict: FAIL

--- Comparison ---
  MenteDB correctly suppressed stale belief. Mem0 did not.
  Speedup: MenteDB is 4,459x faster
```

Mem0 returns both the current and stale belief because flat vector search has no
concept of supersession. MenteDB's graph edges suppress the outdated memory
before it reaches the LLM.

**Requires `OPENAI_API_KEY`** (Mem0 uses OpenAI internally).

## Requirements

```bash
pip install -r benchmarks/requirements.txt
```

The `openai` package is only needed for the attention budget and noise ratio
tests. The stale belief and delta savings tests depend only on the `mentedb`
Python SDK.

## Interpreting Results

Each test prints a PASS/FAIL verdict along with detailed metrics. Here is the
actual output from a real run (Apple M-series, no LLM tests):

```
============================================================
  Stale Belief Test: PASS
============================================================
  Total memories stored: 9
  Query: user prefers PostgreSQL SQLite database
  SQLite rank: 0
  PostgreSQL rank: NOT FOUND (superseded)
  Belief propagation: Working
  Top results: 5
  Result 1 (score=0.516): The user has switched to SQLite. They no longer use PostgreSQL...

============================================================
  Delta Savings Test: PASS
============================================================
  Turns simulated: 20
  Total tokens (full retrieval): 2,660
  Total tokens (delta): 247
  Token savings: 90.7%

============================================================
  Sustained Conversation Test (100 turns, 3 projects): PASS
============================================================
  Total memories ingested: 100
  Belief changes tracked: 6
  Avg insert time: 0.29ms
  Avg search time: 77us
  Stale beliefs returned: 0% (0/6)
  Delta token savings (20 checkpoints): 90.1%
```

### Criterion Performance Benchmarks

Run with `cargo bench`:

```
insert_throughput/memories/100      time: [13.08 ms  13.34 ms  13.41 ms]
insert_throughput/memories/1000     time: [244.1 ms  244.4 ms  245.5 ms]
insert_throughput/memories/10000    time: [2.644 s   2.655 s   2.701 s ]

context_assembly/100                time: [216.0 µs  217.3 µs  222.6 µs]
context_assembly/1000               time: [340.8 µs  342.1 µs  342.4 µs]
context_assembly/10000              time: [690.1 µs  693.5 µs  694.4 µs]
```

Context assembly stays sub-millisecond even at 10,000 memories.

The summary at the end shows all results:

```
============================================================
  SUMMARY
============================================================
  Stale Belief................................ PASS
  Delta Savings............................... PASS
  Sustained Conversation...................... PASS
  Attention Budget............................ PASS
  Noise Ratio................................. PASS
```

Tests that require an API key will show SKIP if the key is not set. The exit
code is nonzero only if a test explicitly fails; skipped tests do not cause
failure.

## What These Results Prove

1. **Belief propagation works.** MenteDB's graph edges (`supersedes`,
   `contradicts`) correctly suppress stale information. No amount of vector
   similarity tuning achieves this in a flat store.

2. **Token budgets shrink.** Delta serving eliminates redundant context,
   reducing cost and latency proportionally to conversation length.

3. **Ordering matters.** Placing critical memories in high attention zones
   improves LLM compliance without changing the memories themselves.

4. **Structured extraction beats naive storage.** Fewer, higher quality memories
   mean less noise for the LLM to filter through at inference time.

5. **It scales.** 100 memories across 3 projects with 6 belief changes, sub-ms
   search, and correct supersession at every query.

6. **Mem0 cannot do this.** In a direct comparison, Mem0 returned stale data.
   MenteDB returned only the current belief. Graph-based belief propagation is
   fundamentally better than flat vector search for memory that changes over time.

## Caveats

We believe in honest benchmarking. Here is what these results do NOT prove:

1. **Mem0 speed comparison includes API latency.** Mem0's 21s includes OpenAI
   API calls for every `add()`. With a local embedding model, Mem0 would be
   faster. The correctness difference (PASS vs FAIL on stale belief suppression)
   is the real finding, not the raw speed number.

2. **100 memories is small.** The sustained conversation test is a good starting
   point, but production workloads with 10K+ memories would be more convincing.
   The Criterion benchmarks show sub-ms context assembly at 10K, but the
   quality benchmarks have not yet been validated at that scale.

3. **Hash embeddings are not semantic.** The engine benchmarks use `search_text`
   with hash embeddings (character n-gram based), not OpenAI or Cohere vectors.
   Real-world retrieval quality depends on the embedding model. MenteDB now
   supports pluggable providers (OpenAI, Cohere, Voyage) but these benchmarks
   have not yet been run with real semantic embeddings.
