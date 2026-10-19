# MenteDB Quality Benchmarks

This suite measures specific MenteDB capabilities against simple baselines such
as naive RAG or a flat vector store. Each benchmark isolates one capability and
measures it with numbers. The sample outputs below say when and on what build
they were produced.

## Quick Start

```bash
# Install the Python SDK: the published wheel...
pip install mentedb
# ...or build it from the workspace root
cd sdks/python && maturin develop && cd ../..

# Run engine tests (no API key needed)
python benchmarks/run_all.py --no-llm

# Run all tests including LLM evaluations
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
python benchmarks/run_all.py
```

## Benchmarks

### Stale Belief Test (`stale_belief.py`)

Checks that MenteDB's supersession hides a fact once a newer fact
supersedes it. A store without supersession edges has no built in way to do
this.

**Scenario:** A user says "I use PostgreSQL" in turn 1, then "I switched to
SQLite" in turn 5. At turn 10, we query "What database does the user prefer?"

**What a plain vector store returns:** both facts, since nothing marks the
PostgreSQL one as outdated.

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

  - Insert time at 100 memories
  - Retrieval accuracy on 6 queries as memory grows
  - Belief update correctness (stale beliefs returned across 6 supersessions)
  - Delta token savings over 20 checkpoints
  - Project isolation (cross project leakage)

**Pass condition (in the script):** 0% stale returns and an average insert
under 5ms. Retrieval accuracy and leakage are reported but do not decide the
verdict. In the sample run below, on the current published wheel, this test
fails on insert time, and retrieval accuracy is 17% (1 of 6) with the hash
embedder.

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

Head to head on the stale belief scenario. Runs the same test on both engines
side by side.

**Output of one earlier run** (the log is not stored in the repo and the
versions used were not recorded):

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

Mem0 returned both the current and the stale belief in that run. Note how the
two sides are scored: on the MenteDB side the script itself drops the ids it
marked as superseded (`if r.id not in superseded`) before checking the results,
so this comparison does not isolate the engine. `stale_belief.py` is the test
where the engine alone hides the old fact. The timing compares a local hash
embedder with Mem0 making OpenAI calls, so the speed ratio says little about
either engine.

**Requires `OPENAI_API_KEY`** (Mem0 uses OpenAI internally).

## Requirements

```bash
pip install -r benchmarks/requirements.txt
```

The `openai` and `anthropic` packages are only needed for the attention budget
and noise ratio tests, and `mem0_comparison.py` also needs `mem0ai`. The stale
belief, delta savings and sustained conversation tests depend only on the
`mentedb` Python SDK.

## Interpreting Results

Each test prints a PASS/FAIL verdict along with detailed metrics. Here is the
output of `python benchmarks/run_all.py --no-llm` on 2026-10-18 (Apple M-series,
`mentedb` 0.32.2 from PyPI, hash embedder). The Python SDK on `main` did not
compile on that date, so the published wheel was used:

```
============================================================
  Stale Belief Test: PASS
============================================================
  Total memories stored: 9
  Query: user prefers PostgreSQL SQLite database
  SQLite rank: 0
  PostgreSQL rank: NOT FOUND (superseded)
  Supersession: Working
  Top results: 5
  Result 1 (score=0.350): The user has switched to SQLite. They no longer use PostgreSQL and prefer SQLite
  Result 2 (score=0.350): Meeting notes from standup: discussed deployment pipeline.
  Result 3 (score=0.350): The user prefers REST over GraphQL.

============================================================
  Delta Savings Test: PASS
============================================================
  Turns simulated: 20
  Total tokens (full retrieval): 2,660
  Total tokens (delta): 247
  Token savings: 90.7%
  Verdict: Delta serving saves 91% of context tokens

============================================================
  Sustained Conversation Test (100 turns, 3 projects): FAIL
============================================================
  Total memories ingested: 100
  Belief changes tracked: 6
  Avg insert time: 11.59ms
  Avg search time: 285us
  Retrieval accuracy (6 queries): 17%
  Stale beliefs returned: 0% (0/6)
  Delta token savings (20 checkpoints): 90.1%
  Project isolation: 5 relevant, 4 leaked
```

### Criterion Performance Benchmarks

Run with `cargo bench -p mentedb --bench insert_throughput --bench context_assembly`.

Earlier versions of this page listed sub millisecond context assembly at 10,000
memories and 13ms to insert 100 memories. Those figures did not reproduce on
`main` on 2026-10-18 (Apple M-series, other work running on the machine, so
treat these as rough):

```
context_assembly/recall_and_assemble/100     time: [2.4692 ms 3.6415 ms 5.0008 ms]
context_assembly/recall_and_assemble/1000    time: [9.0392 ms 13.204 ms 17.904 ms]
context_assembly/recall_and_assemble/10000   time: [2.0226 ms 2.0876 ms 2.1665 ms]
insert_throughput/memories/100               time: [1.3735 s  1.4593 s  1.5498 s ]
```

The 1,000 and 10,000 insert cases were stopped; at that speed criterion
estimated about 30 minutes for the 1,000 case alone. No latency claim is made
here until a clean run on an idle machine is recorded.

The summary at the end shows all results:

```
============================================================
  SUMMARY
============================================================
  Stale Belief............................ PASS
  Delta Savings........................... PASS
  Sustained Conversation.................. FAIL
```

Two more reruns of the sustained test gave the same 17% accuracy and 4 leaked
results, with average inserts of 17ms and 23ms. With `--no-llm` the attention
budget and noise ratio tests are not run and do not appear in the summary. The
exit code is nonzero if any test fails.

## What These Results Prove

1. **Supersession works in the stale belief test.** MenteDB's graph
   edges (`supersedes`, `contradicts`) hide the stale fact from search. A flat
   store has no equivalent mechanism.

2. **Token budgets shrink.** Delta serving eliminates redundant context,
   reducing cost and latency proportionally to conversation length.

3. **Ordering and extraction are testable.** The attention budget and noise
   ratio tests check whether U curve ordering and structured extraction help.
   No stored run of either is published here, so treat them as tools to run,
   not as results.

4. **Supersession holds at 100 memories.** In the sustained test no stale
   belief came back across 6 supersessions and search stayed under 1ms, but
   retrieval accuracy (17% with the hash embedder) and project isolation (4
   leaked results) are weak.

5. **Mem0 returned the stale fact in one run.** In the head to head, Mem0
   returned both the old and the new belief; see the scoring note in that
   section before drawing a wider conclusion.

## Caveats

We believe in honest benchmarking. Here is what these results do NOT prove:

1. **Mem0 speed comparison includes API latency.** Mem0's 21s includes OpenAI
   API calls for every `add()`. With a local embedding model, Mem0 would be
   faster. The correctness difference (PASS vs FAIL on stale belief suppression)
   is the real finding, not the raw speed number.

2. **100 memories is small.** The sustained conversation test is a starting
   point; production workloads with 10K+ memories would be more convincing,
   and the quality benchmarks have not been validated at that scale.

3. **Hash embeddings are not semantic.** The engine benchmarks use `search_text`
   with hash embeddings (character n-gram based), not OpenAI or Cohere vectors.
   Real world retrieval quality depends on the embedding model, and these
   benchmarks have not been run with real semantic embeddings.
