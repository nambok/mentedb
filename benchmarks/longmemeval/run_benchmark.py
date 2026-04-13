#!/usr/bin/env python3
"""
LongMemEval Benchmark Runner for MenteDB

This script is intentionally simple — a thin client that exercises MenteDB's
engine APIs. ALL retrieval intelligence lives in the engine (search_expanded),
not here. This script only does:

  1. Ingest: feed chat sessions into MenteDB (parallel extraction for speed)
  2. Search: ONE call to db.search_expanded() per question
  3. Read: ONE LLM call to generate an answer from retrieved context
  4. Output: write hypothesis JSONL for evaluation by evaluate.py

The evaluation script (evaluate.py) is UNTOUCHED and uses the official
LongMemEval judge (gpt-4o-2024-08-06) with verbatim prompts.

Usage:
    python benchmarks/longmemeval/run_benchmark.py [options]

Environment:
    OPENAI_API_KEY      For embeddings and/or answer generation
    ANTHROPIC_API_KEY   Alternative for answer generation
    EMBEDDING_PROVIDER  Provider for embeddings (default: openai if key set)
    EMBEDDING_MODEL     Model for embeddings (default: text-embedding-3-small)
    TOP_K               Memories to retrieve per question (default: 20)
    DATASET             Dataset variant: s, m, oracle (default: s)
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

from tqdm import tqdm

# Add parent benchmarks dir for harness utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import get_llm_client, has_llm_key, llm_chat

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

DATASET_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}

# Simple reader prompt — no coaching, no tricks. Just present context and ask.
READER_PROMPT = """I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.

History Chats:

{retrieved_context}

Current Date: {question_date}
Question: {question}
Answer:"""


def load_dataset(variant="s"):
    filename = DATASET_FILES.get(variant)
    if not filename:
        print(f"Unknown dataset variant: {variant}")
        print(f"Options: {', '.join(DATASET_FILES.keys())}")
        sys.exit(1)

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Dataset not found: {filepath}")
        print("Run: python benchmarks/longmemeval/download_data.py")
        sys.exit(1)

    with open(filepath) as f:
        return json.load(f)


def format_session(session, date):
    """Format a chat session into a conversation string with its date."""
    lines = [f"[Date: {date}]"]
    for turn in session:
        role = turn["role"].capitalize()
        lines.append(f"  {role}: {turn['content']}")
    return "\n".join(lines)


def date_to_microseconds(date_str):
    """Convert a date string like '2024-03-15' to microseconds since epoch."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    except (ValueError, TypeError):
        return None


def _extract_one(db, text, llm_provider, max_retries=5):
    """Extract memories from one session (GIL released during HTTP call)."""
    for attempt in range(max_retries):
        try:
            return ("ok", db.extract(text, provider=llm_provider))
        except Exception as e:
            err = str(e).lower()
            if "401" in err or "unauthorized" in err or "invalid_api_key" in err:
                return ("fatal", str(e))
            if "404" in err or "not found" in err or "does not exist" in err:
                return ("model_error", str(e))
            # Retry with exponential backoff (3s, 6s, 12s, 24s)
            if attempt < max_retries - 1:
                time.sleep(3 * (2 ** attempt))
            else:
                return ("fail", str(e))
    return ("fail", "max retries exceeded")


def ingest_sessions(db, question_data, use_cognitive=True, llm_provider=None):
    """Ingest all chat sessions for a question into MenteDB.

    When use_cognitive=True, extractions run in parallel (GIL released during
    HTTP calls), then results are stored sequentially.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    sessions = question_data["haystack_sessions"]
    dates = question_data["haystack_dates"]
    thread = threading.current_thread().name
    total = len(sessions)

    if not (use_cognitive and llm_provider):
        # Baseline: raw session storage only
        memory_ids = []
        for i, (session, date) in enumerate(zip(sessions, dates)):
            text = format_session(session, date)
            ts = date_to_microseconds(date)
            mid = db.store(text, memory_type="episodic",
                          tags=[f"date:{date}", f"session:{i}"], created_at=ts)
            memory_ids.append(mid)
        return memory_ids

    # Phase 1: Extract all sessions in parallel (GIL released during HTTP)
    texts = [format_session(s, d) for s, d in zip(sessions, dates)]
    extracted = [None] * total
    cognitive_ok = 0
    cognitive_fail = 0
    first_error_msg = None

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_extract_one, db, text, llm_provider): i
                   for i, text in enumerate(texts)}
        done = 0
        for f in as_completed(futures):
            idx = futures[f]
            status, payload = f.result()
            done += 1

            if status == "ok":
                extracted[idx] = payload
                cognitive_ok += 1
            else:
                cognitive_fail += 1
                if first_error_msg is None:
                    first_error_msg = payload
                    if status == "fatal":
                        print(f"    [{thread}] ⚠️  FATAL: {payload[:200]}", flush=True)
                        print(f"    [{thread}] ⚠️  Check MENTEDB_LLM_API_KEY and MENTEDB_LLM_PROVIDER", flush=True)
                    elif status == "model_error":
                        print(f"    [{thread}] ⚠️  MODEL ERROR: {payload[:200]}", flush=True)
                    else:
                        print(f"    [{thread}] ⚠️  Extraction error: {payload[:200]}", flush=True)

            if done % 10 == 0 or done == total:
                print(f"    [{thread}] extracted {done}/{total} (ok: {cognitive_ok}, fail: {cognitive_fail})", flush=True)

    # Phase 2: Store all extracted memories sequentially (fast, local only)
    memory_ids = []
    for memories in extracted:
        if memories is not None:
            result = db.store_extracted(memories)
            memory_ids.extend(result.get("stored_ids", []))

    # Phase 3: Store raw sessions as episodic memories for keyword coverage.
    # Extracted facts capture the "gist" but may miss casual details like
    # place names or numbers. BM25 search on raw text catches those.
    # Use store_extracted for batch embedding (1 API call instead of 50).
    raw_memories = []
    for i, (text, date) in enumerate(zip(texts, dates)):
        # Cap content to avoid exceeding node size limits (32KB max)
        content = text[:8000] if len(text) > 8000 else text
        raw_memories.append({
            "content": content,
            "memory_type": "episodic",
            "tags": [f"date:{date}", f"session:{i}", "raw"],
            "confidence": 0.3,
            "embedding_key": text[:500],
        })
    result = db.store_extracted(raw_memories)
    memory_ids.extend(result.get("stored_ids", []))

    # Build community summaries for entity clusters (e.g., "health devices", "musical instruments")
    # These summaries make entities discoverable by abstract category queries.
    if use_cognitive and llm_provider:
        try:
            community_ids = db.build_communities()
            if community_ids:
                print(f"    [{thread}] built {len(community_ids)} community summaries", flush=True)
        except Exception as e:
            print(f"    [{thread}] community build failed: {e}", flush=True)

    print(f"    [{thread}] stored {len(memory_ids)} memories", flush=True)
    return memory_ids


def retrieve_and_answer(db, question_data, llm_client, llm_provider, top_k=20, reader_model=None):
    """Search MenteDB for relevant memories and generate an answer.

    Uses a single call to search_expanded() which handles all retrieval
    intelligence internally (query expansion, multi-pass, adaptive K).
    This function is intentionally simple — all smarts live in the engine.
    """
    question = question_data["question"]
    question_date = question_data["question_date"]

    # Convert question date to microseconds for time-aware filtering.
    before_ts = date_to_microseconds(question_date)

    # Single engine call — all retrieval intelligence is inside search_expanded()
    results = db.search_expanded(question, k=top_k, before=before_ts)

    retrieved_parts = []
    for r in results:
        try:
            mem = db.get_memory(r.id)
            content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
            if content:
                retrieved_parts.append(content)
        except Exception:
            continue

    if not retrieved_parts:
        return "I don't have enough information to answer this question."

    retrieved_context = "\n\n---\n\n".join(retrieved_parts)
    prompt = READER_PROMPT.format(
        retrieved_context=retrieved_context,
        question_date=question_date,
        question=question,
    )
    return llm_chat(llm_client, llm_provider, prompt, temperature=0.0,
                    max_tokens=300, model_override=reader_model).strip()


def process_single_question(question_data, embedding_provider, embedding_api_key,
                            embedding_model, use_cognitive, cognitive_provider,
                            llm_provider, top_k, reader_model=None):
    """Process a single question end-to-end. Designed to run in a thread pool."""
    import mentedb as mentedb_pkg
    import shutil
    import threading

    qid = question_data["question_id"]
    qtype = question_data.get("question_type", "unknown")
    thread = threading.current_thread().name
    n_sessions = len(question_data.get("haystack_sessions", []))
    tmp_dir = tempfile.mkdtemp(prefix=f"longmemeval-{qid}-")

    q_start = time.time()
    print(f"  [{thread}] {qid} ({qtype}) — ingesting {n_sessions} sessions...", flush=True)

    try:
        db = mentedb_pkg.MenteDB(
            tmp_dir,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
        )

        # Each thread needs its own LLM client (not thread-safe)
        llm_client, _ = get_llm_client()

        ingest_start = time.time()
        mids = ingest_sessions(db, question_data, use_cognitive=use_cognitive,
                       llm_provider=cognitive_provider)
        ingest_time = time.time() - ingest_start
        print(f"  [{thread}] {qid} — ingested {len(mids)} memories in {ingest_time:.0f}s, searching...", flush=True)

        search_start = time.time()
        # Retry search+answer on transient connection errors (20 attempts)
        for search_attempt in range(20):
            try:
                hypothesis = retrieve_and_answer(
                    db, question_data, llm_client, llm_provider, top_k=top_k,
                    reader_model=reader_model
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                # Don't retry auth or model-not-found errors
                if "401" in err_str or "authentication" in err_str:
                    raise
                if search_attempt < 19:
                    wait = min(3 * (2 ** min(search_attempt, 4)), 48)  # cap at 48s
                    print(f"  [{thread}] {qid} — search error (attempt {search_attempt+1}/20), retrying in {wait}s: {e}", flush=True)
                    time.sleep(wait)
                else:
                    raise
        search_time = time.time() - search_start
        db.close()

        total = time.time() - q_start
        preview = hypothesis[:80].replace("\n", " ")
        print(f"  [{thread}] {qid} — done in {total:.0f}s (ingest={ingest_time:.0f}s search+answer={search_time:.0f}s)", flush=True)
        print(f"  [{thread}] {qid} — answer: {preview}...", flush=True)
    except Exception as e:
        hypothesis = f"Error: {e}"
        print(f"  [{thread}] {qid} — ERROR: {e}", flush=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"question_id": qid, "hypothesis": hypothesis}


def run_benchmark(variant="s", top_k=60, limit=None, offset=0, resume_from=None,
                  use_cognitive=True, workers=3, reader_model=None):
    """Run the full LongMemEval benchmark."""
    if not has_llm_key():
        print("Need OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_MODEL for answer generation.")
        sys.exit(1)

    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "")
    embedding_api_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    # Default to OpenAI embeddings when API key available (much better retrieval
    # than hash). With parallel extraction, embedding overhead is manageable.
    if not embedding_provider:
        embedding_provider = "openai" if embedding_api_key else "hash"

    # Detect LLM provider for cognitive ingestion
    cognitive_provider = None
    if use_cognitive:
        # If user explicitly set MENTEDB_LLM_PROVIDER, respect it
        explicit_provider = os.environ.get("MENTEDB_LLM_PROVIDER", "")
        if explicit_provider == "openai":
            cognitive_provider = "openai"
            os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        elif explicit_provider == "anthropic":
            cognitive_provider = "anthropic"
            os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
        elif explicit_provider == "ollama":
            from harness import get_ollama_base_url, get_ollama_model
            cognitive_provider = "ollama"
            os.environ.setdefault("MENTEDB_LLM_API_KEY", "ollama")
            os.environ.setdefault("MENTEDB_LLM_API_URL", get_ollama_base_url())
            os.environ.setdefault("MENTEDB_LLM_MODEL", get_ollama_model())
        elif os.environ.get("ANTHROPIC_API_KEY"):
            cognitive_provider = "anthropic"
            os.environ.setdefault("MENTEDB_LLM_PROVIDER", "anthropic")
            os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["ANTHROPIC_API_KEY"])
        elif os.environ.get("OPENAI_API_KEY"):
            cognitive_provider = "openai"
            os.environ.setdefault("MENTEDB_LLM_PROVIDER", "openai")
            os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["OPENAI_API_KEY"])
        elif os.environ.get("OLLAMA_MODEL"):
            from harness import get_ollama_base_url, get_ollama_model
            cognitive_provider = "ollama"
            os.environ.setdefault("MENTEDB_LLM_PROVIDER", "ollama")
            os.environ.setdefault("MENTEDB_LLM_API_KEY", "ollama")
            os.environ.setdefault("MENTEDB_LLM_API_URL", get_ollama_base_url())
            os.environ.setdefault("MENTEDB_LLM_MODEL", get_ollama_model())
        else:
            print("Cognitive mode requires ANTHROPIC_API_KEY, OPENAI_API_KEY, or OLLAMA_MODEL.")
            print("Use --no-cognitive for raw storage baseline.")
            sys.exit(1)

    llm_client, llm_provider = get_llm_client()
    if not llm_client:
        print("Could not initialize LLM client.")
        sys.exit(1)

    # Smart defaults: cheap model for extraction, smart model for reading
    DEFAULTS = {
        "openai":    {"extractor": "gpt-4o-mini",             "reader": "gpt-4o"},
        "anthropic": {"extractor": "claude-haiku-4-5",  "reader": "claude-sonnet-4-20250514"},
        "ollama":    {"extractor": None,                      "reader": None},  # uses OLLAMA_MODEL
    }

    # Cognitive extraction model
    if cognitive_provider == "ollama":
        from harness import get_ollama_model
        cognitive_model = os.environ.get("MENTEDB_LLM_MODEL", get_ollama_model())
    else:
        default_extractor = DEFAULTS.get(cognitive_provider, {}).get("extractor", "gpt-4o-mini")
        cognitive_model = os.environ.get("MENTEDB_LLM_MODEL", default_extractor)
        # Also set the env var so the Rust engine picks it up
        os.environ["MENTEDB_LLM_MODEL"] = cognitive_model

    # Reader model (--reader-model flag overrides defaults)
    if reader_model:
        pass  # explicit override
    elif llm_provider == "ollama":
        from harness import get_ollama_model
        reader_model = get_ollama_model()
    else:
        reader_model = DEFAULTS.get(llm_provider, {}).get("reader", "gpt-4o-mini")

    print(f"{'='*60}")
    print(f"  LongMemEval Benchmark — MenteDB")
    print(f"{'='*60}")
    print(f"  Dataset:          {variant} ({len(load_dataset(variant))} questions)")
    print(f"  Limit:            {limit or 'all'}")
    print(f"  Embedding:        {embedding_provider} / {embedding_model}")
    print(f"  Cognitive:        {'ON' if cognitive_provider else 'OFF (raw storage)'}")
    if cognitive_provider:
        print(f"    Provider:       {cognitive_provider}")
        print(f"    Model:          {cognitive_model}")
    print(f"  Reader:           {llm_provider} / {reader_model}")
    print(f"  Top-K:            {top_k}")
    print(f"  Workers:          {workers}")
    print(f"  Engine features:")
    print(f"    Fact-augmented keys:  ON")
    print(f"    Turn decomposition:   ON")
    print(f"    Multi-query RRF:      ON (via search_expanded)")
    print(f"    Time-aware filter:    ON (before=question_date)")
    print(f"  Reader prompt:    official LongMemEval (verbatim)")
    print(f"  Judge:            gpt-4o-2024-08-06 (official)")
    print(f"{'='*60}")
    print()

    dataset = load_dataset(variant)
    total_available = len(dataset)
    if offset:
        dataset = dataset[offset:]
    if limit:
        dataset = dataset[:limit]

    # Name output file by range for pagination
    range_start = offset
    range_end = offset + len(dataset)
    run_tag = f"q{range_start}-{range_end}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, f"hypotheses_{run_tag}.jsonl")

    print(f"  Questions:        {range_start}..{range_end} of {total_available}")
    print(f"  Output:           {output_file}")
    print()

    # Support resuming from a partial run
    existing_ids = set()
    existing_results = []
    if resume_from and os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                entry = json.loads(line)
                existing_ids.add(entry["question_id"])
                existing_results.append(entry)
        print(f"  Resuming: {len(existing_ids)} questions already completed\n")
    elif os.path.exists(output_file):
        # Fresh run: clear previous results
        os.remove(output_file)
        print(f"  Cleared previous results: {output_file}\n")

    try:
        import mentedb as mentedb_pkg
    except ImportError:
        print("mentedb Python package not installed.")
        print("Build from source: cd sdks/python && maturin develop")
        sys.exit(1)

    # Filter out already-completed questions
    pending = [q for q in dataset if q["question_id"] not in existing_ids]

    results = existing_results[:]
    total_start = time.time()

    if workers <= 1:
        # Sequential mode (original behavior)
        for question_data in tqdm(pending, desc="Questions"):
            entry = process_single_question(
                question_data, embedding_provider, embedding_api_key,
                embedding_model, use_cognitive, cognitive_provider,
                llm_provider, top_k, reader_model=reader_model,
            )
            results.append(entry)
            with open(output_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
    else:
        # Parallel mode
        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for question_data in pending:
                future = executor.submit(
                    process_single_question,
                    question_data, embedding_provider, embedding_api_key,
                    embedding_model, use_cognitive, cognitive_provider,
                    llm_provider, top_k, reader_model,
                )
                futures[future] = question_data["question_id"]

            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                completed += 1
                qid = futures[future]
                try:
                    entry = future.result()
                except Exception as e:
                    entry = {"question_id": qid, "hypothesis": f"Error: {e}"}
                    print(f"\n  Error on {qid}: {e}")

                results.append(entry)
                with open(output_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                elapsed = time.time() - total_start
                avg = elapsed / completed
                remaining = avg * (total - completed)
                print(f"\r  [{completed}/{total}] {qid} done | "
                      f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining",
                      end="", flush=True)
            print()

    # Write final clean file (no duplicates from resume)
    with open(output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    total_time = time.time() - total_start
    n = len(results)
    print(f"\nDone. {n} questions answered in {total_time:.1f}s ({total_time/max(n,1):.1f}s/question)")
    print(f"Results saved to: {output_file}")
    print(f"\nNext step: python benchmarks/longmemeval/evaluate.py {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="LongMemEval Benchmark for MenteDB")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "s"),
                        choices=["s", "m", "oracle"],
                        help="Dataset variant (default: s)")
    parser.add_argument("--top-k", type=int,
                        default=int(os.environ.get("TOP_K", "60")),
                        help="Number of memories to retrieve (default: 60)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N questions (for testing)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions (for pagination, e.g. --offset 50 --limit 50 for page 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous partial run")
    parser.add_argument("--no-cognitive", action="store_true",
                        help="Disable cognitive pipeline (raw storage baseline)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of parallel workers (default: 3)")
    parser.add_argument("--reader-model", type=str, default=None,
                        help="Override reader model (default: gpt-4o for OpenAI, Sonnet for Anthropic)")
    args = parser.parse_args()

    run_benchmark(
        variant=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        offset=args.offset,
        resume_from=args.resume if args.resume else None,
        use_cognitive=not args.no_cognitive,
        workers=args.workers,
        reader_model=args.reader_model,
    )


if __name__ == "__main__":
    main()
