#!/usr/bin/env python3
"""
LongMemEval Benchmark Runner for MenteDB

Ingests chat sessions into MenteDB, retrieves relevant memories for each question,
generates answers via LLM, and outputs the standard hypothesis JSONL for evaluation.

Usage:
    python benchmarks/longmemeval/run_benchmark.py [options]

Environment:
    OPENAI_API_KEY      For embeddings and/or answer generation
    ANTHROPIC_API_KEY   Alternative for answer generation
    EMBEDDING_PROVIDER  Provider for embeddings (default: openai)
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


def ingest_sessions(db, question_data, use_cognitive=True, llm_provider=None):
    """Ingest all chat sessions for a question into MenteDB.

    When use_cognitive=True, each session is processed through MenteDB's
    LLM extraction pipeline (fact extraction, knowledge updates, etc.).
    When False, sessions are stored as raw text (baseline mode).
    """
    memory_ids = []
    sessions = question_data["haystack_sessions"]
    dates = question_data["haystack_dates"]

    thread = threading.current_thread().name
    total = len(sessions)
    for i, (session, date) in enumerate(zip(sessions, dates)):
        text = format_session(session, date)
        ts = date_to_microseconds(date)

        if use_cognitive and llm_provider:
            # Full cognitive pipeline: LLM extracts structured memories
            try:
                result = db.ingest(text, provider=llm_provider)
                memory_ids.extend(result.get("stored_ids", []))
                if (i + 1) % 10 == 0 or i == total - 1:
                    print(f"    [{thread}] session {i+1}/{total} ingested", flush=True)
            except Exception as e:
                print(f"    [{thread}] session {i+1}/{total} FAILED: {e}", flush=True)

            # Also store the raw session for retrieval coverage
            mid = db.store(text, memory_type="episodic",
                          tags=[f"date:{date}", f"session:{i}"],
                          created_at=ts)
            memory_ids.append(mid)
        else:
            # Baseline: raw session storage only
            tags = [f"date:{date}", f"session:{i}"]
            mid = db.store(text, memory_type="episodic", tags=tags, created_at=ts)
            memory_ids.append(mid)

    return memory_ids


def retrieve_and_answer(db, question_data, llm_client, llm_provider, top_k=20):
    """Search MenteDB for relevant memories and generate an answer.

    Uses the engine's search_expanded() for LLM-powered query decomposition + RRF.
    All intelligence lives in the engine, not here.
    """
    question = question_data["question"]
    question_date = question_data["question_date"]

    # Convert question date to microseconds for time-aware filtering.
    # Only retrieve memories created before the question was asked.
    before_ts = date_to_microseconds(question_date)

    # Use engine-native expanded search (query decomposition + RRF happens inside)
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

    answer = llm_chat(llm_client, llm_provider, prompt, temperature=0.0, max_tokens=300)
    return answer.strip()


def process_single_question(question_data, embedding_provider, embedding_api_key,
                            embedding_model, use_cognitive, cognitive_provider,
                            llm_provider, top_k):
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
        hypothesis = retrieve_and_answer(
            db, question_data, llm_client, llm_provider, top_k=top_k
        )
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


def run_benchmark(variant="s", top_k=20, limit=None, resume_from=None,
                  use_cognitive=True, workers=3):
    """Run the full LongMemEval benchmark."""
    if not has_llm_key():
        print("Need OPENAI_API_KEY or ANTHROPIC_API_KEY for answer generation.")
        sys.exit(1)

    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "")
    embedding_api_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    # Auto detect embedding provider: use OpenAI if key available, else hash
    if not embedding_provider:
        embedding_provider = "openai" if embedding_api_key else "hash"

    # Detect LLM provider for cognitive ingestion
    cognitive_provider = None
    if use_cognitive:
        if os.environ.get("ANTHROPIC_API_KEY"):
            cognitive_provider = "anthropic"
            os.environ.setdefault("MENTEDB_LLM_PROVIDER", "anthropic")
            os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["ANTHROPIC_API_KEY"])
        elif os.environ.get("OPENAI_API_KEY"):
            cognitive_provider = "openai"
            os.environ.setdefault("MENTEDB_LLM_PROVIDER", "openai")
            os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["OPENAI_API_KEY"])
        else:
            print("Cognitive mode requires ANTHROPIC_API_KEY or OPENAI_API_KEY.")
            print("Use --no-cognitive for raw storage baseline.")
            sys.exit(1)

    llm_client, llm_provider = get_llm_client()
    if not llm_client:
        print("Could not initialize LLM client.")
        sys.exit(1)

    # Resolve models for display
    cognitive_model = os.environ.get("MENTEDB_LLM_MODEL", "claude-sonnet-4-20250514" if cognitive_provider == "anthropic" else "gpt-4o-mini")
    reader_model = "claude-sonnet-4-20250514" if llm_provider == "anthropic" else "gpt-4o-mini"

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
    if limit:
        dataset = dataset[:limit]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, "hypotheses.jsonl")

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
                llm_provider, top_k,
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
                    llm_provider, top_k,
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
                        default=int(os.environ.get("TOP_K", "20")),
                        help="Number of memories to retrieve (default: 20)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N questions (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from a previous partial run")
    parser.add_argument("--no-cognitive", action="store_true",
                        help="Disable cognitive pipeline (raw storage baseline)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of parallel workers (default: 3)")
    args = parser.parse_args()

    run_benchmark(
        variant=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        resume_from=args.resume if args.resume else None,
        use_cognitive=not args.no_cognitive,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
