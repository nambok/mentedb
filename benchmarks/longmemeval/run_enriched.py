#!/usr/bin/env python3
"""
LongMemEval Benchmark with Sleeptime Enrichment — Two-Phase Architecture

Unlike run_benchmark.py (which creates isolated DBs per question), this script
mirrors how MenteDB is actually used in production:

  Phase 1 — Global Ingest:
    Ingest ALL unique sessions into a single shared DB with LLM extraction,
    then run the full 4-phase enrichment pipeline ONCE on the whole corpus.
    This builds cross-session entity links, community summaries, and a user model.

  Phase 2 — Query:
    For each question, query the shared DB with search_expanded().
    The engine surfaces relevant memories using the enriched graph.

This is ~30x cheaper than per-question enrichment and tests the real value
of enrichment: cross-session knowledge linking.

Usage:
    python benchmarks/longmemeval/run_enriched.py [options]

Environment:
    OPENAI_API_KEY         For embeddings and/or answer generation
    ANTHROPIC_API_KEY      Alternative for answer generation
    MENTEDB_LLM_PROVIDER   LLM provider for extraction + enrichment
    MENTEDB_LLM_API_KEY    API key for extraction + enrichment
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
    filepath = os.path.join(DATA_DIR, DATASET_FILES.get(variant, ""))
    if not os.path.exists(filepath):
        print(f"Dataset not found: {filepath}")
        print("Run: python benchmarks/longmemeval/download_data.py")
        sys.exit(1)
    with open(filepath) as f:
        return json.load(f)


def format_session(session, date):
    lines = [f"[Date: {date}]"]
    for turn in session:
        role = turn["role"].capitalize()
        lines.append(f"  {role}: {turn['content']}")
    return "\n".join(lines)


def date_to_microseconds(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    except (ValueError, TypeError):
        return None


def _extract_one(db, text, llm_provider, max_retries=5):
    for attempt in range(max_retries):
        try:
            return ("ok", db.extract(text, provider=llm_provider))
        except Exception as e:
            err = str(e).lower()
            if "401" in err or "unauthorized" in err or "invalid_api_key" in err:
                return ("fatal", str(e))
            if "404" in err or "not found" in err or "does not exist" in err:
                return ("model_error", str(e))
            if attempt < max_retries - 1:
                time.sleep(3 * (2 ** attempt))
            else:
                return ("fail", str(e))
    return ("fail", "max retries exceeded")


# ─── Phase 1: Global Ingest ─────────────────────────────────────────────────

def collect_unique_sessions(dataset):
    """Deduplicate sessions across all questions. Returns list of (session_id, session, date)."""
    seen = {}
    for q in dataset:
        for sid, sess, date in zip(
            q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]
        ):
            if sid not in seen:
                seen[sid] = (sess, date)
    return [(sid, sess, date) for sid, (sess, date) in seen.items()]


def global_ingest(db, unique_sessions, cognitive_provider, workers=10):
    """Ingest all unique sessions into a single shared DB with LLM extraction."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(unique_sessions)
    print(f"\n  Phase 1a: Extracting {total} unique sessions...")

    texts = []
    dates = []
    session_ids = []
    for sid, sess, date in unique_sessions:
        texts.append(format_session(sess, date))
        dates.append(date)
        session_ids.append(sid)

    # Extract in parallel
    extracted = [None] * total
    ok_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_one, db, text, cognitive_provider): i
            for i, text in enumerate(texts)
        }
        for f in tqdm(as_completed(futures), total=total, desc="Extracting"):
            idx = futures[f]
            status, payload = f.result()
            if status == "ok":
                extracted[idx] = payload
                ok_count += 1
            else:
                fail_count += 1
                if status == "fatal":
                    print(f"\n  FATAL: {payload[:200]}")
                    break

    print(f"  Extraction done: {ok_count} ok, {fail_count} failed")

    # Store extracted memories
    print(f"  Phase 1b: Storing extracted memories...")
    memory_count = 0
    for memories in tqdm(extracted, desc="Storing extracted"):
        if memories is not None:
            result = db.store_extracted(memories)
            memory_count += result.get("stored_ids", []).__len__()

    # Store raw sessions as semantic for BM25 keyword coverage.
    # Using semantic (not episodic) so enrichment_candidates() won't re-extract them.
    print(f"  Phase 1c: Storing raw sessions for BM25...")
    raw_memories = []
    for i, (text, date) in enumerate(zip(texts, dates)):
        content = text[:8000] if len(text) > 8000 else text
        raw_memories.append({
            "content": content,
            "memory_type": "semantic",
            "tags": [f"date:{date}", f"session_id:{session_ids[i]}", "raw"],
            "confidence": 0.3,
            "embedding_key": text[:500],
        })

    # Batch store in chunks to avoid memory pressure
    BATCH_SIZE = 100
    raw_count = 0
    for start in tqdm(range(0, len(raw_memories), BATCH_SIZE), desc="Storing raw"):
        batch = raw_memories[start:start + BATCH_SIZE]
        result = db.store_extracted(batch)
        raw_count += len(result.get("stored_ids", []))

    # Build community summaries (rule-based, from extraction phase)
    print(f"  Phase 1d: Building community summaries...")
    try:
        community_ids = db.build_communities()
        print(f"  Built {len(community_ids)} community summaries")
    except Exception as e:
        print(f"  Community build failed: {e}")
        community_ids = []

    print(f"\n  Ingest complete: {memory_count} extracted + {raw_count} raw memories")
    return memory_count + raw_count


def run_enrichment_phase(db, cognitive_provider):
    """Run enrichment Phases 2-4 on the shared DB.

    Phase 1 (extraction) was already done during ingest, so we call the
    individual building blocks for entity linking, communities, and user model.
    """
    print(f"\n  Phase 2: Running enrichment phases 2-4...")
    start = time.time()

    # Phase 2: Entity linking (rule-based)
    print(f"    Entity linking (rule-based)...")
    try:
        link_result = db.link_entities()
        print(f"      Linked: {link_result['linked']}, "
              f"Ambiguous: {link_result['ambiguous']}, "
              f"Edges: {link_result['edges_created']}")
    except Exception as e:
        print(f"      Failed: {e}")

    # Phase 3: Community detection (already built during ingest, rebuild to catch new entities)
    print(f"    Rebuilding community summaries...")
    try:
        community_ids = db.build_communities()
        print(f"      Built {len(community_ids)} community summaries")
    except Exception as e:
        print(f"      Failed: {e}")

    # Phase 2b + 3b + 4: LLM-powered enrichment (entity linking, community summaries, user model)
    # These require run_enrichment which needs episodic candidates.
    # Store a single marker episodic memory to trigger the pipeline.
    print(f"    Running LLM enrichment (entity linking, user model)...")
    try:
        marker_id = db.store(
            "Enrichment trigger marker",
            memory_type="episodic",
            tags=["enrichment_marker"],
        )
        result = db.run_enrichment(provider=cognitive_provider, current_turn=0)
        # Clean up marker
        try:
            db.forget(marker_id)
        except Exception:
            pass
        elapsed = time.time() - start
        print(f"  Enrichment complete in {elapsed:.0f}s:")
        print(f"    Memories stored:      {result['memories_stored']}")
        print(f"    Entities extracted:   {result['entities_extracted']}")
        print(f"    Duplicates skipped:   {result['duplicates_skipped']}")
        print(f"    Sync entity links:    {result['sync_linked']}")
        print(f"    LLM entity links:     {result['llm_linked']}")
        print(f"    Communities created:   {result['communities_created']}")
        print(f"    User model updated:   {result['user_model_updated']}")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"  LLM enrichment failed after {elapsed:.0f}s: {e}")
        return None


# ─── Phase 2: Query ──────────────────────────────────────────────────────────

def answer_question(db, question_data, llm_client, llm_provider, top_k=60,
                    reader_model=None):
    """Search the shared DB and generate an answer for one question."""
    question = question_data["question"]
    question_date = question_data["question_date"]
    before_ts = date_to_microseconds(question_date)

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


def query_phase(db, dataset, llm_client, llm_provider, top_k, reader_model,
                output_file, workers=3):
    """Answer all questions against the shared enriched DB."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    total_start = time.time()

    if workers <= 1:
        for q in tqdm(dataset, desc="Answering"):
            qid = q["question_id"]
            for attempt in range(10):
                try:
                    hyp = answer_question(db, q, llm_client, llm_provider,
                                          top_k=top_k, reader_model=reader_model)
                    break
                except Exception as e:
                    if attempt < 9:
                        time.sleep(3 * (2 ** min(attempt, 4)))
                    else:
                        hyp = f"Error: {e}"
            entry = {"question_id": qid, "hypothesis": hyp}
            results.append(entry)
            with open(output_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
    else:
        def _answer_one(q):
            qid = q["question_id"]
            # Each thread gets its own LLM client
            client, provider = get_llm_client()
            for attempt in range(10):
                try:
                    hyp = answer_question(db, q, client, provider,
                                          top_k=top_k, reader_model=reader_model)
                    return {"question_id": qid, "hypothesis": hyp}
                except Exception as e:
                    if attempt < 9:
                        time.sleep(3 * (2 ** min(attempt, 4)))
                    else:
                        return {"question_id": qid, "hypothesis": f"Error: {e}"}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_answer_one, q): q["question_id"] for q in dataset}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                entry = future.result()
                results.append(entry)
                with open(output_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                elapsed = time.time() - total_start
                avg = elapsed / completed
                remaining = avg * (len(dataset) - completed)
                print(f"\r  [{completed}/{len(dataset)}] {entry['question_id']} | "
                      f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining",
                      end="", flush=True)
            print()

    total_time = time.time() - total_start
    n = len(results)
    print(f"\n  Query phase done: {n} questions in {total_time:.1f}s "
          f"({total_time/max(n,1):.1f}s/question)")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_enriched_benchmark(variant="s", top_k=60, limit=None, offset=0,
                           workers=3, reader_model=None, skip_enrichment=False,
                           db_dir=None):
    if not has_llm_key():
        print("Need OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_MODEL.")
        sys.exit(1)

    import mentedb as mentedb_pkg

    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "")
    embedding_api_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    if not embedding_provider:
        embedding_provider = "openai" if embedding_api_key else "hash"

    # Detect cognitive provider
    cognitive_provider = None
    explicit = os.environ.get("MENTEDB_LLM_PROVIDER", "")
    if explicit == "openai":
        cognitive_provider = "openai"
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    elif explicit == "anthropic":
        cognitive_provider = "anthropic"
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    elif os.environ.get("ANTHROPIC_API_KEY"):
        cognitive_provider = "anthropic"
        os.environ.setdefault("MENTEDB_LLM_PROVIDER", "anthropic")
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["ANTHROPIC_API_KEY"])
    elif os.environ.get("OPENAI_API_KEY"):
        cognitive_provider = "openai"
        os.environ.setdefault("MENTEDB_LLM_PROVIDER", "openai")
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["OPENAI_API_KEY"])
    else:
        print("Need MENTEDB_LLM_PROVIDER + MENTEDB_LLM_API_KEY for enrichment.")
        sys.exit(1)

    DEFAULTS = {
        "openai":    {"extractor": "gpt-4o-mini",             "reader": "gpt-4o"},
        "anthropic": {"extractor": "claude-haiku-4-5",  "reader": "claude-sonnet-4-20250514"},
    }
    default_extractor = DEFAULTS.get(cognitive_provider, {}).get("extractor", "gpt-4o-mini")
    cognitive_model = os.environ.get("MENTEDB_LLM_MODEL", default_extractor)
    os.environ["MENTEDB_LLM_MODEL"] = cognitive_model

    llm_client, llm_provider = get_llm_client()
    if not reader_model:
        reader_model = DEFAULTS.get(llm_provider, {}).get("reader", "gpt-4o-mini")

    dataset = load_dataset(variant)
    total_questions = len(dataset)

    # Apply offset/limit first
    query_dataset = dataset[offset:] if offset else dataset
    if limit:
        query_dataset = query_dataset[:limit]

    # Collect unique sessions only for the questions we'll query
    unique_sessions = collect_unique_sessions(query_dataset)

    enrichment_label = "SKIP (--skip-enrichment)" if skip_enrichment else "ON (4-phase pipeline)"

    print(f"{'='*60}")
    print(f"  LongMemEval Enriched Benchmark — MenteDB")
    print(f"{'='*60}")
    print(f"  Architecture:     Two-phase (shared DB)")
    print(f"  Dataset:          {variant} ({total_questions} questions)")
    print(f"  Unique sessions:  {len(unique_sessions)}")
    print(f"  Query range:      {offset}..{offset + len(query_dataset)} of {total_questions}")
    print(f"  Embedding:        {embedding_provider} / {embedding_model}")
    print(f"  Extraction:       {cognitive_provider} / {cognitive_model}")
    print(f"  Enrichment:       {enrichment_label}")
    print(f"  Reader:           {llm_provider} / {reader_model}")
    print(f"  Top-K:            {top_k}")
    print(f"  Workers:          {workers}")
    print(f"{'='*60}")

    # Create or reuse DB
    if db_dir:
        tmp_dir = db_dir
        os.makedirs(tmp_dir, exist_ok=True)
        print(f"\n  Using DB dir: {tmp_dir}")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="longmemeval-enriched-")
        print(f"\n  Created DB dir: {tmp_dir}")

    db = mentedb_pkg.MenteDB(
        tmp_dir,
        embedding_provider=embedding_provider,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
    )

    # Check if DB already has data (resuming)
    try:
        test_results = db.search_text("test", k=1)
        has_data = len(test_results) > 0
    except Exception:
        has_data = False

    ingest_start = time.time()
    if has_data:
        print(f"\n  DB already has data, skipping ingest (use a fresh --db-dir to re-ingest)")
    else:
        # Phase 1: Global ingest
        global_ingest(db, unique_sessions, cognitive_provider, workers=workers)

        # Enrichment
        if not skip_enrichment:
            run_enrichment_phase(db, cognitive_provider)
        else:
            print(f"\n  Skipping enrichment (--skip-enrichment)")

    ingest_time = time.time() - ingest_start

    # Phase 2: Query
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "enriched" if not skip_enrichment else "baseline-shared"
    range_tag = f"q{offset}-{offset + len(query_dataset)}"
    output_file = os.path.join(RESULTS_DIR, f"hypotheses_{tag}_{range_tag}.jsonl")

    if os.path.exists(output_file):
        # Resume: load already-answered question IDs
        answered_ids = set()
        with open(output_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    answered_ids.add(entry["question_id"])
                except Exception:
                    continue
        remaining = [q for q in query_dataset if q["question_id"] not in answered_ids]
        print(f"\n  Resuming: {len(answered_ids)} already answered, {len(remaining)} remaining")
        print(f"  Output: {output_file}\n")
        if remaining:
            query_phase(db, remaining, llm_client, llm_provider, top_k,
                        reader_model, output_file, workers=workers)
        else:
            print(f"  All questions already answered!")
    else:
        print(f"\n  Phase 3: Answering {len(query_dataset)} questions...")
        print(f"  Output: {output_file}\n")
        query_phase(db, query_dataset, llm_client, llm_provider, top_k,
                    reader_model, output_file, workers=workers)

    db.close()

    total_time = time.time() - ingest_start
    print(f"\n  Total time: {total_time:.0f}s (ingest+enrichment: {ingest_time:.0f}s)")
    print(f"  DB dir: {tmp_dir}")
    print(f"  Results: {output_file}")
    print(f"\n  Next: python benchmarks/longmemeval/evaluate.py {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="LongMemEval with Sleeptime Enrichment (two-phase)")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "s"),
                        choices=["s", "m", "oracle"])
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit query phase to N questions")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions in query phase")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reader-model", type=str, default=None)
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Skip enrichment (baseline with shared DB)")
    parser.add_argument("--db-dir", type=str, default=None,
                        help="Reuse existing DB dir (skip ingest if populated)")

    args = parser.parse_args()
    run_enriched_benchmark(
        variant=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        offset=args.offset,
        workers=args.workers,
        reader_model=args.reader_model,
        skip_enrichment=args.skip_enrichment,
        db_dir=args.db_dir,
    )


if __name__ == "__main__":
    main()
