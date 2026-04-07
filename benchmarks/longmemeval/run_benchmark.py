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
import time

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


def ingest_sessions(db, question_data, use_cognitive=True, llm_provider=None):
    """Ingest all chat sessions for a question into MenteDB.

    When use_cognitive=True, each session is processed through MenteDB's
    LLM extraction pipeline (fact extraction, knowledge updates, etc.).
    When False, sessions are stored as raw text (baseline mode).
    """
    memory_ids = []
    sessions = question_data["haystack_sessions"]
    dates = question_data["haystack_dates"]

    for i, (session, date) in enumerate(zip(sessions, dates)):
        text = format_session(session, date)

        if use_cognitive and llm_provider:
            # Full cognitive pipeline: LLM extracts structured memories
            try:
                result = db.ingest(text, provider=llm_provider)
                memory_ids.extend(result.get("stored_ids", []))
            except Exception as e:
                # Extraction can fail on some sessions (e.g. too long, parse errors)
                # Fall back to raw storage silently
                pass

            # Also store the raw session for retrieval coverage
            mid = db.store(text, memory_type="episodic", tags=[f"date:{date}", f"session:{i}"])
            memory_ids.append(mid)
        else:
            # Baseline: raw session storage only
            tags = [f"date:{date}", f"session:{i}"]
            mid = db.store(text, memory_type="episodic", tags=tags)
            memory_ids.append(mid)

    return memory_ids


def retrieve_and_answer(db, question_data, llm_client, llm_provider, top_k=20):
    """Search MenteDB for relevant memories and generate an answer.

    Uses the engine's search_expanded() for LLM-powered query decomposition + RRF.
    All intelligence lives in the engine, not here.
    """
    question = question_data["question"]
    question_date = question_data["question_date"]

    # Use engine-native expanded search (query decomposition + RRF happens inside)
    results = db.search_expanded(question, k=top_k)

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


def run_benchmark(variant="s", top_k=20, limit=None, resume_from=None, use_cognitive=True):
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

    print(f"LongMemEval Benchmark")
    print(f"  Dataset: {variant}")
    print(f"  Embedding: {embedding_provider} / {embedding_model}")
    print(f"  Cognitive: {'ON (' + cognitive_provider + ')' if cognitive_provider else 'OFF (raw storage)'}")
    print(f"  Reader: {llm_provider}")
    print(f"  Top-K: {top_k}")
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

    results = existing_results[:]
    total_time = 0

    try:
        import mentedb as mentedb_pkg
    except ImportError:
        print("mentedb Python package not installed.")
        print("Build from source: cd sdks/python && maturin develop")
        sys.exit(1)

    for qi, question_data in enumerate(tqdm(dataset, desc="Questions")):
        qid = question_data["question_id"]

        if qid in existing_ids:
            continue

        q_start = time.time()

        # Each question gets a fresh MenteDB instance (isolated memory)
        tmp_dir = tempfile.mkdtemp(prefix=f"longmemeval-{qid}-")
        try:
            db = mentedb_pkg.MenteDB(
                tmp_dir,
                embedding_provider=embedding_provider,
                embedding_api_key=embedding_api_key,
                embedding_model=embedding_model,
            )

            ingest_sessions(db, question_data, use_cognitive=use_cognitive, llm_provider=cognitive_provider)
            hypothesis = retrieve_and_answer(
                db, question_data, llm_client, llm_provider, top_k=top_k
            )
            db.close()
        except Exception as e:
            hypothesis = f"Error: {e}"
            print(f"\n  Error on {qid}: {e}")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        q_elapsed = time.time() - q_start
        total_time += q_elapsed

        entry = {
            "question_id": qid,
            "hypothesis": hypothesis,
        }
        results.append(entry)

        # Write incrementally so we can resume
        with open(output_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # Write final clean file (no duplicates from resume)
    with open(output_file, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

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
    args = parser.parse_args()

    run_benchmark(
        variant=args.dataset,
        top_k=args.top_k,
        limit=args.limit,
        resume_from=args.resume if args.resume else None,
        use_cognitive=not args.no_cognitive,
    )


if __name__ == "__main__":
    main()
