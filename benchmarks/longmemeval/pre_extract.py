#!/usr/bin/env python3
"""
Pre-extract all unique sessions from LongMemEval dataset.

Saves extraction results to a JSON cache file so the benchmark can
skip redundant LLM calls when processing per-question isolated DBs.

Usage:
    python benchmarks/longmemeval/pre_extract.py [--workers 20] [--cache-file extractions.json]

Environment:
    OPENAI_API_KEY / MENTEDB_LLM_PROVIDER / MENTEDB_LLM_API_KEY
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import has_llm_key

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

DATASET_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


def load_dataset(variant="s"):
    filepath = os.path.join(DATA_DIR, DATASET_FILES.get(variant, ""))
    if not os.path.exists(filepath):
        print(f"Dataset not found: {filepath}")
        sys.exit(1)
    with open(filepath) as f:
        return json.load(f)


def format_session(session, date):
    lines = [f"[Date: {date}]"]
    for turn in session:
        role = turn["role"].capitalize()
        lines.append(f"  {role}: {turn['content']}")
    return "\n".join(lines)


def collect_unique_sessions(dataset):
    """Deduplicate sessions across all questions."""
    seen = {}
    for q in dataset:
        for sid, sess, date in zip(
            q["haystack_session_ids"], q["haystack_sessions"], q["haystack_dates"]
        ):
            if sid not in seen:
                seen[sid] = (sess, date)
    return [(sid, sess, date) for sid, (sess, date) in seen.items()]


def extract_one(db, text, llm_provider, max_retries=5):
    for attempt in range(max_retries):
        try:
            return ("ok", db.extract(text, provider=llm_provider))
        except Exception as e:
            err = str(e).lower()
            if "401" in err or "unauthorized" in err:
                return ("fatal", str(e))
            if "404" in err or "not found" in err:
                return ("model_error", str(e))
            if attempt < max_retries - 1:
                time.sleep(3 * (2 ** attempt))
            else:
                return ("fail", str(e))
    return ("fail", "max retries exceeded")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract LongMemEval sessions")
    parser.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--cache-file", default=os.path.join(SCRIPT_DIR, "extraction_cache.json"))
    args = parser.parse_args()

    if not has_llm_key():
        print("Need OPENAI_API_KEY or MENTEDB_LLM_PROVIDER + MENTEDB_LLM_API_KEY")
        sys.exit(1)

    # Setup LLM provider
    explicit = os.environ.get("MENTEDB_LLM_PROVIDER", "")
    if explicit == "openai":
        cognitive_provider = "openai"
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    elif explicit == "anthropic":
        cognitive_provider = "anthropic"
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    elif os.environ.get("OPENAI_API_KEY"):
        cognitive_provider = "openai"
        os.environ.setdefault("MENTEDB_LLM_PROVIDER", "openai")
        os.environ.setdefault("MENTEDB_LLM_API_KEY", os.environ["OPENAI_API_KEY"])
    else:
        print("Need LLM provider for extraction")
        sys.exit(1)

    cognitive_model = os.environ.get("MENTEDB_LLM_MODEL", "gpt-4o-mini")
    os.environ["MENTEDB_LLM_MODEL"] = cognitive_model

    import mentedb as mentedb_pkg
    import tempfile

    # Create a throwaway DB just for the extract() method
    tmp_dir = tempfile.mkdtemp(prefix="longmemeval-extract-")
    db = mentedb_pkg.MenteDB(tmp_dir, embedding_provider="hash")

    dataset = load_dataset(args.variant)
    unique_sessions = collect_unique_sessions(dataset)

    # Load existing cache for resume support
    cache = {}
    if os.path.exists(args.cache_file):
        with open(args.cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached extractions from {args.cache_file}")

    # Filter out already-extracted sessions
    remaining = [(sid, sess, date) for sid, sess, date in unique_sessions if sid not in cache]

    print(f"{'='*60}")
    print(f"  LongMemEval Pre-Extraction")
    print(f"{'='*60}")
    print(f"  Dataset:     {args.variant} ({len(dataset)} questions)")
    print(f"  Total unique sessions: {len(unique_sessions)}")
    print(f"  Already cached:        {len(cache)}")
    print(f"  Remaining:             {len(remaining)}")
    print(f"  Extractor:   {cognitive_provider} / {cognitive_model}")
    print(f"  Workers:     {args.workers}")
    print(f"  Cache file:  {args.cache_file}")
    print(f"{'='*60}")

    if not remaining:
        print("\nAll sessions already extracted!")
        db.close()
        return

    # Prepare texts
    texts = {}
    for sid, sess, date in remaining:
        texts[sid] = format_session(sess, date)

    # Extract in parallel with periodic saves
    ok_count = 0
    fail_count = 0
    SAVE_INTERVAL = 200

    def do_extract(sid):
        return sid, extract_one(db, texts[sid], cognitive_provider)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(do_extract, sid): sid for sid, _, _ in remaining}
        pbar = tqdm(total=len(remaining), desc="Extracting")

        for future in as_completed(futures):
            sid, (status, payload) = future.result()
            pbar.update(1)

            if status == "ok":
                cache[sid] = payload
                ok_count += 1
            elif status == "fatal":
                print(f"\nFATAL: {payload[:200]}")
                break
            else:
                cache[sid] = None  # mark as failed so we don't retry forever
                fail_count += 1
                if fail_count <= 3:
                    print(f"\nFailed {sid}: {payload[:100]}")

            # Periodic save
            if (ok_count + fail_count) % SAVE_INTERVAL == 0:
                with open(args.cache_file, "w") as f:
                    json.dump(cache, f)

        pbar.close()

    # Final save
    with open(args.cache_file, "w") as f:
        json.dump(cache, f)

    db.close()
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    total_cached = sum(1 for v in cache.values() if v is not None)
    print(f"\nDone! {ok_count} new extractions, {fail_count} failures")
    print(f"Total cached: {total_cached} / {len(unique_sessions)}")
    print(f"Cache saved to: {args.cache_file}")


if __name__ == "__main__":
    main()
