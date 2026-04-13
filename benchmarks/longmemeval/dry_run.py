#!/usr/bin/env python3
"""
Dry run to verify the LongMemEval harness works without spending API credits.

Creates a small synthetic dataset, ingests into MenteDB using hash embeddings,
retrieves memories, and validates the end to end pipeline.

Usage:
    python benchmarks/longmemeval/dry_run.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SYNTHETIC_DATA = [
    {
        "question_id": "test_001",
        "question_type": "single-session-user",
        "question": "What is my dog's name?",
        "answer": "Biscuit",
        "question_date": "2024/06/01",
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_dates": ["2024/01/10", "2024/03/15", "2024/05/20"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I just adopted a golden retriever puppy! His name is Biscuit."},
                {"role": "assistant", "content": "Congratulations! Biscuit is a great name for a golden retriever."},
            ],
            [
                {"role": "user", "content": "Can you recommend a good pasta recipe?"},
                {"role": "assistant", "content": "Sure! Try aglio e olio, it is simple and delicious."},
            ],
            [
                {"role": "user", "content": "I need to buy new running shoes."},
                {"role": "assistant", "content": "What kind of terrain do you usually run on?"},
                {"role": "user", "content": "Mostly pavement, sometimes trails."},
            ],
        ],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "test_002",
        "question_type": "knowledge-update",
        "question": "Where do I work?",
        "answer": "Netflix",
        "question_date": "2024/06/01",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2024/01/05", "2024/04/20"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I just started a new job at Google!"},
                {"role": "assistant", "content": "That is exciting! What team are you on?"},
                {"role": "user", "content": "The search quality team."},
            ],
            [
                {"role": "user", "content": "Big news, I switched jobs. I am now at Netflix."},
                {"role": "assistant", "content": "Wow, congratulations on the move! What will you be working on?"},
                {"role": "user", "content": "Content recommendation systems."},
            ],
        ],
        "answer_session_ids": ["s2"],
    },
    {
        "question_id": "test_003_abs",
        "question_type": "single-session-user",
        "question": "What is my favorite board game?",
        "answer": "",
        "question_date": "2024/06/01",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2024/02/10"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I went hiking last weekend at Mt. Tamalpais."},
                {"role": "assistant", "content": "That is a beautiful area! How was the weather?"},
            ],
        ],
        "answer_session_ids": [],
    },
]


def main():
    print("LongMemEval Dry Run (no API keys needed)\n")

    try:
        import mentedb
    except ImportError:
        print("FAIL: mentedb Python package not installed.")
        print("Build from source: cd sdks/python && maturin develop")
        sys.exit(1)

    print("  mentedb package: OK")

    from run_benchmark import format_session, ingest_sessions

    passed = 0
    failed = 0

    for question_data in SYNTHETIC_DATA:
        qid = question_data["question_id"]
        tmp_dir = tempfile.mkdtemp(prefix=f"longmemeval-dry-{qid}-")

        try:
            db = mentedb.MenteDB(tmp_dir)

            # Ingest
            memory_ids = ingest_sessions(db, question_data)
            n_sessions = len(question_data["haystack_sessions"])
            assert len(memory_ids) == n_sessions, (
                f"Expected {n_sessions} memories, got {len(memory_ids)}"
            )

            # Search (hash embeddings, so results are approximate)
            results = db.search_text(question_data["question"], k=5)
            assert len(results) > 0, "Search returned no results"

            # Verify we can retrieve memory content
            mem = db.get_memory(results[0].id)
            content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
            assert len(content) > 0, "Retrieved memory has no content"

            db.close()

            print(f"  {qid}: PASS (ingested {n_sessions} sessions, retrieved {len(results)} memories)")
            passed += 1

        except Exception as e:
            print(f"  {qid}: FAIL ({e})")
            failed += 1

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("\nPipeline is working. Ready to run the full benchmark with API keys.")


if __name__ == "__main__":
    main()
