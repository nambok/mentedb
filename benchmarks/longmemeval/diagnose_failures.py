#!/usr/bin/env python3
"""
Diagnose benchmark failures — shows exactly WHERE each question fails.

For each question, runs the full pipeline and checks:
  1. EXTRACTION: Did the relevant facts get extracted into the DB?
  2. RETRIEVAL: Did search_expanded() find the relevant memories?
  3. SYNTHESIS: Did the engine's internal synthesis get it right?
  4. READER: Did the final LLM answer match the ground truth?

Usage:
    python benchmarks/longmemeval/diagnose_failures.py [--limit N] [--question-id QID]

Requires: MENTEDB_OPENAI_API_KEY or MENTEDB_ANTHROPIC_API_KEY
"""

import argparse
import json
import os
import re
import sys
import tempfile
import shutil
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness import get_llm_client, has_llm_key, llm_chat

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

READER_PROMPT = """I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.

History Chats:

{retrieved_context}

Current Date: {question_date}
Question: {question}
Answer:"""


def date_to_microseconds(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    except (ValueError, TypeError):
        return None


def search_db_for_keywords(db, keywords, before_ts=None):
    """Search the DB for specific keywords to check if they were extracted."""
    found = {}
    for kw in keywords:
        try:
            results = db.search(kw, k=10, before=before_ts)
            matches = []
            for r in results:
                try:
                    mem = db.get_memory(r.id)
                    content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
                    tags = mem.get("tags", []) if isinstance(mem, dict) else getattr(mem, "tags", [])
                    if content:
                        matches.append({
                            "id": r.id[:8],
                            "score": round(r.score, 3),
                            "content": content[:200],
                            "tags": tags[:5],
                        })
                except Exception:
                    continue
            found[kw] = matches
        except Exception as e:
            found[kw] = [{"error": str(e)}]
    return found


def run_diagnosis(question_data, embedding_provider, embedding_api_key, embedding_model,
                  llm_provider, llm_client, reader_model=None, verbose=True):
    """Run full pipeline on one question and diagnose where it fails."""
    import mentedb as mentedb_pkg

    qid = question_data["question_id"]
    question = question_data["question"]
    question_date = question_data["question_date"]
    ground_truth = question_data["answer"]
    qtype = question_data.get("question_type", "unknown")

    tmp_dir = tempfile.mkdtemp(prefix=f"diag-{qid}-")

    try:
        db = mentedb_pkg.MenteDB(
            tmp_dir,
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
        )

        # --- INGEST ---
        sessions = question_data.get("haystack_sessions", [])
        dates = question_data.get("haystack_dates", [])
        if not dates:
            dates = [question_date] * len(sessions)

        cognitive_provider = os.environ.get("MENTEDB_LLM_PROVIDER", "anthropic")

        texts = []
        for session, date in zip(sessions, dates):
            lines = [f"[Date: {date}]"]
            for turn in session:
                role = turn["role"].capitalize()
                lines.append(f"  {role}: {turn['content']}")
            texts.append("\n".join(lines))

        # Extract with LLM (parallel, like run_benchmark.py)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _extract_one(db, text, provider, max_retries=3):
            for attempt in range(max_retries):
                try:
                    return ("ok", db.extract(text, provider=provider))
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (2 ** attempt))
                    else:
                        return ("fail", str(e))

        extracted_memories = []
        n_sessions = len(texts)
        print(f"    Extracting {n_sessions} sessions (parallel)...", end="", flush=True)
        extracted = [None] * n_sessions
        ok_count = 0
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_extract_one, db, text[:8000], cognitive_provider): i
                       for i, text in enumerate(texts)}
            for f in as_completed(futures):
                idx = futures[f]
                status, payload = f.result()
                if status == "ok":
                    extracted[idx] = payload
                    ok_count += 1
                else:
                    print(f"\n    Session {idx} failed: {payload[:100]}", flush=True)
        for mems in extracted:
            if mems is not None:
                extracted_memories.extend(mems)
        print(f" {len(extracted_memories)} facts from {ok_count}/{n_sessions} sessions", flush=True)

        # Store raw sessions too
        raw_memories = []
        for i, (text, date) in enumerate(zip(texts, dates)):
            content = text[:8000] if len(text) > 8000 else text
            raw_memories.append({
                "content": content,
                "memory_type": "episodic",
                "tags": [f"date:{date}", f"session:{i}", "raw"],
                "confidence": 0.3,
                "embedding_key": text[:500],
            })
        print(f"    Storing {len(extracted_memories)} extracted + {len(raw_memories)} raw...", end="", flush=True)
        db.store_extracted(raw_memories)
        print(" done", flush=True)

        # Build communities
        print(f"    Building communities...", end="", flush=True)
        try:
            cids = db.build_communities()
            print(f" {len(cids) if cids else 0} built", flush=True)
        except Exception as e:
            print(f" failed: {e}", flush=True)

        n_extracted = len(extracted_memories)

        # --- RETRIEVAL ---
        print(f"    Searching...", end="", flush=True)
        before_ts = date_to_microseconds(question_date)
        results = db.search_expanded(question, k=20, before=before_ts)
        print(f" {len(results)} results", flush=True)

        retrieved_parts = []
        retrieved_details = []
        for r in results:
            try:
                mem = db.get_memory(r.id)
                content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
                tags = mem.get("tags", []) if isinstance(mem, dict) else getattr(mem, "tags", [])
                if content:
                    retrieved_parts.append(content)
                    retrieved_details.append({
                        "id": r.id[:8],
                        "score": round(r.score, 3),
                        "content": content[:150],
                        "tags": [t for t in tags[:8]],
                    })
            except Exception:
                continue

        # --- KEYWORD SEARCH (check if answer items are in DB at all) ---
        print(f"    Probing DB for answer keywords...", end="", flush=True)
        answer_keywords = extract_answer_keywords(ground_truth, question)
        keyword_search = search_db_for_keywords(db, answer_keywords, before_ts)
        found_count = sum(1 for m in keyword_search.values() if len(m) > 0)
        print(f" {found_count}/{len(answer_keywords)} found", flush=True)

        # --- READER ---
        print(f"    Generating answer...", end="", flush=True)
        if not retrieved_parts:
            hypothesis = "I don't have enough information to answer this question."
        else:
            retrieved_context = "\n\n---\n\n".join(retrieved_parts)
            prompt = READER_PROMPT.format(
                retrieved_context=retrieved_context,
                question_date=question_date,
                question=question,
            )
            hypothesis = llm_chat(llm_client, llm_provider, prompt, temperature=0.0,
                                  max_tokens=300, model_override=reader_model).strip()
        print(" done", flush=True)

        # --- QUICK EVAL (local heuristic, not GPT-4o judge) ---
        passed = local_eval(ground_truth, hypothesis)

        return {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "ground_truth": ground_truth,
            "hypothesis": hypothesis,
            "passed": passed,
            "n_extracted": n_extracted,
            "n_retrieved": len(retrieved_parts),
            "retrieved_top5": retrieved_details[:5],
            "keyword_probe": keyword_search,
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_answer_keywords(answer, question):
    """Pull key searchable terms from the ground truth answer."""
    answer = str(answer)
    keywords = []
    # Numbers with context
    numbers = re.findall(r'\$?\d[\d,.]+', answer)
    # Named items (capitalized words/phrases)
    named = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', answer)
    # Just use the answer itself as one search, plus key terms
    keywords.append(answer[:100])

    for n in named[:5]:
        if len(n) > 3:
            keywords.append(n)

    # Also search for key nouns from the question
    q_nouns = re.findall(r'\b(?:kitchen|antique|bike|bicycle|museum|market|workshop|health|device|pet|class|hobby|instrument)\w*', question.lower())
    for noun in q_nouns[:3]:
        keywords.append(noun)

    return list(dict.fromkeys(keywords))[:8]  # dedupe, max 8


def local_eval(ground_truth, hypothesis):
    """Quick local check — not a replacement for GPT-4o judge."""
    gt = str(ground_truth).lower().strip()
    hyp = str(hypothesis).lower().strip()

    # Number extraction
    gt_numbers = set(re.findall(r'\d+', gt))
    hyp_numbers = set(re.findall(r'\d+', hyp))

    # If ground truth has a clear number, check it
    if gt_numbers:
        if gt_numbers & hyp_numbers:
            return True

    # Substring containment
    if gt in hyp or hyp in gt:
        return True

    # Key phrase overlap
    gt_words = set(gt.split())
    hyp_words = set(hyp.split())
    overlap = gt_words & hyp_words
    if len(overlap) > len(gt_words) * 0.5:
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Diagnose benchmark failures")
    parser.add_argument("--limit", type=int, default=9, help="Number of questions to test")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N questions")
    parser.add_argument("--question-id", type=str, help="Test a specific question ID")
    parser.add_argument("--failures-only", action="store_true", help="Only show detailed output for failures")
    parser.add_argument("--dataset", default="s", choices=["s", "m", "oracle"])
    parser.add_argument("--reader-model", type=str, help="Override reader model")
    args = parser.parse_args()

    # Detect API keys
    oai_key = os.environ.get("MENTEDB_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    ant_key = os.environ.get("MENTEDB_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    if not oai_key and not ant_key:
        print("ERROR: Set MENTEDB_OPENAI_API_KEY or MENTEDB_ANTHROPIC_API_KEY")
        sys.exit(1)

    # Set up providers — mirror run_benchmark.py's env var mapping
    embedding_provider = "openai"
    embedding_api_key = oai_key
    embedding_model = "text-embedding-3-small"

    # Map MENTEDB_ prefixed keys to standard keys (for harness.py)
    if ant_key and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = ant_key
    if oai_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = oai_key

    # Set up MENTEDB_LLM_* env vars for the Rust engine (extract/search_expanded)
    # Prefer OpenAI for extraction (matches run_benchmark.py behavior)
    cognitive_provider = None
    if oai_key:
        cognitive_provider = "openai"
        os.environ["MENTEDB_LLM_PROVIDER"] = "openai"
        os.environ["MENTEDB_LLM_API_KEY"] = oai_key
    elif ant_key:
        cognitive_provider = "anthropic"
        os.environ["MENTEDB_LLM_PROVIDER"] = "anthropic"
        os.environ["MENTEDB_LLM_API_KEY"] = ant_key
    else:
        print("ERROR: Need OPENAI_API_KEY or ANTHROPIC_API_KEY for extraction")
        sys.exit(1)

    # Set extraction model
    DEFAULTS = {"anthropic": {"extractor": "claude-sonnet-4-20250514"}, "openai": {"extractor": "gpt-4o-mini"}}
    cognitive_model = os.environ.get("MENTEDB_LLM_MODEL", DEFAULTS.get(cognitive_provider, {}).get("extractor", "gpt-4o-mini"))
    os.environ["MENTEDB_LLM_MODEL"] = cognitive_model

    llm_client, llm_provider = get_llm_client()
    if llm_client is None:
        print("ERROR: No LLM client available")
        sys.exit(1)

    print(f"  Cognitive: {cognitive_provider} / {cognitive_model}")

    # Load dataset
    dataset_files = {"s": "longmemeval_s_cleaned.json", "m": "longmemeval_m_cleaned.json", "oracle": "longmemeval_oracle.json"}
    filepath = os.path.join(DATA_DIR, dataset_files[args.dataset])
    with open(filepath) as f:
        data = json.load(f)

    # Filter
    if args.question_id:
        data = [q for q in data if q["question_id"] == args.question_id]
        if not data:
            print(f"Question ID '{args.question_id}' not found")
            sys.exit(1)
    else:
        data = data[args.offset:args.offset + args.limit]

    print(f"\n{'='*80}")
    print(f"  MENTEDB FAILURE DIAGNOSIS — {len(data)} questions")
    print(f"  Provider: {llm_provider} | Embedding: {embedding_provider}/{embedding_model}")
    print(f"{'='*80}\n")

    results = []
    passed = 0
    failed = 0

    for i, q in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] {q['question_id']} ({q.get('question_type', '?')})")
        print(f"  Q: {q['question']}")
        print(f"  Expected: {str(q['answer'])[:100]}")

        try:
            result = run_diagnosis(
                q, embedding_provider, embedding_api_key, embedding_model,
                llm_provider, llm_client, reader_model=args.reader_model
            )
            results.append(result)

            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            if result["passed"]:
                passed += 1
            else:
                failed += 1

            print(f"  Answer: {result['hypothesis'][:100]}")
            print(f"  {status}  (extracted: {result['n_extracted']}, retrieved: {result['n_retrieved']})")

            # Show details for failures (or all if not --failures-only)
            if not result["passed"] or not args.failures_only:
                print(f"\n  --- DIAGNOSIS ---")
                print(f"  Ground truth: {result['ground_truth']}")
                print(f"  Hypothesis:   {result['hypothesis']}")

                print(f"\n  📦 Extraction: {result['n_extracted']} memories extracted")

                print(f"\n  🔍 Retrieval: top 5 of {result['n_retrieved']} retrieved:")
                for j, r in enumerate(result["retrieved_top5"]):
                    tags_str = ", ".join(r["tags"]) if r["tags"] else "none"
                    print(f"    [{j+1}] score={r['score']} | {r['content'][:120]}")
                    print(f"        tags: {tags_str}")

                print(f"\n  🔑 Keyword probe (is the answer in the DB?):")
                for kw, matches in result["keyword_probe"].items():
                    n = len(matches)
                    if n == 0:
                        print(f"    '{kw[:50]}' → ❌ NOT FOUND in DB")
                    else:
                        top = matches[0]
                        print(f"    '{kw[:50]}' → ✅ {n} matches (best: score={top.get('score', '?')})")
                        if not result["passed"]:
                            print(f"      best content: {top.get('content', '')[:120]}")

                if not result["passed"]:
                    # Diagnosis
                    all_found = all(len(m) > 0 for m in result["keyword_probe"].values())
                    if not all_found:
                        missing = [kw for kw, m in result["keyword_probe"].items() if len(m) == 0]
                        print(f"\n  💡 DIAGNOSIS: EXTRACTION GAP — these terms not in DB: {missing}")
                    elif result["n_retrieved"] == 0:
                        print(f"\n  💡 DIAGNOSIS: RETRIEVAL FAILURE — items in DB but search_expanded didn't find them")
                    else:
                        print(f"\n  💡 DIAGNOSIS: READER/SYNTHESIS ERROR — data was retrieved but answer is wrong")

        except Exception as e:
            failed += 1
            print(f"  💥 ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    total = passed + failed
    pct = (passed / total * 100) if total > 0 else 0
    print(f"\n{'='*80}")
    print(f"  RESULTS: {passed}/{total} ({pct:.1f}%)")
    print(f"  Passed: {passed} | Failed: {failed}")
    print(f"{'='*80}")

    if results:
        failures = [r for r in results if not r["passed"]]
        if failures:
            print(f"\n  FAILURE SUMMARY:")
            extraction_gaps = []
            retrieval_gaps = []
            reader_errors = []
            for r in failures:
                all_found = all(len(m) > 0 for m in r["keyword_probe"].values())
                if not all_found:
                    extraction_gaps.append(r["question_id"])
                elif r["n_retrieved"] == 0:
                    retrieval_gaps.append(r["question_id"])
                else:
                    reader_errors.append(r["question_id"])

            if extraction_gaps:
                print(f"    📦 Extraction gaps ({len(extraction_gaps)}): {extraction_gaps}")
            if retrieval_gaps:
                print(f"    🔍 Retrieval gaps ({len(retrieval_gaps)}): {retrieval_gaps}")
            if reader_errors:
                print(f"    📖 Reader/synthesis errors ({len(reader_errors)}): {reader_errors}")

    # Save detailed results
    out_path = os.path.join(SCRIPT_DIR, "results", "diagnosis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Detailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
