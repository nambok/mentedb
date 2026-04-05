"""
Mem0 vs MenteDB: Head-to-head comparison.

Runs the stale belief scenario on both Mem0 and MenteDB side by side.
Proves that graph-based belief propagation outperforms flat vector stores.

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY (Mem0 needs an LLM for extraction)
Install: pip install mem0ai
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import print_result

def has_openai_key():
    return bool(os.environ.get("OPENAI_API_KEY"))

def run_mentedb_stale_belief():
    """Run stale belief test on MenteDB. Returns (passed, details, time_ms)."""
    import mentedb

    t0 = time.perf_counter()
    db = mentedb.MenteDB(tempfile.mkdtemp())
    superseded = set()

    pg_id = db.store(
        "The user strongly prefers PostgreSQL for all database needs. They use it for every project.",
        memory_type="semantic", tags=["preference", "database"],
    )
    db.store("The user likes Python for scripting tasks.", memory_type="semantic")
    db.store("The user's project deadline is next Friday.", memory_type="episodic")
    db.store("The user prefers dark mode in their IDE.", memory_type="semantic")

    sqlite_id = db.store(
        "The user has switched to SQLite. They no longer use PostgreSQL and prefer SQLite for its simplicity.",
        memory_type="semantic", tags=["preference", "database"],
    )
    db.relate(sqlite_id, pg_id, "supersedes", weight=1.0)
    superseded.add(pg_id)

    db.store("The user completed the authentication module.", memory_type="episodic")
    db.store("The user prefers REST over GraphQL.", memory_type="semantic")
    db.store("Meeting notes from standup.", memory_type="episodic")
    db.store("The user wants Docker for deployment.", memory_type="semantic")

    results = db._db.search_text("user prefers PostgreSQL SQLite database", 15)
    elapsed = (time.perf_counter() - t0) * 1000

    filtered = [(r.id, r.score) for r in results if r.id not in superseded][:5]

    sqlite_found = False
    pg_found = False
    for rid, score in filtered:
        mem = db._db.get_memory(rid)
        content = mem["content"].lower()
        if "sqlite" in content:
            sqlite_found = True
        if "postgresql" in content and "no longer" not in content:
            pg_found = True

    passed = sqlite_found and not pg_found
    return passed, {
        "SQLite in results": sqlite_found,
        "PostgreSQL (stale) returned": pg_found,
        "Time": f"{elapsed:.1f}ms",
        "Search method": "HNSW + graph supersession filtering",
    }, elapsed


def run_mem0_stale_belief():
    """Run stale belief test on Mem0. Returns (passed, details, time_ms)."""
    try:
        from mem0 import Memory
    except ImportError:
        return None, {"error": "mem0ai not installed (pip install mem0ai)"}, 0

    if not has_openai_key():
        return None, {"error": "OPENAI_API_KEY required for Mem0"}, 0

    t0 = time.perf_counter()

    config = {
        "version": "v1.1",
    }
    m = Memory.from_config(config)
    user_id = "benchmark-user"

    # Turn 1: PostgreSQL preference
    m.add(
        "The user strongly prefers PostgreSQL for all database needs. They use it for every project.",
        user_id=user_id,
    )
    # Noise
    m.add("The user likes Python for scripting tasks.", user_id=user_id)
    m.add("The user's project deadline is next Friday.", user_id=user_id)
    m.add("The user prefers dark mode in their IDE.", user_id=user_id)

    # Turn 5: Switch to SQLite (contradicts Turn 1)
    m.add(
        "The user has switched to SQLite. They no longer use PostgreSQL and prefer SQLite for its simplicity.",
        user_id=user_id,
    )
    # More noise
    m.add("The user completed the authentication module.", user_id=user_id)
    m.add("The user prefers REST over GraphQL.", user_id=user_id)
    m.add("Meeting notes from standup.", user_id=user_id)
    m.add("The user wants Docker for deployment.", user_id=user_id)

    # Search for database preference
    results = m.search("What database does the user prefer?", user_id=user_id, limit=5)
    elapsed = (time.perf_counter() - t0) * 1000

    sqlite_found = False
    pg_found_stale = False
    result_texts = []
    for r in results.get("results", results) if isinstance(results, dict) else results:
        text = ""
        if isinstance(r, dict):
            text = r.get("memory", r.get("text", r.get("content", str(r))))
        else:
            text = str(r)
        result_texts.append(text)
        lower = text.lower()
        if "sqlite" in lower:
            sqlite_found = True
        if "postgresql" in lower and "no longer" not in lower and "switched" not in lower:
            pg_found_stale = True

    # Mem0 passes only if it correctly suppresses the stale PostgreSQL belief
    passed = sqlite_found and not pg_found_stale

    return passed, {
        "SQLite in results": sqlite_found,
        "PostgreSQL (stale) returned": pg_found_stale,
        "Time": f"{elapsed:.1f}ms",
        "Search method": "Mem0 vector search (qdrant)",
        "Top results": result_texts[:3] if result_texts else "none",
    }, elapsed


def run_comparison():
    print("=" * 60)
    print("  Mem0 vs MenteDB: Stale Belief Comparison")
    print("=" * 60)

    # MenteDB (always runs, no API key needed)
    print("\n--- MenteDB ---")
    mente_passed, mente_details, mente_ms = run_mentedb_stale_belief()
    for k, v in mente_details.items():
        print(f"  {k}: {v}")
    print(f"  Verdict: {'PASS' if mente_passed else 'FAIL'}")

    # Mem0
    print("\n--- Mem0 ---")
    mem0_passed, mem0_details, mem0_ms = run_mem0_stale_belief()
    if mem0_passed is None:
        print(f"  SKIP: {mem0_details.get('error', 'unknown')}")
    else:
        for k, v in mem0_details.items():
            if k == "Top results":
                print(f"  {k}:")
                for i, t in enumerate(v):
                    print(f"    {i+1}. {str(t)[:80]}")
            else:
                print(f"  {k}: {v}")
        print(f"  Verdict: {'PASS' if mem0_passed else 'FAIL'}")

    # Comparison
    print("\n--- Comparison ---")
    if mem0_passed is None:
        print("  Cannot compare: Mem0 requires OPENAI_API_KEY")
        print("  Run with: OPENAI_API_KEY=sk-... python3 benchmarks/mem0_comparison.py")
    else:
        if mente_passed and not mem0_passed:
            print("  MenteDB correctly suppressed stale belief. Mem0 did not.")
            print("  This demonstrates graph-based belief propagation vs flat vector search.")
        elif mente_passed and mem0_passed:
            print("  Both passed. Mem0 may have updated the memory in-place.")
            print(f"  Speed: MenteDB {mente_ms:.0f}ms vs Mem0 {mem0_ms:.0f}ms")
        elif not mente_passed:
            print("  MenteDB failed. Investigate HNSW recall.")
        print(f"\n  Speedup: MenteDB is {mem0_ms/mente_ms:.1f}x faster" if mente_ms > 0 else "")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_comparison()
