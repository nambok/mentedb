"""
Candle (local) vs OpenAI embedding quality comparison.

Runs the same retrieval accuracy tests with both providers to show
how local all-MiniLM-L6-v2 compares to OpenAI text-embedding-3-small.

No LLM required. Requires OPENAI_API_KEY for the OpenAI side.
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import print_result, has_openai_key


MEMORIES = [
    ("The user strongly prefers PostgreSQL for all database needs.", "semantic", ["database"]),
    ("The user likes Python for scripting tasks.", "semantic", ["language"]),
    ("The user's project deadline is next Friday.", "episodic", ["deadline"]),
    ("The user prefers dark mode in their IDE.", "semantic", ["ide"]),
    ("The user has switched to SQLite. They no longer use PostgreSQL.", "semantic", ["database"]),
    ("The user completed the authentication module.", "episodic", ["auth"]),
    ("The user prefers REST over GraphQL.", "semantic", ["api"]),
    ("Meeting notes from standup: discussed deployment pipeline.", "episodic", ["meeting"]),
    ("The user wants Docker for deployment.", "semantic", ["deployment"]),
    ("The project uses React for the frontend.", "semantic", ["frontend"]),
    ("CI/CD pipeline runs on GitHub Actions.", "semantic", ["ci"]),
    ("The team decided on JWT for authentication tokens.", "semantic", ["auth"]),
    ("Redis is used for session caching.", "semantic", ["caching"]),
    ("The API rate limit is set to 1000 requests per minute.", "semantic", ["api"]),
    ("The user prefers VS Code over JetBrains.", "semantic", ["ide"]),
]

QUERIES = [
    ("What database does the user prefer?", ["sqlite"], ["postgresql"]),
    ("What language for scripting?", ["python"], []),
    ("What IDE does the user use?", ["vs code"], []),
    ("How is authentication handled?", ["jwt"], []),
    ("What is the deployment strategy?", ["docker"], []),
    ("What caching solution is used?", ["redis"], []),
    ("What API framework?", ["rest"], ["graphql"]),
    ("What frontend framework?", ["react"], []),
]


def run_provider(provider_name, api_key=None):
    """Run accuracy test for a given provider. Returns (hits, total, avg_ms, details)."""
    import mentedb

    kwargs = {"embedding_provider": provider_name}
    if api_key:
        kwargs["embedding_api_key"] = api_key
        kwargs["embedding_model"] = "text-embedding-3-small"

    db = mentedb.MenteDB(tempfile.mkdtemp(), **kwargs)

    superseded = set()
    ids = []
    for content, mtype, tags in MEMORIES:
        mid = db.store(content, memory_type=mtype, tags=tags)
        ids.append(mid)

    # SQLite supersedes PostgreSQL
    db.relate(ids[4], ids[0], "supersedes", weight=1.0)
    superseded.add(ids[0])

    hits = 0
    total = len(QUERIES)
    search_times = []
    details = []

    for query, expected_present, expected_absent in QUERIES:
        t0 = time.perf_counter()
        results = db.search_text(query, 5)
        elapsed = (time.perf_counter() - t0) * 1000
        search_times.append(elapsed)

        filtered = [r for r in results if r.id not in superseded][:3]
        texts = []
        for r in filtered:
            mem = db.get_memory(r.id)
            texts.append(mem["content"].lower() if isinstance(mem, dict) else str(mem).lower())

        combined = " ".join(texts)
        found_expected = all(kw in combined for kw in expected_present)
        no_stale = all(kw not in combined for kw in expected_absent)
        passed = found_expected and no_stale
        if passed:
            hits += 1
        details.append((query, passed, texts[0][:60] if texts else "no results"))

    avg_ms = sum(search_times) / len(search_times)
    return hits, total, avg_ms, details


def main():
    print("=" * 60)
    print("  Candle vs OpenAI Embedding Quality Comparison")
    print("=" * 60)

    # Candle (always available)
    print("\n--- Candle (all-MiniLM-L6-v2, local) ---")
    t0 = time.perf_counter()
    c_hits, c_total, c_avg, c_details = run_provider("candle")
    c_total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Accuracy: {c_hits}/{c_total} ({100*c_hits/c_total:.0f}%)")
    print(f"  Avg search: {c_avg:.1f}ms")
    print(f"  Total time: {c_total_ms:.0f}ms")
    for q, passed, top in c_details:
        print(f"  {'PASS' if passed else 'FAIL'} | {q} -> {top}")

    # OpenAI (requires key)
    if not has_openai_key():
        print("\n--- OpenAI (skipped, no OPENAI_API_KEY) ---")
        print_result("Candle vs OpenAI", True, {
            "Candle accuracy": f"{c_hits}/{c_total} ({100*c_hits/c_total:.0f}%)",
            "OpenAI": "skipped (no API key)",
            "Candle avg search": f"{c_avg:.1f}ms",
        })
        return

    print("\n--- OpenAI (text-embedding-3-small) ---")
    t0 = time.perf_counter()
    o_hits, o_total, o_avg, o_details = run_provider("openai", os.environ["OPENAI_API_KEY"])
    o_total_ms = (time.perf_counter() - t0) * 1000
    print(f"  Accuracy: {o_hits}/{o_total} ({100*o_hits/o_total:.0f}%)")
    print(f"  Avg search: {o_avg:.1f}ms")
    print(f"  Total time: {o_total_ms:.0f}ms")
    for q, passed, top in o_details:
        print(f"  {'PASS' if passed else 'FAIL'} | {q} -> {top}")

    # Comparison
    print("\n--- Comparison ---")
    diff = c_hits - o_hits
    if diff > 0:
        verdict = f"Candle wins by {diff} queries"
    elif diff < 0:
        verdict = f"OpenAI wins by {-diff} queries"
    else:
        verdict = "Tied"

    speed_ratio = o_total_ms / c_total_ms if c_total_ms > 0 else 0

    print_result("Candle vs OpenAI", True, {
        "Candle accuracy": f"{c_hits}/{c_total} ({100*c_hits/c_total:.0f}%)",
        "OpenAI accuracy": f"{o_hits}/{o_total} ({100*o_hits/o_total:.0f}%)",
        "Verdict": verdict,
        "Candle avg search": f"{c_avg:.1f}ms",
        "OpenAI avg search": f"{o_avg:.1f}ms",
        "Candle total": f"{c_total_ms:.0f}ms (no API calls)",
        "OpenAI total": f"{o_total_ms:.0f}ms (includes API latency)",
    })


if __name__ == "__main__":
    main()
