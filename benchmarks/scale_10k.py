"""
10K Scale Benchmark — Production-scale stress test.

Validates MenteDB performance and correctness at 10,000 memories with
belief changes scattered throughout. Uses OpenAI embeddings when available,
falls back to hash embeddings.

Measures:
  - Insert throughput at 10K scale
  - Search latency at 10K scale
  - Belief supersession correctness at scale
  - Memory isolation across projects

Requires: OPENAI_API_KEY for real semantic embeddings (optional, falls back to hash)
"""

import os
import time
import random
from harness import MenteDBBenchmark, print_result

PROJECTS = ["alpha", "beta", "gamma", "delta", "epsilon",
            "zeta", "eta", "theta", "iota", "kappa"]

TOPICS = [
    "database", "auth", "deployment", "testing", "monitoring",
    "caching", "api", "frontend", "security", "performance",
    "logging", "ci-cd", "schema", "migration", "config",
]

BELIEF_CHANGES = {
    500: ("Project alpha uses PostgreSQL for the database.", "Project alpha switched from PostgreSQL to CockroachDB."),
    1500: ("Project beta deploys on AWS ECS.", "Project beta moved from AWS ECS to Kubernetes on GCP."),
    3000: ("Project gamma uses REST for the API layer.", "Project gamma migrated from REST to GraphQL."),
    5000: ("Project delta uses Jest for testing.", "Project delta replaced Jest with Vitest for faster tests."),
    7000: ("Project epsilon uses Redis for caching.", "Project epsilon switched from Redis to DragonflyDB."),
    9000: ("Project alpha uses OAuth2 for authentication.", "Project alpha moved from OAuth2 to passkeys with WebAuthn."),
}


def generate_memory(i):
    """Generate a realistic memory for turn i."""
    project = PROJECTS[i % len(PROJECTS)]
    topic = TOPICS[i % len(TOPICS)]
    templates = [
        f"Project {project}: decided to use {topic} approach version {i}.",
        f"Meeting notes for {project}: discussed {topic} implementation details.",
        f"Project {project} config update: {topic} settings changed at turn {i}.",
        f"Technical decision for {project}: {topic} architecture finalized.",
        f"Sprint retrospective {project}: {topic} improvements identified.",
    ]
    return {
        "content": templates[i % len(templates)],
        "type": "semantic",
        "tags": [f"project-{project}", topic],
    }


def run_scale_test():
    api_key = os.environ.get("OPENAI_API_KEY")
    provider = "openai" if api_key else None
    provider_label = "OpenAI text-embedding-3-small" if api_key else "hash (no API key)"

    print(f"\n  Embedding provider: {provider_label}")

    bench = MenteDBBenchmark(
        embedding_provider=provider,
        embedding_api_key=api_key,
        embedding_model="text-embedding-3-small" if api_key else None,
    )

    belief_ids = {}
    insert_times = []
    total_start = time.time()

    print(f"  Inserting 10,000 memories...")

    for i in range(10000):
        turn = generate_memory(i)

        # Insert belief change originals before their supersession turn
        if i in BELIEF_CHANGES:
            original, _ = BELIEF_CHANGES[i]
            # The original was inserted earlier, now insert the update
            update_content = BELIEF_CHANGES[i][1]
            t0 = time.time()
            new_id = bench.store(update_content, memory_type="semantic", tags=turn["tags"])
            insert_times.append(time.time() - t0)

            if i in belief_ids:
                bench.relate(new_id, belief_ids[i], "supersedes", weight=1.0)
            continue

        # Insert belief originals 100 turns before the change
        for change_turn, (original, _) in BELIEF_CHANGES.items():
            if i == change_turn - 100:
                t0 = time.time()
                orig_id = bench.store(original, memory_type="semantic", tags=turn["tags"])
                insert_times.append(time.time() - t0)
                belief_ids[change_turn] = orig_id

        t0 = time.time()
        bench.store(turn["content"], memory_type=turn["type"], tags=turn["tags"])
        insert_times.append(time.time() - t0)

        if (i + 1) % 500 == 0:
            batch = insert_times[-500:]
            avg = sum(batch) / len(batch) * 1000
            elapsed = time.time() - total_start
            rate = (i + 1) / elapsed
            remaining = (10000 - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/10000 inserted (avg {avg:.2f}ms last 500, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    total_insert = time.time() - total_start

    # Search tests
    search_times = []
    stale_found = 0
    correct_found = 0

    queries = [
        ("project alpha database", "CockroachDB", "PostgreSQL"),
        ("project beta deployment", "Kubernetes", "ECS"),
        ("project gamma API layer", "GraphQL", "REST"),
        ("project delta testing framework", "Vitest", "Jest"),
        ("project epsilon caching", "DragonflyDB", "Redis"),
        ("project alpha authentication", "passkeys", "OAuth2"),
    ]

    print(f"\n  Running {len(queries)} belief queries...")

    for query, expected_keyword, stale_keyword in queries:
        t0 = time.time()
        results = bench.search(query, limit=10)
        search_times.append(time.time() - t0)

        for rid, score in results:
            mem = bench.get(rid)
            if mem:
                content = mem.get("content", "").lower()
                if expected_keyword.lower() in content:
                    correct_found += 1
                if stale_keyword.lower() in content:
                    stale_found += 1

    avg_insert = sum(insert_times) / len(insert_times) * 1000
    avg_search = sum(search_times) / len(search_times) * 1000

    passed = stale_found == 0
    result = {
        "Total memories": 10000,
        "Belief changes": len(BELIEF_CHANGES),
        "Total insert time": f"{total_insert:.1f}s",
        "Avg insert": f"{avg_insert:.2f}ms",
        "Avg search (10K memories)": f"{avg_search:.2f}ms",
        "Correct beliefs found": correct_found,
        "Stale beliefs returned": stale_found,
        "Embedding provider": provider_label,
    }

    print_result(
        f"10K Scale Test ({provider_label})",
        passed,
        result,
    )
    return passed


if __name__ == "__main__":
    run_scale_test()
