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

# Texts embedded per provider call. Remote embedders accept many inputs per
# request, so one call per batch amortizes the network round trip that would
# otherwise be paid once per memory.
EMBED_BATCH = 512

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


def run_scale_test(force_provider=None):
    if force_provider == "candle":
        provider = "candle"
        api_key = None
        provider_label = "Candle all-MiniLM-L6-v2 (local)"
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        provider = "openai" if api_key else None
        provider_label = "OpenAI text-embedding-3-small" if api_key else "hash (no API key)"

    print(f"\n  Embedding provider: {provider_label}")

    bench = MenteDBBenchmark(
        embedding_provider=provider,
        embedding_api_key=api_key,
        embedding_model="text-embedding-3-small" if api_key else None,
    )

    # Build the full ordered insert plan first so embeddings can be batched.
    # Order within a turn matches the original: any belief original first, then
    # the regular memory. Belief-change turns insert only the update.
    ops = []  # (content, tags, role, change_turn)
    for i in range(10000):
        turn = generate_memory(i)
        if i in BELIEF_CHANGES:
            update_content = BELIEF_CHANGES[i][1]
            ops.append((update_content, turn["tags"], "update", i))
            continue
        for change_turn, (original, _) in BELIEF_CHANGES.items():
            if i == change_turn - 100:
                ops.append((original, turn["tags"], "original", change_turn))
        ops.append((turn["content"], turn["tags"], "regular", None))

    # Phase 1: batch embed every memory. One provider call per EMBED_BATCH
    # inputs amortizes the network round trip that a naive per item insert pays
    # once per memory. This time is the embedding provider's, not the engine's.
    print(f"  Embedding {len(ops)} memories in batches of {EMBED_BATCH}...")
    contents = [c for (c, _, _, _) in ops]
    embeddings = []
    embed_start = time.time()
    for j in range(0, len(contents), EMBED_BATCH):
        embeddings.extend(bench.embed_batch(contents[j:j + EMBED_BATCH]))
    total_embed = time.time() - embed_start

    # Phase 2: store with precomputed embeddings. No provider call on the hot
    # path, so this times the ENGINE write (storage + index + graph) only.
    print(f"  Storing {len(ops)} memories (engine write only)...")
    belief_ids = {}
    insert_times = []
    total_start = time.time()
    for idx, ((content, tags, role, change_turn), emb) in enumerate(zip(ops, embeddings)):
        t0 = time.time()
        new_id = bench.store(content, memory_type="semantic", tags=tags, embedding=emb)
        insert_times.append(time.time() - t0)
        if role == "original":
            belief_ids[change_turn] = new_id
        elif role == "update" and change_turn in belief_ids:
            bench.relate(new_id, belief_ids[change_turn], "supersedes", weight=1.0)
        if (idx + 1) % 2000 == 0:
            print(f"    {idx+1}/{len(ops)} stored")
    total_insert = time.time() - total_start

    # Search tests. Time the provider embed and the engine search SEPARATELY:
    # a single figure would be dominated by the provider round trip and hide
    # the engine's real latency.
    search_times = []
    embed_query_times = []
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
        qvec = bench.embed(query)
        embed_query_times.append(time.time() - t0)

        t0 = time.time()
        results = bench.search_vec(qvec, 10)
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
    avg_query_embed = sum(embed_query_times) / len(embed_query_times) * 1000
    embed_per_mem = total_embed / len(ops) * 1000
    insert_rate = len(ops) / total_insert if total_insert > 0 else 0

    passed = stale_found == 0
    result = {
        "Total memories": len(ops),
        "Belief changes": len(BELIEF_CHANGES),
        "Engine insert (write only)": f"{avg_insert:.2f}ms/mem ({insert_rate:.0f} mem/s)",
        "Batch embed (provider)": f"{embed_per_mem:.2f}ms/mem amortized, {total_embed:.1f}s total",
        "Engine search at 10K (no embed)": f"{avg_search:.2f}ms",
        "Query embed (provider round trip)": f"{avg_query_embed:.2f}ms",
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
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else None
    run_scale_test(force_provider=provider)
