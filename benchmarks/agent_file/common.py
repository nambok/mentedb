"""Shared helpers: suite loading, Cohere embeddings, persistent stores."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STORES = os.environ.get("MDB_BENCH_STORES", "/tmp/agent-file-bench-stores")

SUITES = {
    "codex": ("tasks_codex.json", "codex_agents", "llm"),
    "kiali": ("tasks_kiali.json", "kiali_agents", "llm"),
    "temporal": ("tasks_temporal.json", "temporal_agents", "llm"),
    "freerouting": ("tasks_freerouting.json", "freerouting_agents", "llm"),
    "dev": ("tasks_dev.json", "dev_claude", "det"),
    "ops": ("tasks_ops.json", "ops_meridian", "det"),
}


def load_suite(name):
    tasks_file, fixture, mode = SUITES[name]
    spec = json.load(open(os.path.join(HERE, tasks_file)))
    rules = json.load(open(os.path.join(HERE, f"fixtures/{fixture}.rules.json")))["rules"]
    content = open(os.path.join(HERE, f"fixtures/{fixture}.md")).read()
    atoms = None
    cache = os.path.join(HERE, f"fixtures/{fixture}.atoms.json")
    if mode == "llm" and os.path.exists(cache):
        atoms = json.load(open(cache))
    return spec, rules, content, atoms


def norm_text(t):
    return " ".join(t.replace("`", "").lower().split())


def cohere_vectors(texts, input_type):
    import boto3

    client = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"),
                           region_name="us-east-1").client("bedrock-runtime")
    out = []
    for i in range(0, len(texts), 96):
        body = {"texts": [t[:2000] for t in texts[i:i + 96]],
                "input_type": input_type, "embedding_types": ["float"]}
        r = client.invoke_model(modelId="cohere.embed-english-v3", body=json.dumps(body),
                                contentType="application/json", accept="application/json")
        out.extend(json.loads(r["body"].read())["embeddings"]["float"])
    return out


def store_dir(suite):
    return os.path.join(STORES, suite)


def ensure_store(suite):
    """Build the suite's persistent store from the cached parse on first use."""
    import mentedb

    d = store_dir(suite)
    if os.path.isdir(d):
        return d
    spec, rules, content, atoms = load_suite(suite)
    if atoms is None:
        raise SystemExit(f"{suite}: no atoms cache; regenerate with llm_ingest.py first")
    os.makedirs(d, exist_ok=True)
    db = mentedb.MenteDB(d, dimension=1024)
    vecs = cohere_vectors([a["content"] for a in atoms], "search_document")
    for a, v in zip(atoms, vecs):
        db.store(a["content"], memory_type=a["memory_type"], embedding=v,
                 tags=a["tags"] or None)
    db.close()
    return d
