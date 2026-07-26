"""Targeted config tuning on persistent stores: build each suite's store
once, then evaluate any number of injection configs by reopening, which
costs seconds instead of a rebuild. Iterate against the known misses; the
winning config gets a final fresh-rebuild multi-seed confirmation in
sweep.py before any PR.

Usage:
  python3 fast_tune.py --build             # one-time store build (parallel)
  python3 fast_tune.py --config '{"own_agent_boost":1.5}' [--k 12]
  python3 fast_tune.py --grid              # quick opinionated grid
"""

import argparse
import itertools
import json
import os

from sweep import SUITES, load_suite

STORES = "/tmp/sweepstores"
SPACE = os.environ.get("TUNE_SPACE", "candle")


def store_dir(suite):
    base = STORES + ("-cohere" if SPACE == "cohere" else "")
    return os.path.join(base, suite)


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


def rule_vecs(suite, rules):
    cache = os.path.join(HERE_Q, f"{suite}.rules.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    keys = sorted(rules.keys())
    texts = []
    for rid in keys:
        m = rules[rid]["atom_match"]
        texts.append(m[0] if isinstance(m, list) else m)
    vecs = cohere_vectors(texts, "search_document")
    out = dict(zip(keys, vecs))
    json.dump(out, open(cache, "w"))
    return out


def atom_vecs(suite):
    p = os.path.join(HERE_Q, f"{suite}.atomvecs.json")
    return json.load(open(p)) if os.path.exists(p) else None


def norm_text(t):
    return " ".join(t.replace("`", "").lower().split())


def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def query_vecs(suite):
    cache = os.path.join(HERE_Q, f"{suite}.queries.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    spec, _, _, _ = load_suite(suite)
    vecs = cohere_vectors([t["prompt"] for t in spec["tasks"]], "search_query")
    json.dump(vecs, open(cache, "w"))
    return vecs


HERE_Q = "/tmp/sweepstores-queries"
os.makedirs(HERE_Q, exist_ok=True)


def build(suite):
    import mentedb

    d = store_dir(suite)
    if os.path.exists(os.path.join(d, "LOCK")) or os.path.isdir(d):
        return f"{suite}: exists"
    os.makedirs(d, exist_ok=True)
    spec, rules, content, atoms = load_suite(suite)
    if SPACE == "cohere":
        if atoms is None:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
            return f"{suite}: skipped (no llm cache for cohere space)"
        db = mentedb.MenteDB(d, dimension=1024)
        vecs = cohere_vectors([a["content"] for a in atoms], "search_document")
        import hashlib

        json.dump(
            {hashlib.sha1(a["content"].encode()).hexdigest(): v
             for a, v in zip(atoms, vecs)},
            open(os.path.join(HERE_Q, f"{suite}.atomvecs.json"), "w"))
        for a, v in zip(atoms, vecs):
            db.store(a["content"], memory_type=a["memory_type"], embedding=v,
                     tags=a["tags"] or None)
    else:
        db = mentedb.MenteDB(d, embedding_provider="candle")
        if atoms is not None:
            for a in atoms:
                db.store(a["content"], memory_type=a["memory_type"], tags=a["tags"] or None)
        else:
            db.ingest_agent_file(content)
    db.close()
    return f"{suite}: built"


def score(suite, overrides, k):
    import mentedb

    spec, rules, content, atoms = load_suite(suite)
    if SPACE == "cohere":
        db = mentedb.MenteDB(store_dir(suite), dimension=1024,
                             injection_overrides=overrides or None)
        qvecs = query_vecs(suite)
    else:
        db = mentedb.MenteDB(store_dir(suite), embedding_provider="candle",
                             injection_overrides=overrides or None)
        qvecs = None
    covs, misses, toks = [], [], []
    for ti, task in enumerate(spec["tasks"]):
        if qvecs is not None:
            delivered = db.recall_for_injection(task["prompt"], k=k,
                                                query_embedding=qvecs[ti])
        else:
            delivered = db.recall_for_injection(task["prompt"], k=k)
        blob = norm_text("\n".join(m.get("content", "") for m in delivered))
        if task.get("action"):
            for rule in db.recall_for_action(task["action"], k=6):
                blob += "\n" + norm_text(rule.get("content", ""))
        governing = task["governing_rules"]
        rv = rule_vecs(suite, rules) if SPACE == "cohere" else None
        av = atom_vecs(suite) if SPACE == "cohere" else None
        delivered_vecs = []
        if av is not None:
            import hashlib

            for m in delivered:
                h = hashlib.sha1(m.get("content", "").encode()).hexdigest()
                if h in av:
                    delivered_vecs.append(av[h])
        hit = 0
        for rid in governing:
            needles = rules[rid]["atom_match"]
            if isinstance(needles, str):
                needles = [needles]
            if any(norm_text(n) in blob for n in needles):
                hit += 1
            elif rv is not None and any(cos(v, rv[rid]) >= 0.80 for v in delivered_vecs):
                hit += 1
        covs.append(hit / len(governing))
        toks.append(sum(max(1, len(m.get("content", "")) // 4) for m in delivered))
        if hit < len(governing):
            misses.append(f"{task['id']}:{hit}/{len(governing)}")
    db.close()
    return sum(covs) / len(covs), sum(toks) / len(toks), misses


def report(tag, overrides, k, suites):
    rows = [(s, *score(s, overrides, k)) for s in suites]
    worst = min(r[1] for r in rows)
    print(f"\n== {tag} k={k} | worst {worst:.0%}")
    for s, c, t, m in rows:
        print(f"  {s:9s} {c:4.0%} ~{t:5.0f}t  miss: {' '.join(m) if m else '-'}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--suites", default="codex,kiali,temporal,dev,ops")
    args = ap.parse_args()
    suites = args.suites.split(",")

    if args.build:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=5) as pool:
            for r in pool.map(build, suites):
                print(r)
        return

    if args.config is not None:
        report("config", json.loads(args.config), args.k, suites)
        return

    if args.grid:
        best = (0.0, None)
        for boost, frac, k in itertools.product([1.0, 1.25, 1.5], [0.0, 0.34], [12, 16, 20]):
            cfg = {"own_agent_boost": boost, "cluster_dominant_fraction": frac}
            worst = report(f"boost={boost} frac={frac}", cfg, k, suites)
            if worst > best[0]:
                best = (worst, (boost, frac, k))
        print(f"\nBEST: boost={best[1][0]} frac={best[1][1]} k={best[1][2]} worst={best[0]:.0%}")


if __name__ == "__main__":
    main()
