"""Coverage in the production embedding space.

The hosted platform embeds with Cohere embed-english-v3 (1024 dim) on
Bedrock; the local harness floor is candle MiniLM (384 dim). This runner
stores the cached LLM parse atoms with explicit Cohere embeddings and
retrieves by explicit query embeddings, so retrieval quality is measured in
the space production actually runs in. Same ground truth, same tasks, same
coverage definition as run_coverage.py.

Usage:
  AWS_PROFILE=... python3 cohere_space.py --tasks tasks_kiali.json \
      --atoms fixtures/kiali_agents.atoms.json [--ks 4,6,8,12]
"""

import argparse
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "cohere.embed-english-v3"


def toks(s: str) -> int:
    return max(1, len(s) // 4)


def embed(client, texts, input_type):
    out = []
    for i in range(0, len(texts), 96):
        body = {
            "texts": [t[:2000] for t in texts[i : i + 96]],
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        r = client.invoke_model(
            modelId=MODEL,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        data = json.loads(r["body"].read())
        out.extend(data["embeddings"]["float"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--atoms", required=True)
    ap.add_argument("--ks", default="4,6,8,12")
    args = ap.parse_args()

    import boto3
    import mentedb

    client = boto3.Session(
        profile_name=os.environ.get("AWS_PROFILE"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    ).client("bedrock-runtime")

    spec = json.load(open(os.path.join(HERE, args.tasks)))
    rules = json.load(
        open(os.path.join(HERE, spec["file"].replace(".md", ".rules.json")))
    )["rules"]
    atoms = json.load(open(os.path.join(HERE, args.atoms)))
    file_tokens = toks(open(os.path.join(HERE, spec["file"])).read())
    ks = [int(k) for k in args.ks.split(",")]

    vectors = embed(client, [a["content"] for a in atoms], "search_document")
    db = mentedb.MenteDB(tempfile.mkdtemp(), dimension=1024)
    ids = []
    for a, v in zip(atoms, vectors):
        ids.append(db.store(a["content"], memory_type=a["memory_type"], embedding=v,
                            tags=a["tags"] or None))
    id_content = dict(zip(ids, (a["content"] for a in atoms)))
    print(f"stored {len(ids)} atoms with Cohere-1024 embeddings")

    rule_atom, unmapped = {}, []
    for rid, r in rules.items():
        needles = r["atom_match"]
        if isinstance(needles, str):
            needles = [needles]
        hit = next(
            (i for i, c in id_content.items() if any(n.lower() in c.lower() for n in needles)),
            None,
        )
        if hit:
            rule_atom[rid] = hit
        else:
            unmapped.append(rid)
    if unmapped:
        print(f"MAPPING GAPS: {unmapped}")

    qvecs = embed(client, [t["prompt"] for t in spec["tasks"]], "search_query")
    results = {k: {"cov": [], "tok": []} for k in ks}
    for task, qv in zip(spec["tasks"], qvecs):
        governing = [r for r in task["governing_rules"] if r in rule_atom]
        if not governing:
            continue
        gov_ids = {rule_atom[r] for r in governing}
        line = f"{task['id']:9s} rules={len(governing)}"
        for k in ks:
            got = {mid for mid, _ in db.recall_similar(qv, k)}
            cov = len(gov_ids & got) / len(gov_ids)
            tok = sum(toks(id_content.get(i, "")) for i in got)
            results[k]["cov"].append(cov)
            results[k]["tok"].append(tok)
            line += f" | k={k}: {cov:.0%}"
        print(line)

    print("\nCohere-1024 space, coverage vs tokens (mean across tasks):")
    print("k | coverage | tokens/turn | vs whole file")
    for k in ks:
        cov = sum(results[k]["cov"]) / len(results[k]["cov"])
        tok = sum(results[k]["tok"]) / len(results[k]["tok"])
        print(f"{k:2d} | {cov:8.0%} | ~{tok:6.0f} | {file_tokens / max(tok, 1):.1f}x less")
    db.close()


if __name__ == "__main__":
    main()
