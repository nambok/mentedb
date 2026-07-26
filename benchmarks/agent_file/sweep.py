"""Offline retrieval tuning sweep: prove a config reaches 100 percent
delivery locally through the REAL injection pipeline before any default
changes ship. No CI, no prod, no model calls (cached parses, candle
embeddings).

Per (suite, config, seed): fresh engine with injection_overrides, cached
atoms stored through the write pipeline, every task scored by needle
presence in what recall_for_injection actually returns. Seeds re-ingest
from scratch so HNSW graph randomness is averaged, not trusted.

Usage:
  python3 sweep.py --stage boost      # max_items x own_agent_boost grid
  python3 sweep.py --stage cluster    # cluster knobs on top of stage 1 best
  python3 sweep.py --config '{"own_agent_boost": 1.25}' --suites codex,kiali
"""

import argparse
import itertools
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

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


def run_once(suite, overrides, ks, seed):
    import mentedb

    spec, rules, content, atoms = load_suite(suite)
    tmp = tempfile.mkdtemp(prefix=f"sweep-{suite}-{seed}-")
    db = mentedb.MenteDB(tmp, embedding_provider="candle",
                         injection_overrides=overrides or None)
    if atoms is not None:
        for a in atoms:
            db.store(a["content"], memory_type=a["memory_type"], tags=a["tags"] or None)
    else:
        db.ingest_agent_file(content)

    out = {}
    for k in ks:
        covs, toks = [], []
        for task in spec["tasks"]:
            delivered = db.recall_for_injection(task["prompt"], k=k)
            blob = "\n".join(m.get("content", "") for m in delivered).lower()
            if task.get("action"):
                for rule in db.recall_for_action(task["action"], k=6):
                    blob += "\n" + rule.get("content", "").lower()
            governing = task["governing_rules"]
            hit = 0
            for rid in governing:
                needles = rules[rid]["atom_match"]
                if isinstance(needles, str):
                    needles = [needles]
                if any(n.lower() in blob for n in needles):
                    hit += 1
            covs.append(hit / len(governing))
            toks.append(sum(max(1, len(m.get("content", "")) // 4) for m in delivered))
        out[k] = (sum(covs) / len(covs), sum(toks) / len(toks),
                  [(t["id"], c) for t, c in zip(spec["tasks"], covs) if c < 1.0])
    db.close()
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    return out


def _job(args):
    suite, config, ks, seed = args
    return suite, run_once(suite, config, ks, seed)


def evaluate_multi(config, ks, suites, seeds, pool):
    jobs = [(s, config, ks, seed) for s in suites for seed in range(seeds)]
    per = {}
    for suite, out in pool.map(_job, jobs):
        per.setdefault(suite, []).append(out)
    tables = {}
    for k in ks:
        rows = []
        for suite in suites:
            outs = [o[k] for o in per[suite]]
            cs = [o[0] for o in outs]
            ts = [o[1] for o in outs]
            rows.append((suite, sum(cs) / len(cs), min(cs),
                         sum(ts) / len(ts), outs[-1][2]))
        tables[k] = rows
    return tables


def show(tag, rows):
    worst = min(r[1] for r in rows)
    print(f"\n== {tag} | worst-suite mean {worst:.0%}")
    for suite, mean_c, min_c, tok, missed in rows:
        mstr = " ".join(f"{i}:{c:.0%}" for i, c in missed) if missed else "-"
        print(f"  {suite:9s} mean {mean_c:4.0%} min {min_c:4.0%} ~{tok:5.0f}t  miss: {mstr}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["boost", "cluster"], default=None)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--suites", default="codex,kiali,temporal,dev,ops")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    suites = args.suites.split(",")

    from concurrent.futures import ProcessPoolExecutor

    ks = [12, 16, 20]
    with ProcessPoolExecutor(max_workers=8) as pool:
        if args.stage == "boost":
            best = (0.0, None)
            for boost in [1.0, 1.25, 1.5]:
                tables = evaluate_multi({"own_agent_boost": boost}, ks, suites,
                                        args.seeds, pool)
                for k in ks:
                    worst = show(f"k={k} boost={boost}", tables[k])
                    if worst > best[0]:
                        best = (worst, (k, boost))
            print(f"\nSTAGE1 BEST: k={best[1][0]} boost={best[1][1]} worst={best[0]:.0%}")
        elif args.stage == "cluster":
            base = json.loads(args.config)
            best = (0.0, None)
            for frac, fill, span in itertools.product([0.0, 0.25, 0.34], [4, 6], [12, 20]):
                cfg = dict(base)
                cfg.update({"cluster_dominant_fraction": frac, "cluster_fill_max": fill,
                            "cluster_max_span": span})
                tables = evaluate_multi(cfg, [args.k], suites, args.seeds, pool)
                worst = show(f"frac={frac} fill={fill} span={span}", tables[args.k])
                if worst > best[0]:
                    best = (worst, (frac, fill, span))
            print(f"\nSTAGE2 BEST: {best[1]} worst={best[0]:.0%}")
        else:
            tables = evaluate_multi(json.loads(args.config), [args.k], suites,
                                    args.seeds, pool)
            show("single", tables[args.k])


if __name__ == "__main__":
    main()
