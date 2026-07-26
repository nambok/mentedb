#!/usr/bin/env python3
"""Instruction coverage for agent file ingestion, Level 1 of the legit test.

For each task prompt, ground truth names the rules that govern it. The file
is ingested into a real local engine (deterministic parser, candle
embeddings), the prompt retrieves k memories, and coverage@k is the fraction
of governing rules present in the injected subset. The whole file is 100%
coverage by definition; the question is how close retrieval gets at a
fraction of the tokens. The key output is the coverage vs tokens curve.

Honesty notes printed with results: token counts are len/4 estimates, the
deterministic parser is the floor (the LLM parser produces cleaner atoms),
and raw search_text is used rather than the gated injection path, so
production injection quality should be at or above these numbers.

Usage:
  python3 run_coverage.py [--tasks tasks_dev.json] [--ks 2,4,6,8,12]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def toks(s: str) -> int:
    return max(1, len(s) // 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks_dev.json"))
    ap.add_argument("--ks", default="2,4,6,8,12")
    ap.add_argument("--parser", default="deterministic", choices=["deterministic", "llm"])
    ap.add_argument("--atoms", default=None, help="cache path for the LLM parse")
    ap.add_argument("--retrieval", default="search", choices=["search", "injection"])
    ap.add_argument("--llm-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    args = ap.parse_args()

    import mentedb

    spec = json.load(open(args.tasks))
    fixture_path = os.path.join(HERE, spec["file"])
    rules_path = fixture_path.replace(".md", ".rules.json")
    content = open(fixture_path).read()
    rules = json.load(open(rules_path))["rules"]
    ks = [int(k) for k in args.ks.split(",")]

    tmp = tempfile.mkdtemp(prefix="agent-file-coverage-")
    db = mentedb.MenteDB(tmp, embedding_provider="candle")
    if args.parser == "llm":
        from llm_ingest import llm_ingest

        report = llm_ingest(
            db, content, getattr(args, "llm_model", getattr(args, "model", "")), args.atoms
        )
    else:
        report = db.ingest_agent_file(content)
    print(f"ingest: {json.dumps(report)}")
    file_tokens = toks(content)
    print(f"whole file every turn: ~{file_tokens} tok (len/4 estimate)\n")

    # Map every ground truth rule to its stored memory id by substring.
    # A rule whose atom cannot be found is a SEGMENTATION failure and is
    # reported as such rather than silently dropped.
    all_ids = []
    id_content = {}
    res = db.search_text("the entire file", 500)
    for r in res:
        m = db.get_memory(r.id)
        c = m.get("content", "") if isinstance(m, dict) else ""
        all_ids.append(r.id)
        id_content[r.id] = c
    rule_atom = {}
    unmapped = []
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
        print(f"SEGMENTATION GAPS, rules with no stored atom: {unmapped}\n")

    results = {k: {"cov": [], "tok": []} for k in ks}
    for task in spec["tasks"]:
        governing = [r for r in task["governing_rules"] if r in rule_atom]
        if not governing:
            continue
        gov_ids = {rule_atom[r] for r in governing}
        line = f"{task['id']:8s} rules={len(governing)}"
        for k in ks:
            if args.retrieval == "injection":
                got = {c["id"] for c in db.recall_for_injection(task["prompt"], k=k)}
            else:
                got = {r.id for r in db.search_text(task["prompt"], k)}
            # Action tasks also receive the action channel, exactly like the
            # hook does at the moment of the action.
            if task.get("action"):
                for rule in db.recall_for_action(task["action"], k=6):
                    got.add(rule["id"])
            cov = len(gov_ids & got) / len(gov_ids)
            tok = sum(toks(id_content.get(i, "")) for i in got)
            results[k]["cov"].append(cov)
            results[k]["tok"].append(tok)
            line += f" | k={k}: {cov:.0%} ~{tok}t"
        print(line)

    print("\ncoverage vs tokens (mean across tasks):")
    print("k | coverage | tokens/turn | vs whole file")
    for k in ks:
        cov = sum(results[k]["cov"]) / len(results[k]["cov"])
        tok = sum(results[k]["tok"]) / len(results[k]["tok"])
        print(f"{k:2d} | {cov:8.0%} | ~{tok:6.0f} | {file_tokens / max(tok, 1):.1f}x less")

    db.close()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
