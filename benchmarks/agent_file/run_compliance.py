#!/usr/bin/env python3
"""Compliance end to end, Level 2 of the legit test.

A model performs each task under three conditions: the whole agent file in
context, the MenteDB retrieved subset (search plus the action channel,
exactly what injection provides), and nothing. Output is judged by the
MECHANICAL checkers declared in the rules file, string predicates, never an
LLM judge, so a pass is a pass anyone can reproduce.

The nothing condition is the capability baseline: rules a model follows with
no instructions at all say nothing about retrieval. The interesting numbers
are mentedb vs whole file compliance at their respective token costs.

Providers: ANTHROPIC_API_KEY (plain HTTPS) or --provider bedrock (boto3,
AWS_PROFILE respected). Token usage and cost are printed at the end.

Usage:
  python3 run_compliance.py [--tasks tasks_dev.json] [--k 6]
                            [--model claude-haiku-4-5-20251001] [--trials 3]
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request

USAGE = {"in": 0, "out": 0}

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.anthropic.com/v1/messages"

EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002700-\U000027bf\U0001f000-\U0001f0ff"
    "\U00002600-\U000026ff\U0001f900-\U0001f9ff]"
)


def toks(s: str) -> int:
    return max(1, len(s) // 4)


def check(spec: dict, out: str) -> bool:
    t = spec["type"]
    if t == "commit_single_line":
        # The commit message itself: first non empty block must be one line.
        lines = [l for l in out.strip().splitlines() if l.strip()]
        return len(lines) == 1
    if t == "regex_must_match":
        first = next((l for l in out.strip().splitlines() if l.strip()), "")
        return re.search(spec["pattern"], first) is not None
    if t == "no_emoji":
        return EMOJI.search(out) is None
    if t == "no_emoji_or_emdash":
        return EMOJI.search(out) is None and "—" not in out and "–" not in out
    if t == "contains_all":
        return all(n.lower() in out.lower() for n in spec["needles"])
    if t == "contains_any":
        return any(n.lower() in out.lower() for n in spec["needles"])
    if t == "not_contains":
        return not any(n in out for n in spec["needles"])
    raise ValueError(f"unknown checker {t}")


BEDROCK = None


def call_bedrock(model: str, system: str, prompt: str) -> str:
    global BEDROCK
    if BEDROCK is None:
        import boto3

        BEDROCK = boto3.Session(
            profile_name=os.environ.get("AWS_PROFILE"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        ).client("bedrock-runtime")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    for attempt in range(4):
        try:
            r = BEDROCK.invoke_model(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            data = json.loads(r["body"].read())
            u = data.get("usage", {})
            USAGE["in"] += u.get("input_tokens", 0)
            USAGE["out"] += u.get("output_tokens", 0)
            return "".join(b.get("text", "") for b in data.get("content", []))
        except Exception:
            if attempt < 3:
                time.sleep(2**attempt * 2)
                continue
            raise
    return ""


def call_model(model: str, system: str, prompt: str, key: str) -> str:
    body = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            return "".join(b.get("text", "") for b in data.get("content", []))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 529) and attempt < 3:
                time.sleep(2**attempt * 2)
                continue
            raise
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks_dev.json"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--model", default=os.environ.get("MODEL", "claude-haiku-4-5-20251001"))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "bedrock"])
    ap.add_argument("--parser", default="deterministic", choices=["deterministic", "llm"])
    ap.add_argument("--atoms", default=None, help="cache path for the LLM parse")
    ap.add_argument("--retrieval", default="search", choices=["search", "injection"])
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if args.provider == "anthropic" and not key:
        sys.exit("ANTHROPIC_API_KEY required, or pass --provider bedrock")

    import mentedb

    spec = json.load(open(args.tasks))
    fixture_path = os.path.join(HERE, spec["file"])
    rules_path = fixture_path.replace(".md", ".rules.json")
    content = open(fixture_path).read()
    rules = json.load(open(rules_path))["rules"]

    tmp = tempfile.mkdtemp(prefix="agent-file-compliance-")
    db = mentedb.MenteDB(tmp, embedding_provider="candle")
    if args.parser == "llm":
        from llm_ingest import llm_ingest

        report = llm_ingest(
            db, content, getattr(args, "llm_model", getattr(args, "model", "")), args.atoms
        )
    else:
        report = db.ingest_agent_file(content)
    print(f"ingest: stored={report.get('stored')} triggers={report.get('triggers')}")

    def mentedb_context(task) -> str:
        got = {}
        if args.retrieval == "injection":
            for c in db.recall_for_injection(task["prompt"], k=args.k):
                got[c["id"]] = c.get("content", "")
        else:
            for r in db.search_text(task["prompt"], args.k):
                m = db.get_memory(r.id)
                got[r.id] = m.get("content", "") if isinstance(m, dict) else ""
        if task.get("action"):
            for rule in db.recall_for_action(task["action"], k=6):
                got[rule["id"]] = rule.get("content", "")
        return "\n".join(f"- {c}" for c in got.values() if c)

    conditions = {
        "whole_file": lambda t: content,
        "mentedb": mentedb_context,
        "nothing": lambda t: "",
    }
    frame = (
        "You are a coding agent working in this repository. Follow your "
        "instructions exactly. Output ONLY the artifact asked for, no "
        "explanations.\n\nYour instructions:\n"
    )

    tally = {c: {"pass": 0, "total": 0, "tok": []} for c in conditions}
    for task in spec["tasks"]:
        governing = [(rid, rules[rid]["checker"]) for rid in task["governing_rules"] if rid in rules]
        for cname, builder in conditions.items():
            ctx = builder(task)
            system = frame + ctx if ctx else ""
            tally[cname]["tok"].append(toks(ctx))
            passes = {rid: 0 for rid, _ in governing}
            for _ in range(args.trials):
                if args.provider == "bedrock":
                    out = call_bedrock(args.model, system, task["prompt"])
                else:
                    out = call_model(args.model, system, task["prompt"], key)
                for rid, checker in governing:
                    if check(checker, out):
                        passes[rid] += 1
            line = f"{task['id']:8s} {cname:10s}"
            for rid, _ in governing:
                ok = passes[rid] > args.trials // 2
                tally[cname]["pass"] += 1 if ok else 0
                tally[cname]["total"] += 1
                line += f" | {rid}:{'PASS' if ok else 'fail'}({passes[rid]}/{args.trials})"
            print(line)

    print(f"\ncompliance by condition, model={args.model}, majority of {args.trials} trials:")
    print("condition  | compliance | instruction tokens/turn")
    for cname, t in tally.items():
        rate = t["pass"] / max(t["total"], 1)
        tok = sum(t["tok"]) / max(len(t["tok"]), 1)
        print(f"{cname:10s} | {rate:9.0%} | ~{tok:.0f}")
    print(
        "\nhonesty: mechanical checkers only, no judge; len/4 token estimates; "
        "the nothing condition is the capability baseline, subtract it mentally "
        "before crediting either instruction condition."
    )

    if USAGE["in"] or USAGE["out"]:
        cost = USAGE["in"] / 1e6 * 1.0 + USAGE["out"] / 1e6 * 5.0
        print(f"spend: {USAGE['in']} in + {USAGE['out']} out tokens, ~${cost:.2f} at haiku rates")

    db.close()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
