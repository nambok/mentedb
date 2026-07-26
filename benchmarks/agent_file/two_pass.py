"""Two-pass compliance: the definitive instrument.

Pass 1: memories retrieved for the task prompt, model writes the artifact.
Pass 2: retrieval runs AGAIN on what the model wrote (prompt + draft),
newly surfaced rules are delivered, the model revises. Mechanical checkers
judge the revision. This catches latent relevance rules whose applicability
is invisible in the prompt and only appears in the output (interface{}, px
margins, plain go test), which no prompt-side retrieval can fetch.

Conditions: whole_file (best case for the file), two_pass (ours), nothing.
"""

import json
import os
import sys
import time

os.environ.setdefault("TUNE_SPACE", "cohere")
from fast_tune import store_dir, cohere_vectors, norm_text
from run_compliance import check, call_bedrock, USAGE
from sweep import load_suite
import mentedb

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
REVISER_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
FRAME = ("You are a coding agent working in this repository. Follow your "
         "instructions exactly. Output ONLY the artifact asked for, no "
         "explanations.\n\nYour instructions:\n")
REVISE = ("You are a coding agent reviewing your own draft. Below is a "
          "list of CANDIDATE repository rules; only some apply to this "
          "draft. Identify every rule that applies to what the draft does, "
          "then fix ONLY the parts of the draft that violate an applicable "
          "rule; keep every other character exactly as it is. If the draft "
          "already complies with every applicable rule, output the draft "
          "EXACTLY unchanged. Output ONLY the artifact.\n\n"
          "Candidate rules:\n{rules}\n\nTask:\n{task}\n\nDraft:\n{draft}")


def retrieved(db, text, k):
    # Union of the injection pipeline and the raw vector head: measured on
    # these stores, the union covers rules either channel alone drops.
    qv = cohere_vectors([text[:2000]], "search_query")[0]
    out = [m.get("content", "") for m in db.recall_for_injection(text, k=k, query_embedding=qv)]
    for mid, _ in db.recall_similar(qv, 8):
        m = db.get_memory(mid)
        c = m.get("content", "") if isinstance(m, dict) else ""
        if c and c not in out:
            out.append(c)
    return out


def main():
    suites = (sys.argv[1] if len(sys.argv) > 1 else "codex,kiali,temporal").split(",")
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for suite in suites:
        spec, rules, content, atoms = load_suite(suite)
        db = mentedb.MenteDB(store_dir(suite), dimension=1024,
                             injection_overrides={"cluster_dominant_fraction": 0.34,
                                                  "cluster_fill_max": 6,
                                                  "cluster_max_span": 40})
        tally = {c: [0, 0] for c in ("whole_file", "two_pass", "nothing")}
        tok_two = []
        for task in spec["tasks"]:
            governing = [(rid, rules[rid]["checker"]) for rid in task["governing_rules"]]
            for cond in ("whole_file", "two_pass", "nothing"):
                passes = {rid: 0 for rid, _ in governing}
                for _ in range(trials):
                    if cond == "whole_file":
                        out = call_bedrock(MODEL, FRAME + content, task["prompt"])
                    elif cond == "nothing":
                        out = call_bedrock(MODEL, "", task["prompt"])
                    else:
                        ctx1 = retrieved(db, task["prompt"], 16)
                        draft = call_bedrock(
                            MODEL, FRAME + "\n".join(f"- {c}" for c in ctx1), task["prompt"])
                        ctx2 = retrieved(db, task["prompt"] + "\n" + draft[:1500], 40)
                        new_rules = [c for c in ctx2 if c not in ctx1]
                        if new_rules:
                            # The reviser sees ALL rules (pass 1 and pass 2):
                            # revising against only the new ones loses
                            # compliance the draft already had.
                            all_rules = ctx1 + new_rules
                            out = call_bedrock(
                                REVISER_MODEL, "",
                                REVISE.format(rules="\n".join(f"- {c}" for c in all_rules),
                                              task=task["prompt"], draft=draft))
                        else:
                            out = draft
                        if os.environ.get("TRACE_TASK") == task["id"]:
                            print(f"--- DRAFT {task['id']}:\n{draft[:400]}\n--- OUT:\n{out[:400]}")
                        tok_two.append(sum(max(1, len(c) // 4) for c in ctx1 + new_rules))
                    for rid, checker in governing:
                        if check(checker, out):
                            passes[rid] += 1
                for rid, _ in governing:
                    ok = passes[rid] > trials // 2
                    tally[cond][0] += 1 if ok else 0
                    tally[cond][1] += 1
                    if cond == "two_pass" and not ok:
                        print(f"  MISS {suite} {task['id']} {rid} ({passes[rid]}/{trials})")
        db.close()
        wf = tally["whole_file"]
        tp = tally["two_pass"]
        no = tally["nothing"]
        file_tok = max(1, len(content) // 4)
        two_tok = sum(tok_two) / max(len(tok_two), 1)
        print(f"{suite:9s} whole_file {wf[0]}/{wf[1]} ({wf[0]/wf[1]:.0%}) ~{file_tok}t | "
              f"two_pass {tp[0]}/{tp[1]} ({tp[0]/tp[1]:.0%}) ~{two_tok:.0f}t | "
              f"nothing {no[0]}/{no[1]} ({no[0]/no[1]:.0%})")
    cost = USAGE["in"] / 1e6 * 1.0 + USAGE["out"] / 1e6 * 5.0
    print(f"spend: ~${cost:.2f}")


if __name__ == "__main__":
    main()
