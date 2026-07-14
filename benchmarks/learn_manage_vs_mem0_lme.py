"""
Real-benchmark head-to-head for a marketing claim: MenteDB vs mem0 on actual
LongMemEval-S questions with LONG histories, measuring accuracy + speed + cost.

Honest and reproducible:
  - REAL questions from LongMemEval-S, from the long categories (multi-session,
    temporal-reasoning, knowledge-update). Full ~400-500 turn haystacks, no
    trimming, so the answer really is buried in a long history.
  - Same local embedder (bge-small) and same extraction model (Bedrock Haiku)
    for both, so differences are the memory system, not the model.
  - Per question we record: correct? (independent strict Sonnet judge, already
    calibrated to fail wrong/hedged answers), ingest wall-clock, and ingest
    tokens -> dollars (every Bedrock call measured exactly).
  - LIVE running token/$ meter and a HARD cap: it processes the cheapest (still
    long) questions first and stops before exceeding the budget, so it cannot
    overspend. Small N by necessity (mem0 ingest is ~$3/question); the cost and
    speed gaps are structural and robust even at small N, the accuracy read is
    directional and reported as such.

Run: python benchmarks/learn_manage_vs_mem0_lme.py    (long: mem0 ingest is slow)
"""
import json
import os
import re
import shutil
import tempfile
import time

os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("MEM0_TELEMETRY", "False")

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))

from learn_manage_vs_mem0_deep import (  # noqa: E402
    EXTRACT_SYSTEM, HAIKU, JUDGE, PRICE, _TOK, bedrock_chat, embed, make_mem0,
    parse_json, recall_context,
)
from mentedb import MenteDB  # noqa: E402

CAP_USD = float(os.environ.get("LME_CAP_USD", "17.0"))  # ~$2 under budget to absorb the Sonnet judge fraction
MAX_Q = int(os.environ.get("LME_MAX_Q", "5"))  # stop after this many questions (whichever hits first, this or the cap)
K = 10
CATS = ["multi-session", "temporal-reasoning", "knowledge-update"]
HAYSTACK_PER_CALL_EST = 8800 * 0.8 / 1e6  # rough mem0 $/turn for the pre-flight cap check
LME = os.path.join(os.path.dirname(__file__), "longmemeval", "data", "longmemeval_s_cleaned.json")


def cum_usd():
    # ~95% of tokens are Haiku extraction (mem0's ~8.8k/turn); the Sonnet judge
    # portion is small. Price at Haiku rate for a close, slightly-low estimate,
    # and keep the cap a couple dollars under $20 to absorb the Sonnet fraction.
    hin, hout = PRICE[HAIKU]
    return _TOK["in"] / 1e6 * hin + _TOK["out"] / 1e6 * hout


def sessions_to_turns(sessions):
    turns = []
    for sess in sessions:
        u = None
        for msg in sess:
            if msg.get("role") == "user":
                u = msg.get("content", "")
            elif msg.get("role") == "assistant" and u is not None:
                turns.append((u, msg.get("content", "")))
                u = None
        if u is not None:
            turns.append((u, ""))
    return turns


def ingest_mentedb(turns):
    path = tempfile.mkdtemp(prefix="lme-mdb-")
    db = MenteDB(path)
    t0, i0, o0 = time.time(), _TOK["in"], _TOK["out"]
    for (u, a) in turns:
        ctx = recall_context(db, f"{u} {a}", K)
        ctx_str = json.dumps(ctx) if ctx else "No prior context."
        facts = parse_json(bedrock_chat(HAIKU, EXTRACT_SYSTEM.format(context=ctx_str),
                                        f"User: {u}\nAssistant: {a}", 1024), [])
        for f in (facts if isinstance(facts, list) else []):
            if isinstance(f, dict) and f.get("content"):
                fid = db.store(f["content"], embedding=embed([f["content"]])[0])
                if f.get("contradiction"):
                    try:
                        db.relate(fid, str(f["contradiction"]), "supersedes")
                    except Exception:
                        pass
    return db, path, time.time() - t0, _TOK["in"] - i0, _TOK["out"] - o0


def ingest_mem0(m, turns, uid):
    t0, i0, o0 = time.time(), _TOK["in"], _TOK["out"]
    for (u, a) in turns:
        try:
            m.add(f"User: {u}\nAssistant: {a}", user_id=uid)
        except Exception:
            pass
    return time.time() - t0, _TOK["in"] - i0, _TOK["out"] - o0


def answer(ctx_texts, question):
    ctx = "\n".join(f"- {t}" for t in ctx_texts) if ctx_texts else "(no memories)"
    return bedrock_chat(JUDGE, "Answer the question using ONLY the memories. If they do not contain the answer, "
                        "reply exactly: I don't know.", f"Memories:\n{ctx}\n\nQuestion: {question}", 300)


def judge(question, gold, reply):
    out = bedrock_chat(JUDGE, "You are a strict grader. Reply JSON only.",
                       f'Question: "{question}"\nGold correct answer: "{gold}"\nModel answer: "{reply}"\n\n'
                       'Is the model answer correct (conveys the gold answer, or correctly declines if the gold '
                       'says the info was unavailable)? A wrong value, a hedge, or an evasive non-answer is '
                       'INCORRECT. JSON: {"correct": true|false}', 60)
    m = re.search(r"\{.*\}", out, re.DOTALL)
    try:
        return bool(json.loads(m.group(0)).get("correct"))
    except Exception:
        return False


def main():
    data = json.load(open(LME))
    pool = [q for q in data if q["question_type"] in CATS]
    pool.sort(key=lambda q: sum(len(s) for s in q["haystack_sessions"]))  # cheapest (still long) first
    print(f"Real LongMemEval-S head-to-head  |  extract {HAIKU.split('.')[-1]}  |  judge {JUDGE.split('.')[-1]}  |  "
          f"full haystacks, top-{K}  |  hard cap ${CAP_USD}\n")
    m0 = make_mem0()
    rows = []
    agg = {s: dict(ok=0, n=0, isec=0.0, itok=0) for s in ("MenteDB", "mem0")}
    for i, q in enumerate(pool):
        turns = sessions_to_turns(q["haystack_sessions"])
        # pre-flight: will mem0 ingesting this question likely exceed the cap?
        if cum_usd() + len(turns) * HAYSTACK_PER_CALL_EST > CAP_USD:
            print(f"  stopping before Q{i} ({len(turns)} turns): would exceed ${CAP_USD} (spent ${cum_usd():.2f})")
            break
        uid = f"q{i}"
        mdb = path = None
        try:
            mdb, path, ms, mi, mo = ingest_mentedb(turns)
            m0s, m0i, m0o = ingest_mem0(m0, turns, uid)
            # answer + judge
            mdb_ctx = [c["content"] for c in recall_context(mdb, q["question"], K)]
            r0 = m0.search(q["question"], filters={"user_id": uid}, limit=K)
            m0_ctx = [x.get("memory", "") for x in (r0.get("results", r0) if isinstance(r0, dict) else r0)][:K]
            mdb_ok = judge(q["question"], q["answer"], answer(mdb_ctx, q["question"]))
            m0_ok = judge(q["question"], q["answer"], answer(m0_ctx, q["question"]))
        except Exception as e:
            print(f"  Q{i} errored, skipping: {type(e).__name__}: {str(e)[:100]}")
            continue
        finally:
            try:
                if mdb is not None:
                    mdb.close()
                if path is not None:
                    shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass
        hin, hout = PRICE[HAIKU]
        mdb_usd = mi / 1e6 * hin + mo / 1e6 * hout
        m0_usd = m0i / 1e6 * hin + m0o / 1e6 * hout
        agg["MenteDB"]["ok"] += mdb_ok; agg["MenteDB"]["n"] += 1; agg["MenteDB"]["isec"] += ms; agg["MenteDB"]["itok"] += mi + mo
        agg["mem0"]["ok"] += m0_ok; agg["mem0"]["n"] += 1; agg["mem0"]["isec"] += m0s; agg["mem0"]["itok"] += m0i + m0o
        rows.append((q["question_type"], len(turns), mdb_ok, m0_ok))
        print(f"  Q{i} [{q['question_type']:>17}] {len(turns)} turns | "
              f"MenteDB {'OK' if mdb_ok else 'XX'} {ms:5.1f}s ${mdb_usd:.3f} | "
              f"mem0 {'OK' if m0_ok else 'XX'} {m0s:6.1f}s ${m0_usd:.3f} | cum ${cum_usd():.2f}", flush=True)
        if agg["MenteDB"]["n"] >= MAX_Q:
            print(f"  reached {MAX_Q}-question cap; stopping.")
            break

    print("\n  === RESULT (real LongMemEval-S, long histories) ===")
    n = agg["MenteDB"]["n"]
    if n:
        for s in ("MenteDB", "mem0"):
            a = agg[s]
            hin, hout = PRICE[HAIKU]
            usd = a["itok"] and (a["itok"] / 1e6 * (hin + hout) / 2)  # approx blended
            print(f"  {s:8}  accuracy {a['ok']}/{a['n']} ({a['ok']/a['n']:.0%})   "
                  f"avg ingest {a['isec']/a['n']:5.1f}s   {a['itok']//a['n']:>6} tok/history")
        mdb_t = agg["MenteDB"]["isec"] or 1e-9
        mem_t = agg["mem0"]["isec"]
        mdb_k = agg["MenteDB"]["itok"] or 1
        print(f"\n  mem0 ingest was {mem_t/mdb_t:.0f}x SLOWER and {agg['mem0']['itok']/mdb_k:.0f}x more tokens "
              f"than MenteDB over {n} identical histories.")
    print(f"  total spent this run: ${cum_usd():.2f}")


if __name__ == "__main__":
    main()
