"""
STRESS accuracy probe: one big corpus, unguessable facts, deep update chains,
many same-topic distractors, tight retrieval. Built to actually make retrieval
fail so the 100%/100% tie either breaks or is proven robust.

Why the earlier ties were suspect and this is harder:
  - ONE combined corpus (~70 memories), not a fresh 15-memory store per question,
    so top-k must find the needle among many.
  - UNGUESSABLE values (arbitrary regions, ports, dollar amounts, codenames), so
    the answer LLM cannot reason it from priors, it MUST come from retrieval.
    (Grounding was already verified: empty context -> "I don't know".)
  - DEEP update chains (value changes 2-3 times); the system must surface the
    LATEST, not any earlier link.
  - MANY same-topic distractors (a dozen other regions/ports) competing for the
    top-k slots.
  - Tight k. The validated strict judge grades; every question is PRINTED with
    each system's retrieved context and answer so it can be eyeballed.

Reuses the validated deep/hard harness. Run: python benchmarks/learn_manage_vs_mem0_stress.py
"""

import shutil

from learn_manage_vs_mem0_deep import (  # noqa: E402
    HAIKU, JUDGE, ingest_mem0, ingest_mentedb, make_mem0, recall_context,
)
from learn_manage_vs_mem0_hard import answer, strict_score  # noqa: E402

K = 5

# One long history. Target facts have update chains; everything else is a
# same-topic distractor so retrieval has to discriminate. All values arbitrary.
def _ack(_):
    return "Noted."


TURNS = [
    # --- datacenter region: chain ap-southeast-2 -> us-east-1 -> eu-west-3 ---
    ("Our primary datacenter region is ap-southeast-2.",),
    ("The backup region is sa-east-1.",),
    ("Our analytics warehouse runs in af-south-1.",),
    ("We migrated the primary region to us-east-1.",),
    ("The CDN edge nodes are mostly in eu-north-1.",),
    ("A partner team runs their service in ca-central-1.",),
    ("We ran a DR test in me-south-1 last month.",),
    ("Primary region moved again, it is eu-west-3 now.",),
    ("The staging environment lives in ap-northeast-1.",),
    ("Our secondary analytics replica is in us-west-2.",),
    # --- auth service port: chain 8410 -> 8443 -> 8500 ---
    ("The auth service listens on port 8410.",),
    ("The metrics endpoint is on port 9090.",),
    ("We changed the auth service port to 8443.",),
    ("The admin dashboard runs on port 7000.",),
    ("Auth moved again to port 8500 for the TLS upgrade.",),
    ("The healthcheck endpoint is on port 8081.",),
    ("The message broker listens on port 5672.",),
    ("The internal gRPC gateway uses port 50051.",),
    # --- monthly cloud budget: chain 47000 -> 63000 -> 58000 ---
    ("Our monthly cloud budget is 47000 dollars.",),
    ("The marketing budget is 12000 dollars.",),
    ("Cloud budget was raised to 63000 dollars.",),
    ("The travel budget is 9000 dollars a month.",),
    ("After review the cloud budget is now capped at 58000 dollars.",),
    ("The training budget is 4000 dollars per quarter.",),
    # --- billing service owner: Falcon -> Peregrine ---
    ("The billing service is owned by the Falcon team.",),
    ("The search service is owned by the Osprey team.",),
    ("The notifications service is owned by the Kestrel team.",),
    ("Ownership changed: the billing service is now owned by the Peregrine team.",),
    ("The payments gateway is owned by the Harrier team.",),
    ("The identity service is owned by the Merlin team.",),
    # --- stable facts (should never be lost) ---
    ("Our on-call rotation is weekly, handed over on Mondays.",),
    ("We deploy on Tuesdays and Thursdays only.",),
    ("The incident sev-1 pager is a PagerDuty schedule named Nightwatch.",),
    # --- pure distractors (unrelated) ---
    ("The office coffee machine is a La Marzocco.",),
    ("The all-hands is the first Friday of the month.",),
    ("The design team uses Figma.",),
    ("Lunch is catered on Wednesdays.",),
    ("The company mascot is a red panda named Pixel.",),
]
CONV = {"turns": [(t[0], _ack(t)) for t in TURNS]}

QUESTIONS = [
    dict(type="chain_current", q="What is our PRIMARY datacenter region right now?", gt="eu-west-3",
         stale="ap-southeast-2 or us-east-1"),
    dict(type="chain_history", q="What was our primary datacenter region ORIGINALLY, before any migrations?",
         gt="ap-southeast-2", stale=None),
    dict(type="distractor", q="Which region is the BACKUP region?", gt="sa-east-1", stale=None),
    dict(type="chain_current", q="What port does the auth service listen on now?", gt="8500", stale="8410 or 8443"),
    dict(type="distractor", q="What port does the metrics endpoint use?", gt="9090", stale=None),
    dict(type="chain_current", q="What is the current monthly cloud budget?", gt="58000", stale="47000 or 63000"),
    dict(type="distractor", q="What is the monthly marketing budget?", gt="12000", stale=None),
    dict(type="chain_current", q="Which team owns the billing service now?", gt="Peregrine", stale="Falcon"),
    dict(type="distractor", q="Which team owns the search service?", gt="Osprey", stale=None),
    dict(type="stable", q="Which days does the team deploy?", gt="Tuesdays and Thursdays", stale=None),
    dict(type="stable", q="What is the sev-1 pager schedule called?", gt="Nightwatch", stale=None),
    dict(type="absent", q="What region does the machine-learning training cluster run in?", gt=None, stale=None),
    dict(type="absent", q="What port does the WebSocket service use?", gt=None, stale=None),
]


def main():
    print(f"STRESS probe  |  extract {HAIKU.split('.')[-1]}  |  judge {JUDGE.split('.')[-1]}  |  "
          f"one corpus of {len(TURNS)} turns  |  retrieve top-{K}  |  strict, unguessable values\n")
    mdb, path, mdb_i, mdb_o = ingest_mentedb(CONV)
    m0 = make_mem0()
    ingest_mem0(m0, CONV, "s")
    by_type = {}
    for q in QUESTIONS:
        mdb_ctx = [c["content"] for c in recall_context(mdb, q["q"], K)]
        r0 = m0.search(q["q"], filters={"user_id": "s"}, limit=K)
        m0_ctx = [x.get("memory", "") for x in (r0.get("results", r0) if isinstance(r0, dict) else r0)][:K]
        ma, m0a = answer(mdb_ctx, q["q"]), answer(m0_ctx, q["q"])
        mok, m0ok = strict_score(q, ma), strict_score(q, m0a)
        for s, ok in (("MenteDB", mok), ("mem0", m0ok)):
            d = by_type.setdefault(q["type"], {"MenteDB": [0, 0], "mem0": [0, 0]})
            d[s][0] += ok
            d[s][1] += 1
        print(f"[{q['type']}] {q['q']}  (GT={q['gt']})")
        print(f"   MenteDB {'OK ' if mok else 'XX '} {ma[:75]!r}")
        print(f"           ctx: {[c[:48] for c in mdb_ctx]}")
        print(f"   mem0    {'OK ' if m0ok else 'XX '} {m0a[:75]!r}")
        print(f"           ctx: {[c[:48] for c in m0_ctx]}")
    try:
        mdb.close()
    except Exception:
        pass
    shutil.rmtree(path, ignore_errors=True)

    print("\n  === PER-TYPE ACCURACY (strict, one big corpus, top-5) ===")
    order = ["chain_current", "chain_history", "distractor", "stable", "absent"]
    tot = {"MenteDB": [0, 0], "mem0": [0, 0]}
    print(f"  {'type':<16}{'MenteDB':>12}{'mem0':>12}")
    for t in order:
        if t not in by_type:
            continue
        d = by_type[t]
        cells = []
        for s in ("MenteDB", "mem0"):
            c, n = d[s]
            cells.append(f"{c}/{n}")
            tot[s][0] += c
            tot[s][1] += n
        print(f"  {t:<16}{cells[0]:>12}{cells[1]:>12}")
    print(f"  {'TOTAL':<16}{f'{tot['MenteDB'][0]}/{tot['MenteDB'][1]}':>12}{f'{tot['mem0'][0]}/{tot['mem0'][1]}':>12}")
    print(f"\n  MenteDB ingest tokens: {mdb_i + mdb_o}   (per system totals differ; see deep eval for cost)")


if __name__ == "__main__":
    main()
