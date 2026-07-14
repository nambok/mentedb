"""
HARDER accuracy probe: MenteDB vs mem0, typed questions built to BREAK a tie.

The deep eval scored 100%/100%, which usually means the questions were too easy
(clean "I switched to Y" updates) and/or the judge was lenient. This probe uses
question types designed to expose real differences and real MenteDB weaknesses,
retrieves at a tight k=3, and judges strictly (must state the exact answer; a
hedge or a stale value fails):

  explicit_update  - "I switched to Y", ask current           (baseline)
  implicit_update  - later contradicting statement, no marker  (which is newer?)
  temporal_history - "what did you use BEFORE the switch?"      (suppression may LOSE this -> MenteDB risk, mem0 edge)
  stable           - a fact that never changed                 (don't lose it under noise)
  scoped           - two sub-scopes of one attribute           (Postgres for app, BigQuery for analytics)
  buried           - stale fact repeated many times, one update (retrieval precision at k=3)
  multi_hop        - combine two memories                       (reasoning over context)
  absent           - never stated                               (must abstain)

Reuses the validated deep harness (same Haiku extraction, same embedder, same
platform prompt, LLM judge). Reports accuracy PER TYPE so we see who wins where.
Run: python benchmarks/learn_manage_vs_mem0_hard.py
"""

import shutil

from learn_manage_vs_mem0_deep import (  # noqa: E402
    JUDGE, PRICE, HAIKU, _TOK, bedrock_chat, embed, ingest_mem0, ingest_mentedb,
    make_mem0, parse_json, recall_context,
)

K = 3  # tight retrieval budget to stress precision

CONV = [
    dict(turns=[
        ("I use PostgreSQL as my main database.", "Noted."),
        ("For caching we rely on Redis.", "Ok."),
        ("The backend is written in Python.", "Ok."),
        ("We migrated the main database from Postgres to SQLite last quarter.", "Big migration."),
        ("The backend got rewritten, it's all Go now instead of Python.", "Nice."),
        ("Redis is still our cache, unchanged.", "Good."),
    ], qs=[
        dict(type="explicit_update", q="What is the user's main database now?", gt="SQLite", stale="PostgreSQL"),
        dict(type="temporal_history", q="What database did the user use BEFORE the migration to their current one?",
             gt="PostgreSQL", stale=None),
        dict(type="explicit_update", q="What language is the backend written in now?", gt="Go", stale="Python"),
        dict(type="stable", q="What does the user use for caching?", gt="Redis", stale=None),
        dict(type="multi_hop", q="Was the backend rewritten in the same language it originally used?",
             gt="No (it went from Python to Go)", stale=None),
        dict(type="absent", q="What cloud provider does the user use?", gt=None, stale=None),
    ]),
    dict(turns=[
        ("My manager is Sarah.", "Ok."),
        ("I live in Seattle.", "Ok."),
        ("I'm now reporting to David.", "Ok."),
        ("These days I'm based in Austin.", "Ok."),
    ], qs=[
        dict(type="implicit_update", q="Who is the user's manager now?", gt="David", stale="Sarah"),
        dict(type="implicit_update", q="Which city does the user live in now?", gt="Austin", stale="Seattle"),
        dict(type="absent", q="What is the user's job title?", gt=None, stale=None),
    ]),
    dict(turns=[
        ("I always indent with tabs, never spaces.", "Ok."),
        ("Tabs are just better for indentation, that's my rule.", "Understood."),
        ("Every file I write uses tabs.", "Ok."),
        ("Our style guide says tabs and I follow it.", "Ok."),
        ("I've used tabs for years without exception.", "Ok."),
        ("Change of heart: I switched to spaces for indentation.", "Noted the switch."),
    ], qs=[
        dict(type="buried", q="Does the user indent with tabs or spaces now?", gt="spaces", stale="tabs"),
    ]),
    dict(turns=[
        ("For analytics workloads I use BigQuery.", "Ok."),
        ("The app's transactional data lives in Postgres.", "Ok."),
        ("For search we run Elasticsearch.", "Ok."),
    ], qs=[
        dict(type="scoped", q="What does the user use for the app's transactional data?", gt="Postgres", stale=None),
        dict(type="scoped", q="What does the user use for analytics?", gt="BigQuery", stale=None),
        dict(type="scoped", q="What does the user use for search?", gt="Elasticsearch", stale=None),
    ]),
]


def strict_score(q, reply):
    if q["type"] == "absent":
        v = parse_json(bedrock_chat(JUDGE, "Reply JSON only.",
                       f'Answer: "{reply}"\nDid the assistant decline / say it does not know, rather than assert a '
                       'specific fact? JSON: {"abstained": true|false}', 60), {})
        return bool(v.get("abstained"))
    prompt = (f'Question: "{q["q"]}"\nAssistant answer: "{reply}"\n'
              f'The ONLY correct answer is: "{q["gt"]}".\n'
              + (f'Stating "{q["stale"]}" as current would be WRONG.\n' if q.get("stale") else "")
              + 'Score correct=true ONLY if the answer clearly and unambiguously conveys the correct answer. '
              'If it states a wrong/outdated value, hedges between two values, or is vague/evasive, score false. '
              'JSON: {"correct": true|false}')
    v = parse_json(bedrock_chat(JUDGE, "You are a strict grader. Reply JSON only.", prompt, 60), {})
    return bool(v.get("correct"))


def answer(ctx_texts, question):
    ctx = "\n".join(f"- {t}" for t in ctx_texts) if ctx_texts else "(no memories)"
    return bedrock_chat(JUDGE, "Answer the question using ONLY the memories. If they do not contain the answer, "
                        "reply exactly: I don't know.", f"Memories:\n{ctx}\n\nQuestion: {question}", 300)


def main():
    print(f"HARD probe  |  extract {HAIKU.split('.')[-1]}  |  judge {JUDGE.split('.')[-1]}  |  retrieve top-{K}  |  strict grading\n")
    by_type = {}  # type -> {system -> [correct, total]}
    m0 = make_mem0()
    for ci, conv in enumerate(CONV):
        uid = f"h{ci}"
        mdb, path, _, _ = ingest_mentedb(conv)
        ingest_mem0(m0, conv, uid)
        for q in conv["qs"]:
            mdb_ctx = [c["content"] for c in recall_context(mdb, q["q"], K)]
            r0 = m0.search(q["q"], filters={"user_id": uid}, limit=K)
            m0_ctx = [x.get("memory", "") for x in (r0.get("results", r0) if isinstance(r0, dict) else r0)]
            for name, ctx in (("MenteDB", mdb_ctx), ("mem0", m0_ctx)):
                ok = strict_score(q, answer(ctx, q["q"]))
                d = by_type.setdefault(q["type"], {"MenteDB": [0, 0], "mem0": [0, 0]})
                d[name][0] += ok
                d[name][1] += 1
        try:
            mdb.close()
        except Exception:
            pass
        shutil.rmtree(path, ignore_errors=True)
        print(f"  conversation {ci + 1}/{len(CONV)} done")

    order = ["explicit_update", "implicit_update", "temporal_history", "buried", "scoped", "stable", "multi_hop", "absent"]
    print(f"\n  {'question type':<18}{'MenteDB':>12}{'mem0':>12}")
    print("  " + "-" * 40)
    tot = {"MenteDB": [0, 0], "mem0": [0, 0]}
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
        print(f"  {t:<18}{cells[0]:>12}{cells[1]:>12}")
    print("  " + "-" * 40)
    print(f"  {'TOTAL':<18}{f'{tot['MenteDB'][0]}/{tot['MenteDB'][1]}':>12}{f'{tot['mem0'][0]}/{tot['mem0'][1]}':>12}")
    print("\n  Watch temporal_history (does suppression lose 'what came before'?) and buried/implicit")
    print("  (retrieval precision). If a column drops there, that is the real, honest weakness to report.")


if __name__ == "__main__":
    main()
