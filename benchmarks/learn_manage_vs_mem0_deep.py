"""
Deep, apples-to-apples head-to-head: MenteDB vs mem0.

The first pass was unfair to itself: it fed MenteDB pre-extracted facts (0 write
tokens) while mem0 extracted from raw text. This pass fixes that so nothing is
made up:

  - Both systems ingest the SAME raw multi-turn conversations.
  - Both EXTRACT with the SAME model (Bedrock Haiku 4.5, MenteDB's real cloud
    extraction model), so the token/cost numbers are on equal footing.
  - MenteDB uses the platform's ACTUAL extraction prompt, verbatim, including its
    bounded prior-context + contradiction detection (one call extracts AND
    reconciles). We store the extracted facts and apply supersedes edges from the
    contradiction field, exactly as production does.
  - mem0 runs its own add() pipeline on the same Haiku model.
  - Correctness is judged by an LLM (Sonnet) answering from each system's
    retrieved context, then scoring current-vs-stale, NOT keyword matching (which
    unfairly flagged mem0's "switched from X to Y" memory as stale last time).
  - Every Bedrock token both systems spend on ingest is measured exactly.

Reports, per system: answer accuracy on updated / stable / absent questions, and
ingest token + dollar cost. Same embedder (bge-small) for both.

Needs: mentedb (maturin develop), mem0ai, fastembed, AWS creds. Slow (many LLM
calls) and costs a couple dollars of Bedrock. Run: python benchmarks/learn_manage_vs_mem0_deep.py
"""

import io
import json
import os
import re
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("MEM0_TELEMETRY", "False")  # avoid the shared ~/.mem0 telemetry qdrant lock

import boto3  # noqa: E402
import botocore.client  # noqa: E402
from botocore.response import StreamingBody  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402
from mem0 import Memory  # noqa: E402
from mentedb import MenteDB  # noqa: E402

HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"      # both systems' extraction (MenteDB's real model)
JUDGE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"     # neutral answer + scoring
PRICE = {  # Bedrock on-demand list price, USD per 1M tokens (in, out). Tokens are measured exactly.
    HAIKU: (float(os.environ.get("HAIKU_IN", "0.80")), float(os.environ.get("HAIKU_OUT", "4.00"))),
    JUDGE: (3.00, 15.00),
}

_EMB = TextEmbedding("BAAI/bge-small-en-v1.5")
_TOK = {"in": 0, "out": 0}
_bedrock = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])


def embed(texts):
    return [list(map(float, v)) for v in _EMB.embed(list(texts))]


# --- exact token accounting across converse + invoke_model -------------------
_ORIG = botocore.client.BaseClient._make_api_call


def _counting(self, op, params):
    resp = _ORIG(self, op, params)
    try:
        if op in ("Converse", "ConverseStream"):
            u = resp.get("usage", {})
            _TOK["in"] += u.get("inputTokens", 0)
            _TOK["out"] += u.get("outputTokens", 0)
        elif op == "InvokeModel":
            raw = resp["body"].read()
            resp["body"] = StreamingBody(io.BytesIO(raw), len(raw))
            u = json.loads(raw).get("usage", {})
            _TOK["in"] += u.get("input_tokens", 0)
            _TOK["out"] += u.get("output_tokens", 0)
    except Exception:
        pass
    return resp


botocore.client.BaseClient._make_api_call = _counting


def bedrock_chat(model, system, user, max_tokens=512):
    body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
                       "system": system, "messages": [{"role": "user", "content": user}]})
    r = _bedrock.invoke_model(modelId=model, contentType="application/json",
                              accept="application/json", body=body)
    return json.loads(r["body"].read())["content"][0]["text"]


def parse_json(text, default):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0] if "\n" in text else text[3:]
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else text)
    except Exception:
        return default


# --- MenteDB's ACTUAL platform extraction prompt (verbatim from bedrock.rs) ---
EXTRACT_SYSTEM = (
    "You extract structured memories from conversations. Given a user/assistant exchange and "
    "prior context, identify facts worth remembering.\n\n"
    "Output ONLY a JSON array. Each item has:\n"
    '- "content": the fact to remember (concise, self-contained)\n'
    '- "memory_type": one of "semantic", "procedural", "correction", "anti_pattern"\n'
    '- "tags": array of 1-3 short tags\n'
    '- "contradiction": null, or the ID of a prior memory this contradicts\n\n'
    "Rules:\n"
    "- Only extract genuinely useful facts (preferences, decisions, corrections, procedures)\n"
    "- Do NOT extract greetings, acknowledgments, or chitchat\n"
    "- If the user corrects something, mark it as \"correction\" type\n"
    "- If nothing worth remembering, return []\n\n"
    "Prior context:\n{context}"
)


def make_mem0():
    cfg = {"llm": {"provider": "aws_bedrock", "config": {"model": HAIKU, "aws_region": os.environ["AWS_REGION"]}},
           "embedder": {"provider": "fastembed", "config": {"model": "BAAI/bge-small-en-v1.5"}},
           "vector_store": {"provider": "qdrant",
                            "config": {"path": tempfile.mkdtemp(prefix="mem0q-"), "collection_name": "d",
                                       "embedding_model_dims": 384}}}
    return Memory.from_config(cfg)


# --- conversations: raw turns with genuine updates, stable facts, and gaps ----
CONversations = [
    dict(turns=[
        ("I just started as a data scientist at Acme Corp.", "Congrats on the new role."),
        ("For our pipelines I rely on PostgreSQL as the main database.", "Postgres is solid."),
        ("We're headquartered in Seattle, I work from the office there.", "Nice."),
        ("Update: I left Acme and I'm now an ML engineer at Globex.", "Big move, congrats."),
        ("At Globex we dropped Postgres, I use SQLite as my primary database now.", "Got it."),
        ("I also relocated to Austin for the job.", "Welcome to Austin."),
    ], questions=[
        dict(q="What is the user's current job and employer?", gt="ML engineer at Globex", stale="data scientist at Acme", kind="update"),
        dict(q="Which database is the user's primary now?", gt="SQLite", stale="PostgreSQL", kind="update"),
        dict(q="What city is the user based in now?", gt="Austin", stale="Seattle", kind="update"),
        dict(q="What is the user's astrological sign?", gt=None, stale=None, kind="absent"),
    ]),
    dict(turns=[
        ("My main editor is VS Code and I love it.", "A popular choice."),
        ("I drive a Honda Civic to work every day.", "Reliable car."),
        ("I've been vegetarian for about five years.", "Good to know."),
        ("Actually I recently switched editors, I use Neovim as my daily driver now.", "Neovim is powerful."),
        ("I started eating meat again a couple months ago, so no longer vegetarian.", "Understood."),
    ], questions=[
        dict(q="What editor does the user use day to day now?", gt="Neovim", stale="VS Code", kind="update"),
        dict(q="Does the user eat meat?", gt="yes, they eat meat now", stale="vegetarian", kind="update"),
        dict(q="What car does the user drive?", gt="Honda Civic", stale=None, kind="stable"),
        dict(q="What is the user's shoe size?", gt=None, stale=None, kind="absent"),
    ]),
    dict(turns=[
        ("My manager is Sarah and she's great.", "Nice."),
        ("Our sprint length is two weeks.", "Standard."),
        ("Reorg happened, my new manager is David now.", "Change can be good."),
        ("We also moved to one-week sprints after the reorg.", "Faster cadence."),
    ], questions=[
        dict(q="Who is the user's manager currently?", gt="David", stale="Sarah", kind="update"),
        dict(q="How long are the user's sprints now?", gt="one week", stale="two weeks", kind="update"),
        dict(q="What is the user's home address?", gt=None, stale=None, kind="absent"),
    ]),
    dict(turns=[
        ("I'm learning French in my spare time.", "Bonne chance."),
        ("My primary language at work is Python.", "Great language."),
        ("I decided to pivot from French, I'm learning Japanese instead now.", "Ganbatte."),
        ("At the new project my primary language is Go, not Python anymore.", "Go is fast."),
    ], questions=[
        dict(q="Which human language is the user learning now?", gt="Japanese", stale="French", kind="update"),
        dict(q="What is the user's primary programming language now?", gt="Go", stale="Python", kind="update"),
        dict(q="What is the user's phone number?", gt=None, stale=None, kind="absent"),
    ]),
]


def recall_context(db, query, k=5):
    out = []
    for r in db.search(embed([query])[0], k):
        mem = db.get_memory(r.id)
        content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", "")
        out.append({"id": str(r.id), "content": content})
    return out


def ingest_mentedb(conv):
    path = tempfile.mkdtemp(prefix="mdb-")
    db = MenteDB(path)
    _TOK["in"] = _TOK["out"] = 0
    for (u, a) in conv["turns"]:
        ctx = recall_context(db, f"{u} {a}")
        ctx_str = json.dumps(ctx) if ctx else "No prior context."
        facts = parse_json(bedrock_chat(HAIKU, EXTRACT_SYSTEM.format(context=ctx_str),
                                        f"User: {u}\nAssistant: {a}", 1024), [])
        if not isinstance(facts, list):
            facts = []
        for f in facts:
            if not isinstance(f, dict) or not f.get("content"):
                continue
            fid = db.store(f["content"], embedding=embed([f["content"]])[0])
            con = f.get("contradiction")
            if con:
                try:
                    db.relate(fid, str(con), "supersedes")
                except Exception:
                    pass
    return db, path, _TOK["in"], _TOK["out"]


def ingest_mem0(m, conv, uid):
    _TOK["in"] = _TOK["out"] = 0
    for (u, a) in conv["turns"]:
        m.add(f"User: {u}\nAssistant: {a}", user_id=uid)
    return _TOK["in"], _TOK["out"]


def answer(ctx_texts, question):
    ctx = "\n".join(f"- {t}" for t in ctx_texts) if ctx_texts else "(no memories)"
    return bedrock_chat(JUDGE, "Answer the question using ONLY the memories. If they do not contain "
                        "the answer, reply exactly: I don't know.",
                        f"Memories:\n{ctx}\n\nQuestion: {question}", 300)


def score(q, reply):
    if q["kind"] == "absent":
        v = parse_json(bedrock_chat(JUDGE, "Reply JSON only.",
                       f'Assistant answer: "{reply}"\nDid it decline / say it does not know rather than assert a '
                       'fact? JSON: {"abstained": true|false}', 60), {})
        return bool(v.get("abstained"))
    v = parse_json(bedrock_chat(JUDGE, "Reply JSON only.",
                   f'Question: "{q["q"]}"\nAnswer: "{reply}"\nCorrect current answer: "{q["gt"]}".'
                   + (f'\nOutdated answer would be: "{q["stale"]}".' if q.get("stale") else "")
                   + '\nDoes the answer convey the correct current answer? JSON: {"correct": true|false}', 60), {})
    return bool(v.get("correct"))


def main():
    print(f"DEEP MenteDB vs mem0  |  both extract with {HAIKU.split('.')[-1]}  |  judge {JUDGE.split('.')[-1]}")
    print(f"  same raw conversations, same embedder, same extraction model. Haiku price "
          f"${PRICE[HAIKU][0]}/${PRICE[HAIKU][1]} per 1M in/out (measured tokens).\n")
    tallies = {s: dict(update=[0, 0], stable=[0, 0], absent=[0, 0], tin=0, tout=0) for s in ("mem0", "MenteDB")}
    m0 = make_mem0()  # one instance, a distinct user_id per conversation (avoids the shared qdrant lock)
    for ci, conv in enumerate(CONversations):
        uid = f"c{ci}"
        mdb, path, mdb_in, mdb_out = ingest_mentedb(conv)
        m0_in, m0_out = ingest_mem0(m0, conv, uid)
        tallies["MenteDB"]["tin"] += mdb_in
        tallies["MenteDB"]["tout"] += mdb_out
        tallies["mem0"]["tin"] += m0_in
        tallies["mem0"]["tout"] += m0_out
        for q in conv["questions"]:
            # retrieve from each
            mdb_ctx = [c["content"] for c in recall_context(mdb, q["q"], 5)]
            r0 = m0.search(q["q"], filters={"user_id": uid}, limit=5)
            m0_ctx = [x.get("memory", "") for x in (r0.get("results", r0) if isinstance(r0, dict) else r0)]
            for name, ctx in (("MenteDB", mdb_ctx), ("mem0", m0_ctx)):
                ok = score(q, answer(ctx, q["q"]))
                tallies[name][q["kind"]][0] += ok
                tallies[name][q["kind"]][1] += 1
        try:
            mdb.close()
        except Exception:
            pass
        shutil.rmtree(path, ignore_errors=True)
        print(f"  conversation {ci + 1}/{len(CONversations)} done")

    print("\n  === ACCURACY (LLM-judged answers from each system's retrieved context) ===")
    print(f"  {'':9}{'updated facts':>16}{'stable facts':>15}{'absent (abstain)':>19}")
    for name in ("MenteDB", "mem0"):
        t = tallies[name]
        def frac(p):
            return f"{p[0]}/{p[1]}" + (f" ({p[0]/p[1]:.0%})" if p[1] else "")
        print(f"  {name:9}{frac(t['update']):>16}{frac(t['stable']):>15}{frac(t['absent']):>19}")

    print("\n  === INGEST COST (same model, measured Bedrock tokens) ===")
    for name in ("MenteDB", "mem0"):
        t = tallies[name]
        pin, pout = PRICE[HAIKU]
        usd = (t["tin"] / 1e6) * pin + (t["tout"] / 1e6) * pout
        turns = sum(len(c["turns"]) for c in CONversations)
        print(f"  {name:9}  {t['tin']:>7} in + {t['tout']:>5} out tokens over {turns} turns  "
              f"= {(t['tin']+t['tout'])//turns:>5} tok/turn   ${usd:.4f} total   ${usd/turns*1000:.2f}/1k turns")
    print("\n  Honest: same model + same input, so the token gap is purely how each system uses the LLM.")
    print("  mem0 typically issues multiple calls per add (extract, then add/update/delete decisions);")
    print("  MenteDB does one extract+contradiction call per turn. Accuracy tells you if the cheaper path")
    print("  costs quality. Read both columns together, not the cost alone.")


if __name__ == "__main__":
    main()
