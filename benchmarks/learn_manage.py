"""
Learn / Manage eval: does the memory system actually MANAGE facts over time,
or just store-and-retrieve?

Popular memory products win LOCOMO-style recall benchmarks yet buyers still say
"it stores but doesn't learn, I have to re-tell it things." That failure lives in
the write/manage stages that recall benchmarks do not test:

  1. KNOWLEDGE UPDATE   - when a fact changes, does a query return the CURRENT
                          fact, or resurface the stale one?
  2. CONTRADICTION      - does it flag GENUINE contradictions while NOT flagging
                          merely topic-similar-but-compatible facts (precision)?
  3. HONEST ABSTENTION  - asked something not in memory, does it say "I don't
                          know" or confabulate a confident wrong answer?

Each suite runs MenteDB's managed path (graph-aware recall that suppresses
superseded/contradicted facts + an LLM conflict judge that mirrors the cloud's
BedrockJudge + an honest-abstention answer prompt) against a naive-RAG baseline
(same engine, no conflict management, mem0-style "always answer" prompt) so the
delta isolates the value of the MANAGE layer, not the retriever.

This is synthetic data in a temp dir (no real user memory touched). It needs the
mentedb Python SDK (cd sdks/python && maturin develop) and an LLM: Bedrock (AWS
creds, the provider the cloud uses), or ANTHROPIC_API_KEY, or OPENAI_API_KEY.

Run:  python benchmarks/learn_manage.py
"""

import json
import os
import re
import sys

from harness import MenteDBBenchmark, print_result


# --------------------------------------------------------------------------
# LLM client: Bedrock (default, matches the cloud) -> Anthropic -> OpenAI.
# The memory system's differentiation is the MANAGE layer, not the model, so we
# use the same model for both the managed and naive paths.
# --------------------------------------------------------------------------

def get_llm():
    """Return a callable llm(prompt, json_mode) -> str, plus a label."""
    model = os.environ.get("MENTEDB_BENCH_MODEL")

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        m = model or "claude-sonnet-4-20250514"

        def call(prompt, json_mode=False):
            if json_mode:
                prompt += "\n\nRespond with valid JSON only, no markdown fences."
            r = client.messages.create(
                model=m, max_tokens=512, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text.strip()

        return call, f"anthropic:{m}"

    if os.environ.get("OPENAI_API_KEY"):
        import openai
        client = openai.OpenAI()
        m = model or "gpt-4o-mini"

        def call(prompt, json_mode=False):
            kwargs = dict(model=m, temperature=0.0, max_tokens=512,
                          messages=[{"role": "user", "content": prompt}])
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs).choices[0].message.content.strip()

        return call, f"openai:{m}"

    # Bedrock via boto3 Converse (the provider the platform runs on).
    try:
        import boto3
    except ImportError:
        return None, None
    region = os.environ.get("AWS_REGION", "us-east-1")
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
    except Exception:
        return None, None
    m = model or os.environ.get("MENTEDB_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")

    def call(prompt, json_mode=False):
        if json_mode:
            prompt += "\n\nRespond with valid JSON only, no markdown fences."
        r = client.converse(
            modelId=m,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.0},
        )
        return r["output"]["message"]["content"][0]["text"].strip()

    return call, f"bedrock:{m}"


def parse_json(text):
    """Best-effort JSON extraction from an LLM reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else text)
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Synthetic scenarios (facts about "the user"). Fully made up; no real data.
# --------------------------------------------------------------------------

# Each: an OLD fact, then later a NEW fact for the same attribute. Both are
# phrased as plain present-tense statements with NO recency cue in the text
# ("switched", "now", "moved last month" are deliberately removed) so the model
# cannot infer which is current from wording. The only signal for "which is
# true now" is order of arrival, which MenteDB encodes as a supersedes edge and
# its graph-aware recall then suppresses the stale node. A naive store that
# keeps both and retrieves by similarity has no way to tell them apart. This is
# where the MANAGE layer, not the LLM, has to do the work.
KNOWLEDGE_UPDATE = [
    dict(old="The user works as a data scientist at Acme Corp.",
         new="The user works as a machine learning engineer at Globex.",
         query="What is the user's current job?",
         current="machine learning engineer at Globex", stale="data scientist at Acme"),
    dict(old="The user lives in Seattle.",
         new="The user lives in Austin.",
         query="Where does the user currently live?",
         current="Austin", stale="Seattle"),
    dict(old="The user uses PostgreSQL as their primary database.",
         new="The user uses SQLite as their primary database.",
         query="Which database does the user use as their primary?",
         current="SQLite", stale="PostgreSQL"),
    dict(old="The user's primary programming language is Python.",
         new="The user's primary programming language is Go.",
         query="What is the user's primary programming language?",
         current="Go", stale="Python"),
    dict(old="The user drives a Honda Civic.",
         new="The user drives a Tesla Model 3.",
         query="What car does the user drive?",
         current="Tesla Model 3", stale="Honda Civic"),
    dict(old="The user's main laptop is a MacBook Pro.",
         new="The user's main laptop is a ThinkPad running Linux.",
         query="What is the user's main laptop?",
         current="ThinkPad running Linux", stale="MacBook Pro"),
    dict(old="The user's manager is Sarah.",
         new="The user's manager is David.",
         query="Who is the user's manager?",
         current="David", stale="Sarah"),
    dict(old="The user's phone number ends in 4471.",
         new="The user's phone number ends in 8892.",
         query="What does the user's phone number end in?",
         current="8892", stale="4471"),
]

# Realistic context volume: unrelated facts stored alongside every scenario so
# the stale and current facts do not sit alone in a two-item context. Present in
# BOTH the managed and naive stores, so they do not change the comparison, only
# make it non-trivial.
DISTRACTORS = [
    "The user enjoys hiking on weekends.",
    "The user is learning to play the guitar.",
    "The user has a golden retriever named Biscuit.",
    "The user prefers dark mode in their editor.",
    "The user usually works late in the evening.",
    "The user drinks two cups of coffee a day.",
    "The user is allergic to pollen.",
    "The user listens to podcasts while commuting.",
]

# Positive = genuine contradiction/update (should flag). Negative = topic-similar
# but perfectly compatible (must NOT flag -- this is the precision test).
CONTRADICTION = [
    dict(a="The user is allergic to peanuts.",
         b="The user happily ate a peanut butter sandwich with no reaction.", contradicts=True),
    dict(a="The user is a vegetarian.",
         b="The user had a ribeye steak for dinner.", contradicts=True),
    dict(a="The user lives in Seattle.",
         b="The user lives in Austin.", contradicts=True),
    dict(a="The user prefers tabs over spaces for indentation.",
         b="The user now prefers spaces over tabs.", contradicts=True),
    dict(a="The project deadline is this Friday.",
         b="The project deadline was pushed to next Wednesday.", contradicts=True),
    dict(a="The user likes programming in Python.",
         b="The user likes programming in Rust.", contradicts=False),
    dict(a="The user uses VS Code as their editor.",
         b="The user uses vim keybindings.", contradicts=False),
    dict(a="The user drinks coffee in the morning.",
         b="The user drinks tea in the afternoon.", contradicts=False),
    dict(a="The user works mostly on backend systems.",
         b="The user also does some frontend work.", contradicts=False),
    dict(a="The user has a cat named Milo.",
         b="The user has a dog named Rex.", contradicts=False),
]

# A fixed profile, then questions that ARE / ARE NOT answerable from it.
ABSTENTION_PROFILE = [
    "The user works as a machine learning engineer at Globex.",
    "The user lives in Austin, Texas.",
    "The user has a golden retriever named Biscuit.",
    "The user prefers SQLite and codes mostly in the evening.",
    "The user is learning Japanese.",
]
ABSTENTION_QUESTIONS = [
    dict(q="What is the user's job?", answerable=True),
    dict(q="Where does the user live?", answerable=True),
    dict(q="What is the user's blood type?", answerable=False),
    dict(q="What was the user's SAT score?", answerable=False),
    dict(q="What is the user's mother's maiden name?", answerable=False),
    dict(q="How many siblings does the user have?", answerable=False),
    dict(q="What car does the user's neighbor drive?", answerable=False),
]


# --------------------------------------------------------------------------
# LLM roles
# --------------------------------------------------------------------------

def judge_supersedes(llm, older, newer):
    """Mirror of the cloud BedrockJudge: does `newer` update/contradict `older`?"""
    out = parse_json(llm(
        "Two facts about the same user were recorded at different times.\n"
        f'OLDER: "{older}"\n'
        f'NEWER: "{newer}"\n\n'
        "Does the NEWER fact make the OLDER one obsolete (an update to the same "
        "attribute) OR directly contradict it? Answer no if both can be true at "
        'once (different attributes/topics).\n'
        'Return JSON: {"supersedes": true|false}', json_mode=True))
    return bool(out.get("supersedes"))


def answer(llm, context, query, honest):
    """Answer from retrieved context. honest=abstain if absent; else mem0-style."""
    ctx = "\n".join(f"- {c}" for c in context) if context else "(no memories retrieved)"
    if honest:
        instruction = ("Answer ONLY from the memories. If they do not contain the "
                       "answer, reply exactly: I don't know.")
    else:
        instruction = ("Answer the question. If no relevant memory is found, do not "
                       "say information is missing; provide your best general answer.")
    return llm(f"Memories about the user:\n{ctx}\n\nQuestion: {query}\n\n{instruction}").strip()


def judge_answer(llm, query, reply, current, stale):
    """Did the reply state the CURRENT fact (not the stale one)?"""
    out = parse_json(llm(
        f'Question: "{query}"\n'
        f'Assistant reply: "{reply}"\n\n'
        f'The correct, current answer is: "{current}".\n'
        f'An outdated/stale answer would be: "{stale}".\n\n'
        'Does the reply convey the CURRENT answer (not the stale one)?\n'
        'Return JSON: {"correct": true|false}', json_mode=True))
    return bool(out.get("correct"))


def judge_abstained(llm, reply):
    out = parse_json(llm(
        f'Assistant reply: "{reply}"\n\n'
        'Did the assistant decline to answer / say it does not know, rather than '
        'assert a specific factual answer?\n'
        'Return JSON: {"abstained": true|false}', json_mode=True))
    return bool(out.get("abstained"))


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------

def _retrieve(bench, query, k=12):
    """Full-recall retrieval over the small synthetic set. k exceeds the number
    of stored facts on purpose: it holds retrieval at 100% recall so the eval
    isolates the MANAGE layer (the managed DB's graph-aware recall drops
    superseded/contradicted nodes) rather than embedding-ranking quality, which
    the default hash embeddings would otherwise dominate. Retrieval ranking is a
    separate benchmark (longmemeval)."""
    out = []
    for rid, _score in bench.search(query, limit=k):
        mem = bench.get(rid)
        content = mem.get("content", "") if isinstance(mem, dict) else getattr(mem, "content", str(mem))
        if content:
            out.append(content)
    return out


def suite_knowledge_update(llm):
    managed_ok = naive_ok = 0
    rows = []
    for sc in KNOWLEDGE_UPDATE:
        # MANAGED: store, judge+supersede, graph-aware recall, honest answer.
        mb = MenteDBBenchmark()
        try:
            old_id = mb.store(sc["old"])
            for n in DISTRACTORS:
                mb.store(n)
            new_id = mb.store(sc["new"])
            if judge_supersedes(llm, sc["old"], sc["new"]):
                mb.relate(new_id, old_id, "supersedes")
                mb.relate(new_id, old_id, "contradicts")
            ctx = _retrieve(mb, sc["query"])
            m_ok = judge_answer(llm, sc["query"], answer(llm, ctx, sc["query"], honest=True),
                                sc["current"], sc["stale"])
        finally:
            mb.cleanup()

        # NAIVE: store everything, no management, similarity recall, always-answer.
        nb = MenteDBBenchmark()
        try:
            nb.store(sc["old"])
            for n in DISTRACTORS:
                nb.store(n)
            nb.store(sc["new"])
            ctx = _retrieve(nb, sc["query"])
            n_ok = judge_answer(llm, sc["query"], answer(llm, ctx, sc["query"], honest=False),
                                sc["current"], sc["stale"])
        finally:
            nb.cleanup()

        managed_ok += m_ok
        naive_ok += n_ok
        rows.append(f"{'OK ' if m_ok else 'MISS'} managed / {'OK ' if n_ok else 'MISS'} naive  :: {sc['query']}")

    n = len(KNOWLEDGE_UPDATE)
    details = {"MenteDB (managed)": f"{managed_ok}/{n} return current fact",
               "Naive RAG": f"{naive_ok}/{n} return current fact"}
    for r in rows:
        details[r[:4]] = r
    print_result("Knowledge Update (does it return the CURRENT fact?)",
                 managed_ok >= naive_ok and managed_ok >= 0.75 * n, details)
    return managed_ok, naive_ok, n


def suite_contradiction(llm):
    tp = fp = tn = fn = 0
    rows = []
    for sc in CONTRADICTION:
        flagged = judge_supersedes(llm, sc["a"], sc["b"])
        gold = sc["contradicts"]
        if gold and flagged:
            tp += 1; verdict = "TP"
        elif gold and not flagged:
            fn += 1; verdict = "FN (missed)"
        elif not gold and flagged:
            fp += 1; verdict = "FP (false alarm)"
        else:
            tn += 1; verdict = "TN"
        rows.append(f"{verdict:16} {sc['a'][:34]} | {sc['b'][:34]}")
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    details = {"Precision (no false alarms on compatible facts)": f"{prec:.2f}",
               "Recall (catches genuine contradictions)": f"{rec:.2f}",
               "F1": f"{f1:.2f}",
               "Confusion": f"TP={tp} FP={fp} TN={tn} FN={fn}"}
    for r in rows:
        details[r[:20]] = r
    print_result("Contradiction Detection (precision on topic-similar facts)",
                 prec >= 0.8 and rec >= 0.8, details)
    return prec, rec, f1


def suite_abstention(llm):
    mb = MenteDBBenchmark()
    honest_abstain = honest_answer = 0
    naive_confab = 0
    n_absent = sum(1 for q in ABSTENTION_QUESTIONS if not q["answerable"])
    n_answerable = len(ABSTENTION_QUESTIONS) - n_absent
    rows = []
    try:
        for f in ABSTENTION_PROFILE:
            mb.store(f)
        for q in ABSTENTION_QUESTIONS:
            ctx = _retrieve(mb, q["q"])
            m_reply = answer(llm, ctx, q["q"], honest=True)
            n_reply = answer(llm, ctx, q["q"], honest=False)
            m_abst = judge_abstained(llm, m_reply)
            if q["answerable"]:
                # Should ANSWER, not abstain.
                honest_answer += (not m_abst)
                rows.append(f"answerable: managed {'ANSWERED' if not m_abst else 'over-abstained'} :: {q['q']}")
            else:
                honest_abstain += m_abst
                naive_confab += (not judge_abstained(llm, n_reply))
                rows.append(f"absent    : managed {'abstained' if m_abst else 'CONFABULATED'} / "
                            f"naive {'abstained' if judge_abstained(llm, n_reply) else 'confabulated'} :: {q['q']}")
    finally:
        mb.cleanup()
    details = {
        "MenteDB honest abstention (absent info)": f"{honest_abstain}/{n_absent}",
        "MenteDB answers when it can (control)": f"{honest_answer}/{n_answerable}",
        "Naive RAG confabulates (absent info)": f"{naive_confab}/{n_absent}",
    }
    for r in rows:
        details[r[:24]] = r
    print_result("Honest Abstention (say 'I don't know' vs confabulate)",
                 honest_abstain >= 0.8 * n_absent and honest_answer >= 0.8 * n_answerable, details)
    return honest_abstain, n_absent, naive_confab


def main():
    llm, label = get_llm()
    if llm is None:
        print("No LLM available. Set ANTHROPIC_API_KEY / OPENAI_API_KEY, or configure "
              "AWS creds for Bedrock (optionally MENTEDB_BENCH_MODEL).")
        sys.exit(2)
    print(f"learn/manage eval  |  judge+answer model: {label}\n")

    ku_m, ku_n, ku_total = suite_knowledge_update(llm)
    prec, rec, f1 = suite_contradiction(llm)
    ab_ok, ab_total, ab_confab = suite_abstention(llm)

    print("\n" + "=" * 60)
    print("  SUMMARY  (MenteDB managed vs naive RAG, same model)")
    print("=" * 60)
    print(f"  Knowledge update : MenteDB {ku_m}/{ku_total} vs naive {ku_n}/{ku_total} return the CURRENT fact")
    print(f"  Contradiction    : precision {prec:.2f} / recall {rec:.2f} / F1 {f1:.2f} on topic-similar pairs")
    print(f"  Abstention       : MenteDB abstains {ab_ok}/{ab_total} on absent info; naive confabulates {ab_confab}/{ab_total}")
    print()
    print("  INTERPRETATION (read before quoting any number):")
    print("  - Abstention is the real, defensible differentiator here: a naive store")
    print("    with a mem0-style 'always answer' prompt confabulates absent facts while")
    print("    MenteDB abstains. mem0's own answer prompt discourages 'I don't know'.")
    print("  - Knowledge update usually shows little delta at THIS scale: with full")
    print("    recall and a capable model, the LLM reasons over both facts, so graph")
    print("    suppression is hard to see. The management win must be measured where")
    print("    full recall is impossible (many memories, small-k retrieval, real")
    print("    embeddings). Do not claim a knowledge-update edge from this run alone.")
    print("  - Contradiction precision here measures the LLM conflict JUDGE that the")
    print("    cloud uses, NOT the engine heuristic. Any LLM-judge system scores similar;")
    print("    it is not a MenteDB-specific result.")
    print()


if __name__ == "__main__":
    main()
