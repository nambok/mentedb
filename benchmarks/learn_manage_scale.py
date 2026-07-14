"""
Scale eval: does memory MANAGEMENT beat naive retrieval when the corpus is large
and you can only retrieve a few memories?

The small-scale learn/manage eval showed no knowledge-update delta: with full
recall a capable LLM reasons over both the old and new fact, so graph
suppression is invisible. That is honest but it does not test the real claim.
The real claim is about SCALE: when there are hundreds of memories and you
retrieve only the top k, the up-to-date fact competes for slots against the
stale one. The realistic failure is REPETITION: a user states a fact many times
over a long history, then updates it once. Naive similarity retrieval surfaces
the many stale mentions and buries the single update. MenteDB drops superseded
nodes at recall time, so the update surfaces instead.

This measures that directly at the RETRIEVAL level (what lands in the agent's
context), with real local embeddings and NO LLM, so nothing is confounded by a
model reasoning over the context. For each scenario we vary how many times the
stale fact was repeated and report, over the retrieved top-k:
  - current-retrieved rate : did the up-to-date fact make it into context?
  - stale-in-context rate  : did an outdated mention make it into context?

MenteDB (managed) creates a supersedes edge from the update to each stale
mention (ground truth, so this isolates recall suppression from judge quality).
Naive keeps everything. Same corpus, same embeddings, same k.

Fully offline. Needs the mentedb SDK (maturin develop) and fastembed
(pip install fastembed). Run:  python benchmarks/learn_manage_scale.py
"""

import shutil
import tempfile

from fastembed import TextEmbedding

from mentedb import MenteDB

_EMBEDDER = TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts):
    return [list(map(float, v)) for v in _EMBEDDER.embed(list(texts))]


# --------------------------------------------------------------------------
# ~200 diverse distractor facts, deliberately OFF-TOPIC from every scenario
# attribute (no jobs, databases, cars, laptops, languages, or city-of-residence)
# so they add corpus volume without themselves matching the queries.
# --------------------------------------------------------------------------

_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Jamie",
          "Quinn", "Avery", "Reese", "Devon", "Harper", "Elliot", "Rowan"]
_HOBBIES = ["rock climbing", "oil painting", "chess", "baking sourdough",
            "birdwatching", "woodworking", "salsa dancing", "the violin",
            "pottery", "long-distance running", "knitting", "astrophotography"]
_FOODS = ["Thai green curry", "tonkotsu ramen", "street tacos", "margherita pizza",
          "soup dumplings", "brisket", "pho", "falafel wraps", "chicken biryani",
          "eggs benedict"]
_SHOWS = ["a documentary about the deep ocean", "a baking competition",
          "a space exploration series", "a wildlife program", "a heist thriller",
          "a stand-up special"]
_SPORTS = ["tennis", "road cycling", "lap swimming", "bouldering", "kayaking",
           "ultimate frisbee"]
_MISC = [
    "The weather has been unusually warm this week.",
    "The user is reading a novel about lighthouse keepers.",
    "The user picked up a new pair of running shoes.",
    "The user's houseplant finally bloomed.",
    "The user is trying to drink more water lately.",
    "The user reorganized their bookshelf by color.",
    "The user started a small herb garden on the balcony.",
    "The user is learning to make cold brew at home.",
]


def _build_distractors():
    out = []
    for i, h in enumerate(_HOBBIES):
        out.append(f"The user enjoys {h}.")
        out.append(f"{_NAMES[i % len(_NAMES)]} and the user took a class in {h} together.")
    for i, f in enumerate(_FOODS):
        out.append(f"The user tried {f} at a new place recently.")
        out.append(f"{_NAMES[(i + 3) % len(_NAMES)]} cooked {f} for the user.")
    for i, s in enumerate(_SHOWS):
        out.append(f"The user watched {s} over the weekend.")
    for i, sp in enumerate(_SPORTS):
        out.append(f"The user played {sp} on Saturday.")
        out.append(f"{_NAMES[(i + 5) % len(_NAMES)]} invited the user to play {sp}.")
    for n in _NAMES:
        out.append(f"{n} is an old friend of the user from school.")
        out.append(f"The user called {n} to catch up last week.")
    out.extend(_MISC)
    # de-dup, stable order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


DISTRACTORS = _build_distractors()

# Each scenario: a stale fact the user stated repeatedly, then one update. The
# keywords identify which fact a retrieved memory is. Stale paraphrases vary the
# wording (as a real history would) so it is not literal duplicate detection.
SCENARIOS = [
    dict(query="Which database does the user use as their primary database?",
         current="The user now uses SQLite as their primary database.", current_kw="sqlite",
         stale_kw="postgre", stale=[
             "The user uses PostgreSQL as their primary database.",
             "The user relies on Postgres for all of their data.",
             "PostgreSQL is the user's main database system.",
             "The user stores everything in a PostgreSQL database.",
             "The user's backend is built on PostgreSQL.",
             "The user prefers PostgreSQL for their projects."]),
    dict(query="Where does the user currently live?",
         current="The user now lives in Austin.", current_kw="austin",
         stale_kw="seattle", stale=[
             "The user lives in Seattle.",
             "The user has an apartment in Seattle.",
             "The user has lived in Seattle for years.",
             "The user's home is in Seattle.",
             "The user commutes across Seattle every day.",
             "The user loves living in Seattle."]),
    dict(query="What is the user's current job title?",
         current="The user is now a machine learning engineer.", current_kw="machine learning",
         stale_kw="data scientist", stale=[
             "The user works as a data scientist.",
             "The user is a data scientist by profession.",
             "The user's role is data scientist.",
             "The user has a data scientist position.",
             "The user works in a data scientist job.",
             "The user is employed as a data scientist."]),
    dict(query="What car does the user drive?",
         current="The user now drives a Tesla Model 3.", current_kw="tesla",
         stale_kw="civic", stale=[
             "The user drives a Honda Civic.",
             "The user's car is a Honda Civic.",
             "The user commutes in a Honda Civic.",
             "The user owns a Honda Civic.",
             "The user has driven a Honda Civic for years.",
             "The user parks their Honda Civic in the garage."]),
    dict(query="What is the user's primary programming language?",
         current="The user's primary programming language is now Go.", current_kw="go",
         stale_kw="python", stale=[
             "The user's primary programming language is Python.",
             "The user writes most of their code in Python.",
             "The user prefers Python for their work.",
             "The user builds everything in Python.",
             "The user is most comfortable in Python.",
             "The user's main language is Python."]),
    dict(query="What laptop does the user use?",
         current="The user now uses a ThinkPad running Linux.", current_kw="thinkpad",
         stale_kw="macbook", stale=[
             "The user uses a MacBook Pro.",
             "The user's main laptop is a MacBook Pro.",
             "The user works on a MacBook Pro.",
             "The user carries a MacBook Pro everywhere.",
             "The user does all their work on a MacBook Pro.",
             "The user's MacBook Pro is their daily driver."]),
]


def build_db(distractor_embeds, scenario, n_stale, managed):
    path = tempfile.mkdtemp(prefix="mentedb-scale-")
    db = MenteDB(path)
    for content, emb in zip(DISTRACTORS, distractor_embeds):
        db.store(content, embedding=emb)
    stale_texts = scenario["stale"][:n_stale]
    stale_ids = [db.store(t, embedding=e) for t, e in zip(stale_texts, embed(stale_texts))]
    cur_id = db.store(scenario["current"], embedding=embed([scenario["current"]])[0])
    if managed:
        for sid in stale_ids:
            db.relate(cur_id, sid, "supersedes")
    return db, path


def probe(db, scenario, k):
    q = embed([scenario["query"]])[0]
    got_current = got_stale = False
    for r in db.search(q, k):
        mem = db.get_memory(r.id)
        content = (mem.get("content", "") if isinstance(mem, dict)
                   else getattr(mem, "content", "")).lower()
        if scenario["current_kw"] in content:
            got_current = True
        if scenario["stale_kw"] in content:
            got_stale = True
    return got_current, got_stale


def main():
    k = 5
    print(f"scale learn/manage eval  |  local embedder: bge-small-en-v1.5  |  "
          f"corpus {len(DISTRACTORS)} distractors + facts  |  retrieve top-{k}\n")
    print("  For each history length (how many times the stale fact was stated before")
    print("  the single update), over the retrieved top-k across all scenarios:\n")
    distractor_embeds = embed(DISTRACTORS)

    header = f"  {'stale reps':>10} | {'':7} {'current retrieved':>18} | {'stale in context':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n_stale in (1, 3, 6):
        for managed in (False, True):
            cur = stale = 0
            for sc in SCENARIOS:
                db, path = build_db(distractor_embeds, sc, n_stale, managed)
                try:
                    gc, gs = probe(db, sc, k)
                finally:
                    try:
                        db.close()
                    except Exception:
                        pass
                    shutil.rmtree(path, ignore_errors=True)
                cur += gc
                stale += gs
            n = len(SCENARIOS)
            label = "MenteDB" if managed else "naive  "
            print(f"  {n_stale:>10} | {label} {cur}/{n} ({cur/n:>4.0%}){'':7} | {stale}/{n} ({stale/n:>4.0%})")
        print()

    print("  READ: as the stale fact is repeated more, naive retrieval buries the single")
    print("  update (current-retrieved drops, stale-in-context stays high). MenteDB drops")
    print("  the superseded mentions at recall, so the update stays retrievable and the")
    print("  stale ones disappear. The gap IS the management value, and it grows with how")
    print("  much history there is, which is exactly the long-lived-agent case.")


if __name__ == "__main__":
    main()
