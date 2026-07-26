# Agent file compliance benchmark

Does an agent follow its instruction file better when the whole file sits in
context, or when the file is ingested into MenteDB once and rules are retrieved
as they become relevant? This suite measures exactly that, with mechanical
string checkers instead of judges. Full write-up:
https://mentedb.com/blog/infinite-context-window-for-ai-with-memory

## Layout

- `fixtures/` — four real public agent files, copied verbatim; upstream links
  in `fixtures/SOURCES.md`: OpenAI codex (22 KB), Kiali (70 KB), Temporal (8 KB), and
  Freerouting (51 KB, held out: its rules were written before the pipeline
  ever saw the file). Plus two synthetic files (`dev_claude`, `ops_meridian`).
  Each fixture has a `.rules.json` (rule text, needle strings, checkers) and an
  `.atoms.json` parse cache so runs are reproducible without re-parsing.
- `tasks_*.json` — realistic tasks per fixture, each listing the rules that
  govern it.
- `two_pass.py` — the compliance instrument. Three conditions per task:
  whole file in context, MenteDB two pass retrieval, and nothing (the control).
  Mechanical checkers score the output.
- `sweep.py` — offline retrieval tuning through the real injection pipeline
  (no model calls, candle embeddings, multi seed).
- `fast_tune.py` — same scoring on persistent stores so a config evaluates in
  seconds; `--build` once, then iterate.
- `run_compliance.py`, `run_coverage.py`, `cohere_space.py`, `llm_ingest.py` —
  shared helpers (Bedrock calls, checkers, store builders).

## Running it

```bash
pip install mentedb boto3
python3 sweep.py --config '{}' --suites codex,kiali,temporal   # offline, no model
python3 fast_tune.py --build && python3 fast_tune.py --config '{}'
python3 two_pass.py codex,kiali,temporal 5                     # full instrument
```

`two_pass.py` and `fast_tune.py` call AWS Bedrock (Claude Haiku, Claude Sonnet,
Cohere embed-english-v3), so they need AWS credentials with Bedrock access; set
`AWS_PROFILE` if you use a named profile. `sweep.py` runs fully offline.

Rules anchor to verbatim text from the files, checkers are plain string
predicates, and the `nothing` condition is the honesty control: any rule the
model passes with no instructions proves nothing about delivery. Write your own
`tasks_*.json` against any fixture and rerun.
