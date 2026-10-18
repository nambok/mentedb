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
  Each fixture has a `.rules.json` (rule text, needle strings, checkers). The
  four real fixtures also ship an `.atoms.json` parse cache so runs are
  reproducible without re-parsing; the two synthetic files have no parse cache,
  so `two_pass.py` cannot run the `dev` and `ops` suites until one is generated
  with `llm_ingest.py`.
- `tasks_*.json` — realistic tasks per fixture, each listing the rules that
  govern it.
- `two_pass.py` — the compliance instrument. Three conditions per task:
  whole file in context, MenteDB two pass retrieval, and nothing (the control).
  Mechanical checkers score the output. Builds each suite's store from the
  cached parse on first run.
- `common.py`, `run_compliance.py`, `run_coverage.py`, `cohere_space.py`,
  `llm_ingest.py` — shared helpers (suite loading, Bedrock calls, checkers,
  store building, parse cache generation).

## Running it

```bash
pip install mentedb boto3
cd benchmarks/agent_file
python3 two_pass.py codex,kiali,temporal 5   # suites, trials per task (default 3)
```

`two_pass.py` calls AWS Bedrock in us-east-1 (Claude Haiku 4.5 writes the
artifact, Claude Sonnet 4.5 revises it, Cohere embed-english-v3 embeds), so it
needs AWS credentials with Bedrock access; set `AWS_PROFILE` if you use a named
profile. Stores are built under `/tmp/agent-file-bench-stores` (override with
`MDB_BENCH_STORES`).

Rules anchor to verbatim text from the files, checkers are plain string
predicates, and the `nothing` condition is the honesty control: any rule the
model passes with no instructions proves nothing about delivery. Write your own
`tasks_*.json` against any fixture and rerun.
