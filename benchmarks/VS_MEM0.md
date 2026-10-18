# MenteDB vs mem0: reproducible head-to-head

This is the exact harness and methodology behind the numbers on the MenteDB
comparison page. Everything here is runnable. If a number looks wrong, run it
yourself and open an issue.

## What we claim (and what we do not)

- **Ingest cost and speed**: on real LongMemEval-S conversations, MenteDB
  ingested the same histories for roughly **7x fewer tokens, ~6x lower dollar
  cost, and ~1.4x faster** than mem0. The gap is structural (mem0 issues
  several LLM calls per `add`; the MenteDB path in this harness makes one
  extraction call per turn, which also names any fact it contradicts).
  The output of that run is not committed to this repository, so these
  figures cannot be checked against a stored log; rerun the harness to
  reproduce them.
- **Accuracy**: **comparable**. On the sample below MenteDB answered 3/5 and
  mem0 2/5. That is a one-question difference on five questions, i.e. within
  noise. We do **not** claim an accuracy advantage.

This page is about measured cost and efficiency on identical inputs, nothing more.

## Fairness controls

- **Same questions**: real items from LongMemEval-S, from the multi-session,
  temporal-reasoning and knowledge-update categories, full histories, no
  trimming. To fit the budget the harness takes the questions with the
  shortest histories first, so with the default of 5 it runs the five shortest
  in those categories (197 to 209 user/assistant exchanges each; the median
  LongMemEval-S history is 243). They are not a random sample.
- **Same extraction model** for both systems: AWS Bedrock Claude Haiku 4.5.
- **Same embedder** for both: `fastembed` BAAI/bge-small-en-v1.5 (local).
- **Same judge**: Claude Sonnet 4.5 answers from each system's retrieved context
  and grades strictly (a wrong value, a hedge, or an evasive non-answer is
  marked incorrect). The answerer is told to use only the retrieved memories
  and to reply "I don't know" otherwise. The control checks on the judge and
  answerer were run by hand and are not part of this repository.
- **Exact token accounting**: every Bedrock call each system makes is
  intercepted (Converse `usage` and the InvokeModel response body) and summed.
  MenteDB `store` + `relate` make no LLM call, so its write path is deterministic.
- **What the MenteDB side is**: the MenteDB path is written in the harness
  (`ingest_mentedb`): one Haiku extraction call per turn, then `store` and a
  `supersedes` edge for any contradiction. It is not `process_turn`.

### The honest caveats

- **N is small** (5 questions, one run). The cost/speed gap is structural and
  stable per turn; the accuracy read is directional. Scale it up if you want
  tighter accuracy numbers.
- **mem0.add() also extracts** facts from raw text; MenteDB's production
  `process_turn` extracts too. The isolated cost difference is the
  **reconciliation** approach (mem0 re-reads similar memories via LLM on every
  write; MenteDB reconciles with a single call plus deterministic supersede
  edges), not "MenteDB does memory for free."

## Reproduce it

```bash
# 1. Python env with the SDK, mem0, and a local embedder
python3.13 -m venv .venv && source .venv/bin/activate
pip install maturin mem0ai fastembed boto3
(cd sdks/python && maturin develop)     # build the MenteDB Python SDK

# 2. AWS credentials for Bedrock (Haiku extraction + Sonnet judge)
export AWS_REGION=us-east-1
eval "$(aws configure export-credentials --format env)"   # or set the AWS_* vars

# 3. Download LongMemEval-S into benchmarks/longmemeval/data/ (see that folder's README)

# 4. Run the head-to-head (defaults: 5 questions, $17 hard cap)
python benchmarks/learn_manage_vs_mem0_lme.py
```

The harness prints, per question, each system's answer correctness, ingest
wall-clock, and measured token cost, then the aggregate. `LME_MAX_Q` and
`LME_CAP_USD` control the sample size and the hard budget cap.

## The harnesses

- `learn_manage_vs_mem0_lme.py` — the real LongMemEval head-to-head above.
- `learn_manage_vs_mem0_deep.py` — apples-to-apples on synthetic conversations
  (both extract via the same model); shared helpers.
- `learn_manage_vs_mem0.py` — first pass (measures cost cleanly; superseded by
  the deep/lme versions for accuracy).
- `learn_manage_vs_mem0_hard.py`, `learn_manage_vs_mem0_stress.py` — typed and
  large-corpus stress probes used to break a false 100%/100% tie and find each
  system's real failure modes (mem0 can surface a stale intermediate on a deep
  update chain; MenteDB's default recall drops superseded facts, which loses
  "what did I use before" history).
