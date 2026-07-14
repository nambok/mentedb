# MenteDB vs mem0: reproducible head-to-head

This is the exact harness and methodology behind the numbers on the MenteDB
comparison page. Everything here is runnable. If a number looks wrong, run it
yourself and open an issue.

## What we claim (and what we do not)

- **Ingest cost and speed**: on real LongMemEval-S conversations, MenteDB
  ingested the same histories for roughly **7x fewer tokens, ~6x lower dollar
  cost, and ~1.4x faster** than mem0. This is the robust result and it is
  structural (mem0 issues multiple LLM calls per `add`; MenteDB does one
  extract-plus-contradiction call per turn).
- **Accuracy**: **comparable**. On the sample below MenteDB answered 3/5 and
  mem0 2/5. That is a one-question difference on five questions, i.e. within
  noise. We do **not** claim an accuracy advantage.

We also concede where mem0 leads: distribution, ecosystem breadth, framework
integrations, and adoption. This page is about measured cost/efficiency on
identical inputs, nothing more.

## Fairness controls

- **Same questions**: real items from LongMemEval-S, from the long categories
  (multi-session, temporal-reasoning, knowledge-update), full ~200-exchange
  histories, no trimming.
- **Same extraction model** for both systems: AWS Bedrock Claude Haiku 4.5.
- **Same embedder** for both: `fastembed` BAAI/bge-small-en-v1.5 (local).
- **Same judge**: Claude Sonnet 4.5 answers from each system's retrieved context
  and grades strictly (a wrong value, a hedge, or an evasive non-answer is
  marked incorrect). The judge and the answerer are validated: on a control set
  the judge fails stale/hedged/wrong/confabulated answers, and the answerer says
  "I don't know" on empty context and repeats the stale fact when given only
  that, i.e. it follows retrieval rather than its own knowledge.
- **Exact token accounting**: every Bedrock call each system makes is
  intercepted (Converse `usage` and the InvokeModel response body) and summed.
  MenteDB `store` + `relate` make no LLM call, so its write path is deterministic.

### The honest caveats

- **N is small** (5 questions, one run, ~$11 of Bedrock at the $17 cap). The
  cost/speed gap is structural and stable per turn; the accuracy read is
  directional. Scale it up if you want tighter accuracy numbers.
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
