# Self-hosting MenteDB

Run the engine yourself. You own the data, the LLM keys, and the schedule. If
you'd rather not run anything, use the managed [cloud](./CLOUD.md) instead, one
API key and you're done.

There are two moving parts to understand:

1. **The server** handles every conversation turn (store, recall, extract).
2. **The maintenance sweep** runs periodically to keep memory healthy
   (merge duplicates, forget stale facts, decay old ones). This is the part
   people forget, so it is built in, see [Step 5](#step-5-overnight-maintenance).

---

## Step 1: Install the server

**With Cargo:**

```bash
cargo install mentedb-server
mentedb-server --data-dir ./data
```

**With Docker** (the image bundles the local embedding model):

```bash
docker run -p 6677:6677 -v mentedb-data:/data ghcr.io/nambok/mentedb:latest
```

The server listens on `6677` (REST) and `6678` (gRPC). Data lives in `--data-dir`.

## Step 2: Configure an LLM (recommended)

**What it does:** the LLM extracts what matters from each turn (facts,
preferences, corrections), detects contradictions, and powers consolidation.
Without it you still get storage and keyword/vector recall, but no automatic
fact extraction.

```bash
export MENTEDB_LLM_PROVIDER=anthropic      # openai | anthropic | ollama | none
export MENTEDB_LLM_API_KEY=sk-ant-...
export MENTEDB_LLM_MODEL=claude-sonnet-4-6 # optional, provider default otherwise
mentedb-server --data-dir ./data --auto-extract
```

`--auto-extract` (or `MENTEDB_AUTO_EXTRACT=1`) turns on per-turn extraction. For
a fully local setup, point `MENTEDB_LLM_PROVIDER=ollama` and
`MENTEDB_LLM_BASE_URL=http://localhost:11434`.

## Step 3: Choose an embedding provider

**What it does:** embeddings power semantic search, auto-linking, and
contradiction detection. Without a real embedder only keyword recall works.

- `candle` (default in the Docker image): local, semantic (all-MiniLM-L6-v2,
  384 dims), no API key. Requires the `local-embeddings` build feature.
- `hash`: deterministic but **not** semantic. Fine for testing, weak for recall.
- `none`: disables embeddings (not recommended).

```bash
mentedb-server --data-dir ./data --embedding-provider candle
```

## Step 4: Use it

Every turn goes through one call. **`process_turn`** embeds the message, runs
hybrid recall (vector + keyword + graph), stores the turn, extracts facts,
detects contradictions, and returns attention-ordered context for your prompt.

**REST:**

```bash
curl -X POST http://localhost:6677/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"user_message":"I switched from Postgres to SQLite","assistant_response":"Noted.","turn_id":0}'
```

**SDK** (Python / TypeScript / Rust):

```python
from mentedb import MenteDB
db = MenteDB("./data")
result = db.process_turn(user_message="...", assistant_response="...", turn_id=0)
# result.context -> memories for your prompt; result.facts_extracted -> what was learned
```

## Step 5: Overnight maintenance

This is what keeps memory from turning into a junk drawer. The sweep runs, in
order:

| Job | What it does |
|-----|--------------|
| Consolidation | Merges near-duplicate memories into one richer memory (sources kept, linked). |
| Decay | Ages salience so old, unused memories rank lower over time. |
| Archival | Forgets cold memories flagged for deletion (capped per sweep). |
| Cache eviction | Clears stale speculative pre-assembly cache entries. |

**It is built into the server and on by default (every 24h):**

```bash
mentedb-server --data-dir ./data                    # sweeps every 24h
mentedb-server --data-dir ./data --maintenance-interval-hours 12
mentedb-server --data-dir ./data --maintenance-interval-hours 0   # disable
```

**Or run it one-shot from cron** (set the interval to 0 to avoid double-running):

```bash
# /etc/cron.d/mentedb  — 2am daily
0 2 * * *  mentedb-server maintenance --data-dir /var/lib/mentedb
```

**If you never run it:** duplicates pile up, stale facts keep getting recalled,
and old low-value memories never decay, recall quality degrades over time. Don't
skip it.

## Environment variable reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `MENTEDB_LLM_PROVIDER` | `none` | `openai`, `anthropic`, `ollama`, or `none` |
| `MENTEDB_LLM_API_KEY` | — | LLM API key |
| `MENTEDB_LLM_MODEL` | provider default | Model name |
| `MENTEDB_LLM_BASE_URL` | — | For Ollama or a proxy |
| `MENTEDB_AUTO_EXTRACT` | `false` | Per-turn fact extraction (`--auto-extract`) |
| `MENTEDB_EMBEDDING_PROVIDER` | `candle`/`hash` | `candle`, `hash`, or `none` |
| `MENTEDB_MAINTENANCE_INTERVAL_HOURS` | `24` | Sweep interval; `0` disables |
| `MENTEDB_JWT_SECRET` | — | Enables auth; required with `--require-auth` |
| `MENTEDB_ADMIN_KEY` | — | Admin token issuance |
| `MENTEDB_EXTRACTION_QUALITY_THRESHOLD` | `0.7` | Min confidence to store an extracted fact |
| `MENTEDB_EXTRACTION_DEDUP_THRESHOLD` | `0.85` | Similarity above which an extraction is a duplicate |

## Production notes

Run with auth on:

```bash
export MENTEDB_JWT_SECRET=$(openssl rand -base64 32)
mentedb-server --data-dir ./data --require-auth
```

Without `--require-auth` every endpoint is open, fine for local, not for a
public host.
