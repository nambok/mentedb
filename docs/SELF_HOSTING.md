# Self-hosting MenteDB

Run the engine yourself. You own the data, the LLM keys, and the schedule. If
you'd rather not run anything, use the managed [cloud](./CLOUD.md) instead, one
API key and you're done.

There are two moving parts to understand:

1. **The server** handles every conversation turn (store, recall, analyze).
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
docker run -p 6677:6677 -p 6678:6678 -v mentedb-data:/var/mentedb/data ghcr.io/nambok/mentedb:latest
```

The server listens on `6677` (REST, `/metrics`, `/console`) and `6678` (gRPC).
Data lives in `--data-dir` (default `./mentedb-data`; the image uses
`/var/mentedb/data`, so mount your volume there). The server takes an exclusive
lock on the data directory: a second process that opens the same directory fails
fast with a "locked" error instead of corrupting it.
The image runs the overnight maintenance sweep automatically (every 24h by
default, see [Step 5](#step-5-overnight-maintenance)), so a plain `docker run`
already keeps memory healthy with nothing extra to schedule.

## Step 2: Configure an LLM (recommended)

**What it does:** the LLM reads conversations and pulls out what matters (facts,
preferences, corrections) as separate memories. It powers `POST /v1/ingest`,
`--auto-extract` on stored memories, and topic labels in the maintenance sweep.
Without it you still get the whole `process_turn` pipeline (storage, hybrid
recall, rule based fact extraction, embedding based contradiction detection),
just no LLM extraction.

```bash
export MENTEDB_LLM_PROVIDER=anthropic      # openai | anthropic | ollama | bedrock | none
export MENTEDB_LLM_API_KEY=sk-ant-...
export MENTEDB_LLM_MODEL=claude-haiku-5-5  # optional, this is the anthropic default
mentedb-server --data-dir ./data --auto-extract
```

`--auto-extract` (or `MENTEDB_AUTO_EXTRACT=1`) queues LLM extraction in the
background whenever a memory stored through `POST /v1/memories` looks like a
conversation. For
a fully local setup, set `MENTEDB_LLM_PROVIDER=ollama`; it calls
`http://localhost:11434/api/chat` with model `llama3` by default. If you set
`MENTEDB_LLM_BASE_URL`, give the full endpoint URL (for example
`http://localhost:11434/api/chat`), not just the host.

`bedrock` needs a server built with `--features bedrock`; it signs requests with
the standard AWS environment credentials and reads the region from
`MENTEDB_LLM_REGION` (or `AWS_REGION`).

## Step 3: Choose an embedding provider

**What it does:** embeddings power semantic search, auto-linking, and
contradiction detection. Without a real embedder only keyword recall works.

- `candle` (the default when the server is built with the `local-embeddings`
  feature, as the Docker image is): local, semantic (all-MiniLM-L6-v2, 384
  dims), no API key. The model downloads on first start.
- `hash` (the default for a plain `cargo install mentedb-server`, which does not
  include `local-embeddings`): deterministic but **not** semantic. Fine for
  testing, weak for recall. For semantic recall from Cargo, install with
  `cargo install mentedb-server --features local-embeddings`.
- `none`: disables embeddings (not recommended).

```bash
mentedb-server --data-dir ./data --embedding-provider candle
```

## Step 4: Use it

Every turn goes through one call. **`process_turn`** embeds the message, runs
hybrid recall (vector + keyword), stores the turn, extracts simple facts with
rules, links related memories, checks for contradictions, and returns
attention-ordered context for your prompt. It needs no LLM.

**REST:**

```bash
curl -X POST http://localhost:6677/v1/process_turn \
  -H "Content-Type: application/json" \
  -d '{"user_message":"I switched from Postgres to SQLite","assistant_response":"Noted.","turn_id":0}'
```

The response carries `context` (a list of `{id, content, memory_type,
relevance_score, created_at}`), `stored_ids`, `contradictions`,
`facts_extracted`, `pain_warnings` (a count), `predicted_topics`, and the delta
fields `delta_added` / `delta_removed`. (`POST /v1/ingest` is a different call:
it takes a whole `conversation` string plus an `agent_id` and needs an LLM
configured.)

**SDK** (Python / TypeScript / Rust):

```python
from mentedb import MenteDB
db = MenteDB("./data")
result = db.process_turn(user_message="...", assistant_response="...", turn_id=0)
# result.context -> memories for your prompt; result.facts_extracted -> what was learned
```

For the full turn-by-turn agent loop in Python, Node, and Rust, see
[Build an agent](./BUILD_AN_AGENT.md).

## Step 5: Overnight maintenance

This is what keeps memory from turning into a junk drawer. The sweep runs, in
order:

| Job | What it does |
|-----|--------------|
| Consolidation | Merges near-duplicate memories into one richer memory (sources kept, linked). |
| Decay | Ages salience so old, unused memories rank lower over time. |
| Archival | Forgets cold memories flagged for deletion (capped per sweep). |
| Cache eviction | Clears stale speculative pre-assembly cache entries. |
| Topic labels | With an LLM configured, gives recent conversation topics a stable label. |

**It is built into the server and on by default (every 24h):**

```bash
mentedb-server --data-dir ./data                    # sweeps every 24h
mentedb-server --data-dir ./data --maintenance-interval-hours 12
mentedb-server --data-dir ./data --maintenance-interval-hours 0   # disable
```

**Or run it one-shot from cron.** The `maintenance` subcommand opens the data
directory itself, so it only runs when no server holds that directory (a running
server makes it exit with a "locked" error). Use it when you stop the server for
a maintenance window, or for a directory an embedded SDK uses between runs. For
an always-on server, use the built-in interval instead.

```bash
# /etc/cron.d/mentedb  — 2am daily, against a directory no server is holding
0 2 * * *  mentedb-server maintenance --data-dir /var/lib/mentedb
```

**If you never run it:** duplicates pile up, stale facts keep getting recalled,
and old low-value memories never decay, recall quality degrades over time. Don't
skip it.

## Environment variable reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `MENTEDB_LLM_PROVIDER` | `none` | `openai`, `anthropic`, `ollama`, `bedrock` (needs `--features bedrock`), or `none` |
| `MENTEDB_LLM_API_KEY` | — | LLM API key |
| `MENTEDB_LLM_MODEL` | provider default | Model name (`gpt-4o-mini`, `claude-haiku-5-5`, `llama3`) |
| `MENTEDB_LLM_BASE_URL` | provider endpoint | Full endpoint URL, for Ollama on another host or a proxy |
| `MENTEDB_LLM_REGION` | `AWS_REGION` | AWS region for the `bedrock` provider |
| `MENTEDB_AUTO_EXTRACT` | `false` | Background LLM extraction for stored memories (`--auto-extract`); `true` or `1` |
| `MENTEDB_EMBEDDING_PROVIDER` | `candle` if built with `local-embeddings`, else `hash` | `candle`, `hash`, or `none` |
| `MENTEDB_MAINTENANCE_INTERVAL_HOURS` | `24` | Sweep interval; `0` disables |
| `MENTEDB_JWT_SECRET` | — | Enables auth; required with `--require-auth` |
| `MENTEDB_ADMIN_KEY` | — | Admin token issuance and the console's admin tabs |
| `MENTEDB_EXTRACTION_QUALITY_THRESHOLD` | `0.7` | Min confidence to store an extracted fact |
| `MENTEDB_EXTRACTION_DEDUP_THRESHOLD` | `0.85` | Similarity above which an extraction is a duplicate |
| `MENTEDB_SHARDING` | off | `1` or `true` to shard agents across several servers (see below) |
| `MENTEDB_NODE_ID` | `$HOSTNAME` | This node's id in the sharded fleet |
| `MENTEDB_NODE_ADDR` | — | Base URL peers reach this node at; required for sharding |
| `MENTEDB_SEEDS` | — | Comma separated peer base URLs to discover the fleet from |

Each variable also has a flag (`--llm-provider`, `--llm-api-key`, `--llm-model`,
`--llm-base-url`, `--llm-region`, `--embedding-provider`, `--jwt-secret`,
`--admin-key`, `--maintenance-interval-hours`), plus `--port` (default `6677`)
and `--grpc-port` (default `6678`). A flag wins over its variable.
## Production notes

Run with auth on:

```bash
export MENTEDB_JWT_SECRET=$(openssl rand -base64 32)
mentedb-server --data-dir ./data --require-auth
```

Without a JWT secret every endpoint is open, fine for local, not for a public
host. `--require-auth` makes the server refuse to start when no secret is set.

## Running several servers (sharding)

One server owns one data directory. To spread many agents over several servers,
set `MENTEDB_SHARDING=1`, `MENTEDB_NODE_ADDR`, and `MENTEDB_SEEDS` on each node.
The nodes gossip to agree on who is alive, every node computes the same owner
for each agent (rendezvous hashing), and a request that lands on the wrong node
is forwarded to the owner, so each agent's database is only written by one node.
Routing reads the agent id from the JWT, so run sharded nodes with a JWT secret.
`/metrics` reports `mentedb_cluster_live_nodes`.
