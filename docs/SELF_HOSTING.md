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
docker run -p 6677:6677 -v mentedb-data:/var/mentedb/data ghcr.io/nambok/mentedb:latest
```

The image runs as a non-root user, uid and gid `10001`, and keeps its data in
`/var/mentedb/data`. A named volume picks up that ownership automatically; for
a bind mount, give the directory to that user first:

```bash
mkdir -p ./data && sudo chown 10001:10001 ./data
docker run -p 6677:6677 -v "$PWD/data:/var/mentedb/data" ghcr.io/nambok/mentedb:latest
```

The local embedding model (about 90 MB) downloads on first start into
`/var/mentedb/.cache/huggingface`, under the service user's home. Mount a
volume at `/var/mentedb/.cache` too if you want to skip the download when the
container is recreated.

The image has a Docker `HEALTHCHECK` that calls `GET /v1/ready` on port 6677
every 30 seconds (60 second start period, 3 retries), so `docker ps` and
orchestrators that read Docker health see when the server can take traffic.
If you turn on TLS (see [HTTPS](#https)), override the check, for example
`--health-cmd "curl -fsSk https://localhost:6677/v1/ready || exit 1"`.

Release images are tagged with the version (for example `0.38.0`), the short
commit (`sha-1a2b3c4`) and `latest`, and carry the OCI labels
`org.opencontainers.image.revision`, `version` and `source`, so you can pin a
deployment to an exact release and trace it back to the commit it was built
from.

The server listens on `6677` (REST) and `6678` (gRPC). Data lives in `--data-dir`.
The image runs the overnight maintenance sweep automatically (every 24h by
default, see [Step 5](#step-5-overnight-maintenance)), so a plain `docker run`
already keeps memory healthy with nothing extra to schedule.

## Step 2: Configure an LLM (recommended)

**What it does:** the LLM extracts what matters from each turn (facts,
preferences, corrections), detects contradictions, and powers consolidation.
Without it you still get storage and keyword/vector recall, but no automatic
fact extraction.

```bash
export MENTEDB_LLM_PROVIDER=anthropic      # openai | anthropic | ollama | none
export MENTEDB_LLM_API_KEY=sk-ant-...
export MENTEDB_LLM_MODEL=claude-sonnet-5-5 # optional, provider default otherwise
mentedb-server --data-dir ./data --auto-extract
```

`--auto-extract` (or `MENTEDB_AUTO_EXTRACT=1`) turns on per-turn extraction. For
a fully local setup, point `MENTEDB_LLM_PROVIDER=ollama` and
`MENTEDB_LLM_BASE_URL=http://localhost:11434`.

Claude 5.5 models think before answering by default, and the thinking counts
against the output limit. Two optional settings control this:

- `MENTEDB_LLM_MAX_OUTPUT_TOKENS` (default 16000): output ceiling per call,
  thinking included. You are billed only for tokens actually produced. A call
  that hits the ceiling fails with a clear error instead of returning a
  partial answer.
- `MENTEDB_LLM_EFFORT` (`low`, `medium`, `high`; unset keeps the model
  default): lower effort thinks less, which is faster and cheaper.

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

### Contradictions: supersede or flag

When a new memory contradicts an older one (the same sentence with a changed
value, like "my favorite coffee is a cortado" then "... a flat white", or an
LLM verdict that one fact replaces the other), the engine resolves it at write
time. `MENTEDB_CONTRADICTION_POLICY` picks how:

- `supersede` (default): the older memory is closed. Its `valid_until` is set,
  a `supersedes` edge links the two, and recall serves only the newer one.
- `flag`: both memories stay valid and recallable, and a `contradicts` edge
  links them. Use this when your application wants to show the conflict to a
  person or an agent and let it decide.

Memories stored with the explicit `correction` type always supersede, under
either policy, because the caller already made the decision. Any other value
of the variable stops the server at startup.

```bash
MENTEDB_CONTRADICTION_POLICY=flag mentedb-server --data-dir ./data
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

**It is built into the server and on by default (every 24h):**

```bash
mentedb-server --data-dir ./data                    # sweeps every 24h
mentedb-server --data-dir ./data --maintenance-interval-hours 12
mentedb-server --data-dir ./data --maintenance-interval-hours 0   # disable
```

**Or run it one-shot from cron** (set the interval to 0 to avoid double-running):

```bash
# /etc/cron.d/mentedb, 2am daily
0 2 * * *  mentedb-server maintenance --data-dir /var/lib/mentedb
```

**If you never run it:** duplicates pile up, stale facts keep getting recalled,
and old low-value memories never decay, recall quality degrades over time. Don't
skip it.

## Environment variable reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `MENTEDB_LLM_PROVIDER` | `none` | `openai`, `anthropic`, `ollama`, or `none` |
| `MENTEDB_LLM_API_KEY` | unset | LLM API key |
| `MENTEDB_LLM_MODEL` | provider default | Model name |
| `MENTEDB_LLM_BASE_URL` | unset | For Ollama or a proxy |
| `MENTEDB_AUTO_EXTRACT` | `false` | Per-turn fact extraction (`--auto-extract`) |
| `MENTEDB_EMBEDDING_PROVIDER` | `candle`/`hash` | `candle`, `hash`, or `none` |
| `MENTEDB_MAINTENANCE_INTERVAL_HOURS` | `24` | Sweep interval; `0` disables |
| `MENTEDB_JWT_SECRET` | unset | Enables auth; required with `--require-auth` |
| `MENTEDB_ADMIN_KEY` | unset | Admin token issuance |
| `MENTEDB_EXTRACTION_QUALITY_THRESHOLD` | `0.7` | Min confidence to store an extracted fact |
| `MENTEDB_EXTRACTION_DEDUP_THRESHOLD` | `0.85` | Similarity above which an extraction is a duplicate |
| `MENTEDB_CONTRADICTION_POLICY` | `supersede` | `supersede` closes the older of two contradicting memories, `flag` keeps both and links them with a `contradicts` edge |
| `MENTEDB_TLS_CERT_FILE` | unset | PEM certificate chain; with the key, serves HTTPS and gRPC over TLS (`--tls-cert`) |
| `MENTEDB_TLS_KEY_FILE` | unset | PEM private key for the certificate (`--tls-key`) |
| `MENTEDB_SHUTDOWN_DRAIN_SECS` | `10` | Seconds open HTTPS connections get to finish on shutdown |

## Production notes

Run with auth on:

```bash
export MENTEDB_JWT_SECRET=$(openssl rand -base64 32)
mentedb-server --data-dir ./data --require-auth
```

Without `--require-auth` every endpoint is open, fine for local, not for a
public host.

### HTTPS

The server can terminate TLS itself, no proxy needed. Pass a PEM certificate
chain and its private key:

```bash
mentedb-server --data-dir ./data \
  --tls-cert /etc/mentedb/fullchain.pem \
  --tls-key /etc/mentedb/privkey.pem
```

or set `MENTEDB_TLS_CERT_FILE` and `MENTEDB_TLS_KEY_FILE`. With both set, the
REST API on 6677 serves HTTPS only (HTTP/1.1 and HTTP/2) and gRPC on 6678 uses
TLS with the same certificate. With neither set the server speaks plain HTTP.
Setting only one of the two, an unreadable file, or a key that does not match
the certificate stops the server at startup with an error, so a typo never
falls back to plaintext. The files are read once at start; restart the server
after renewing the certificate.

In Docker, mount the files read only and point the variables at them:

```bash
docker run -p 6677:6677 -p 6678:6678 \
  -v mentedb-data:/var/mentedb/data \
  -v /etc/mentedb/tls:/tls:ro \
  -e MENTEDB_TLS_CERT_FILE=/tls/fullchain.pem \
  -e MENTEDB_TLS_KEY_FILE=/tls/privkey.pem \
  --health-cmd "curl -fsSk https://localhost:6677/v1/ready || exit 1" \
  ghcr.io/nambok/mentedb:latest
```

The files must be readable by uid `10001`. Without TLS, keep the server on a
private network or behind a proxy that terminates TLS, since tokens and memory
content would otherwise cross the network in the clear.
