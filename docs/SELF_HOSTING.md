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
It tries plain HTTP and then HTTPS, so it works with [TLS](#https) on or off.

Release images are tagged with the version (for example `0.38.0`), the short
commit (`sha-1a2b3c4`) and `latest`, and carry the OCI labels
`org.opencontainers.image.revision`, `version` and `source`, so you can pin a
deployment to an exact release and trace it back to the commit it was built
from.

The server listens on `6677` (REST, `/metrics`, `/console`) and `6678` (gRPC;
add `-p 6678:6678` to the `docker run` above to reach it). Data lives in
`--data-dir` (default `./mentedb-data`; the image uses `/var/mentedb/data`, so
mount your volume there). The server takes an exclusive lock on each database
directory: a second process that opens the same directory fails fast with a
"locked" error instead of corrupting it.
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
conversation. For a fully local setup, set `MENTEDB_LLM_PROVIDER=ollama`; it
calls `http://localhost:11434/api/chat` with model `llama3` by default. If you
set `MENTEDB_LLM_BASE_URL`, give the full endpoint URL (for example
`http://localhost:11434/api/chat`), not just the host. On AWS, `bedrock` calls Claude
through Bedrock with the task role and no API key, see
[Bedrock on AWS](#bedrock-on-aws-with-a-task-role).

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

- `candle` (the default when the server is built with the `local-embeddings`
  feature, as the Docker image is): local, semantic (all-MiniLM-L6-v2, 384
  dims), no API key. The model downloads on first start.
- `bedrock`: AWS Bedrock, Cohere Embed v3 by default (1024 dims), credentials
  from the task role. Requires the `bedrock` build feature, which the Docker
  image has. See [Bedrock on AWS](#bedrock-on-aws-with-a-task-role).
- `hash` (the default for a plain `cargo install mentedb-server`, which does not
  include `local-embeddings`): deterministic but **not** semantic. Fine for
  testing, weak for recall. For semantic recall from Cargo, install with
  `cargo install mentedb-server --features local-embeddings`.
- `none`: disables embeddings (not recommended).

Without a flag the server uses `candle` when it was built with
`local-embeddings`, else `hash`. Pick the embedder before you store data: each
one produces vectors of its own size, and the server does not re-embed
memories already stored.

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
[Build an agent](./BUILD_AN_AGENT.md). Every REST endpoint (hybrid search,
heads-up context, pins, history, graph, documents, maintenance, backups) is
described in the [server API reference](./SERVER_API.md).

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

**It is built into the server and on by default (every 24h).** The scheduled
sweep covers every database open at that moment, the default and the named
ones. The same jobs can be run on one database on demand through
`POST /v1/maintenance/...` (see the [API reference](./SERVER_API.md#maintenance-and-enrichment)).

```bash
mentedb-server --data-dir ./data                    # sweeps every 24h
mentedb-server --data-dir ./data --maintenance-interval-hours 12
mentedb-server --data-dir ./data --maintenance-interval-hours 0   # disable
```

**Or run it one-shot from cron** (set the interval to 0 to avoid double-running).
The `maintenance` subcommand opens the data directory itself, so it only runs
when no server holds that directory (a running server makes it exit with a
"locked" error), and it sweeps the default database only. Use it during a
maintenance window or for a directory an embedded SDK uses between runs. For an
always-on server, and with named databases, use the built-in scheduler or the
per database endpoint instead:

```bash
# /etc/cron.d/mentedb, 2am daily, against a directory no server is holding
0 2 * * *  mentedb-server maintenance --data-dir /var/lib/mentedb
```

**If you never run it:** duplicates pile up, stale facts keep getting recalled,
and old low-value memories never decay, recall quality degrades over time. Don't
skip it.

## Production notes

Run with auth on:

```bash
export MENTEDB_JWT_SECRET=$(openssl rand -base64 32)
mentedb-server --data-dir ./data --require-auth
```

Auth turns on as soon as the server has a secret or a public key (see
[Tokens and scopes](#tokens-and-scopes)); without either, every endpoint except
the admin ones is open, fine for local, not for a public host.
`--require-auth` makes that mistake impossible: the server refuses to start
when neither is configured.

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
  ghcr.io/nambok/mentedb:latest
```

The files must be readable by uid `10001`. Without TLS, keep the server on a
private network or behind a proxy that terminates TLS, since tokens and memory
content would otherwise cross the network in the clear.

## Named databases

One server can hold a separate database per tenant (a person, a team, an
organization). Each lives in its own directory with its own file lock and
indexes, so tenants share nothing but the process.

- The **default database** sits at the data directory root and backs the
  routes directly under `/v1`, such as `POST /v1/memories`.
- A **named database** sits in `<data-dir>/dbs/<name>` and backs the same
  routes under `/v1/db/<name>`, such as `POST /v1/db/alice/memories`.

```bash
curl -X POST http://localhost:6677/v1/db/alice/memories \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"agent_id":"00000000-0000-0000-0000-000000000000","memory_type":"semantic","content":"Prefers tea"}'
```

Names may use letters, digits, `-`, `_` and `.`, up to 128 characters, and may
not start with `.`.

| Variable | Default | Effect |
|----------|---------|--------|
| `MENTEDB_AUTO_CREATE_DATABASES` | on | A request to a named database that does not exist creates it. Set `0`, `false`, `no` or `off` to answer `404` instead, so only an admin can create databases (`POST /v1/databases`). |
| `MENTEDB_MAX_OPEN_DATABASES` | `256` | Named databases kept open at once. Past this, the least recently used idle ones are flushed and closed; they reopen on their next request. The default database is always open. |

With the admin key, `GET /v1/databases` lists databases, `POST /v1/databases`
with `{"name": "alice"}` creates one, and `DELETE /v1/databases/alice` closes
and permanently deletes one. The gRPC API, the WebSocket stream and `/metrics`
serve the default database only. See the
[API reference](./SERVER_API.md#databases-and-url-layout) for details.

## Tokens and scopes

With auth on, every request carries `Authorization: Bearer <token>`, except
health, readiness, `/metrics`, `/console` and the token endpoint, which are
open, and the admin routes, which take the admin key instead. A token is a JWT
with these claims:

| Claim | Meaning |
|-------|---------|
| `exp` | Expiry, Unix seconds. Required. |
| `dbs` | Databases the token may use, for example `["alice"]`. `"*"` or no claim means all. The default database is `default`. |
| `read_only` | `true` allows reads and searches only; writes get `403`. |
| `agent_id` | Optional. Binds the token to one agent: request bodies must use that agent, and the token sees only that agent's memories plus shared ones. |
| `admin` | Admin token: not limited by `dbs`, may use database management and backups. |

A request outside the token's databases is `403`, so one deployment can give
each tenant a token that reaches only its own database.

### Shared secret (the server issues tokens)

Set `MENTEDB_JWT_SECRET` and `MENTEDB_ADMIN_KEY`, then mint tokens with the
admin key:

```bash
curl -X POST http://localhost:6677/v1/auth/token \
  -H "x-api-key: $MENTEDB_ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"dbs": ["alice"], "read_only": false, "expiry_hours": 24}'
# {"token":"eyJ...","agent_id":"","dbs":["alice"],"read_only":false}
```

The body also takes `agent_id`. This endpoint never issues admin tokens.

### Public key (tokens signed elsewhere)

To keep signing keys off the server, give it only a public key. Another
service signs tokens with the private half; the server verifies them and holds
no secret.

```bash
# Ed25519 key pair. RSA (RS256) and EC P-256 (ES256) keys work too.
openssl genpkey -algorithm ed25519 -out signing.pem
openssl pkey -in signing.pem -pubout -out verify.pem

export MENTEDB_JWT_PUBLIC_KEY_FILE=/etc/mentedb/verify.pem
export MENTEDB_JWT_ISSUER=my-token-service     # optional, checks `iss`
export MENTEDB_JWT_AUDIENCE=mentedb            # optional, checks `aud`
mentedb-server --data-dir ./data --require-auth
```

The algorithm follows the key: RSA keys verify RS256, EC P-256 keys ES256,
Ed25519 keys EdDSA. A file that is unreadable or not one of those stops the
server at startup. `MENTEDB_JWT_ISSUER` and `MENTEDB_JWT_AUDIENCE` apply to
shared secret tokens too; when unset, `iss` and `aud` are not checked.

Minting a scoped token, here in Python with PyJWT (`pip install pyjwt cryptography`):

```python
import time, jwt

claims = {
    "dbs": ["alice"],
    "read_only": False,
    "exp": int(time.time()) + 3600,
    "iss": "my-token-service",   # must match MENTEDB_JWT_ISSUER when set
    "aud": "mentedb",            # must match MENTEDB_JWT_AUDIENCE when set
}
token = jwt.encode(claims, open("signing.pem").read(), algorithm="EdDSA")
```

When `MENTEDB_JWT_PUBLIC_KEY_FILE` is set the server checks tokens against
that key only, so tokens from `/v1/auth/token` (signed with the shared secret)
are refused. Mint all tokens with the private key.

### Admin key

`MENTEDB_ADMIN_KEY` gates token issuance, database management, backups and the
console endpoints. Send it as `x-api-key` or `Authorization: Bearer`. Database
management and backups also accept a token with `"admin": true`. Without an
admin key those endpoints are disabled, whether or not auth is on.

## Bedrock on AWS with a task role

On ECS or EKS the server can embed and extract through Amazon Bedrock with the
task's IAM role: no API keys, no model download. The Docker image is built
with the `bedrock` feature; for `cargo install`, add `--features bedrock` (and
`local-embeddings` if you also want `candle`).

```bash
export AWS_REGION=us-east-1
export MENTEDB_EMBEDDING_PROVIDER=bedrock
export MENTEDB_EMBEDDING_MODEL=cohere.embed-english-v3   # the default
export MENTEDB_LLM_PROVIDER=bedrock
export MENTEDB_LLM_MODEL=us.anthropic.claude-haiku-5-5   # the default
mentedb-server --data-dir /var/mentedb/data --auto-extract
```

**Embeddings.** `MENTEDB_EMBEDDING_MODEL` picks the model. Supported, with
their dimensions: `cohere.embed-english-v3` (1024, the default),
`cohere.embed-multilingual-v3` (1024), `amazon.titan-embed-text-v2:0` (1024)
and `amazon.titan-embed-text-v1` (1536). Any other value stops the server at
startup. Cohere models embed up to 96 texts per call. The region is
`MENTEDB_EMBEDDING_REGION`, else `AWS_REGION`, else `us-east-1`. At startup the
server makes one embedding call; if it fails (missing permission, model not
enabled) the error is logged, the server still starts, and
[`/v1/ready`](#health-and-readiness) reports the embedder as failing until it
works.

**Extraction.** `MENTEDB_LLM_PROVIDER=bedrock` calls Claude on Bedrock with
`InvokeModel`. The model defaults to `us.anthropic.claude-haiku-5-5`
(`MENTEDB_LLM_MODEL` overrides it). The region is `--llm-region`, else
`MENTEDB_LLM_REGION`, else `AWS_REGION`, else `us-east-1`. No
`MENTEDB_LLM_API_KEY` is needed. `MENTEDB_LLM_MAX_OUTPUT_TOKENS` and
`MENTEDB_LLM_EFFORT` (see [Step 2](#step-2-configure-an-llm-recommended)) apply
here too.

**Credentials** are resolved in this order:

1. Static keys: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, plus
   `AWS_SESSION_TOKEN` for temporary credentials.
2. The container credentials endpoint used by ECS task roles and EKS Pod
   Identity: `AWS_CONTAINER_CREDENTIALS_FULL_URI` (with
   `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE` or
   `AWS_CONTAINER_AUTHORIZATION_TOKEN` when set), else
   `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` on `169.254.170.2`. ECS and EKS set
   these variables for you.

Endpoint credentials are cached and fetched again five minutes before they
expire. Shared config profiles and EC2 instance metadata are not read; on a
plain EC2 host, export keys. With no credentials at all, `bedrock` embeddings
stop the server at startup, and Bedrock extraction requests fail with an error
naming the variables above.

**IAM.** The task role needs `bedrock:InvokeModel` on the models it uses. With
the defaults (Cohere v3 for embeddings, the Claude Haiku 5.5 cross region
inference profile for extraction):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-english-v3",
        "arn:aws:bedrock:us-east-1:<account-id>:inference-profile/us.anthropic.claude-haiku-5-5",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-5-5"
      ]
    }
  ]
}
```

An inference profile routes requests to the model in several regions, so the
foundation model statement covers every region the profile may use. Enable
access to each model in the Bedrock console for the account first. If the
task runs in private subnets without internet access, give it a VPC endpoint
for `bedrock-runtime`.

## Backups and restore

`POST /v1/backup` (or `/v1/db/<name>/backup`) flushes the database and writes
one file, `<database>-<UTC time>.mentebackup`, into
`<backup dir>/<database>/`. `GET /v1/backups` lists them, newest first. Both
need the admin key or an admin token.

```bash
curl -X POST http://localhost:6677/v1/db/alice/backup -H "x-api-key: $MENTEDB_ADMIN_KEY"
```

The backup directory is `MENTEDB_BACKUP_DIR`, default `<data-dir>/backups`. It
must not contain a database's directory (pointing it at the data directory
itself fails). A backup of the default database leaves out the named databases
and a backup directory that sits inside the data directory. Writes that land while a backup runs may or may not be in it; pause
writers when you need an exact point in time. Copy backup files off the host
on your own schedule.

There is no restore endpoint, since replacing the files of an open database is
unsafe. Restore offline:

1. Stop the server.
2. Unpack the backup into an empty directory with
   `mentedb::storage::backup::BackupManager::restore_backup`:

   ```rust
   // Cargo.toml: mentedb = "<the server's version>"
   use std::path::Path;
   use mentedb::storage::backup::BackupManager;

   fn main() -> Result<(), Box<dyn std::error::Error>> {
       let args: Vec<String> = std::env::args().collect();
       let manifest = BackupManager::restore_backup(Path::new(&args[1]), Path::new(&args[2]))?;
       println!("restored {} files", manifest.memory_count);
       Ok(())
   }
   ```

3. Put the directory in place. For a named database, move the old
   `<data-dir>/dbs/<name>` away and put the restored directory there. For the
   default database, restore into a new data directory, move the `dbs/`
   directory over from the old one if you want to keep the named databases,
   and start the server with `--data-dir` pointing at it.
4. Start the server.

## Health and readiness

- `GET /v1/health` is liveness: it answers as long as the HTTP server runs and
  does not touch the database. Restart the process when it stops answering.
- `GET /v1/ready` is readiness: `200` with `{"status": "ready", ...}` when the
  default database answers and the embedder (if any) can embed, `503` with
  `{"status": "not_ready", "failed": [...]}` otherwise. Each check times out
  after 5 seconds and the embed result is reused for 30 seconds. Route traffic
  only to instances that answer `200`.

Neither needs a token. The Docker image's `HEALTHCHECK` uses `/v1/ready`. A
load balancer health check should use `/v1/ready` too, so an instance whose
Bedrock access broke is taken out of rotation instead of failing writes.

## Running several servers (sharding)

One server owns one data directory. To spread many agents over several servers,
set `MENTEDB_SHARDING=1`, `MENTEDB_NODE_ADDR`, and `MENTEDB_SEEDS` on each node.
The nodes gossip to agree on who is alive, every node computes the same owner
for each agent (rendezvous hashing), and a request that lands on the wrong node
is forwarded to the owner, so each agent's database is only written by one node.
Routing reads the `agent_id` claim of the request's token, so run sharded nodes
with auth on and give each agent an agent bound token. Requests without a token,
and tokens scoped to databases with no `agent_id`, are not spread by agent.
`/metrics` reports `mentedb_cluster_live_nodes`.

## Environment variable reference

Command line flags, where they exist, win over the variable, except for
`MENTEDB_MAINTENANCE_INTERVAL_HOURS`, which wins over its flag.

**Server and auth**

| Variable | Flag | Default | Purpose |
|----------|------|---------|---------|
| | `--data-dir` | `./mentedb-data` | Data directory (`/var/mentedb/data` in the image) |
| | `--port` | `6677` | REST port |
| | `--grpc-port` | `6678` | gRPC port |
| `MENTEDB_JWT_SECRET` | `--jwt-secret` | unset | HS256 secret; turns auth on and enables `/v1/auth/token` |
| `MENTEDB_JWT_PUBLIC_KEY_FILE` | | unset | PEM public key (RSA, EC P-256 or Ed25519); turns auth on and replaces the secret for verification |
| `MENTEDB_JWT_ISSUER` | | unset | Required `iss` claim |
| `MENTEDB_JWT_AUDIENCE` | | unset | Required `aud` claim |
| | `--require-auth` | off | Refuse to start without a secret or public key |
| `MENTEDB_ADMIN_KEY` | `--admin-key` | unset | Admin key for tokens, databases, backups and console endpoints |
| `MENTEDB_TLS_CERT_FILE` | `--tls-cert` | unset | PEM certificate chain; with the key, serves HTTPS and gRPC over TLS |
| `MENTEDB_TLS_KEY_FILE` | `--tls-key` | unset | PEM private key for the certificate |
| `MENTEDB_SHUTDOWN_DRAIN_SECS` | | `10` | Seconds open HTTPS connections get to finish on shutdown |
| `RUST_LOG` | | `info` | Log filter |

**Databases, maintenance and backups**

| Variable | Flag | Default | Purpose |
|----------|------|---------|---------|
| `MENTEDB_MAX_OPEN_DATABASES` | | `256` | Named databases kept open at once |
| `MENTEDB_AUTO_CREATE_DATABASES` | | on | Create named databases on first use; `0`, `false`, `no` or `off` disables |
| `MENTEDB_MAINTENANCE_INTERVAL_HOURS` | `--maintenance-interval-hours` | `24` | Sweep interval; `0` disables |
| `MENTEDB_BACKUP_DIR` | | `<data-dir>/backups` | Where backups are written |
| `MENTEDB_CONTRADICTION_POLICY` | | `supersede` | `supersede` closes the older of two contradicting memories, `flag` keeps both and links them with a `contradicts` edge |

**LLM and extraction**

| Variable | Flag | Default | Purpose |
|----------|------|---------|---------|
| `MENTEDB_LLM_PROVIDER` | `--llm-provider` | `none` | `openai`, `anthropic`, `ollama`, `bedrock`, or `none` |
| `MENTEDB_LLM_API_KEY` | `--llm-api-key` | unset | API key (not used by `ollama` or `bedrock`) |
| `MENTEDB_LLM_MODEL` | `--llm-model` | provider default | `gpt-4o-mini`, `claude-haiku-5-5`, `llama3`, or `us.anthropic.claude-haiku-5-5` for Bedrock |
| `MENTEDB_LLM_BASE_URL` | `--llm-base-url` | provider endpoint | Full endpoint URL, for Ollama on another host or a proxy |
| `MENTEDB_LLM_REGION` | `--llm-region` | `AWS_REGION`, else `us-east-1` | Bedrock region for extraction |
| `MENTEDB_LLM_MAX_OUTPUT_TOKENS` | | `16000` | Output ceiling per LLM call, thinking included |
| `MENTEDB_LLM_EFFORT` | | model default | `low`, `medium` or `high` |
| `MENTEDB_AUTO_EXTRACT` | `--auto-extract` | `false` | Background LLM extraction for conversation shaped `POST /v1/memories` bodies; `true` or `1` enables |
| `MENTEDB_EXTRACTION_QUALITY_THRESHOLD` | | `0.7` | Min confidence to store an extracted fact |
| `MENTEDB_EXTRACTION_DEDUP_THRESHOLD` | | `0.85` | Similarity above which an extraction is a duplicate |

**Embeddings and AWS**

| Variable | Flag | Default | Purpose |
|----------|------|---------|---------|
| `MENTEDB_EMBEDDING_PROVIDER` | `--embedding-provider` | `candle` when built with `local-embeddings`, else `hash` | `candle`, `bedrock`, `hash`, or `none` |
| `MENTEDB_EMBEDDING_MODEL` | | `cohere.embed-english-v3` | Bedrock embedding model |
| `MENTEDB_EMBEDDING_REGION` | | `AWS_REGION`, else `us-east-1` | Bedrock region for embeddings |
| `AWS_REGION` | | unset | Region fallback for both Bedrock clients |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | | unset | Static AWS credentials, checked first |
| `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`, `AWS_CONTAINER_CREDENTIALS_FULL_URI` | | set by ECS or EKS | Task role or Pod Identity credentials endpoint |
| `AWS_CONTAINER_AUTHORIZATION_TOKEN`, `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE` | | set by EKS | Authorization for the full URI endpoint |

**Sharding** (off by default)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MENTEDB_SHARDING` | off | `1` or `true` turns on the gossip fleet |
| `MENTEDB_NODE_ADDR` | unset | Base URL peers reach this node at; required, sharding stays off without it |
| `MENTEDB_NODE_ID` | `$HOSTNAME`, else `node-<pid>` | This node's id |
| `MENTEDB_SEEDS` | unset | Comma separated peer base URLs |
