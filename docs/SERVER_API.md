# Server REST API reference

Every HTTP endpoint `mentedb-server` serves, with the request fields it reads,
the response it returns, who may call it, and the errors it gives. For
installing and configuring the server, see [Self-hosting](./SELF_HOSTING.md).

The server listens on port `6677` (`--port`). Examples use
`http://localhost:6677`; with TLS on (see [HTTPS](./SELF_HOSTING.md#https)) use
`https://`. Request bodies are JSON, so send `Content-Type: application/json`
on every `POST`.

- [Conventions](#conventions)
- [Databases and URL layout](#databases-and-url-layout)
- [Authentication and access](#authentication-and-access)
- [Errors](#errors)
- [Global endpoints](#global-endpoints)
- [Memories](#memories)
- [Search and recall](#search-and-recall)
- [Turns and extraction](#turns-and-extraction)
- [Graph, history and conflicts](#graph-history-and-conflicts)
- [Pins, profile, entities and recent](#pins-profile-entities-and-recent)
- [Documents](#documents)
- [Maintenance and enrichment](#maintenance-and-enrichment)
- [Backups](#backups)
- [Console endpoints](#console-endpoints)
- [Spaces](#spaces)
- [WebSocket stream](#websocket-stream)

## Conventions

- **Times** are Unix timestamps in **microseconds** (`created_at`, `as_of`,
  `valid_from`, `valid_until`, `time_range`).
- **IDs** (memory, agent, user, space, document) are UUID strings.
- **Memory types**: `episodic`, `semantic`, `procedural`, `antipattern` (also
  accepted as `anti_pattern`), `reasoning`, `correction`. Case does not matter.
- **Edge types**: `caused`, `before`, `related`, `contradicts`, `supports`,
  `supersedes`, `derived`, `partof` (also accepted as `part_of`). Responses
  spell them in lower case, for example `partof`.

### The memory object

Endpoints that return a memory use this shape. Some add fields such as
`score`, `reason`, `hops` or `pinned`, noted per endpoint.

```json
{
  "id": "6f1c0d0e-2a51-4a8e-9a57-3f0a3c2b9e11",
  "agent_id": "00000000-0000-0000-0000-000000000000",
  "memory_type": "semantic",
  "embedding": [0.012, -0.044, ...],
  "content": "User prefers dark mode",
  "created_at": 1760745600000000,
  "accessed_at": 1760745600000000,
  "access_count": 0,
  "salience": 0.5,
  "confidence": 1.0,
  "space_id": "00000000-0000-0000-0000-000000000000",
  "attributes": {},
  "tags": ["preferences"]
}
```

## Databases and URL layout

One server holds many databases. Each has its own directory, file lock and
indexes, so nothing is shared between them but the process.

- **The default database** lives at the data directory root and backs the
  routes directly under `/v1`, for example `POST /v1/memories`.
- **Named databases** live under `<data-dir>/dbs/<name>` and back the same
  routes under `/v1/db/{db}`, for example `POST /v1/db/alice/memories`. Every
  per database route below works under both prefixes. `/v1/db/default/...` is
  the default database.

A named database opens on first use. When it does not exist yet it is created,
unless `MENTEDB_AUTO_CREATE_DATABASES` is off, in which case the request gets
`404` until an admin creates it with [`POST /v1/databases`](#post-v1databases).
At most `MENTEDB_MAX_OPEN_DATABASES` (default `256`) named databases stay open;
past that, the least recently used idle ones are flushed and closed, and reopen
on their next request.

Database names are letters, digits, `-`, `_` and `.`, at most 128 characters,
not starting with `.`. Any other name is a `400`.

The WebSocket stream, gRPC and `/metrics` serve the default database only.

## Authentication and access

Auth is off until the server has a way to check tokens: an HS256 secret
(`--jwt-secret` / `MENTEDB_JWT_SECRET`) or a public key
(`MENTEDB_JWT_PUBLIC_KEY_FILE`). With neither, every endpoint except the admin
ones is open. `--require-auth` makes the server refuse to start without one.
See [Tokens and scopes](./SELF_HOSTING.md#tokens-and-scopes) for setup.

With auth on, send `Authorization: Bearer <token>`. These paths need no token:
`/v1/health`, `/v1/ready`, `/v1/auth/token`, `/metrics`, `/console`, and
`/v1/cluster/gossip`.

### Token claims

| Claim | Type | Meaning |
|-------|------|---------|
| `exp` | number | Expiry, Unix seconds. Required. |
| `agent_id` | string | Agent the token is bound to. Empty or absent for a token scoped to databases only. |
| `dbs` | array of strings | Databases the token may use. `"*"` means all. Absent means all. The default database is named `default`. |
| `read_only` | bool | Read only access. Default `false`. |
| `admin` | bool | Admin token. Default `false`. |
| `iss`, `aud` | string | Checked only when `MENTEDB_JWT_ISSUER` / `MENTEDB_JWT_AUDIENCE` are set. |

How the claims are applied:

- **Database scope.** A request to a database outside `dbs` is `403`
  (`token is not scoped to database '<name>'`). Admin tokens are not limited
  by `dbs`.
- **Read only.** A read only token may use `GET` routes and these `POST`
  routes, which only read: `/recall`, `/search`, `/search/hybrid`, `/context`.
  Every other `POST` and every `DELETE` is `403` (`token is read-only`), even
  for an admin token.
- **Agent binding.** When the token carries an `agent_id`, an `agent_id` in a
  request body that differs from it is `403`. Reading, changing, pinning or
  deleting a memory owned by another agent is `403`, unless the token is an
  admin token. List and search endpoints show such a caller its own memories
  plus shared ones (owned by the nil agent `00000000-0000-0000-0000-000000000000`).

### Admin access

Admin routes are database management (`/v1/databases`), backups (`/backup`,
`/backups`) and the console endpoints (`/admin/...`). They need the admin key
(`--admin-key` / `MENTEDB_ADMIN_KEY`), sent as `x-api-key: <key>` or
`Authorization: Bearer <key>`. Database management and backups also accept a
valid token with `"admin": true`; the console endpoints accept only the admin
key. With no admin key configured, these routes answer `401`
(`admin endpoints disabled: no admin key`), whether or not auth is on. A
missing key is `401`, a wrong one `403`.

## Errors

Errors are JSON with one field:

```json
{ "error": "missing or invalid 'content'" }
```

| Status | When |
|--------|------|
| `400` | Missing or invalid field, unknown type name, invalid database name, no embedder for a text query, or auth disabled at `/v1/auth/token`. |
| `401` | Missing or invalid bearer token, expired token, missing admin key, or admin endpoints disabled. |
| `403` | Token not scoped to the database, read only token on a write, `agent_id` mismatch, memory owned by another agent, wrong admin key. |
| `404` | Memory, document or database not found (a database only when auto create is off). |
| `409` | Database already exists or in use, backup file name collision, or a pin change refused because the memory duplicates another. |
| `429` | Rate limited: `{"error": "too many requests"}`. Each client IP gets a bucket of 100 requests refilled at 10 per second. |
| `500` | Engine or storage failure. |
| `503` | Readiness check failed, embedding call failed, or an endpoint that needs an LLM or an embedder has none configured. |

A request without `Content-Type: application/json`, or with a body that is not
valid JSON, is rejected before it reaches the handler (`415`, `400` or `422`)
with a plain text body instead of the JSON error object.

---

## Global endpoints

### `GET /v1/health`

Liveness. Answers as soon as the HTTP server runs, without touching the
database. No auth.

```json
{ "status": "ok", "version": "0.1.0", "uptime_seconds": 3600 }
```

`version` is a fixed string, not the release version.

### `GET /v1/ready`

Readiness. No auth. Checks that the default database answers and, when an
embedder is configured, that it can embed a short probe text. The embed result
is reused for 30 seconds so frequent probes stay cheap; each check times out
after 5 seconds. The Docker image's `HEALTHCHECK` calls this route.

`200`:

```json
{ "status": "ready", "checks": { "database": "ok", "embedder": "ok" } }
```

`embedder` is `"not_configured"` when the server runs without one; that still
counts as ready.

`503`:

```json
{
  "status": "not_ready",
  "failed": ["embedder"],
  "checks": { "database": "ok", "embedder": "error: embed failed: ..." }
}
```

Use `/v1/health` for liveness (restart the process when it stops answering)
and `/v1/ready` for readiness (send traffic only when it answers `200`). A
Bedrock embedder that cannot reach Bedrock, or lacks permission, makes the
server not ready without making it unhealthy.

### `POST /v1/auth/token`

Issue an HS256 token signed with the server's secret. Needs the admin key and
a configured `--jwt-secret`. No bearer token.

Request:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `agent_id` | string | `""` | Agent to bind the token to. Leave empty for a token scoped to databases only. |
| `expiry_hours` | number | `24` | Hours until the token expires. |
| `dbs` | array of strings | absent (all) | Databases the token may use; `"*"` for all. Names are validated. |
| `read_only` | bool | `false` | Issue a read only token. |

```bash
curl -X POST http://localhost:6677/v1/auth/token \
  -H "x-api-key: $MENTEDB_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dbs": ["alice"], "read_only": true, "expiry_hours": 1}'
```

```json
{ "token": "eyJ0eXAi...", "agent_id": "", "dbs": ["alice"], "read_only": true }
```

This endpoint never issues admin tokens. Errors: `400` when no secret is
configured (`auth is disabled (no jwt-secret configured)`) or the body is
invalid, `403` when no admin key is configured or the key is wrong, `401` when
the key is missing.

When `MENTEDB_JWT_PUBLIC_KEY_FILE` is set the server verifies tokens with that
key only, so tokens from this endpoint are refused. Sign tokens with your own
private key instead.

### `GET /v1/databases`

Admin. Lists databases.

```json
{ "default": "default", "databases": ["alice", "bob"], "open": ["default", "alice"] }
```

`databases` is every named database on disk, sorted. `open` is what is open
right now, the default first.

### `POST /v1/databases`

Admin. Creates a named database.

```json
{ "name": "alice" }
```

`201`:

```json
{ "name": "alice", "status": "created" }
```

`409` when it already exists (including `default`), `400` for an invalid name.

### `DELETE /v1/databases/{name}`

Admin. Closes the database and permanently deletes its directory. Backups are
kept.

```json
{ "name": "alice", "status": "deleted" }
```

`400` for `default` or an invalid name, `404` when it does not exist, `409`
when a request is using it right now (retry).

### `GET /metrics`

Prometheus metrics for the default database and the process. No auth.

### `GET /console`

The bundled management console (HTML). No auth to load the page; its memory
browser calls the [console endpoints](#console-endpoints) with the admin key.

### `POST /v1/cluster/gossip`

Peer to peer membership exchange used when `MENTEDB_SHARDING` is on. `404`
when sharding is off. Not for clients.

---

## Memories

### `POST /memories`

Store one memory. Write.

| Field | Type | Required | Default | Meaning |
|-------|------|----------|---------|---------|
| `agent_id` | UUID | yes | | Owning agent. Must match a token bound to an agent. |
| `memory_type` | string | yes | | One of the memory types. |
| `content` | string | yes | | The memory text. |
| `embedding` | array of numbers | no | server embeds `content` | A vector of the database's dimension. |
| `tags` | array of strings | no | `[]` | |
| `attributes` | object | no | `{}` | String, number and boolean values are kept; others are dropped. |
| `space_id` | UUID | no | nil | |
| `user_id` | UUID | no | nil | |
| `salience` | number | no | `0.5` | |
| `confidence` | number | no | `1.0` | |
| `valid_from`, `valid_until` | number | no | none | Microseconds. |

`201`:

```json
{ "id": "6f1c0d0e-2a51-4a8e-9a57-3f0a3c2b9e11", "status": "stored" }
```

Write time inference runs on store: a memory that contradicts an older one
supersedes it or is flagged, per `MENTEDB_CONTRADICTION_POLICY`. When the
server runs with `--auto-extract` and an LLM, content that looks like a
conversation (at least two of `user:`, `assistant:`, `human:`, `ai:`, `q:`,
`a:` line markers) is also queued for fact extraction in the background.

### `POST /memories/batch`

Store up to 1000 memories in one call. Write. Every item takes the same fields
as `POST /memories`, and every item is validated before any is stored. Items
without an `embedding` are embedded together in one batch call.

```json
{
  "memories": [
    { "agent_id": "...", "memory_type": "semantic", "content": "Prefers tea" },
    { "agent_id": "...", "memory_type": "semantic", "content": "Lives in Lisbon", "tags": ["home"] }
  ]
}
```

`201`:

```json
{ "ids": ["...", "..."], "stored": 2 }
```

Validation errors name the item, for example `memories[1]: missing 'content'`.
A vector of the wrong dimension is `400`.

### `GET /memories/{id}`

Read one memory. Returns the [memory object](#the-memory-object). `404` when
it does not exist, `403` when it belongs to another agent.

### `DELETE /memories/{id}`

Forget one memory. Write.

```json
{ "status": "deleted" }
```

### `POST /edges`

Create a typed, weighted edge between two memories. Write.

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `source` | UUID | yes | |
| `target` | UUID | yes | |
| `edge_type` | string | yes | |
| `weight` | number | no | `1.0` |
| `label` | string | no | none |
| `valid_from`, `valid_until` | number | no | none |

`201`:

```json
{ "status": "created" }
```

### `GET /stats`

```json
{ "database": "alice", "memory_count": 1284, "uptime_seconds": 3600 }
```

---

## Search and recall

### `POST /search`

Nearest neighbors by vector. Read (allowed for read only tokens).

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `embedding` | array of numbers | | Query vector. |
| `query` | string | | Text the server embeds, used when `embedding` is absent. |
| `k` | number | `10` | Results. |

One of `embedding` or `query` is required. A text query on a server without an
embedder is `400`.

```json
{ "results": [ { "id": "...", "score": 0.83 } ] }
```

### `POST /search/hybrid`

Vector similarity fused with BM25 keyword matching, at a point in time,
optionally scoped. Read.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `query` | string | | Text. Embedded by the server when `embedding` is absent, and used for keyword matching. |
| `embedding` | array of numbers | | Query vector. Wins over `query` for the vector half. |
| `k` | number | `10` | Results, at most `200`. |
| `as_of` | number | now | Only memories valid at this time (microseconds). |
| `time_range` | `[from, until]` | none | Time window, microseconds. |
| `tags` | array of strings | none | Tag filter. |
| `tags_mode` | `"and"` or `"or"` | `"and"` | Whether every tag or any tag must match. |
| `agent_id` | UUID | the token's agent | Limit to this agent's memories plus shared ones. Must match a token bound to an agent. |
| `user_id` | UUID | none | Limit to this user's memories plus shared ones. |
| `project` | string | none | Current project. Memories tagged `scope:project:<other>` rank lower; they are not hidden. |

One of `query` or `embedding` is required.

```json
{
  "database": "alice",
  "as_of": 1760745600000000,
  "results": [ { "id": "...", "content": "...", "score": 0.031, ... } ]
}
```

Each result is a [memory object](#the-memory-object) plus `score`.

### `POST /context`

Heads-up context for a prompt: pinned memories plus the most relevant ones,
each with its score and the reason it was picked. Read.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `query` | string | | Prompt text. |
| `embedding` | array of numbers | | Query vector instead of text. |
| `max_items` | number | `10` | Ranked items beyond pins, at most `100`. |
| `max_episodic` | number | `3` | Verbatim episodic items, at most `max_items`. |
| `session_id` | string | none | |
| `project` | string | none | Current project. |
| `exclude_ids` | array of UUIDs | `[]` | Memories to leave out, for example ones already in the prompt. |
| `agent_id` | UUID | the token's agent | |
| `user_id` | UUID | none | |

```json
{
  "database": "alice",
  "memories": [
    { "id": "...", "content": "Always answer in Portuguese", "score": 1.0, "reason": "pinned", ... },
    { "id": "...", "content": "...", "score": 0.42, "reason": "relevant", ... }
  ]
}
```

`reason` is `pinned`, `relevant` or `mode_rule`.

### `POST /recall`

Run an MQL query and get an assembled context window. Read.

```json
{ "query": "RECALL memories WHERE tag = \"preferences\" LIMIT 10" }
```

```json
{ "context": "...", "total_tokens": 212, "memory_count": 4 }
```

`context` is the formatted context text. An MQL parse error is `500`.

---

## Turns and extraction

### `POST /process_turn`

The full per turn pipeline in one call: embed the message, recall context,
store the turn, extract and reconcile facts. Write. Delta serving (only what
changed since the agent's last turn) is tracked per database and agent across
requests.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `user_message` | string | yes | |
| `assistant_response` | string | no | |
| `turn_id` | number | no (`0`) | |
| `project_context` | string | no | |
| `session_id` | string | no | |
| `agent_id` | UUID | no | Must match a token bound to an agent. |
| `user_id` | UUID | no | |

```bash
curl -X POST http://localhost:6677/v1/process_turn \
  -H "Content-Type: application/json" \
  -d '{"user_message":"I switched from Postgres to SQLite","assistant_response":"Noted.","turn_id":0}'
```

```json
{
  "ok": true,
  "context": [
    { "id": "...", "content": "...", "memory_type": "semantic", "relevance_score": 0.71, "created_at": 1760745600000000 }
  ],
  "cache_hit": false,
  "stored_ids": ["..."],
  "episodic_id": "...",
  "contradictions": 1,
  "facts_extracted": 2,
  "edges_created": 3,
  "inference_actions": 1,
  "pain_warnings": 0,
  "predicted_topics": ["sqlite"],
  "delta_added": ["..."],
  "delta_removed": []
}
```

`contradictions` and `pain_warnings` are counts.

### `POST /ingest`

Extract memories from a conversation with the configured LLM and store the
ones that pass the quality threshold. Write. `503` when no LLM provider is
configured.

| Field | Type | Required |
|-------|------|----------|
| `agent_id` | UUID | yes |
| `conversation` | string | yes |
| `space_id` | UUID | no |

```json
{
  "memories_stored": 3,
  "rejected_low_quality": 1,
  "rejected_duplicate": 0,
  "contradictions": 0,
  "stored_ids": ["...", "...", "..."]
}
```

`rejected_duplicate` and `contradictions` are always `0` here; duplicates and
contradictions are resolved by the engine when each memory is stored.

---

## Graph, history and conflicts

### `GET /memories/{id}/history`

The chain of older facts this memory replaced, newest predecessor first.

Query: `depth` (default and maximum `64`).

```json
{ "memory": { ... }, "history": [ { ... }, { ... } ] }
```

### `GET /memories/{id}/neighbors`

Memories within `depth` hops in the graph, with the edges between them.

| Query | Default | Meaning |
|-------|---------|---------|
| `depth` | `1` | Hops, at most `4`. |
| `edge_types` | all | Comma separated edge types to walk, for example `caused,supersedes`. |
| `direction` | `both` | `out`, `in` or `both`. |

At most 500 neighbors are returned.

```json
{
  "memory": { ... },
  "depth": 1,
  "neighbors": [ { "id": "...", "hops": 1, ... } ],
  "edges": [
    {
      "source": "...", "target": "...", "edge_type": "supersedes", "weight": 1.0,
      "created_at": 1760745600000000, "valid_from": null, "valid_until": null, "label": null
    }
  ]
}
```

### `GET /conflicts`

Contradiction and supersession edges with both memories of each pair. A
contradiction pair appears once.

Query: `limit` (default `50`, at most `1000`).

```json
{
  "contradicts": 2,
  "supersedes": 7,
  "conflicts": [
    { "edge_type": "contradicts", "weight": 0.9, "source": { ... }, "target": { ... } }
  ]
}
```

`contradicts` and `supersedes` count every such edge in the database;
`conflicts` lists the pairs the caller may see, up to `limit`.

---

## Pins, profile, entities and recent

The list endpoints here take `limit` (default `50`, at most `1000`) and return
newest first.

### `GET /pins`

Pinned memories, which every [`/context`](#post-context) response includes.

```json
{ "pins": [ { ... } ] }
```

### `POST /memories/{id}/pin` and `DELETE /memories/{id}/pin`

Pin or unpin a memory. Write. A pin is the tag `scope:always`; pinning also
adds `source:manual`. Pinned memories do not decay and are never archived.
Unpinning keeps the memory.

```json
{ "id": "...", "tags": ["scope:always", "source:manual"], "pinned": true, ... }
```

`409` when the engine refuses the change because the memory duplicates another
memory of the same owner.

### `GET /profile`

The user profile and the facts it is built from.

Query: `user_id` (default nil), `agent_id` (default the token's agent, else nil).

```json
{
  "user_id": "...",
  "agent_id": "...",
  "profile": { ... },
  "facts": ["Lives in Lisbon", "Prefers tea"]
}
```

`profile` is `null` when none has been built yet.

### `GET /entities`

Entities named by entity memories, with how many memories carry each, most
first, plus the names the entity resolver knows.

```json
{
  "entities": [ { "name": "max", "memory_count": 4 } ],
  "resolver": ["Max"]
}
```

`resolver` is empty for a caller bound to one agent, since the resolver is
shared by the whole database.

### `GET /entities/{name}/memories`

Memories tagged with an entity, after resolving the name to its canonical
form.

```json
{
  "name": "my dog",
  "canonical": "Max",
  "confidence": 0.9,
  "source": "rulebased",
  "memories": [ { ... } ]
}
```

`source` is `cache`, `rulebased`, `llm` or `identity`.

### `GET /recent`

The newest memories.

```json
{ "memories": [ { ... } ] }
```

---

## Documents

### `POST /documents`

Load a long text as overlapping chunks, each stored as a memory and embedded
in one batch. Write. Paragraphs are kept whole when they fit, longer ones are
split on sentence boundaries, and each chunk after the first starts with the
end of the previous one.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `content` | string | required | The document text. |
| `title` | string | none | Stored as the `title` attribute. |
| `source` | string | none | Stored as the `source` attribute. |
| `agent_id` | UUID | the token's agent, else nil | |
| `user_id` | UUID | nil | |
| `memory_type` | string | `semantic` | |
| `tags` | array of strings | `[]` | Added to every chunk. |
| `chunk_chars` | number | `1500` | Between `100` and `8000`. |
| `overlap_chars` | number | `200` | At most half of `chunk_chars`. |

A document may produce at most 10000 chunks. Every chunk carries the tag
`doc:<document_id>` and the attributes `document_id`, `chunk_index` and
`chunk_count`.

`201`:

```json
{
  "database": "alice",
  "document_id": "0b7e...",
  "chunk_count": 12,
  "chunk_ids": ["...", "..."]
}
```

### `DELETE /documents/{document_id}`

Forget every chunk of a document. Write. Ownership of every chunk is checked
before any is forgotten.

```json
{ "database": "alice", "document_id": "0b7e...", "deleted": 12 }
```

`404` when no chunk carries that document id.

---

## Maintenance and enrichment

These run the server's upkeep jobs on one database on demand. They are writes,
so a read only token gets `403`; they do not need the admin key. The scheduled
sweep (`--maintenance-interval-hours`) runs the same work on every open
database.

### `POST /maintenance/decay`

```json
{ "database": "alice", "decayed": 1284 }
```

Decay is derived when memories are read, so this persists nothing; it reports
what the engine reports.

### `POST /maintenance/archive`

Evaluates every memory and forgets the ones flagged for deletion, capped per
run. Memories flagged for archiving or consolidation are only counted.

```json
{
  "database": "alice",
  "evaluated": 1284,
  "keep": 1270,
  "flagged_archive": 6,
  "flagged_consolidate": 3,
  "flagged_delete": 5,
  "forgotten": 5,
  "capped": false
}
```

### `POST /maintenance/consolidate`

Folds paraphrased restatements into the older memory, then merges clusters of
near duplicates into one memory with the sources invalidated and linked.

```json
{
  "database": "alice",
  "duplicates_merged": 4,
  "clusters_found": 2,
  "clusters_consolidated": 2
}
```

### `POST /maintenance/sweep`

The full sweep the scheduler runs.

```json
{ "database": "alice", "consolidated": 2, "decayed": 1284, "forgotten": 5 }
```

### `POST /enrich`

Runs the LLM enrichment pipeline over memories stored since the last run:
extraction, entity linking, communities and the user model. Write. `503` when
no LLM provider or no embedder is configured.

```json
{
  "database": "alice",
  "memories_stored": 9,
  "edges_created": 14,
  "entities_extracted": 3,
  "duplicates_skipped": 1,
  "contradictions_found": 0,
  "sync_linked": 5,
  "llm_linked": 2,
  "communities_created": 1,
  "user_model_updated": true
}
```

---

## Backups

Admin. Backups are written under `MENTEDB_BACKUP_DIR` (default
`<data-dir>/backups`), one subdirectory per database. There is no restore
endpoint; see [Backups and restore](./SELF_HOSTING.md#backups-and-restore).

### `POST /backup`

Flushes the database and writes one backup file named
`<database>-<UTC time>.mentebackup`, for example
`alice-20261018T021500.123Z.mentebackup`. Writes that land while the files are
read may or may not be included; pause writers for an exact point in time. A
backup of the default database leaves out the named databases and the backup
directory.

`201`:

```json
{
  "database": "alice",
  "file": "alice-20261018T021500.123Z.mentebackup",
  "manifest": {
    "created_at": 1760753700123000,
    "size_bytes": 696320,
    "memory_count": 9,
    "version": "mentedb-backup-v1"
  }
}
```

In the manifest, `memory_count` is the number of files in the backup and
`size_bytes` their total size. `500` when the backup directory contains the
database's directory.

### `GET /backups`

Backups of this database, newest first.

```json
{
  "database": "alice",
  "backups": [ { "file": "alice-20261018T021500.123Z.mentebackup", "manifest": { ... } } ]
}
```

---

## Console endpoints

Used by `/console` and the `mentedb` command line client's remote mode. They
need the admin key in `x-api-key` or `Authorization: Bearer`; admin tokens are
not accepted here.

### `GET /admin/memories`

A page of memories.

| Query | Default | Meaning |
|-------|---------|---------|
| `limit` | `50` | `1` to `200`. |
| `offset` | `0` | |
| `agent` | none | Owning agent UUID. |
| `type` | none | Memory type. |
| `q` | none | Case insensitive content substring. |

```json
{ "total": 1284, "limit": 50, "offset": 0, "memories": [ { ... } ] }
```

### `DELETE /admin/memories/{id}`

```json
{ "status": "deleted", "id": "..." }
```

### `POST /admin/mql`

```json
{ "mql": "RECALL memories WHERE type = semantic LIMIT 5" }
```

```json
{ "count": 5, "memories": [ { "score": 0.8, ... } ] }
```

An MQL error is `400`.

---

## Spaces

Memory spaces group memories for sharing between agents. The space registry
is kept in memory and is not saved across restarts. These routes are global,
not per database.

### `POST /v1/spaces`

```json
{ "name": "team-notes" }
```

The owner is the token's agent; without auth, send `agent_id` in the body.
`201`:

```json
{ "id": "...", "name": "team-notes", "owner": "..." }
```

### `GET /v1/spaces`

Spaces the token's agent can access. Needs a token bound to an agent; without
one it is `401`.

```json
{ "spaces": [ { "id": "...", "name": "team-notes", "owner": "..." } ] }
```

### `POST /v1/spaces/{id}/grant`

Grant another agent access. Needs a token bound to an agent (without auth it
is `401`), and the caller must hold admin permission on the space or use an
admin token.

```json
{ "agent_id": "...", "permission": "read" }
```

`permission` is `read`, `write`, `readwrite` (or `read_write`) or `admin`.

```json
{ "status": "granted" }
```

---

## WebSocket stream

### `GET /v1/ws/stream`

Upgrades to a WebSocket on the default database. Needs a bearer token when
auth is on. Send text frames as JSON:

```json
{ "type": "query", "data": "RECALL memories WHERE tag = \"preferences\" LIMIT 5" }
```

The reply is `{"type": "result", "data": {"context": "...", "total_tokens": 120, "memory_count": 3}}`,
or `{"type": "error", "data": "..."}`. A `{"type": "token", "data": ...}`
message is echoed back as `{"type": "ack", "data": ...}`.
