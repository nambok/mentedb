# Provenance

Every memory records where it came from at the moment it is first written,
and that record never changes afterwards. With it you can see where a fact
came from, and forget everything a session wrote in one call.

## What is recorded

| Field | Meaning |
|---|---|
| `session_id` | The session the memory was written in, when the writer named one |
| `source` | The kind of source: `chat`, `file`, `api`, `mcp`, `import`, `derived`, `unspecified`, or any other name |
| `source_ref` | A reference into the source: a file path, a URL, a message id |
| `written_by` | The agent that wrote it |
| `written_at` | When it was first written, microseconds since the Unix epoch. Separate from `created_at`, which a caller may set to another time (an import keeps the original one) |
| `derived_from` | For a derived memory, the ids of the memories it was made from |

Writers that set it for you:

| Path | `source` | `session_id` | `source_ref` |
|---|---|---|---|
| `process_turn` (the turn's episodic memory) | `chat` | the turn's `session_id` | |
| REST `POST /memories`, `POST /memories/batch`, gRPC `Store`, SDK `store` | `api` unless the request names one | from the request | from the request |
| REST `POST /ingest` (memories extracted from a conversation) | `chat` unless the request names one | from the request | from the request |
| REST `POST /documents` (each chunk) | `file` | from the request | the document's `source`, unless the request gives one |
| Agent file ingestion | `file` | | the ingest's `source_tag` |
| Facts and entity nodes from enrichment, consolidation (rule based and LLM), community summaries, server side auto extraction | `derived` | the origins' session, when they all share one | |
| The user profile | `derived` | none (it summarizes everything) | |
| A store that sets nothing | `unspecified` | | |

## It never changes

The engine stamps provenance on the first write of an id. Every later write
of the same id keeps the stored one, whatever the new node carries:

- a re-store through `store` or `store_batch` (with new content, a different
  provenance, or none),
- every in place update: `update_memory_fields` (REST `PATCH`), validity
  changes, decay, access counts, confidence updates, supersession,
  consolidation, profile and community rebuilds.

All of these go through two places, the store path and the private
`update_node`, and both load the stored record and write its provenance
back. Tests in `crates/mentedb/tests/provenance.rs` cover each path and a
reopen.

## Derived memories

A memory the engine makes from others gets `source = derived` and
`derived_from` set to its origins. It inherits a session only when every
origin has the same one: a consolidation of two memories from session `s1`
is in `s1`; one that merges a memory from `s1` with one from `s2` is in no
session. A source reference the writer gave a derived memory is kept.

`derived_from` is a record of where the memory came from; the `Derived`
graph edges remain what the erase cascade follows.

## Finding memories by provenance

Session and source are indexed as reserved keys of the bitmap index, the
same way memory type and owner are, so every lookup below costs
O(matching memories), never a scan.

Rust:

```rust,no_run
use mentedb::{HybridOptions, MenteDb, ProvenanceFilter, SourceKind};
# fn run(db: &MenteDb, query: &[f32]) -> mentedb::prelude::MenteResult<()> {
let ids = db.memory_ids_in_session("conv-7");
let page = db.list_memories_page_by(
    None, None, &ProvenanceFilter::session("conv-7"), false, 0, 0, 50,
);
let filter = ProvenanceFilter { session_id: None, source: Some(SourceKind::File) };
let hits = db.recall_hybrid_scoped_with(
    query, None, 10, 0, None, false, None, None, None, None,
    &HybridOptions { provenance: Some(&filter), ..HybridOptions::default() },
)?;
# Ok(()) }
```

The filter is resolved to an id set before the search runs, so a narrow
filter searches only its own memories instead of being cut out of a global
top k.

MQL:

```sql
RECALL memories WHERE session = "conv-7" LIMIT 50
RECALL memories WHERE session = "conv-7" AND source = file
RECALL memories WHERE source IN ["chat", "mcp"] LIMIT 20
```

`session` and `source` take `=`, `!=` or `IN`; values are names, quoted or
bare, compared exactly as written.

REST:

```bash
# Write with provenance (a top level session_id works as a shorthand).
curl -X POST localhost:6677/v1/memories -H "Authorization: Bearer $TOKEN" \
  -d '{"agent_id": "...", "memory_type": "semantic", "content": "...",
       "provenance": {"session_id": "conv-7", "source": "chat", "source_ref": "msg-42"}}'

# Every read returns it.
curl localhost:6677/v1/memories/$ID -H "Authorization: Bearer $TOKEN"
# {"id": "...", ..., "provenance": {"session_id": "conv-7", "source": "chat",
#   "source_ref": "msg-42", "written_by": "...", "written_at": 1793845123000000,
#   "derived_from": []}}

# Filter a listing or a search.
curl "localhost:6677/v1/memories?session_id=conv-7&source=chat" -H "Authorization: Bearer $TOKEN"
curl -X POST localhost:6677/v1/search/hybrid -H "Authorization: Bearer $TOKEN" \
  -d '{"query": "trip plans", "session_id": "conv-7"}'
```

The server sets `written_by` and `written_at`; a request cannot. A memory
written before provenance existed returns `"provenance": null`.

gRPC: `StoreRequest` carries `session_id`, `source` and `source_ref`;
`SearchRequest` takes `session_id` and `source` filters and each
`SearchResult` carries the memory's `Provenance`; `GetProvenance` returns
one memory's.

Python and TypeScript SDKs: `store(..., session_id=, source=, source_ref=)`
(`sessionId`, `source`, `sourceRef` in TypeScript), `get_memory` returns
`provenance`, `search_text(..., session_id=, source=)` (`search(embedding,
k, { sessionId, source })`), and `memory_ids_in_session`.

## Forget a session

`erase_session` erases every memory written in a session and everything
derived from them, through the verified erase described in
[FORGETTING.md](FORGETTING.md), as one erase: one cascade over all of them,
one scrub of the record file, WAL and snapshots, one verification scan, and
one receipt.

```rust,no_run
use mentedb::{EraseOptions, MenteDb};
# fn run(db: &MenteDb) -> mentedb::prelude::MenteResult<()> {
let receipt = db.erase_session("conv-7", EraseOptions::default())?;
assert!(!receipt.verification.unwrap().residual_found);
# Ok(()) }
```

```bash
curl -X POST localhost:6677/v1/sessions/conv-7/erase -H "Authorization: Bearer $TOKEN"
```

```python
receipt = db.erase_session("conv-7")
```

```typescript
const receipt = db.eraseSession('conv-7');
```

The receipt is the single memory receipt with `session_id` and the list of
the session's memories in `requested`:

```json
{
  "session_id": "conv-7",
  "requested": ["6f1c...", "a3d0..."],
  "erased": [
    {"id": "6f1c...", "reason": "requested", "content_sha256": "9b2e...", "embedding_sha256": "40aa..."},
    {"id": "c2e9...", "reason": "derived", "content_sha256": "c71f...", "embedding_sha256": "e5b2..."}
  ],
  "ancestors_kept": [],
  "profiles_to_review": [],
  "erased_at_us": 1793845123000000,
  "steps": {"records_zeroed": 3, "wal_scrubbed": true, "snapshots_rewritten": true, "...": "..."},
  "verification": {"files_scanned": 18, "residual_found": false, "hits": [], "...": "..."}
}
```

- A memory derived from the session's memories is erased even when it
  carries no session or another one (a consolidation across sessions), with
  reason `derived`.
- `max_cascade` applies to the whole session; when it would be exceeded,
  nothing is erased.
- A session with no stored memories returns an empty receipt and changes
  nothing (`verification` is null).
- Over REST and gRPC, a token bound to an agent may erase a session only
  when every memory in it is its own; otherwise the call is refused and
  nothing is erased.
- Memories written before provenance existed carry no session. A
  `session:<id>` tag that `process_turn` wrote on older turns is only a tag:
  those memories are not matched. Erase them by id.

## Storage compatibility

A record stores the memory as raw embedding bytes followed by JSON of the
other fields (`crates/mentedb-storage/src/record.rs`). The JSON is
self describing, so provenance is one more optional key:

- Records written before provenance have no `provenance` key and read back
  with none. Opening such a database rewrites nothing; the test
  `a_database_from_the_previous_release_opens_and_reads_unchanged` opens a
  database written by the 0.47.1 release (`crates/mentedb/tests/fixtures/`)
  and checks the record files are byte for byte unchanged after open.
- A memory with no provenance is written without the key, so its bytes are
  what the previous release wrote.
- A source kind this build does not know (written by a later release)
  decodes as that name and is written back unchanged.
- A previous release reading a record with provenance ignores the key, but
  rewrites the record without it if it updates that memory. Do not move a
  database back to an older release if its provenance matters.
