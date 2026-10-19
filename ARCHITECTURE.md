# MenteDB Architecture

## Overview

MenteDB is a cognition aware database engine for AI agent memory, built from scratch
in Rust. Rather than treating memory as a retrieval problem (store text, search later),
MenteDB pre digests knowledge for single pass transformer consumption. Every write
triggers inference. Every read is shaped by attention budgets. The engine thinks about
what it stores so the LLM doesn't have to.

The system is organized as a Cargo workspace of 14 crates, each owning a distinct
responsibility. The facade crate (`mentedb`) wires them together behind a small,
coherent API: `open`, `store`, `recall`, `relate`, `forget`, `close`, and the
one call turn pipeline `process_turn`.

## Crate Map

```
mentedb (workspace root)
 |
 +-- mentedb-core            Fundamental types shared by every crate
 +-- mentedb-storage          Page manager, WAL, buffer pool, single-writer lock, backup
 +-- mentedb-index            HNSW vectors, BM25 keyword, bitmap tags, temporal, salience
 +-- mentedb-graph            CSR/CSC knowledge graph, traversal, belief propagation
 +-- mentedb-query            MQL lexer, parser, query planner
 +-- mentedb-context          Token budgets, U curve attention, delta serving, serializers
 +-- mentedb-cognitive        Stream cognition, write time inference, phantoms, pain signals
 +-- mentedb-consolidation    Decay, compression, fact extraction, archival, GDPR forget
 +-- mentedb-extraction       LLM-powered memory extraction pipeline
 +-- mentedb-embedding        Embedding providers (HTTP APIs, AWS Bedrock, local Candle, hash)
 +-- mentedb-replication      Raft replication layer (openraft); not used by the server
 +-- mentedb-server           REST + gRPC + WebSocket server wrapping the facade
 +-- mentedb-cli              Command line client: run MQL locally or against a server
 +-- mentedb                  Facade: subsystems, process_turn pipeline, sharding
```

### Dependency Graph

```
mentedb-server ----> mentedb, mentedb-extraction, mentedb-embedding,
                     mentedb-cognitive, mentedb-core
mentedb-cli -------> mentedb
    |
    v
mentedb  (facade)
    |
    +-----> mentedb-core
    +-----> mentedb-storage ---------> mentedb-core
    +-----> mentedb-index -----------> mentedb-core
    +-----> mentedb-graph -----------> mentedb-core
    +-----> mentedb-query -----------> mentedb-core
    +-----> mentedb-context ---------> mentedb-core
    +-----> mentedb-cognitive -------> mentedb-core
    +-----> mentedb-consolidation ---> mentedb-core
    +-----> mentedb-embedding -------> mentedb-core
    +-----> mentedb-extraction ------> mentedb-core, mentedb-cognitive, mentedb-embedding

mentedb-replication ----------> mentedb-core, mentedb-storage
```

The subsystem crates depend on `mentedb-core` and, with two exceptions
(`mentedb-extraction` builds on cognitive and embedding, `mentedb-replication`
on storage), on nothing else in the workspace, so each can be compiled and
tested on its own. The facade crate pulls them all in. `mentedb-replication`
is a standalone crate: neither the facade nor the server depends on it.

## Crate Details

### mentedb-core

The foundation. Every other crate depends on it. Contains zero business logic,
only shared type definitions and error handling.

**Key types:**

| Type | Module | Purpose |
|------|--------|---------|
| `MemoryNode` | `memory.rs` | Atomic unit of knowledge: embedding, content, salience, confidence, tags, attributes |
| `MemoryEdge` | `edge.rs` | Typed weighted relationship between two memories |
| `MemoryTier` | `tier.rs` | Cognitive hierarchy: Working, Episodic, Semantic, Procedural, Archival |
| `MemoryType` | `memory.rs` | Episodic, Semantic, Procedural, AntiPattern, Reasoning, Correction |
| `EdgeType` | `edge.rs` | Caused, Before, Related, Contradicts, Supports, Supersedes, Derived, PartOf |
| `Agent` | `agent.rs` | An AI agent identity with metadata and a default space |
| `AgentRegistry` | `agent.rs` | Registry for managing agent identities |
| `MemorySpace` | `space.rs` | Isolation boundary for memories with access control |
| `SpaceManager` | `space.rs` | CRUD for spaces with permission grants |
| `MenteConfig` | `config.rs` | Unified configuration (storage, index, context, cognitive, consolidation, server) |
| `EventBus` | `event.rs` | Publish/subscribe for system events (memory created, belief changed, contradiction) |
| `VersionStore` | `mvcc.rs` | MVCC version tracking for concurrent writes |
| `ConflictResolver` | `conflict.rs` | Resolves conflicting writes (keep latest, keep highest confidence, merge) |
| `MenteError` | `error.rs` | Unified error enum with `MenteResult<T>` alias |

`AgentRegistry`, `EventBus`, `VersionStore`, and `ConflictResolver` are library
building blocks: the `MenteDb` engine does not use them on its own write path
today (they are exercised by tests and available to embedders). The server uses
`SpaceManager` for its `/v1/spaces` routes.

**Identifiers and scalar aliases** (in `types.rs`). The ids are distinct
newtypes over a UUID, so the compiler rejects passing one kind where another is
expected:

```rust
pub struct MemoryId(pub Uuid);
pub struct AgentId(pub Uuid);
pub struct UserId(pub Uuid);     // end user, an owner axis orthogonal to AgentId
pub struct SpaceId(pub Uuid);
pub type Embedding  = Vec<f32>;
pub type Timestamp  = u64;       // microseconds since epoch
pub type Salience   = f32;       // 0.0 .. 1.0
pub type Confidence = f32;       // 0.0 .. 1.0
```

### mentedb-storage

Owns all durable state. Implements a page oriented storage engine with
write ahead logging and a buffer pool.

**Page Manager** (`page.rs`):
Pages are 64 KB, sized to fit large embedding vectors and long content in a single page. Each page carries a header
with its ID, LSN (log sequence number), checksum, free space counter, slot count,
and page type (Free, Data, Index, Overflow). The page manager maps a directory
to a single data file, maintains a free list for page allocation, and provides
`allocate_page`, `read_page`, `write_page`, `free_page`.

**Write Ahead Log** (`wal.rs`):
Every mutation goes to the WAL before touching data pages. Each entry is
serialized as:

```
[length: u32][lsn: u64][type: u8][page_id: u64][LZ4-compressed data][CRC32: u32]
```

LZ4 compression reduces I/O for large payloads. CRC32 checksums catch torn writes
on recovery. The WAL supports `append`, `sync`, `iterate` (for crash recovery),
and `truncate` (after checkpoint).

Entry types: `PageWrite` (data mutation), `PageFree` (a deleted memory's page
returned to the free list), `Commit` (reserved, not emitted), `Checkpoint`
(safe truncation point; its payload is the token of the memory map persisted
with it).

Durability is the WAL fsync on every write, not the flush. A checkpoint
flushes cached pages, fsyncs the data file, appends a `Checkpoint` entry, and
truncates the WAL to it, so after a checkpoint the WAL holds exactly the
changes since. The engine checkpoints on its own when the WAL passes 8 MB
after a store; `MenteDb::flush` checkpoints on every call and saves the index,
graph, and cognitive snapshots every `flush_snapshot_interval` flushes.

**Memory Map** (`memory_map.rs`):
Which page holds which memory, persisted as `memory_map.bin` at every
checkpoint that changed it (written to a temp file, fsynced, renamed). The
snapshot carries a random token that is also the payload of the checkpoint's
WAL entry. On open the engine finds the checkpoint entry carrying that token
and applies every later entry to the map: a `PageWrite` maps its page to the
memory its data decodes to (or to nothing), a `PageFree` unmaps the page and
marks it free. That is the same rule a full page scan applies, so a reused
page never keeps a stale id. When no entry carries the token (the WAL was
truncated without a map, the map write failed, or the file is damaged or from
an older version), open reads every page once instead and persists a fresh
map. Each mapped page also carries a write stamp derived from the token and
the entry's LSN, identical whether computed live, at checkpoint, or by replay
after a crash; derived caches (mode probes) use it to tell, without loading a
page, that a memory has not changed. After crash recovery the free list is
rebuilt from the map's free pages (each verified to be a free page before the
list is written) instead of from a scan of every page.

**Buffer Pool** (`buffer.rs`):
Sits between the page manager and the rest of the system. Uses CLOCK eviction
(an approximation of LRU that avoids the overhead of maintaining a linked list
on every access). Pages can be pinned to prevent eviction during active use.
The pool is wrapped in a `parking_lot::Mutex` for thread safety. Default capacity is 1024 pages
(64 MB of buffered data at 64 KB per page); frames are allocated on demand up to
that cap.

**Storage Engine** (`engine.rs`):
Wires page manager, WAL, and buffer pool together. Provides `store_memory`
(serializes a `MemoryNode` to JSON, writes to an allocated page via WAL) and
`load_memory` (reads a page, deserializes). Memory layout per page:
`[length: u32 LE][JSON bytes]`.

**Single writer** (`lock.rs`): opening a directory takes an exclusive lock on a
`LOCK` file in it, held until close or process exit. A second open, from
another process or another handle in the same process, fails fast with a
"locked" error instead of silently corrupting state. On Linux the lock is a
`fcntl` open file description lock (`F_OFD_SETLK`), which the NFSv4 lock
manager enforces across hosts, so the rule holds when the directory sits on
NFS or EFS mounted by several machines; a crashed holder frees it once the NFS
lease expires. On macOS and Windows it is a `flock`, which covers one host.
Each write transaction additionally runs under a `flock` on the WAL file.

**Backup** (`backup.rs`): backup and restore to a single file, a JSON manifest
followed by length prefixed file entries.

### mentedb-index

Five index structures, each optimized for a different access pattern, managed
by a unified `IndexManager` that coordinates hybrid queries.

**HNSW Vector Index** (`hnsw.rs`):
A from scratch implementation of Hierarchical Navigable Small World graphs for
approximate nearest neighbor search. Configuration: M=16 (max connections per
layer), ef_construction=200, ef_search=50, cosine distance by default. Supports
insert, search (returns k nearest with distances), remove (tombstone), and
serialize/deserialize for persistence.

Why build HNSW from scratch instead of wrapping a library? Control. MenteDB
serializes the index in its own format, filters search to a candidate set (tag
or time prefilters), and integrates distance calculations with the salience
scoring pipeline. A dependency like `hnswlib` would force an opaque memory model.

**Bitmap Index** (`bitmap.rs`):
Uses roaring bitmaps for set operations on tags. Each tag maps to a bitmap of
integer encoded memory IDs. Supports AND (intersection) and OR (union) queries
across multiple tags. Roaring bitmaps compress sparse integer sets far better than
raw bitsets while still supporting O(1) membership tests.

**Temporal Index** (`temporal.rs`):
A `BTreeMap<Timestamp, Vec<MemoryId>>` that supports range queries ("give me
everything from the last hour") and `latest(n)`. B-trees give O(log n) point
lookups and efficient range iteration.

**Salience Index** (`salience.rs`):
Another B-tree, keyed by salience score. Supports `top_k` (highest salience
first) and `update` (move a memory when its salience decays). This index is
the backbone of the "what matters most" question that context assembly asks
on every read.

**BM25 Keyword Index** (`bm25.rs`):
An inverted index with proper BM25 scoring (IDF, term-frequency saturation, and
document-length normalization). It complements vector search for the exact entity
names, dates, and numbers that embeddings blur together. Indexing is upsert safe,
so re-indexing a memory after an edit replaces its posting rather than double
counting it. It indexes each memory's context-prefixed text (the contextual
retrieval hook), so a memory is findable by situating terms the caller attached
even when the content itself never contains them.

**Index Manager** (`manager.rs`):
`index_memory` fans out a single `MemoryNode` to every index. `hybrid_search`
retrieves candidates from HNSW (vector) and BM25 (keyword) independently and fuses
them with Reciprocal Rank Fusion (k = 60), then applies tag, temporal, and validity
filters. A magnitude aware fusion (`FusionConfig::magnitude_aware`, set through
`MenteDb::set_fusion`) is available as an alternative: it min-max normalizes each
leg's scores and takes a weighted sum, so a much stronger match cannot fall
below a weaker one the way adjacent RRF ranks can. It is off by default. The recall layer above it (in the `mentedb` crate) finishes the ranking:
it blends in decay-adjusted salience, applies optional project scope weighting
(down-weighting memories tagged for a different project than the query's), then
runs two optional, config-gated passes over the fused pool, a pluggable reranker
(relevance re-scoring, e.g. a cross-encoder) and MMR (diversity selection so
near-duplicates do not fill the budget), before truncating to k. Every stage is
off by default and tunable through `CognitiveConfig`, so the base path stays a
simple, predictable RRF fusion.

### mentedb-graph

Stores the knowledge graph in CSR (Compressed Sparse Row) and CSC (Compressed
Sparse Column) format for efficient outgoing and incoming edge traversal.

**CSR Graph** (`csr.rs`):
Nodes are mapped to contiguous u32 indices. Edges are stored in a delta buffer
during writes and merged into the compressed representation on `compact()`.
The dual CSR/CSC layout allows both forward and backward traversal without
scanning. This is the same representation used in high performance graph
analytics libraries.

Why CSR instead of an adjacency list? Memory density. An adjacency list
(`HashMap<NodeId, Vec<Edge>>`) wastes space on per-entry allocations and
pointer chasing. CSR packs edges contiguously, giving much better cache
behavior for traversals.

**Traversal** (`traversal.rs`):
BFS and DFS with depth limits. `bfs_filtered` restricts traversal to specific
edge types (e.g., only follow `Caused` and `Supports` edges). `extract_subgraph`
pulls a neighborhood around a center node. `shortest_path` finds the shortest
path between two memories using BFS.

**Belief Propagation** (`belief.rs`):
When a memory's confidence changes, the update propagates through the graph:

- `Caused` edges: propagate with dampening (0.9x per hop)
- `Supports` edges: propagate a positive boost (0.5x factor)
- `Contradicts` edges: propagate a negative adjustment (0.7x factor)
- `Supersedes` edges: floor the superseded memory's confidence (to 0.1)

Propagation is bounded by `max_depth` (default 5) to prevent runaway cascades.
All factors are configurable via `PropagationConfig`.

**Contradiction Detection** (`contradiction.rs`):
`find_contradictions` follows direct `Contradicts` edges and also discovers
transitive contradictions (A supports B, B contradicts C, therefore A indirectly
contradicts C). `detect_cycles` finds circular reasoning chains.

### mentedb-query

Implements MQL (Mente Query Language), a domain specific query language for
memory operations.

**Lexer** (`lexer.rs`):
Hand written tokenizer that recognizes keywords (`RECALL`, `RELATE`, `FORGET`,
`CONSOLIDATE`, `TRAVERSE`, `WHERE`, `AND`, `OR`, `NEAR`, `LIMIT`, etc.),
literals (strings, integers, floats, UUIDs), and operators (`=`, `!=`, `>`,
`<`, `>=`, `<=`, `~`).

**Parser** (`parser.rs`):
Recursive descent parser producing a typed AST. Statement types: `Recall`
(with filters, NEAR clause, LIMIT, ORDER BY), `Relate` (source, target,
edge type, optional weight), `Forget`, `Consolidate`, `Traverse` (start node,
depth, optional edge filter).

Why a hand written parser instead of a parser combinator library? Diagnostics.
When an MQL query fails to parse, the user needs a clear error message pointing
at the exact token. Combinator libraries like `nom` or `pest` produce generic
errors that are hard to map back to the original query. A hand written parser
gives full control over error reporting.

**Planner** (`planner.rs`):
Translates AST statements into `QueryPlan` variants:

| Statement | Plan |
|-----------|------|
| `RECALL ... NEAR [vector]` | `VectorSearch` |
| `RECALL WHERE tag = ...` | `TagScan` |
| `RECALL WHERE created > ...` | `TemporalScan` |
| `TRAVERSE from ... depth ...` | `GraphTraversal` |
| `FORGET ...` | `Delete` |
| `RELATE ... AS ...` | `EdgeInsert` |
| `CONSOLIDATE ...` | `Consolidate` |

### mentedb-context

Assembles the final context window that gets sent to the LLM. This is where
MenteDB's opinion about attention shows up.

**Token Budget** (`budget.rs`):
`estimate_tokens` approximates token count as `word_count * 1.3` (close enough
for English text without pulling in a tokenizer dependency). `TokenBudget`
tracks consumption. `BudgetAllocation` divides the total budget into zones
with configurable percentages (system, critical, primary, supporting, reference).

**Attention Layout** (`layout.rs`):
Implements the U curve attention pattern. Research shows transformers attend
most strongly to the beginning and end of the context, with a trough in the
middle. MenteDB exploits this:

```
[Opening]     Anti-patterns, corrections      (highest attention)
[Critical]    Score >= 0.8 AND salience >= 0.7
[Primary]     Score >= 0.5
[Supporting]  Score >= 0.2
[Closing]     Low scoring memories             (reinforcement zone)
```

Anti-patterns and corrections go first because they represent past mistakes
the agent must not repeat. Critical memories come next. The middle (lowest
attention) holds supporting context. The closing zone reinforces key points.

**Delta Tracker** (`delta.rs`):
Tracks which memories were served in the previous turn. On the next turn,
`compute_delta` identifies added, removed, and unchanged memories. Only
changes are serialized, saving 40 to 60 percent of tokens in multi-turn
conversations where context evolves slowly.

**Serializers** (`serializer.rs`):
Three output formats:

- `CompactFormat`: Pipe delimited, roughly 3x fewer tokens than prose.
  `M|episodic|0.8|User prefers dark mode|tags:ui,preferences`
- `StructuredFormat`: Markdown with headers and bullets for readability.
- `DeltaFormat`: Shows only what changed since the previous turn.

**Context Assembler** (`assembler.rs`):
Orchestrates layout, budget, and serialization. `assemble` takes scored
memories and produces a `ContextWindow` with blocks, token counts, format
string, and metadata (candidates considered, included, excluded, zones used).

### mentedb-cognitive

The heart of what makes MenteDB different. Seven cognitive features that
run at write time and read time, giving the database a form of awareness
about the knowledge it holds.

**Stream Cognition** (`stream.rs`):
Monitors the LLM's output token stream in real time. As tokens arrive,
they are buffered and compared against known facts. Detects:

- Contradictions (the LLM says something that conflicts with stored memory)
- Forgotten facts (the LLM omits something it should know)
- Corrections (the LLM revises a previous statement)
- Reinforcements (the LLM confirms a stored fact)

Alerts are returned as `StreamAlert` variants so the host application
can decide how to intervene.

**Write Time Inference** (`write_inference.rs`):
When a new memory is stored, the inference engine compares it against
existing memories and automatically:

- Flags contradictions (cosine similarity above 0.95 with conflicting content)
- Marks obsolete memories (high similarity, newer timestamp, correction type)
- Creates `Related` edges (similarity between 0.6 and 0.85)
- Updates confidence (decays superseded memories by 0.5x, floors at 0.1)
- Triggers belief propagation when confidence changes

All thresholds are configurable via `WriteInferenceConfig`.

**Trajectory Tracking** (`trajectory.rs`):
Records the conversation's reasoning trajectory as a sequence of
`TrajectoryNode` entries, each with a topic summary, decision state
(Investigating, NarrowedTo, Decided, Interrupted, Completed), open
questions, and an embedding. `get_resume_context` produces a summary
that lets an agent resume a conversation after interruption.
`predict_next_topics` guesses what the agent will ask about next.

**Phantom Memory Detection** (`phantom.rs`):
Detects references to knowledge the database does not hold: quoted terms,
capitalized multi-word entities, and technical terms (containing hyphens,
dots, or ALL_CAPS patterns). These "phantom memories" are logged with a
priority level so the agent can decide whether to acquire the missing
knowledge.

**Pain Signals** (`pain.rs`):
Records negative experiences (failed actions, user corrections, repeated
mistakes) as `PainSignal` entries with intensity, decay rate, and trigger
keywords. When assembling context, matching pain signals are surfaced as
warnings so the agent avoids repeating past mistakes. Intensity decays
over time so old pain fades naturally.

**Interference Detection** (`interference.rs`):
Identifies pairs of memories that are so similar they might confuse the
LLM (similarity above 0.8). Generates disambiguation strings and can
reorder memories to maximize separation between confusing pairs.

**Speculative Pre-assembly** (`speculative.rs`):
Predicts upcoming queries based on conversation trajectory and pre-builds
context windows. Matches incoming queries against cached predictions by
embedding cosine similarity when both sides have an embedding, and by Jaccard
keyword overlap otherwise. LRU eviction keeps the cache bounded, and the
maintenance sweep evicts stale entries.

### mentedb-consolidation

Background processes that maintain memory health over time.

**Decay Engine** (`decay.rs`):
Applies exponential salience decay:

```
decayed = salience * 2^(-time_since_access / half_life) + boost * ln(1 + access_count)
```

Default half life is 7 days. Frequently accessed memories resist decay.

Decay is derived on read: the stored `salience` is a base as of the last
access and is never rewritten by decay. Everything that reads salience reads
the decayed value at the current time: recall scoring, archival, the health
shown by dashboards, and MQL (`WHERE salience ...` and `ORDER BY salience`
compare the decayed value, computed once per query at a single instant, only
for memories the query already loads). Pinned memories (a decay exempt tag
or attribute) keep their stored value. The stored base is internal.

The stored base is not an upper bound on the decayed value: the access bonus
and the `min_salience` floor both lift it above the base. So `ORDER BY
salience DESC LIMIT k` (`salience_topk.rs`) caches every input of the decayed
value per memory (base, created and accessed times, access count, exemption),
computes each candidate's exact decayed value from the cache, and loads
candidates from the highest down until the next one is below the k-th result.
A write drops the memory's cache entry (`note_write`); entries are persisted
in `salience_bounds.bin`, stamped with page and write stamp like the mode
probes, and restored on open only when unchanged.

**Archival Pipeline** (`archival.rs`):
Evaluates memories for archival or deletion:

- Delete: salience below 0.05, age over 30 days, fewer than 2 accesses
- Archive: salience below 0.1, age over 7 days
- Consolidate: cluster of similar memories ready for merging
- Keep: everything else

**Consolidation Engine** (`consolidation.rs`):
Groups similar episodic memories using union-find clustering on cosine
similarity. Merged clusters produce a single semantic memory with a
combined embedding (averaged) and combined confidence (max). Converts
episodic experience into semantic knowledge.

**Memory Compressor** (`compression.rs`):
Extracts key sentences (those containing signal words like "decided",
"uses", "prefers") and drops filler. Reports a compression ratio.

**Fact Extractor** (`extraction.rs`):
Pattern based extraction of subject/predicate/object triples from memory
content. Recognizes patterns like "X uses Y", "X prefers Y", "X switched
to Y". Extracted facts support contradiction detection and knowledge
graph construction.

**Forget Engine** (`forget.rs`):
GDPR compliant deletion. Given a `ForgetRequest` (by agent, by space,
or by specific memory IDs), plans the complete removal of memories,
their edges, and extracted facts. Generates an audit log entry documenting
what was deleted and why.

### mentedb-server

Binary crate. Axum-based REST API + Tonic gRPC + WebSocket server.
Each database is an `Arc<MenteDb>` shared across requests (no external lock).
REST listens on 6677 and gRPC on 6678 by default.

Key subsystems:
- **Named databases** (`databases.rs`): one `MenteDb` per tenant, each in its own
  directory with its own lock. The default database sits at the data directory
  root and backs the routes under `/v1`; a named one sits in `<data-dir>/dbs/<name>`
  and backs the same routes under `/v1/db/<name>`. Named databases open on first
  use and the least recently used idle ones close past `MENTEDB_MAX_OPEN_DATABASES`
- **REST API** (`routes.rs`, `handlers.rs`, `handlers_memory.rs`, `handlers_ops.rs`),
  per database: `/memories` (store, get, forget, batch), `/recall` (MQL),
  `/search` (vector), `/search/hybrid`, `/context`, `/edges`, `/stats`,
  `/process_turn` (the full turn pipeline), `/ingest` (LLM extraction from a
  conversation), memory history, neighbors and pins, `/conflicts`, `/profile`,
  `/entities`, `/recent`, `/maintenance/*`, `/enrich`, `/documents`, `/backup`
  and `/backups`, plus the admin routes under `/admin`. Global: `/v1/databases`,
  `/v1/spaces`, `/v1/auth/token`, `/v1/health`, `/v1/live` and `/v1/ready`. The
  full list is in the [server API reference](docs/SERVER_API.md)
- **Hosting** (`hosting.rs`): the routes a shared multi tenant deployment needs:
  a paged, token scoped `GET /memories` narrowed by the tag and agent bitmaps,
  `PATCH /memories/{id}` and `POST /memories/{id}/supersede` (both on engine
  primitives that keep ids and write through the WAL), `/recall_for_action`,
  `/consolidation_candidates` and `/consolidate`, `DELETE /v1/db/{db}` for
  maintenance scope tokens, `POST /v1/batch/search` (one embedding, many
  databases), the request deadline middleware (`x-mentedb-deadline-ms`,
  `MENTEDB_REQUEST_TIMEOUT_MS`, `504`), and the enrichment pool: a separate
  small tokio runtime with bounded concurrency, so LLM pipelines never hold the
  threads that serve requests
- **gRPC** (`grpc.rs`): Bidirectional streaming cognition + memory services on
  the default database, with the same token, `dbs` and scope checks as REST.
  `--grpc-port 0` or `MENTEDB_GRPC=off` leaves it unbound
- **WebSocket** (`websocket.rs`): `/v1/ws/stream` cognition stream
- **Auth** (`auth.rs`): JWT middleware. Tokens are verified with a shared secret
  (HS256) or a public key (RS256, ES256, EdDSA; from a file or inline as PEM or
  base64 DER). The `dbs` claim (names or globs) scopes a token to databases and
  the `scope` claim (`read` < `write` < `maintenance`) to what it may do. A
  `TokenPolicy` can require `dbs` (on by default with a public key) and bind
  `dbs` entries to the token's `sub` through a pattern such as
  `tenant.{sub}.*`
- **TLS** (`tls.rs`): with `--tls-cert` and `--tls-key` (or the PEM text in
  `MENTEDB_TLS_CERT_PEM` / `MENTEDB_TLS_KEY_PEM`), REST is served over HTTPS
  and gRPC over TLS with the same certificate
- **Rate limiting** (`rate_limit.rs`): Token bucket per caller, keyed by the
  token's `sub` (else `jti`, else agent) and by client IP when
  unauthenticated. It sits inside the auth middleware so it can see the
  caller. Size and refill come from `MENTEDB_RATE_LIMIT_BURST` and
  `MENTEDB_RATE_LIMIT_PER_SEC`; `0` turns it off
- **Extraction queue** (`extraction_queue.rs`): Bounded mpsc channel (capacity 64)
  with semaphore-limited concurrency (max 4 parallel LLM calls). Drains
  on shutdown (up to 10 seconds).
- **Maintenance** (`maintenance.rs`): a timed sweep (every 24h by default) that
  consolidates, decays, archives, and evicts stale cache entries; also runnable
  once with the `maintenance` subcommand
- **Metrics and console** (`metrics.rs`, `console.rs`): Prometheus `/metrics`
  (aggregate only) and a bundled web console at `/console`
- **Cluster** (`cluster.rs`): optional sharding across several servers, see
  [Sharding](#sharding)

Configurable via CLI flags (`--data-dir`, `--port`, `--grpc-port`, `--jwt-secret`,
`--require-auth`, `--admin-key`, `--llm-*` including `--llm-reader-model`,
`--auto-extract`, `--embedding-provider`, `--maintenance-interval-hours`,
`--tls-cert`, `--tls-key`) and `MENTEDB_*` environment variables, some of which
have no flag (for example `MENTEDB_JWT_PUBLIC_KEY_FILE`); see
[Self-hosting](docs/SELF_HOSTING.md). Shuts down gracefully on Ctrl+C or
SIGTERM: it stops accepting connections, lets requests in flight finish for up
to `MENTEDB_SHUTDOWN_DRAIN_SECS`, then flushes and closes every open database
and exits with status 0.

Memory JSON leaves out the embedding vector unless a read asks for it with
`?fields=embedding`, and carries `user_id`, `valid_from` and `valid_until`.

The engine side of these routes lives in `mentedb::hosting`:
`update_memory_fields` (merge attributes, replace tags and re index the tag
bitmap, set or clear `valid_until`), `supersede_memory` (store, close, link
with a `Supersedes` edge), `list_memories_page` (bitmap narrowed, creation
ordered), `recall_hybrid_scoped_with` and `HybridOptions` (per call MMR
weight, reranker, relevance first selection), and `note_access`, which batches
search hit access statistics in memory and writes them at the next flush
rather than once per hit.

### mentedb (facade)

The entry point for library consumers. Re-exports subsystem crates:

```rust
pub use mentedb_core as core;
pub use mentedb_storage as storage;
pub use mentedb_index as index;
pub use mentedb_graph as graph;
pub use mentedb_query as query;
pub use mentedb_context as context;
pub use mentedb_cognitive as cognitive;
pub use mentedb_consolidation as consolidation;
```

Provides a `prelude` module with the most common types and the `MenteDb`
struct itself.

The core `MenteDb` methods (the struct has many more for recall variants,
maintenance, and the cognitive features):

| Method | Description |
|--------|-------------|
| `open(path)` | Opens or creates a database at the given directory |
| `open_with_embedder(path, embedder)` | Same, with an embedding provider for semantic recall |
| `process_turn(input, delta)` | The full turn pipeline: recall context, store the turn, infer, analyze |
| `store(node)` | Persists a memory, indexes it, registers it in the graph |
| `recall(mql)` | Parses an MQL query, executes it, assembles a context window |
| `recall_similar(embedding, k)` | Vector similarity search returning top-k matches |
| `relate(edge)` | Adds a typed, weighted edge between two memories |
| `forget(id)` | Removes a memory from storage, indexes, and graph |
| `close()` | Checkpoints and closes the database |

## Data Flow

### Write Path

```
Client
  |
  v
MenteDb::store(node)
  |
  +---> StorageEngine::store_memory(node)
  |       |
  |       +---> WAL::append(PageWrite, page_id, serialized_json)
  |       +---> PageManager::allocate_page()
  |       +---> BufferPool::update_page(page_id, page)
  |
  +---> IndexManager::index_memory(node)
  |       |
  |       +---> HnswIndex::insert(id, embedding)
  |       +---> BitmapIndex::add_tag(id, tag)  // for each tag
  |       +---> TemporalIndex::insert(id, created_at)
  |       +---> SalienceIndex::insert(id, salience)
  |
  +---> GraphManager::add_memory(id)
          |
          +---> CsrGraph::add_node(id)
```

At write time, the cognitive subsystem (if wired in) runs
`WriteInferenceEngine::infer_on_write` to detect contradictions,
create edges, and propagate belief changes.

### Read Path

```
Client
  |
  v
MenteDb::recall(mql_string)
  |
  +---> Mql::parse(query)
  |       |
  |       +---> tokenize(query)          // Lexer
  |       +---> Parser::parse(tokens)     // Recursive descent -> AST
  |       +---> plan(statement)           // AST -> QueryPlan
  |
  +---> execute_plan(plan)
  |       |
  |       +---> (match on plan type)
  |       |       VectorSearch  -> IndexManager::hybrid_search
  |       |       TagScan       -> BitmapIndex + hybrid_search
  |       |       TemporalScan  -> TemporalIndex + hybrid_search
  |       |       GraphTraversal -> GraphManager::get_context_subgraph
  |       |       PointLookup   -> StorageEngine::load_memory
  |       |
  |       +---> load_scored_memories(hits)
  |               +---> StorageEngine::load_memory(page_id) for each hit
  |
  +---> ContextAssembler::assemble(scored_memories, edges, config)
          |
          +---> ContextLayout::arrange(memories)     // U curve zones
          +---> TokenBudget::consume(block)           // Budget enforcement
          +---> Serializer::serialize(blocks)         // Compact/Structured/Delta
          +---> ContextWindow { blocks, total_tokens, format, metadata }
```

## Key Design Decisions

### Why Rust

Memory safety without garbage collection. AI agent memory databases run as
long lived services processing concurrent requests. Rust's ownership model
eliminates use after free, data races, and null pointer bugs at compile time.
The zero cost abstractions mean hand written HNSW and CSR implementations
run as fast as C equivalents. The type system catches misuse of MemoryId vs
AgentId vs SpaceId at compile time.

### Why Custom Storage vs SQLite

SQLite is an excellent general purpose embedded database, but it imposes
decisions that conflict with MenteDB's requirements:

1. **Page format control**: MenteDB needs to co-locate memory embeddings
   with metadata on the same page for cache friendly access. SQLite's
   B-tree pages don't support this layout.
2. **WAL integration**: MenteDB's WAL carries LZ4 compressed payloads
   and integrates with the cognitive checkpoint system. SQLite's WAL is
   a black box.
3. **Buffer pool policy**: CLOCK eviction is chosen specifically because
   MenteDB's access pattern (repeated scans during hybrid search) would
   thrash an LRU pool. CLOCK gives a second chance to recently accessed
   pages.

### Why HNSW from Scratch

HNSW is the standard algorithm for approximate nearest neighbor search in
high dimensional spaces. Building it from scratch gives MenteDB:

1. **Serialization control**: The index serializes to its own compact
   binary snapshot (bincode), written next to the data files and reconciled
   with the WAL protected memory pages on open.
2. **Filtered search**: Search can be restricted to a precomputed candidate
   set (from tag or time filters) instead of searching globally and filtering
   afterwards.
3. **Custom distance functions**: Cosine, Euclidean, and dot product are
   built in, but the architecture allows plugging in domain specific
   metrics.

### Why CSR for Graphs

Compressed Sparse Row (with CSC for reverse lookups) is the gold standard
for static graph analytics. MenteDB's knowledge graph is append heavy with
periodic compaction, which matches the CSR model well:

1. **Cache locality**: Edges for a node are stored contiguously, so
   traversal hits sequential memory addresses.
2. **Compact representation**: No per-edge heap allocations. The entire
   graph fits in a few arrays.
3. **Delta buffer**: New edges accumulate in an uncompressed buffer and
   merge into CSR on `compact()`, amortizing the cost of insertion.

### Why Hand Written Parser

Parser combinator libraries like `nom` and `pest` are powerful but produce
opaque error messages. MQL is simple enough (five statement types, a handful
of operators) that a recursive descent parser is straightforward to write
and maintain. The hand written approach gives:

1. **Precise diagnostics**: Error messages point at the exact token and
   position where parsing failed.
2. **Zero dependencies**: No procedural macros, no generated code, no
   build time cost.
3. **Easy extension**: Adding a new statement type means adding one parse
   function and one AST variant.

### Configurable Heuristics

Most thresholds in MenteDB are configurable: contradiction similarity (0.95),
related edge range (0.6 to 0.85), salience decay half life (7 days), budget
zone percentages, HNSW parameters, belief propagation factors. The defaults
are reasonable starting points, but AI agent behavior varies widely. A coding
assistant has different memory patterns than a customer support bot. Making
everything configurable lets operators tune without forking the engine.

## Cognitive Features: The Seven Radical Capabilities

### 1. Stream Cognition

Traditional RAG is fire and forget: retrieve context, generate, done.
Stream cognition monitors the LLM's output in real time, comparing each
token against stored knowledge. If the model starts contradicting known
facts, the system raises an alert before the response completes. This
enables mid-generation intervention, something no retrieval only system
can do.

### 2. Write Time Inference

Most databases are passive stores. MenteDB is active. When a new memory
arrives, the inference engine examines it against the existing knowledge
base and takes action: flagging contradictions, creating relationship
edges, marking obsolete information, propagating confidence changes. The
database maintains its own consistency without requiring the agent to
manage relationships explicitly.

### 3. Trajectory Tracking

Conversations have narrative structure: topics evolve, decisions are made,
questions are raised and answered. Trajectory tracking records this structure
as a sequence of decision states. When a conversation is interrupted and
resumed, `get_resume_context` produces a summary of where things left off.
`predict_next_topics` anticipates what the agent will need, enabling
speculative pre-assembly of context.

### 4. Phantom Memory Detection

When the LLM references an entity or concept that does not exist in the
database, that gap is a "phantom memory." Detecting these gaps lets the
agent decide whether to acquire the missing knowledge (ask the user, call
an API, search the web) rather than hallucinating an answer. Phantoms are
prioritized: a missing critical entity is more urgent than a missing
tangential reference.

### 5. Pain Signals

Learning from mistakes requires remembering that they happened. Pain
signals record negative experiences with intensity and trigger keywords.
When context assembly encounters matching keywords, it surfaces warnings:
"Last time you used this API endpoint, it returned stale data." Pain
decays over time, so ancient mistakes don't clutter every context window
forever.

### 6. Interference Detection

When two memories are very similar (but not identical), they can confuse
the LLM. Interference detection identifies these confusing pairs and
either generates disambiguation text or reorders the context to maximize
separation. This is inspired by the psychological concept of proactive
interference, where old memories interfere with new learning.

### 7. Speculative Pre-assembly

Based on trajectory predictions, the speculative cache pre-builds context
windows for anticipated queries. If the prediction hits (Jaccard keyword
overlap above the threshold), the pre-assembled context is served
immediately, skipping the search/score/assemble pipeline entirely. Cache
misses fall back to the normal path.

## Configuration

`MenteConfig` (in `mentedb-core`) describes the subsystem settings and their
defaults. Note that the engine does not read it today: `MenteDb` is tuned through
`mentedb::CognitiveConfig` (write inference, decay, recall passes such as MMR and
the reranker pool, injection and dedup thresholds), the storage engine uses its
built in defaults, and the server takes flags and environment variables.

```rust
MenteConfig {
    storage: StorageConfig {
        data_dir: PathBuf,
        buffer_pool_size: 1024,       // pages
        page_size: 65536,             // bytes
    },
    index: IndexConfig {
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 50,
    },
    context: ContextConfig {
        default_token_budget: 4096,
        token_multiplier: 1.3,
        // zone percentages...
    },
    cognitive: CognitiveConfig {
        contradiction_threshold: 0.95,
        // related thresholds, cache size, etc.
    },
    consolidation: ConsolidationConfig {
        decay_half_life_hours: 168,   // 7 days
        min_salience: 0.01,
        // archival settings...
    },
    server: ServerConfig {
        host: "0.0.0.0",
        port: 6677,
    },
}
```

Configuration can be loaded from a file via `MenteConfig::from_file(path)`.

## Error Handling

All fallible operations return `MenteResult<T>`, which is
`Result<T, MenteError>`. The error enum covers:

| Variant | When |
|---------|------|
| `MemoryNotFound(MemoryId)` | Lookup or delete of a nonexistent memory |
| `Storage(String)` | Page I/O, WAL corruption, serialization failure |
| `Index(String)` | HNSW dimension mismatch, bitmap overflow |
| `Query(String)` | MQL parse error with position information |
| `Serialization(String)` | Encode/decode failure (JSON or bincode) |
| `CapacityExceeded(String)` | Buffer pool full, space limit reached |
| `EmbeddingDimensionMismatch { .. }` | An embedding's length does not match the index |
| `PermissionDenied { agent_id, space_id }` | Access control violation |
| `ResourceExhausted(String)` | A configured resource limit was hit |
| `Io(std::io::Error)` | Underlying filesystem error |

## Thread Safety & Concurrency

MenteDB uses **interior mutability** throughout — each component manages its own
fine-grained locking. The server holds one `Arc<MenteDb>` per open database (no external lock).

| Component | Lock Type | Scope |
|-----------|-----------|-------|
| `StorageEngine` | `parking_lot::Mutex` on PageManager + WAL | Per-operation |
| `BufferPool` | `parking_lot::Mutex` on inner state | Page fetch/evict |
| `GraphManager` | `parking_lot::RwLock` on CsrGraph | Read: traversal/search, Write: add/remove |
| `IndexManager` | `parking_lot::RwLock` per index (5 indexes) | Read: search, Write: insert/remove |
| `page_map` | `parking_lot::RwLock` on HashMap | Read: lookups, Write: store/forget |
| `EventBus` | `RwLock` for subscriber list | Publish/subscribe |
| `CognitionStream` | `Mutex` on ring buffer | Token feed/drain |

Passes that visit many memories (MQL scans, archival, consolidation, recall's
candidate loads, write inference) copy page ids out of `page_map` and release
it before loading pages. A reader holding it across page loads would stall
every store waiting for the write lock, and with the lock's writer preference
every later reader too.

**Lock ordering** (prevents deadlocks):
1. `page_map` → `StorageEngine` (never reverse)
2. WAL lock → PageManager lock (released before buffer pool)
3. Graph lock is independent (never held with page_map or storage)

**Concurrency benefits:**
- All read operations (recall, search, get_memory, stats) run concurrently with zero contention
- Write operations only lock the specific component being mutated
- LLM extraction runs in a background `tokio::spawn` task — no lock held during API calls

## Sharding

One database directory has exactly one writer, so the way to scale out is to
give each account (agent) its own directory and spread accounts over nodes.
`mentedb::sharding` owns that logic and leaves storage of the shared state to
the embedder:

- **Placement** (`sharding/placement.rs`): rendezvous (highest random weight)
  hashing with a fixed FNV-1a hash, so every node computes the same owner for a
  key, and adding or removing a node moves only about 1/N of the keys.
- **Leases** (`Coordinator`, `LeaseStore`, `NodeRegistry`): ownership is an
  expiring lease with an epoch that fences writes, so a node that lost
  ownership is rejected even if it has not noticed. `resolve(key)` answers
  "serve locally" or "forward to this address". The lease and membership
  backends are traits the embedder implements (for example over a database with
  conditional writes), so the engine takes no external dependency.
- **Gossip** (`sharding/gossip.rs`): membership without a coordinator. Each
  round a node exchanges its view with a few peers and keeps the newest
  announcement per node; a node that stops announcing is dropped after a
  timeout.

`mentedb-server` uses the gossip and placement parts: with `MENTEDB_SHARDING=1`
plus `MENTEDB_NODE_ADDR` and `MENTEDB_SEEDS`, nodes find each other on
`/v1/cluster/gossip` and forward each request to the agent's owner (the agent
id comes from the token's `agent_id` claim). This is separate from `mentedb-replication`, which
replicates one dataset with Raft and is not wired into the server.

## Index Persistence

Indexes are persisted as **bincode** binary format for fast serialization (~5x smaller,
~10x faster than JSON). On load, indexes auto-detect and migrate from legacy JSON format.
MemoryNode pages and the knowledge graph use JSON due to `skip_serializing_if` compatibility.

## Testing Strategy

Each crate has its own unit tests (`#[cfg(test)]` modules). Most crates also
have integration tests in their own `tests/` directory. The workspace is
tested with `cargo test --workspace`.

Key test categories:

- **Storage**: Page allocation round trips, WAL crash recovery, buffer
  pool eviction under pressure
- **Index**: HNSW recall accuracy, bitmap set operations, temporal range
  correctness
- **Graph**: CSR compaction correctness, traversal depth limits, belief
  propagation convergence
- **Query**: MQL parsing of every statement type, planner output
  verification
- **Context**: Token budget enforcement, zone classification, delta
  correctness
- **Cognitive**: Contradiction detection precision, phantom gap coverage,
  trajectory resume accuracy
- **Consolidation**: Decay formula correctness, archival decision rules,
  fact extraction patterns
