# MenteDB Architecture

## Overview

MenteDB is a cognition aware database engine for AI agent memory, built from scratch
in Rust. Rather than treating memory as a retrieval problem (store text, search later),
MenteDB pre digests knowledge for single pass transformer consumption. Every write
triggers inference. Every read is shaped by attention budgets. The engine thinks about
what it stores so the LLM doesn't have to.

The system is organized as a Cargo workspace of 13 crates, each owning a distinct
responsibility. The facade crate (`mentedb`) wires them together behind a small,
coherent API: `open`, `store`, `recall`, `relate`, `forget`, `close`.

## Crate Map

```
mentedb (workspace root)
 |
 +-- mentedb-core            Fundamental types shared by every crate
 +-- mentedb-storage          Page manager, WAL, buffer pool
 +-- mentedb-index            HNSW vectors, bitmap tags, temporal, salience
 +-- mentedb-graph            CSR/CSC knowledge graph, traversal, belief propagation
 +-- mentedb-query            MQL lexer, parser, query planner
 +-- mentedb-context          Token budgets, U curve attention, delta serving, serializers
 +-- mentedb-cognitive        Stream cognition, write time inference, phantoms, pain signals
 +-- mentedb-consolidation    Decay, compression, fact extraction, archival, GDPR forget
 +-- mentedb-extraction       LLM-powered entity/relation extraction pipeline
 +-- mentedb-embedding        Embedding provider abstraction (OpenAI, local)
 +-- mentedb-replication      Multi-node replication protocol
 +-- mentedb-server           HTTP/TCP server wrapping the facade
 +-- mentedb                  Unified facade re-exporting the subsystems
```

### Dependency Graph

```
mentedb-server
    |
    v
mentedb  (facade)
    |
    +-----> mentedb-core
    +-----> mentedb-storage -----> mentedb-core
    +-----> mentedb-index -------> mentedb-core
    +-----> mentedb-graph -------> mentedb-core
    +-----> mentedb-query -------> mentedb-core
    +-----> mentedb-context -----> mentedb-core
    
mentedb-cognitive ------------> mentedb-core
mentedb-consolidation --------> mentedb-core
mentedb-extraction -----------> mentedb-core
mentedb-embedding ------------> mentedb-core
mentedb-replication ----------> mentedb-core
```

All subsystem crates depend only on `mentedb-core`. The facade crate pulls in
storage, index, graph, query, and context. The cognitive and consolidation crates
are deliberately decoupled, they share types through core but can be compiled and
tested independently.

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

**Scalar aliases** (in `types.rs`):

```rust
pub type MemoryId   = Uuid;
pub type AgentId    = Uuid;
pub type SpaceId    = Uuid;
pub type Embedding  = Vec<f32>;
pub type Timestamp  = u64;       // microseconds since epoch
pub type Salience   = f32;       // 0.0 .. 1.0
pub type Confidence = f32;       // 0.0 .. 1.0
```

### mentedb-storage

Owns all durable state. Implements a page oriented storage engine with
write ahead logging and a buffer pool.

**Page Manager** (`page.rs`):
Pages are 16 KB, the same size PostgreSQL uses. Each page carries a header
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

Entry types: `PageWrite` (data mutation), `Commit` (transaction boundary),
`Checkpoint` (safe truncation point).

**Buffer Pool** (`buffer.rs`):
Sits between the page manager and the rest of the system. Uses CLOCK eviction
(an approximation of LRU that avoids the overhead of maintaining a linked list
on every access). Pages can be pinned to prevent eviction during active use.
The pool is wrapped in a `parking_lot::Mutex` for thread safety. Default capacity is 1024 pages
(16 MB of buffered data).

**Storage Engine** (`engine.rs`):
Wires page manager, WAL, and buffer pool together. Provides `store_memory`
(serializes a `MemoryNode` to JSON, writes to an allocated page via WAL) and
`load_memory` (reads a page, deserializes). Memory layout per page:
`[length: u32 LE][JSON bytes]`.

### mentedb-index

Four index structures, each optimized for a different access pattern, managed
by a unified `IndexManager` that coordinates hybrid queries.

**HNSW Vector Index** (`hnsw.rs`):
A from scratch implementation of Hierarchical Navigable Small World graphs for
approximate nearest neighbor search. Configuration: M=16 (max connections per
layer), ef_construction=200, ef_search=50, cosine distance by default. Supports
insert, search (returns k nearest with distances), remove (tombstone), and
serialize/deserialize for persistence.

Why build HNSW from scratch instead of wrapping a library? Control. MenteDB needs
to serialize the index into its own page format, evict graph layers under memory
pressure, and integrate distance calculations with the salience scoring pipeline.
A dependency like `hnswlib` would force an opaque memory model.

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

**Index Manager** (`manager.rs`):
`index_memory` fans out a single `MemoryNode` to all four indexes. `hybrid_search`
combines results with a weighted score:

```
final_score = vector_similarity * 0.6 + salience * 0.3 + recency * 0.1
```

This formula is intentionally simple and configurable. Vector similarity dominates
because embeddings capture semantic intent. Salience rewards memories the agent
has flagged as important. Recency provides a mild preference for fresh information.

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
context windows. Uses Jaccard keyword overlap to match incoming queries
against cached predictions. LRU eviction keeps the cache bounded.

### mentedb-consolidation

Background processes that maintain memory health over time.

**Decay Engine** (`decay.rs`):
Applies exponential salience decay:

```
decayed = salience * 2^(-time_since_access / half_life) + boost * ln(1 + access_count)
```

Default half life is 7 days. Frequently accessed memories resist decay.

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
Wraps `MenteDb` in `Arc<MenteDb>` for concurrent access (no external lock).

Key subsystems:
- **REST API** (`handlers.rs`): Memory CRUD, MQL recall, vector search, ingest
- **gRPC** (`grpc.rs`): Bidirectional streaming cognition + memory services
- **WebSocket** (`websocket.rs`): Real-time memory event streaming
- **Auth** (`auth.rs`): JWT token creation/validation middleware
- **Rate limiting** (`rate_limit.rs`): Token bucket rate limiter
- **Extraction queue** (`extraction_queue.rs`): Bounded mpsc channel (capacity 64)
  with semaphore-limited concurrency (max 4 parallel LLM calls). Drains
  gracefully on shutdown.

Configurable via CLI args (`--data-dir`, `--port`, `--grpc-port`, `--jwt-secret`).
Handles graceful shutdown on SIGTERM/SIGINT.

### mentedb (facade)

The entry point for library consumers. Re-exports subsystem crates:

```rust
pub use mentedb_core as core;
pub use mentedb_storage as storage;
pub use mentedb_index as index;
pub use mentedb_graph as graph;
pub use mentedb_query as query;
pub use mentedb_context as context;
```

Provides a `prelude` module with the most common types and the `MenteDb`
struct itself.

`MenteDb` exposes six public methods:

| Method | Description |
|--------|-------------|
| `open(path)` | Opens or creates a database at the given directory |
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

1. **Serialization control**: The index serializes into MenteDB's own page
   format, so it participates in WAL, buffer pool, and checkpoint.
2. **Memory pressure awareness**: Individual graph layers can be evicted
   under memory pressure instead of keeping the entire index in RAM.
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

Every threshold in MenteDB is configurable: contradiction similarity (0.95),
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
misses fall back to the normal path with no penalty.

## Configuration

`MenteConfig` unifies all subsystem configurations:

```rust
MenteConfig {
    storage: StorageConfig {
        data_dir: PathBuf,
        buffer_pool_size: 1024,       // pages
        page_size: 16384,             // bytes
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
| `PermissionDenied { agent_id, space_id }` | Access control violation |
| `Io(std::io::Error)` | Underlying filesystem error |

## Thread Safety & Concurrency

MenteDB uses **interior mutability** throughout — each component manages its own
fine-grained locking. The server holds `Arc<MenteDb>` (no external lock).

| Component | Lock Type | Scope |
|-----------|-----------|-------|
| `StorageEngine` | `parking_lot::Mutex` on PageManager + WAL | Per-operation |
| `BufferPool` | `parking_lot::Mutex` on inner state | Page fetch/evict |
| `GraphManager` | `parking_lot::RwLock` on CsrGraph | Read: traversal/search, Write: add/remove |
| `IndexManager` | `parking_lot::RwLock` per index (5 indexes) | Read: search, Write: insert/remove |
| `page_map` | `parking_lot::RwLock` on HashMap | Read: lookups, Write: store/forget |
| `EventBus` | `RwLock` for subscriber list | Publish/subscribe |
| `CognitionStream` | `Mutex` on ring buffer | Token feed/drain |

**Lock ordering** (prevents deadlocks):
1. `page_map` → `StorageEngine` (never reverse)
2. WAL lock → PageManager lock (released before buffer pool)
3. Graph lock is independent (never held with page_map or storage)

**Concurrency benefits:**
- All read operations (recall, search, get_memory, stats) run concurrently with zero contention
- Write operations only lock the specific component being mutated
- LLM extraction runs in a background `tokio::spawn` task — no lock held during API calls

## Index Persistence

Indexes are persisted as **bincode** binary format for fast serialization (~5x smaller,
~10x faster than JSON). On load, indexes auto-detect and migrate from legacy JSON format.
MemoryNode pages and the knowledge graph use JSON due to `skip_serializing_if` compatibility.

## Testing Strategy

Each crate has its own unit tests (`#[cfg(test)]` modules). Integration
tests live in `tests/` directories within each crate. The workspace is
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
