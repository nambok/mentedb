# MenteDB: Claude Code Instructions

## What is this?

MenteDB is a cognition aware database engine for AI agent memory, written in Rust. It is not a wrapper around an existing database. It is a purpose built engine with custom storage, indexing, graph, query, and cognitive layers.

## Project structure

```
crates/
  mentedb-core/       Core types, config, error, MVCC, multi agent
  mentedb-storage/    Page manager, WAL, buffer pool, backup/restore
  mentedb-index/      HNSW vector index, bitmap, temporal, salience
  mentedb-graph/      CSR/CSC graph, traversal, belief propagation
  mentedb-query/      MQL lexer, parser, planner
  mentedb-context/    U curve attention layout, delta tracker, serializers
  mentedb-cognitive/  Stream cognition, write inference, trajectory, phantoms, interference, pain, speculative
  mentedb-consolidation/ Decay, archival, extraction, compression, GDPR forget
  mentedb-embedding/  Provider trait, hash/HTTP providers, LRU cache
  mentedb-server/     Axum REST API, JWT auth, rate limiting, WebSocket
  mentedb/            Unified facade (MenteDb struct)
sdks/
  python/             PyO3 bindings + pure Python client
  typescript/         napi-rs bindings + TypeScript client
  python/integrations/langchain/  LangChain memory, retriever, chat history
  python/integrations/crewai/     CrewAI memory and tool adapter
```

## Build and test commands

```bash
# Check, lint, format (always run before committing)
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace

# Run the server
cargo run --bin mentedb-server -- --port 8080

# Build Python SDK (requires maturin)
cd sdks/python && maturin develop && pytest tests/

# Build TypeScript SDK (requires napi-rs CLI)
cd sdks/typescript && npm run build
```

## Coding conventions

- Rust edition 2024
- All thresholds and heuristics must be configurable via Config structs, never hardcoded
- Trait based abstractions (e.g. EmbeddingProvider trait)
- No unwrap() in library code, use MenteResult<T> everywhere
- No emojis in code, comments, docs, or commit messages
- No dashes (em dash or en dash) in prose. Use commas instead. Dashes in CLI flags and code are fine.
- Commit style: conventional (feat:, fix:, chore:), single line, no emojis
- NEVER include Co-authored-by or Authored-by trailers in commits

## Key types

- `MemoryNode`: The fundamental storage unit (id, content, memory_type, embedding, metadata, timestamps)
- `MemoryEdge`: Typed relationship between memories (caused, contradicts, relates_to, obsoletes, etc.)
- `MenteDb`: The unified facade that coordinates all subsystems
- `MenteConfig`: Top level config with sub configs for every subsystem
- `MenteError` / `MenteResult<T>`: Error handling throughout

## Architecture notes

- Storage uses 16KB pages with CLOCK eviction buffer pool and WAL with LZ4 + CRC32
- HNSW index is built from scratch (not a binding), configurable M and ef parameters
- Graph uses CSR/CSC hybrid with delta log and compact() to merge
- MQL is a custom query language parsed by hand written recursive descent parser
- Context assembly uses U curve attention layout (primacy/recency bias) with delta aware serving
- 7 cognitive features are unique to MenteDB and do not exist in any other database engine

## What NOT to do

- Do not add external database dependencies (SQLite, Postgres, etc.), this IS the database
- Do not remove or weaken any cognitive features
- Do not hardcode language specific heuristics (stopwords, patterns), use config
- Do not modify sdks/python/ or sdks/typescript/ Cargo.toml to join the workspace (they build independently)
