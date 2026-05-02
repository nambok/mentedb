# MenteDB Development Instructions

## Project overview

MenteDB is a purpose built Rust database engine for AI agent memory. It includes custom storage (WAL, buffer pool, pages), HNSW vector indexing, CSR/CSC graph, a custom query language (MQL), context assembly with U curve attention layout, and 7 unique cognitive features (stream cognition, write time inference, trajectory tracking, phantom memories, interference detection, pain signals, speculative pre assembly).

## Workspace structure

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

The SDKs are excluded from the Cargo workspace and build independently.

## Build, test, and lint

Always run these before committing:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

The server binary is `mentedb-server` and runs on axum with JWT auth, rate limiting, and WebSocket support.

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

## Conventions

- Rust edition 2024, Apache 2.0 license
- All heuristics and thresholds must be in Config structs, never hardcoded magic numbers
- Use `MenteResult<T>` for error handling, no `unwrap()` in library code
- Commit messages: conventional style (feat:, fix:, chore:), single line, no emojis
- No emojis or em/en dashes in prose anywhere (docs, README, comments). Use commas instead.
- Trait based design: `EmbeddingProvider`, storage backends, etc.

## Key files

- `crates/mentedb/src/lib.rs`: Unified `MenteDb` facade (open, store, recall, relate, forget, close)
- `crates/mentedb-core/src/config.rs`: All configuration (MenteConfig and sub configs)
- `crates/mentedb-core/src/error.rs`: MenteError enum
- `crates/mentedb-cognitive/src/`: 7 cognitive feature modules
- `crates/mentedb-server/src/main.rs`: Axum server entry point
- `.github/workflows/ci.yml`: CI pipeline (check, test, fmt, clippy)
- `.github/workflows/release.yml`: Release pipeline (crates.io, PyPI, npm)

## Do not

- Add external database dependencies (this IS the database engine)
- Remove cognitive features or weaken their configurability
- Hardcode language specific patterns, use config structs
- Modify SDK Cargo.toml files to join the root workspace
