# MenteDB Copilot Instructions

## Project overview

MenteDB is a purpose built Rust database engine for AI agent memory. It includes custom storage (WAL, buffer pool, pages), HNSW vector indexing, CSR/CSC graph, a custom query language (MQL), context assembly with U curve attention layout, and 7 unique cognitive features (stream cognition, write time inference, trajectory tracking, phantom memories, interference detection, pain signals, speculative pre assembly).

## Workspace structure

11 Rust crates under `crates/`, plus standalone SDKs under `sdks/python/` (PyO3) and `sdks/typescript/` (napi-rs). The SDKs are excluded from the Cargo workspace and build independently.

## Build, test, and lint

Always run these before committing:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

The server binary is `mentedb-server` and runs on axum with JWT auth, rate limiting, and WebSocket support.

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
