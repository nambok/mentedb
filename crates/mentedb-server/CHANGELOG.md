# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.11](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.10...mentedb-server-v0.11.11) - 2026-07-11

### Added

- server maintenance runner + speculative cache entries accessor ([#190](https://github.com/nambok/mentedb/pull/190))

## [0.11.10](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.9...mentedb-server-v0.11.10) - 2026-07-11

### Other

- update Cargo.lock dependencies

## [0.11.9](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.8...mentedb-server-v0.11.9) - 2026-07-11

### Other

- update Cargo.lock dependencies

## [0.11.8](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.7...mentedb-server-v0.11.8) - 2026-07-11

### Other

- live deps.rs dependency-status badge; group Dependabot updates into 1-2 PRs ([#169](https://github.com/nambok/mentedb/pull/169))
- add Dependabot badge to README ([#149](https://github.com/nambok/mentedb/pull/149))

## [0.11.7](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.6...mentedb-server-v0.11.7) - 2026-07-11

### Other

- update Cargo.lock dependencies

## [0.11.5](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.4...mentedb-server-v0.11.5) - 2026-07-11

### Other

- correct LongMemEval to the real 92.0% (460/500), replace cherry-picked composite with the honest baseline run + committed judge labels ([#124](https://github.com/nambok/mentedb/pull/124))

## [0.11.3](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.2...mentedb-server-v0.11.3) - 2026-07-06

### Other

- update Cargo.lock dependencies

## [0.10.8](https://github.com/nambok/mentedb/compare/mentedb-server-v0.10.7...mentedb-server-v0.10.8) - 2026-07-06

### Other

- update Cargo.lock dependencies

## [0.10.6](https://github.com/nambok/mentedb/compare/mentedb-server-v0.10.5...mentedb-server-v0.10.6) - 2026-07-06

### Other

- update Cargo.lock dependencies

## [0.10.5](https://github.com/nambok/mentedb/compare/mentedb-server-v0.10.4...mentedb-server-v0.10.5) - 2026-07-06

### Added

- wire all 9 remaining cognitive subsystems into MenteDb ([#66](https://github.com/nambok/mentedb/pull/66))
- wire cognitive engine subsystems into MenteDb ([#65](https://github.com/nambok/mentedb/pull/65))
- token efficiency benchmark ([#63](https://github.com/nambok/mentedb/pull/63))
- SIMD vector distance (AVX2 + NEON) and extraction queue
- bincode index persistence, async LLM extraction, SDK concurrency
- add LongMemEval benchmark harness ([#48](https://github.com/nambok/mentedb/pull/48))
- add candle embedding provider to Python SDK and quality benchmark
- wire extraction into REST API, auto-extract mode, LLM provider config
- add comprehensive docs.rs documentation across all crates
- add gRPC streaming, dashboard UI, Dockerfile, PyPI trusted publisher, README badges, and npm OIDC publish
- add server integration tests, realistic scenario tests, and crates.io publish metadata
- upgrade server to axum with JWT auth, rate limiting, and WebSocket
- add embedding crate, WAL crash recovery, entity API, and clippy fixes
- add REST API server, index/graph persistence, 8 scenario tests, docs, examples, and CI
- add config system, benchmarks, docs and make all heuristics configurable
- add cognitive engine (7 modules), database facade, and server binary
- scaffold rust workspace with 8 crates

### Fixed

- installable on debian and arm linux, publish mentedb-server ([#102](https://github.com/nambok/mentedb/pull/102))
- wire embedding provider into server and stop silent zero-vector fallback in process_turn
- publish pipeline — npm napi build, integration ordering, changelog ([#70](https://github.com/nambok/mentedb/pull/70))
- sync SDK versions to 0.7.0 and add process_turn ([#69](https://github.com/nambok/mentedb/pull/69))
- collapse nested if to satisfy clippy collapsible_if
- remove global RwLock — each component uses interior mutability
- clarify delta token reduction claims
- clean up unused imports and variables in tests
- complete audit remediation, 427 tests passing
- space ACLs, embedding validation, cleanup unused imports
- address audit findings (partial, server in progress)
- use static license badge and mentedb-core for crates.io badges

### Other

- lead with Claude Code hooks and connector integration paths, bump benchmark version to 0.10.0 ([#92](https://github.com/nambok/mentedb/pull/92))
- remove unimplemented vscode setup target from README
- align README and architecture docs with actual engine behavior
- update LongMemEval benchmark results to 95.2% (476/500)
- add sleeptime enrichment to READMEs
- fix READMEs — process_turn as primary API, remove deprecated ingest, fix class names
- rewrite quick start around process_turn
- add installation section, update Docker and SDK examples
- update README test count, ARCHITECTURE server section, lib.rs module list
- eliminate global write lock for read operations
- update MCP install to npx as primary method
- *(deps)* bump jsonwebtoken from 9.3.1 to 10.3.0 in the cargo group across 1 directory ([#16](https://github.com/nambok/mentedb/pull/16))
- Wire LLM topic canonicalization into TransitionMap ([#42](https://github.com/nambok/mentedb/pull/42))
- Add bi-temporal validity to memories and edges ([#35](https://github.com/nambok/mentedb/pull/35))
- Add LLM accuracy benchmark results to README ([#38](https://github.com/nambok/mentedb/pull/38))
- add issues link to README
- add beta notice to README
- update install to cargo install mentedb-mcp
- update install to cargo install --git
- fix tool count to 32 in README
- update benchmark results with Mem0 comparison and 10K scale
- update README with Candle embeddings and 31 MCP tools
- add honest caveats to benchmark results
- add Mem0 comparison and updated results to README
- add benchmark results to README
- cargo fmt
- link to mentedb-mcp repo in README
- comprehensive README with extraction, security, MCP, SDKs
- apply cargo fmt to all crates
- replace text architecture with mermaid diagram and remove status section
- remove dashes and emojis from README
