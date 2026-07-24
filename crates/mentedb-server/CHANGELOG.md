# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.27.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.27.0...mentedb-server-v0.27.1) - 2026-07-24

### Added

- *(embed)* expose embed and embed_batch, honest benchmark timing ([#365](https://github.com/nambok/mentedb/pull/365))

## [0.27.0](https://github.com/nambok/mentedb/compare/mentedb-server-v0.26.0...mentedb-server-v0.27.0) - 2026-07-24

### Added

- offline dedup sweep profile and multi-arch docker image ([#362](https://github.com/nambok/mentedb/pull/362))
- *(server)* expose process_turn on the REST API and tolerate mid-turn memory loss ([#360](https://github.com/nambok/mentedb/pull/360))

## [0.24.3](https://github.com/nambok/mentedb/compare/mentedb-server-v0.24.2...mentedb-server-v0.24.3) - 2026-07-22

### Other

- update Cargo.lock dependencies

## [0.24.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.24.0...mentedb-server-v0.24.1) - 2026-07-22

### Other

- update Cargo.lock dependencies

## [0.23.0](https://github.com/nambok/mentedb/compare/mentedb-server-v0.22.1...mentedb-server-v0.23.0) - 2026-07-22

### Added

- *(recall)* contextual-retrieval hook via an optional context field ([#334](https://github.com/nambok/mentedb/pull/334))

## [0.22.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.22.0...mentedb-server-v0.22.1) - 2026-07-21

### Other

- update Cargo.lock dependencies

## [0.21.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.21.0...mentedb-server-v0.21.1) - 2026-07-21

### Other

- update Cargo.lock dependencies

## [0.20.3](https://github.com/nambok/mentedb/compare/mentedb-server-v0.20.1...mentedb-server-v0.20.3) - 2026-07-19

### Other

- *(readme)* fix inaccuracies found in the docs audit ([#309](https://github.com/nambok/mentedb/pull/309))

## [0.20.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.20.0...mentedb-server-v0.20.1) - 2026-07-19

### Other

- update Cargo.lock dependencies

## [0.17.11](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.10...mentedb-server-v0.17.11) - 2026-07-19

### Added

- *(server)* add Query tab to the bundled console ([#296](https://github.com/nambok/mentedb/pull/296))

## [0.17.10](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.9...mentedb-server-v0.17.10) - 2026-07-19

### Added

- one-command Prometheus + Grafana observability stack ([#291](https://github.com/nambok/mentedb/pull/291))

### Other

- update architecture diagram (hybrid retrieval + reranker, sharding fleet) ([#292](https://github.com/nambok/mentedb/pull/292))

## [0.17.9](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.8...mentedb-server-v0.17.9) - 2026-07-19

### Added

- *(server)* browse and manage memories from the bundled console ([#289](https://github.com/nambok/mentedb/pull/289))

## [0.17.8](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.7...mentedb-server-v0.17.8) - 2026-07-19

### Added

- *(server)* bundled management console at /console ([#288](https://github.com/nambok/mentedb/pull/288))

## [0.17.7](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.6...mentedb-server-v0.17.7) - 2026-07-19

### Other

- update Cargo.lock dependencies

## [0.17.5](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.4...mentedb-server-v0.17.5) - 2026-07-19

### Added

- mentedb-server self-shards a fleet (gossip membership + placement + routing) ([#279](https://github.com/nambok/mentedb/pull/279))

## [0.17.4](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.2...mentedb-server-v0.17.4) - 2026-07-19

### Other

- README scaling now points at the engine mentedb::sharding primitives ([#277](https://github.com/nambok/mentedb/pull/277))
- release v0.17.3 ([#272](https://github.com/nambok/mentedb/pull/272))
- frame README sharding as DIY-today with elastic auto-scaling as the direction ([#274](https://github.com/nambok/mentedb/pull/274))
- reframe README Scaling as self-host guidance with a sharding example ([#273](https://github.com/nambok/mentedb/pull/273))
- add Scaling section to README ([#271](https://github.com/nambok/mentedb/pull/271))

## [0.17.3](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.2...mentedb-server-v0.17.3) - 2026-07-19

### Other

- frame README sharding as DIY-today with elastic auto-scaling as the direction ([#274](https://github.com/nambok/mentedb/pull/274))
- reframe README Scaling as self-host guidance with a sharding example ([#273](https://github.com/nambok/mentedb/pull/273))
- add Scaling section to README ([#271](https://github.com/nambok/mentedb/pull/271))

## [0.17.2](https://github.com/nambok/mentedb/compare/mentedb-server-v0.17.1...mentedb-server-v0.17.2) - 2026-07-17

### Other

- document MQL AS OF and decay reinforcement/never-delete behavior ([#270](https://github.com/nambok/mentedb/pull/270))

## [0.16.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.16.0...mentedb-server-v0.16.1) - 2026-07-17

### Other

- update Cargo.lock dependencies

## [0.15.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.15.0...mentedb-server-v0.15.1) - 2026-07-17

### Other

- update Cargo.lock dependencies

## [0.14.7](https://github.com/nambok/mentedb/compare/mentedb-server-v0.14.6...mentedb-server-v0.14.7) - 2026-07-17

### Other

- update Cargo.lock dependencies

## [0.14.4](https://github.com/nambok/mentedb/compare/mentedb-server-v0.14.3...mentedb-server-v0.14.4) - 2026-07-16

### Other

- update Cargo.lock dependencies

## [0.14.2](https://github.com/nambok/mentedb/compare/mentedb-server-v0.14.1...mentedb-server-v0.14.2) - 2026-07-15

### Other

- update Cargo.lock dependencies

## [0.14.0](https://github.com/nambok/mentedb/compare/mentedb-server-v0.13.1...mentedb-server-v0.14.0) - 2026-07-15

### Added

- add AWS Bedrock as a self-host LLM provider (feature-gated, SigV4) ([#234](https://github.com/nambok/mentedb/pull/234))

## [0.13.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.13.0...mentedb-server-v0.13.1) - 2026-07-15

### Other

- update Cargo.lock dependencies

## [0.12.6](https://github.com/nambok/mentedb/compare/mentedb-server-v0.12.5...mentedb-server-v0.12.6) - 2026-07-15

### Other

- update Cargo.lock dependencies

## [0.12.3](https://github.com/nambok/mentedb/compare/mentedb-server-v0.12.2...mentedb-server-v0.12.3) - 2026-07-15

### Other

- update architecture diagram (embedding providers incl. Bedrock, consolidation, injection/attention, trajectory + speculative) ([#220](https://github.com/nambok/mentedb/pull/220))
- add AWS Bedrock embeddings to the feature list ([#216](https://github.com/nambok/mentedb/pull/216))

## [0.12.1](https://github.com/nambok/mentedb/compare/mentedb-server-v0.12.0...mentedb-server-v0.12.1) - 2026-07-13

### Other

- update Cargo.lock dependencies

## [0.11.14](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.13...mentedb-server-v0.11.14) - 2026-07-13

### Added

- *(cognitive)* LLM-judged contradiction/supersession detection, replacing the ~0%-precision cosine heuristic ([#203](https://github.com/nambok/mentedb/pull/203))

## [0.11.13](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.12...mentedb-server-v0.11.13) - 2026-07-12

### Added

- canonicalize trajectory topics off the hot path so the speculative cache stops keying on raw message text ([#201](https://github.com/nambok/mentedb/pull/201))

### Other

- use the clean /v1/process_turn cloud endpoint in all examples ([#200](https://github.com/nambok/mentedb/pull/200))

## [0.11.12](https://github.com/nambok/mentedb/compare/mentedb-server-v0.11.11...mentedb-server-v0.11.12) - 2026-07-11

### Added

- *(sdk)* expose processTurn/ingest in Node SDK; rewrite onboarding docs around building an agent ([#195](https://github.com/nambok/mentedb/pull/195))

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
