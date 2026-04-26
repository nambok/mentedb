# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0](https://github.com/nambok/mentedb/compare/mentedb-v0.7.2...mentedb-v0.8.0) - 2026-04-26

### Fixed

- entity resolver hyphen matching, context truncation overflow, doc comments
- drop page_map read lock before loading memories to prevent potential deadlock
- fix duplicate entity link edges on repeated runs, add idempotency test

### Other

- fmt
- LLM-powered entity linking: replace threshold heuristics with EntityResolver + LLM pipeline
- add entity linking: name+embedding disambiguation, entity tags, SDK methods, tests
- add sleeptime enrichment foundation: config, API, always-scope, trigger
- fix READMEs — process_turn as primary API, remove deprecated ingest, fix class names
- rewrite quick start around process_turn
- remove unused MIN_WORD_LEN constant

## [0.7.2](https://github.com/nambok/mentedb/compare/mentedb-v0.7.1...mentedb-v0.7.2) - 2026-04-26

### Other

- add installation section, update Docker and SDK examples

## [0.7.1](https://github.com/nambok/mentedb/compare/mentedb-v0.7.0...mentedb-v0.7.1) - 2026-04-26

### Added

- unified process_turn() engine API ([#67](https://github.com/nambok/mentedb/pull/67))

### Fixed

- sync SDK versions to 0.7.0 and add process_turn ([#69](https://github.com/nambok/mentedb/pull/69))

## [0.6.2](https://github.com/nambok/mentedb/compare/mentedb-v0.6.1...mentedb-v0.6.2) - 2026-04-26

### Added

- wire all 9 remaining cognitive subsystems into MenteDb ([#66](https://github.com/nambok/mentedb/pull/66))
- wire cognitive engine subsystems into MenteDb ([#65](https://github.com/nambok/mentedb/pull/65))
- token efficiency benchmark ([#63](https://github.com/nambok/mentedb/pull/63))

## [0.6.1](https://github.com/nambok/mentedb/compare/mentedb-v0.6.0...mentedb-v0.6.1) - 2026-04-23

### Added

- export VERSION const from mentedb crate

## [0.6.0](https://github.com/nambok/mentedb/compare/mentedb-v0.5.2...mentedb-v0.6.0) - 2026-04-20

### Added

- bincode index persistence, async LLM extraction, SDK concurrency

### Fixed

- remove global RwLock — each component uses interior mutability

### Other

- update README test count, ARCHITECTURE server section, lib.rs module list
- eliminate global write lock for read operations

## [0.5.1](https://github.com/nambok/mentedb/compare/mentedb-v0.5.0...mentedb-v0.5.1) - 2026-04-19

### Fixed

- resolve all clippy warnings across workspace
- use rustls-tls instead of native-tls in extraction crate

### Other

- update MCP install to npx as primary method

## [0.5.0](https://github.com/nambok/mentedb/compare/mentedb-v0.4.2...mentedb-v0.5.0) - 2026-04-13

### Added

- add LongMemEval benchmark harness ([#48](https://github.com/nambok/mentedb/pull/48))

## [0.4.2](https://github.com/nambok/mentedb/compare/mentedb-v0.4.1...mentedb-v0.4.2) - 2026-04-07

### Other

- Add entity resolution with three tier strategy ([#43](https://github.com/nambok/mentedb/pull/43))

## [0.4.1](https://github.com/nambok/mentedb/compare/mentedb-v0.3.2...mentedb-v0.4.1) - 2026-04-07

### Other

- Wire LLM topic canonicalization into TransitionMap ([#42](https://github.com/nambok/mentedb/pull/42))
- release v0.4.0 ([#41](https://github.com/nambok/mentedb/pull/41))
- Add bi-temporal validity to memories and edges ([#35](https://github.com/nambok/mentedb/pull/35))

## [0.4.0](https://github.com/nambok/mentedb/compare/mentedb-v0.3.2...mentedb-v0.4.0) - 2026-04-07

### Other

- Add bi-temporal validity to memories and edges ([#35](https://github.com/nambok/mentedb/pull/35))

## [0.3.2](https://github.com/nambok/mentedb/compare/mentedb-v0.3.1...mentedb-v0.3.2) - 2026-04-07

### Other

- release v0.3.2 ([#39](https://github.com/nambok/mentedb/pull/39))
- Add LLM accuracy benchmark results to README ([#38](https://github.com/nambok/mentedb/pull/38))
