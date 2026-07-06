# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1](https://github.com/nambok/mentedb/compare/mentedb-index-v0.10.8...mentedb-index-v0.11.1) - 2026-07-06

### Other

- release v0.11.0 ([#114](https://github.com/nambok/mentedb/pull/114))
- amortize snapshot writes across flushes, reconcile stale snapshots at open ([#113](https://github.com/nambok/mentedb/pull/113))

## [0.11.0](https://github.com/nambok/mentedb/compare/mentedb-index-v0.10.8...mentedb-index-v0.11.0) - 2026-07-06

### Other

- amortize snapshot writes across flushes, reconcile stale snapshots at open ([#113](https://github.com/nambok/mentedb/pull/113))

## [0.10.5](https://github.com/nambok/mentedb/compare/mentedb-index-v0.10.4...mentedb-index-v0.10.5) - 2026-07-06

### Fixed

- installable on debian and arm linux, publish mentedb-server ([#102](https://github.com/nambok/mentedb/pull/102))

## [0.10.1](https://github.com/nambok/mentedb/compare/mentedb-index-v0.10.0...mentedb-index-v0.10.1) - 2026-07-03

### Other

- lead with Claude Code hooks and connector integration paths, bump benchmark version to 0.10.0 ([#92](https://github.com/nambok/mentedb/pull/92))

## [0.10.0](https://github.com/nambok/mentedb/compare/mentedb-index-v0.9.2...mentedb-index-v0.10.0) - 2026-07-03

### Other

- remove unimplemented vscode setup target from README
- align README and architecture docs with actual engine behavior

## [0.9.2](https://github.com/nambok/mentedb/compare/mentedb-index-v0.9.1...mentedb-index-v0.9.2) - 2026-05-13

### Added

- benchmark improvements + engine fixes

### Fixed

- resolve clippy and fmt warnings in engine/SDK

### Other

- update LongMemEval benchmark results to 95.2% (476/500)

## [0.9.0](https://github.com/nambok/mentedb/compare/mentedb-index-v0.8.2...mentedb-index-v0.9.0) - 2026-04-27

### Fixed

- WAL durability, page checksums, graph cleanup, salience O(1)

## [0.8.2](https://github.com/nambok/mentedb/compare/mentedb-index-v0.8.1...mentedb-index-v0.8.2) - 2026-04-26

### Other

- add sleeptime enrichment to READMEs

## [0.8.0](https://github.com/nambok/mentedb/compare/mentedb-index-v0.7.2...mentedb-index-v0.8.0) - 2026-04-26

### Other

- fix READMEs — process_turn as primary API, remove deprecated ingest, fix class names
- rewrite quick start around process_turn

## [0.7.2](https://github.com/nambok/mentedb/compare/mentedb-index-v0.7.1...mentedb-index-v0.7.2) - 2026-04-26

### Other

- add installation section, update Docker and SDK examples

## [0.6.2](https://github.com/nambok/mentedb/compare/mentedb-index-v0.6.1...mentedb-index-v0.6.2) - 2026-04-26

### Added

- wire all 9 remaining cognitive subsystems into MenteDb ([#66](https://github.com/nambok/mentedb/pull/66))
- wire cognitive engine subsystems into MenteDb ([#65](https://github.com/nambok/mentedb/pull/65))
- token efficiency benchmark ([#63](https://github.com/nambok/mentedb/pull/63))

## [0.6.0](https://github.com/nambok/mentedb/compare/mentedb-index-v0.5.2...mentedb-index-v0.6.0) - 2026-04-20

### Added

- SIMD vector distance (AVX2 + NEON) and extraction queue
- bincode index persistence, async LLM extraction, SDK concurrency

### Other

- update README test count, ARCHITECTURE server section, lib.rs module list

## [0.5.1](https://github.com/nambok/mentedb/compare/mentedb-index-v0.5.0...mentedb-index-v0.5.1) - 2026-04-19

### Other

- update MCP install to npx as primary method
- *(deps)* bump the cargo group across 2 directories with 1 update ([#51](https://github.com/nambok/mentedb/pull/51))

## [0.5.0](https://github.com/nambok/mentedb/compare/mentedb-index-v0.4.2...mentedb-index-v0.5.0) - 2026-04-13

### Added

- add LongMemEval benchmark harness ([#48](https://github.com/nambok/mentedb/pull/48))

## [0.4.1](https://github.com/nambok/mentedb/compare/mentedb-index-v0.4.0...mentedb-index-v0.4.1) - 2026-04-07

### Other

- Wire LLM topic canonicalization into TransitionMap ([#42](https://github.com/nambok/mentedb/pull/42))
