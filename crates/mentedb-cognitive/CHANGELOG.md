# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.2](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.9.1...mentedb-cognitive-v0.9.2) - 2026-05-13

### Other

- update LongMemEval benchmark results to 95.2% (476/500)

## [0.8.2](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.8.1...mentedb-cognitive-v0.8.2) - 2026-04-26

### Other

- Merge pull request #76 from nambok/feat/enrichment-phase3-4
- add sleeptime enrichment to READMEs

## [0.8.1](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.8.0...mentedb-cognitive-v0.8.1) - 2026-04-26

### Added

- add community detection and user model generation (Phase 3 & 4)

### Fixed

- address code review findings for Phase 3 & 4

## [0.8.0](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.7.2...mentedb-cognitive-v0.8.0) - 2026-04-26

### Fixed

- entity resolver hyphen matching, context truncation overflow, doc comments

### Other

- remove unused MIN_WORD_LEN constant
- LLM-powered entity linking: replace threshold heuristics with EntityResolver + LLM pipeline
- fix READMEs — process_turn as primary API, remove deprecated ingest, fix class names
- rewrite quick start around process_turn

## [0.7.2](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.7.1...mentedb-cognitive-v0.7.2) - 2026-04-26

### Other

- add installation section, update Docker and SDK examples

## [0.6.2](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.6.1...mentedb-cognitive-v0.6.2) - 2026-04-26

### Added

- wire all 9 remaining cognitive subsystems into MenteDb ([#66](https://github.com/nambok/mentedb/pull/66))
- wire cognitive engine subsystems into MenteDb ([#65](https://github.com/nambok/mentedb/pull/65))
- token efficiency benchmark ([#63](https://github.com/nambok/mentedb/pull/63))

## [0.6.0](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.5.2...mentedb-cognitive-v0.6.0) - 2026-04-20

### Added

- bincode index persistence, async LLM extraction, SDK concurrency

### Other

- update README test count, ARCHITECTURE server section, lib.rs module list

## [0.5.1](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.5.0...mentedb-cognitive-v0.5.1) - 2026-04-19

### Fixed

- resolve all clippy warnings across workspace

### Other

- update MCP install to npx as primary method

## [0.5.0](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.4.2...mentedb-cognitive-v0.5.0) - 2026-04-13

### Added

- add LongMemEval benchmark harness ([#48](https://github.com/nambok/mentedb/pull/48))

## [0.4.2](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.4.1...mentedb-cognitive-v0.4.2) - 2026-04-07

### Other

- Add entity resolution with three tier strategy ([#43](https://github.com/nambok/mentedb/pull/43))

## [0.4.1](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.4.0...mentedb-cognitive-v0.4.1) - 2026-04-07

### Other

- Wire LLM topic canonicalization into TransitionMap ([#42](https://github.com/nambok/mentedb/pull/42))

## [0.4.0](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.3.2...mentedb-cognitive-v0.4.0) - 2026-04-07

### Other

- Add bi-temporal validity to memories and edges ([#35](https://github.com/nambok/mentedb/pull/35))

## [0.3.2](https://github.com/nambok/mentedb/compare/mentedb-cognitive-v0.3.1...mentedb-cognitive-v0.3.2) - 2026-04-07

### Added

- add CognitiveLlmService for LLM powered memory judgment ([#37](https://github.com/nambok/mentedb/pull/37))
