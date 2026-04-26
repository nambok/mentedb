# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
