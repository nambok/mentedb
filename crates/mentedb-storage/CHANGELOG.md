# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1](https://github.com/nambok/mentedb/compare/mentedb-storage-v0.7.0...mentedb-storage-v0.7.1) - 2026-04-26

### Fixed

- sync SDK versions to 0.7.0 and add process_turn ([#69](https://github.com/nambok/mentedb/pull/69))

## [0.6.2](https://github.com/nambok/mentedb/compare/mentedb-storage-v0.6.1...mentedb-storage-v0.6.2) - 2026-04-26

### Added

- wire all 9 remaining cognitive subsystems into MenteDb ([#66](https://github.com/nambok/mentedb/pull/66))
- wire cognitive engine subsystems into MenteDb ([#65](https://github.com/nambok/mentedb/pull/65))
- token efficiency benchmark ([#63](https://github.com/nambok/mentedb/pull/63))

## [0.6.0](https://github.com/nambok/mentedb/compare/mentedb-storage-v0.5.2...mentedb-storage-v0.6.0) - 2026-04-20

### Added

- bincode index persistence, async LLM extraction, SDK concurrency

### Other

- update README test count, ARCHITECTURE server section, lib.rs module list
- eliminate global write lock for read operations

## [0.5.1](https://github.com/nambok/mentedb/compare/mentedb-storage-v0.5.0...mentedb-storage-v0.5.1) - 2026-04-19

### Fixed

- use sort_by_key with Reverse to satisfy clippy

### Other

- update MCP install to npx as primary method

## [0.5.0](https://github.com/nambok/mentedb/compare/mentedb-storage-v0.4.2...mentedb-storage-v0.5.0) - 2026-04-13

### Added

- add LongMemEval benchmark harness ([#48](https://github.com/nambok/mentedb/pull/48))

## [0.4.1](https://github.com/nambok/mentedb/compare/mentedb-storage-v0.4.0...mentedb-storage-v0.4.1) - 2026-04-07

### Other

- Wire LLM topic canonicalization into TransitionMap ([#42](https://github.com/nambok/mentedb/pull/42))
