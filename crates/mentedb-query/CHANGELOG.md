# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.12](https://github.com/nambok/mentedb/compare/mentedb-query-v0.11.11...mentedb-query-v0.11.12) - 2026-07-11

### Added

- *(sdk)* expose processTurn/ingest in Node SDK; rewrite onboarding docs around building an agent ([#195](https://github.com/nambok/mentedb/pull/195))

## [0.11.8](https://github.com/nambok/mentedb/compare/mentedb-query-v0.11.7...mentedb-query-v0.11.8) - 2026-07-11

### Other

- live deps.rs dependency-status badge; group Dependabot updates into 1-2 PRs ([#169](https://github.com/nambok/mentedb/pull/169))
- add Dependabot badge to README ([#149](https://github.com/nambok/mentedb/pull/149))

## [0.11.5](https://github.com/nambok/mentedb/compare/mentedb-query-v0.11.4...mentedb-query-v0.11.5) - 2026-07-11

### Other

- correct LongMemEval to the real 92.0% (460/500), replace cherry-picked composite with the honest baseline run + committed judge labels ([#124](https://github.com/nambok/mentedb/pull/124))

## [0.10.5](https://github.com/nambok/mentedb/compare/mentedb-query-v0.10.4...mentedb-query-v0.10.5) - 2026-07-06

### Fixed

- installable on debian and arm linux, publish mentedb-server ([#102](https://github.com/nambok/mentedb/pull/102))

## [0.10.1](https://github.com/nambok/mentedb/compare/mentedb-query-v0.10.0...mentedb-query-v0.10.1) - 2026-07-03

### Other

- lead with Claude Code hooks and connector integration paths, bump benchmark version to 0.10.0 ([#92](https://github.com/nambok/mentedb/pull/92))

## [0.10.0](https://github.com/nambok/mentedb/compare/mentedb-query-v0.9.2...mentedb-query-v0.10.0) - 2026-07-03

### Other

- remove unimplemented vscode setup target from README
- align README and architecture docs with actual engine behavior

## [0.9.2](https://github.com/nambok/mentedb/compare/mentedb-query-v0.9.1...mentedb-query-v0.9.2) - 2026-05-13

### Other

- update LongMemEval benchmark results to 95.2% (476/500)

## [0.8.2](https://github.com/nambok/mentedb/compare/mentedb-query-v0.8.1...mentedb-query-v0.8.2) - 2026-04-26

### Other

- add sleeptime enrichment to READMEs

## [0.8.0](https://github.com/nambok/mentedb/compare/mentedb-query-v0.7.2...mentedb-query-v0.8.0) - 2026-04-26

### Other

- fix READMEs — process_turn as primary API, remove deprecated ingest, fix class names
- rewrite quick start around process_turn

## [0.7.2](https://github.com/nambok/mentedb/compare/mentedb-query-v0.7.1...mentedb-query-v0.7.2) - 2026-04-26

### Other

- add installation section, update Docker and SDK examples

## [0.6.2](https://github.com/nambok/mentedb/compare/mentedb-query-v0.6.1...mentedb-query-v0.6.2) - 2026-04-26

### Added

- wire all 9 remaining cognitive subsystems into MenteDb ([#66](https://github.com/nambok/mentedb/pull/66))
- wire cognitive engine subsystems into MenteDb ([#65](https://github.com/nambok/mentedb/pull/65))
- token efficiency benchmark ([#63](https://github.com/nambok/mentedb/pull/63))

## [0.6.0](https://github.com/nambok/mentedb/compare/mentedb-query-v0.5.2...mentedb-query-v0.6.0) - 2026-04-20

### Added

- bincode index persistence, async LLM extraction, SDK concurrency

### Other

- update README test count, ARCHITECTURE server section, lib.rs module list

## [0.5.1](https://github.com/nambok/mentedb/compare/mentedb-query-v0.5.0...mentedb-query-v0.5.1) - 2026-04-19

### Other

- update MCP install to npx as primary method

## [0.5.0](https://github.com/nambok/mentedb/compare/mentedb-query-v0.4.2...mentedb-query-v0.5.0) - 2026-04-13

### Added

- add LongMemEval benchmark harness ([#48](https://github.com/nambok/mentedb/pull/48))

## [0.4.1](https://github.com/nambok/mentedb/compare/mentedb-query-v0.4.0...mentedb-query-v0.4.1) - 2026-04-07

### Other

- Wire LLM topic canonicalization into TransitionMap ([#42](https://github.com/nambok/mentedb/pull/42))
