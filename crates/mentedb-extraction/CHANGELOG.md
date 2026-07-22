# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.24.1](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.23.0...mentedb-extraction-v0.24.1) - 2026-07-22

### Other

- release v0.24.0 ([#337](https://github.com/nambok/mentedb/pull/337))

## [0.24.0](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.23.0...mentedb-extraction-v0.24.0) - 2026-07-22

### Fixed

- *(inference)* value-update corrections supersede the outdated fact ([#341](https://github.com/nambok/mentedb/pull/341))

## [0.23.0](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.22.1...mentedb-extraction-v0.23.0) - 2026-07-22

### Added

- *(recall)* contextual-retrieval hook via an optional context field ([#334](https://github.com/nambok/mentedb/pull/334))

## [0.20.3](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.20.1...mentedb-extraction-v0.20.3) - 2026-07-19

### Other

- *(readme)* fix inaccuracies found in the docs audit ([#309](https://github.com/nambok/mentedb/pull/309))

## [0.17.10](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.17.9...mentedb-extraction-v0.17.10) - 2026-07-19

### Added

- one-command Prometheus + Grafana observability stack ([#291](https://github.com/nambok/mentedb/pull/291))

### Other

- update architecture diagram (hybrid retrieval + reranker, sharding fleet) ([#292](https://github.com/nambok/mentedb/pull/292))

## [0.17.5](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.17.4...mentedb-extraction-v0.17.5) - 2026-07-19

### Added

- mentedb-server self-shards a fleet (gossip membership + placement + routing) ([#279](https://github.com/nambok/mentedb/pull/279))

## [0.17.4](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.17.3...mentedb-extraction-v0.17.4) - 2026-07-19

### Other

- README scaling now points at the engine mentedb::sharding primitives ([#277](https://github.com/nambok/mentedb/pull/277))

## [0.17.3](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.17.2...mentedb-extraction-v0.17.3) - 2026-07-19

### Other

- frame README sharding as DIY-today with elastic auto-scaling as the direction ([#274](https://github.com/nambok/mentedb/pull/274))
- reframe README Scaling as self-host guidance with a sharding example ([#273](https://github.com/nambok/mentedb/pull/273))
- add Scaling section to README ([#271](https://github.com/nambok/mentedb/pull/271))

## [0.17.2](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.17.1...mentedb-extraction-v0.17.2) - 2026-07-17

### Other

- document MQL AS OF and decay reinforcement/never-delete behavior ([#270](https://github.com/nambok/mentedb/pull/270))

## [0.17.1](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.17.0...mentedb-extraction-v0.17.1) - 2026-07-17

### Fixed

- truncate text on char boundaries to avoid UTF-8 panics ([#268](https://github.com/nambok/mentedb/pull/268))

## [0.14.0](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.13.1...mentedb-extraction-v0.14.0) - 2026-07-15

### Added

- add AWS Bedrock as a self-host LLM provider (feature-gated, SigV4) ([#234](https://github.com/nambok/mentedb/pull/234))

## [0.12.3](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.12.2...mentedb-extraction-v0.12.3) - 2026-07-15

### Other

- update architecture diagram (embedding providers incl. Bedrock, consolidation, injection/attention, trajectory + speculative) ([#220](https://github.com/nambok/mentedb/pull/220))
- add AWS Bedrock embeddings to the feature list ([#216](https://github.com/nambok/mentedb/pull/216))

## [0.12.1](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.14...mentedb-extraction-v0.12.1) - 2026-07-13

### Fixed

- *(cognitive)* dedup exact copies with a Derived edge not Supersedes; tighten phantom extraction to reject fragments and data values ([#206](https://github.com/nambok/mentedb/pull/206))

### Other

- release v0.12.0 ([#207](https://github.com/nambok/mentedb/pull/207))

## [0.12.0](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.14...mentedb-extraction-v0.12.0) - 2026-07-13

### Fixed

- *(cognitive)* dedup exact copies with a Derived edge not Supersedes; tighten phantom extraction to reject fragments and data values ([#206](https://github.com/nambok/mentedb/pull/206))

## [0.11.14](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.13...mentedb-extraction-v0.11.14) - 2026-07-13

### Added

- *(cognitive)* LLM-judged contradiction/supersession detection, replacing the ~0%-precision cosine heuristic ([#203](https://github.com/nambok/mentedb/pull/203))

## [0.11.13](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.12...mentedb-extraction-v0.11.13) - 2026-07-12

### Added

- canonicalize trajectory topics off the hot path so the speculative cache stops keying on raw message text ([#201](https://github.com/nambok/mentedb/pull/201))

### Other

- use the clean /v1/process_turn cloud endpoint in all examples ([#200](https://github.com/nambok/mentedb/pull/200))
- fix MenteDBChatHistory example, session_id is a required argument ([#196](https://github.com/nambok/mentedb/pull/196))

## [0.11.12](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.11...mentedb-extraction-v0.11.12) - 2026-07-11

### Added

- *(sdk)* expose processTurn/ingest in Node SDK; rewrite onboarding docs around building an agent ([#195](https://github.com/nambok/mentedb/pull/195))

## [0.11.11](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.10...mentedb-extraction-v0.11.11) - 2026-07-11

### Added

- server maintenance runner + speculative cache entries accessor ([#190](https://github.com/nambok/mentedb/pull/190))

## [0.11.9](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.8...mentedb-extraction-v0.11.9) - 2026-07-11

### Other

- maintain EntityResolver canonical set + disambiguate rule matches (closes #45, #46) ([#182](https://github.com/nambok/mentedb/pull/182))

## [0.11.8](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.7...mentedb-extraction-v0.11.8) - 2026-07-11

### Other

- live deps.rs dependency-status badge; group Dependabot updates into 1-2 PRs ([#169](https://github.com/nambok/mentedb/pull/169))
- add Dependabot badge to README ([#149](https://github.com/nambok/mentedb/pull/149))

## [0.11.5](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.11.4...mentedb-extraction-v0.11.5) - 2026-07-11

### Fixed

- remove redundant format arg borrows (clippy 1.97 useless_borrows_in_formatting) ([#125](https://github.com/nambok/mentedb/pull/125))

### Other

- correct LongMemEval to the real 92.0% (460/500), replace cherry-picked composite with the honest baseline run + committed judge labels ([#124](https://github.com/nambok/mentedb/pull/124))

## [0.10.5](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.10.4...mentedb-extraction-v0.10.5) - 2026-07-06

### Fixed

- installable on debian and arm linux, publish mentedb-server ([#102](https://github.com/nambok/mentedb/pull/102))

## [0.10.1](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.10.0...mentedb-extraction-v0.10.1) - 2026-07-03

### Other

- lead with Claude Code hooks and connector integration paths, bump benchmark version to 0.10.0 ([#92](https://github.com/nambok/mentedb/pull/92))

## [0.10.0](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.9.2...mentedb-extraction-v0.10.0) - 2026-07-03

### Fixed

- dedup write-inference edges, make similarity bands exclusive, run inference in store_batch

### Other

- remove unimplemented vscode setup target from README
- align README and architecture docs with actual engine behavior

## [0.9.2](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.9.1...mentedb-extraction-v0.9.2) - 2026-05-13

### Fixed

- resolve clippy and fmt warnings in engine/SDK

### Other

- update LongMemEval benchmark results to 95.2% (476/500)

## [0.8.2](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.8.1...mentedb-extraction-v0.8.2) - 2026-04-26

### Other

- Merge pull request #76 from nambok/feat/enrichment-phase3-4
- add sleeptime enrichment to READMEs

## [0.8.1](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.8.0...mentedb-extraction-v0.8.1) - 2026-04-26

### Added

- add community detection and user model generation (Phase 3 & 4)

### Fixed

- address code review findings for Phase 3 & 4

## [0.8.0](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.7.2...mentedb-extraction-v0.8.0) - 2026-04-26

### Fixed

- entity resolver hyphen matching, context truncation overflow, doc comments

### Other

- add entity linking: name+embedding disambiguation, entity tags, SDK methods, tests
- fix READMEs — process_turn as primary API, remove deprecated ingest, fix class names
- rewrite quick start around process_turn
- remove unused MIN_WORD_LEN constant
- LLM-powered entity linking: replace threshold heuristics with EntityResolver + LLM pipeline

## [0.7.2](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.7.1...mentedb-extraction-v0.7.2) - 2026-04-26

### Other

- add installation section, update Docker and SDK examples

## [0.6.2](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.6.1...mentedb-extraction-v0.6.2) - 2026-04-26

### Added

- wire all 9 remaining cognitive subsystems into MenteDb ([#66](https://github.com/nambok/mentedb/pull/66))
- wire cognitive engine subsystems into MenteDb ([#65](https://github.com/nambok/mentedb/pull/65))
- token efficiency benchmark ([#63](https://github.com/nambok/mentedb/pull/63))

## [0.6.0](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.5.2...mentedb-extraction-v0.6.0) - 2026-04-20

### Added

- bincode index persistence, async LLM extraction, SDK concurrency

### Other

- update README test count, ARCHITECTURE server section, lib.rs module list

## [0.5.1](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.5.0...mentedb-extraction-v0.5.1) - 2026-04-19

### Fixed

- use rustls-tls instead of native-tls in extraction crate
- resolve all clippy warnings across workspace

### Other

- update MCP install to npx as primary method

## [0.5.0](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.4.2...mentedb-extraction-v0.5.0) - 2026-04-13

### Added

- add LongMemEval benchmark harness ([#48](https://github.com/nambok/mentedb/pull/48))

## [0.4.2](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.4.1...mentedb-extraction-v0.4.2) - 2026-04-07

### Other

- Add entity resolution with three tier strategy ([#43](https://github.com/nambok/mentedb/pull/43))

## [0.4.1](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.3.2...mentedb-extraction-v0.4.1) - 2026-04-07

### Other

- Wire LLM topic canonicalization into TransitionMap ([#42](https://github.com/nambok/mentedb/pull/42))

## [0.3.2](https://github.com/nambok/mentedb/compare/mentedb-extraction-v0.3.1...mentedb-extraction-v0.3.2) - 2026-04-07

### Added

- add CognitiveLlmService for LLM powered memory judgment ([#37](https://github.com/nambok/mentedb/pull/37))
