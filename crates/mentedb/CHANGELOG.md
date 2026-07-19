# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.17.7](https://github.com/nambok/mentedb/compare/mentedb-v0.17.5...mentedb-v0.17.7) - 2026-07-19

### Added

- structured export (fill a JSON schema from memories via a pluggable LLM) ([#284](https://github.com/nambok/mentedb/pull/284))
- pluggable reranker stage (opt-in second-pass recall reorder) ([#282](https://github.com/nambok/mentedb/pull/282))

### Other

- release v0.17.6 ([#283](https://github.com/nambok/mentedb/pull/283))

## [0.17.6](https://github.com/nambok/mentedb/compare/mentedb-v0.17.5...mentedb-v0.17.6) - 2026-07-19

### Added

- structured export (fill a JSON schema from memories via a pluggable LLM) ([#284](https://github.com/nambok/mentedb/pull/284))
- pluggable reranker stage (opt-in second-pass recall reorder) ([#282](https://github.com/nambok/mentedb/pull/282))

## [0.17.5](https://github.com/nambok/mentedb/compare/mentedb-v0.17.4...mentedb-v0.17.5) - 2026-07-19

### Added

- mentedb-server self-shards a fleet (gossip membership + placement + routing) ([#279](https://github.com/nambok/mentedb/pull/279))

## [0.17.4](https://github.com/nambok/mentedb/compare/mentedb-v0.17.2...mentedb-v0.17.4) - 2026-07-19

### Other

- README scaling now points at the engine mentedb::sharding primitives ([#277](https://github.com/nambok/mentedb/pull/277))
- release v0.17.3 ([#272](https://github.com/nambok/mentedb/pull/272))
- relocate elastic sharding to the published mentedb facade ([#276](https://github.com/nambok/mentedb/pull/276))
- frame README sharding as DIY-today with elastic auto-scaling as the direction ([#274](https://github.com/nambok/mentedb/pull/274))
- reframe README Scaling as self-host guidance with a sharding example ([#273](https://github.com/nambok/mentedb/pull/273))
- add Scaling section to README ([#271](https://github.com/nambok/mentedb/pull/271))

## [0.17.3](https://github.com/nambok/mentedb/compare/mentedb-v0.17.2...mentedb-v0.17.3) - 2026-07-19

### Other

- relocate elastic sharding to the published mentedb facade ([#276](https://github.com/nambok/mentedb/pull/276))
- frame README sharding as DIY-today with elastic auto-scaling as the direction ([#274](https://github.com/nambok/mentedb/pull/274))
- reframe README Scaling as self-host guidance with a sharding example ([#273](https://github.com/nambok/mentedb/pull/273))
- add Scaling section to README ([#271](https://github.com/nambok/mentedb/pull/271))

## [0.17.2](https://github.com/nambok/mentedb/compare/mentedb-v0.17.1...mentedb-v0.17.2) - 2026-07-17

### Other

- document MQL AS OF and decay reinforcement/never-delete behavior ([#270](https://github.com/nambok/mentedb/pull/270))

## [0.17.1](https://github.com/nambok/mentedb/compare/mentedb-v0.17.0...mentedb-v0.17.1) - 2026-07-17

### Fixed

- truncate text on char boundaries to avoid UTF-8 panics ([#268](https://github.com/nambok/mentedb/pull/268))

## [0.17.0](https://github.com/nambok/mentedb/compare/mentedb-v0.16.1...mentedb-v0.17.0) - 2026-07-17

### Added

- MQL AS OF <t> for bitemporal point-in-time recall ([#264](https://github.com/nambok/mentedb/pull/264))

## [0.16.1](https://github.com/nambok/mentedb/compare/mentedb-v0.15.1...mentedb-v0.16.1) - 2026-07-17

### Added

- reembed_all for embedding-dimension migration ([#262](https://github.com/nambok/mentedb/pull/262))

### Fixed

- decay demotes but never deletes curated memories ([#260](https://github.com/nambok/mentedb/pull/260))

### Other

- release v0.16.0 ([#261](https://github.com/nambok/mentedb/pull/261))

## [0.16.0](https://github.com/nambok/mentedb/compare/mentedb-v0.15.1...mentedb-v0.16.0) - 2026-07-17

### Fixed

- decay demotes but never deletes curated memories ([#260](https://github.com/nambok/mentedb/pull/260))

## [0.15.1](https://github.com/nambok/mentedb/compare/mentedb-v0.14.7...mentedb-v0.15.1) - 2026-07-17

### Added

- associative recall via graph expansion (gated, default off) ([#258](https://github.com/nambok/mentedb/pull/258))

### Other

- release v0.14.8 ([#256](https://github.com/nambok/mentedb/pull/256))
- end-to-end retention probe for the memory lifecycle ([#257](https://github.com/nambok/mentedb/pull/257))

## [0.15.0](https://github.com/nambok/mentedb/compare/mentedb-v0.14.7...mentedb-v0.15.0) - 2026-07-17

### Added

- associative recall via graph expansion (gated, default off) ([#258](https://github.com/nambok/mentedb/pull/258))

### Other

- end-to-end retention probe for the memory lifecycle ([#257](https://github.com/nambok/mentedb/pull/257))

## [0.14.7](https://github.com/nambok/mentedb/compare/mentedb-v0.14.5...mentedb-v0.14.7) - 2026-07-17

### Added

- add reset_decay_state to repair salience corrupted by compounding decay ([#252](https://github.com/nambok/mentedb/pull/252))

### Other

- release v0.14.6 ([#251](https://github.com/nambok/mentedb/pull/251))

## [0.14.6](https://github.com/nambok/mentedb/compare/mentedb-v0.14.5...mentedb-v0.14.6) - 2026-07-17

### Added

- add reset_decay_state to repair salience corrupted by compounding decay ([#252](https://github.com/nambok/mentedb/pull/252))

## [0.14.5](https://github.com/nambok/mentedb/compare/mentedb-v0.14.4...mentedb-v0.14.5) - 2026-07-17

### Fixed

- derive salience decay on read instead of persisting it ([#249](https://github.com/nambok/mentedb/pull/249))

## [0.14.4](https://github.com/nambok/mentedb/compare/mentedb-v0.14.2...mentedb-v0.14.4) - 2026-07-16

### Added

- engine-native standing-rules cleanup (prune_standing_rules) ([#243](https://github.com/nambok/mentedb/pull/243))

### Other

- release v0.14.3 ([#241](https://github.com/nambok/mentedb/pull/241))

## [0.14.3](https://github.com/nambok/mentedb/compare/mentedb-v0.14.2...mentedb-v0.14.3) - 2026-07-16

### Added

- engine-native standing-rules cleanup (prune_standing_rules) ([#243](https://github.com/nambok/mentedb/pull/243))

## [0.14.2](https://github.com/nambok/mentedb/compare/mentedb-v0.14.0...mentedb-v0.14.2) - 2026-07-15

### Added

- owner-scoped user profile accessors (user_profile_for, profile_owners) ([#237](https://github.com/nambok/mentedb/pull/237))

### Other

- release v0.14.1 ([#238](https://github.com/nambok/mentedb/pull/238))

## [0.14.1](https://github.com/nambok/mentedb/compare/mentedb-v0.14.0...mentedb-v0.14.1) - 2026-07-15

### Added

- owner-scoped user profile accessors (user_profile_for, profile_owners) ([#237](https://github.com/nambok/mentedb/pull/237))

## [0.14.0](https://github.com/nambok/mentedb/compare/mentedb-v0.13.1...mentedb-v0.14.0) - 2026-07-15

### Added

- add AWS Bedrock as a self-host LLM provider (feature-gated, SigV4) ([#234](https://github.com/nambok/mentedb/pull/234))

### Other

- release v0.14.0 ([#233](https://github.com/nambok/mentedb/pull/233))

## [0.13.1](https://github.com/nambok/mentedb/compare/mentedb-v0.12.6...mentedb-v0.13.1) - 2026-07-15

### Fixed

- scope enrichment, profile, entity-linking and community detection by agent (full memory isolation) ([#229](https://github.com/nambok/mentedb/pull/229))

### Other

- release v0.13.0 ([#230](https://github.com/nambok/mentedb/pull/230))

## [0.13.0](https://github.com/nambok/mentedb/compare/mentedb-v0.12.6...mentedb-v0.13.0) - 2026-07-15

### Fixed

- scope enrichment, profile, entity-linking and community detection by agent (full memory isolation) ([#229](https://github.com/nambok/mentedb/pull/229))

## [0.12.6](https://github.com/nambok/mentedb/compare/mentedb-v0.12.4...mentedb-v0.12.6) - 2026-07-15

### Fixed

- prevent cross-user memory merge in consolidation (scope clustering by agent) ([#227](https://github.com/nambok/mentedb/pull/227))

### Other

- release v0.12.5 ([#226](https://github.com/nambok/mentedb/pull/226))

## [0.12.5](https://github.com/nambok/mentedb/compare/mentedb-v0.12.4...mentedb-v0.12.5) - 2026-07-15

### Fixed

- prevent cross-user memory merge in consolidation (scope clustering by agent) ([#227](https://github.com/nambok/mentedb/pull/227))

## [0.12.4](https://github.com/nambok/mentedb/compare/mentedb-v0.12.3...mentedb-v0.12.4) - 2026-07-15

### Fixed

- prevent cross-user memory leak in process_turn proactive recall and write inference ([#224](https://github.com/nambok/mentedb/pull/224))

## [0.12.3](https://github.com/nambok/mentedb/compare/mentedb-v0.12.2...mentedb-v0.12.3) - 2026-07-15

### Other

- update architecture diagram (embedding providers incl. Bedrock, consolidation, injection/attention, trajectory + speculative) ([#220](https://github.com/nambok/mentedb/pull/220))
- add AWS Bedrock embeddings to the feature list ([#216](https://github.com/nambok/mentedb/pull/216))

## [0.12.1](https://github.com/nambok/mentedb/compare/mentedb-v0.11.14...mentedb-v0.12.1) - 2026-07-13

### Fixed

- *(cognitive)* dedup exact copies with a Derived edge not Supersedes; tighten phantom extraction to reject fragments and data values ([#206](https://github.com/nambok/mentedb/pull/206))

### Other

- release v0.12.0 ([#207](https://github.com/nambok/mentedb/pull/207))

## [0.12.0](https://github.com/nambok/mentedb/compare/mentedb-v0.11.14...mentedb-v0.12.0) - 2026-07-13

### Fixed

- *(cognitive)* dedup exact copies with a Derived edge not Supersedes; tighten phantom extraction to reject fragments and data values ([#206](https://github.com/nambok/mentedb/pull/206))

## [0.11.14](https://github.com/nambok/mentedb/compare/mentedb-v0.11.13...mentedb-v0.11.14) - 2026-07-13

### Added

- *(cognitive)* LLM-judged contradiction/supersession detection, replacing the ~0%-precision cosine heuristic ([#203](https://github.com/nambok/mentedb/pull/203))

## [0.11.13](https://github.com/nambok/mentedb/compare/mentedb-v0.11.12...mentedb-v0.11.13) - 2026-07-12

### Added

- canonicalize trajectory topics off the hot path so the speculative cache stops keying on raw message text ([#201](https://github.com/nambok/mentedb/pull/201))

### Other

- use the clean /v1/process_turn cloud endpoint in all examples ([#200](https://github.com/nambok/mentedb/pull/200))
- fix MenteDBChatHistory example, session_id is a required argument ([#196](https://github.com/nambok/mentedb/pull/196))

## [0.11.12](https://github.com/nambok/mentedb/compare/mentedb-v0.11.11...mentedb-v0.11.12) - 2026-07-11

### Added

- *(sdk)* expose processTurn/ingest in Node SDK; rewrite onboarding docs around building an agent ([#195](https://github.com/nambok/mentedb/pull/195))

## [0.11.11](https://github.com/nambok/mentedb/compare/mentedb-v0.11.10...mentedb-v0.11.11) - 2026-07-11

### Added

- server maintenance runner + speculative cache entries accessor ([#190](https://github.com/nambok/mentedb/pull/190))

## [0.11.9](https://github.com/nambok/mentedb/compare/mentedb-v0.11.8...mentedb-v0.11.9) - 2026-07-11

### Other

- maintain EntityResolver canonical set + disambiguate rule matches (closes #45, #46) ([#182](https://github.com/nambok/mentedb/pull/182))

## [0.11.8](https://github.com/nambok/mentedb/compare/mentedb-v0.11.7...mentedb-v0.11.8) - 2026-07-11

### Other

- live deps.rs dependency-status badge; group Dependabot updates into 1-2 PRs ([#169](https://github.com/nambok/mentedb/pull/169))
- add Dependabot badge to README ([#149](https://github.com/nambok/mentedb/pull/149))

## [0.11.7](https://github.com/nambok/mentedb/compare/mentedb-v0.11.5...mentedb-v0.11.7) - 2026-07-11

### Added

- engine-native LLM memory consolidation (MenteDb::consolidate_memories) ([#128](https://github.com/nambok/mentedb/pull/128))

### Other

- release v0.11.6 ([#129](https://github.com/nambok/mentedb/pull/129))

## [0.11.6](https://github.com/nambok/mentedb/compare/mentedb-v0.11.5...mentedb-v0.11.6) - 2026-07-11

### Added

- engine-native LLM memory consolidation (MenteDb::consolidate_memories) ([#128](https://github.com/nambok/mentedb/pull/128))

## [0.11.5](https://github.com/nambok/mentedb/compare/mentedb-v0.11.4...mentedb-v0.11.5) - 2026-07-11

### Fixed

- remove redundant format arg borrows (clippy 1.97 useless_borrows_in_formatting) ([#125](https://github.com/nambok/mentedb/pull/125))

### Other

- correct LongMemEval to the real 92.0% (460/500), replace cherry-picked composite with the honest baseline run + committed judge labels ([#124](https://github.com/nambok/mentedb/pull/124))

## [0.11.4](https://github.com/nambok/mentedb/compare/mentedb-v0.11.3...mentedb-v0.11.4) - 2026-07-10

### Fixed

- decay actually affects recall ranking (recompute decayed salience in hybrid recall hot path) ([#121](https://github.com/nambok/mentedb/pull/121))

## [0.11.3](https://github.com/nambok/mentedb/compare/mentedb-v0.11.1...mentedb-v0.11.3) - 2026-07-06

### Added

- close_quick for fast lock release on shutdown ([#119](https://github.com/nambok/mentedb/pull/119))

### Other

- release v0.11.2 ([#118](https://github.com/nambok/mentedb/pull/118))

## [0.11.2](https://github.com/nambok/mentedb/compare/mentedb-v0.11.1...mentedb-v0.11.2) - 2026-07-06

### Added

- close_quick for fast lock release on shutdown ([#119](https://github.com/nambok/mentedb/pull/119))

## [0.11.1](https://github.com/nambok/mentedb/compare/mentedb-v0.11.0...mentedb-v0.11.1) - 2026-07-06

### Fixed

- ghost memories never inject as knowledge ([#116](https://github.com/nambok/mentedb/pull/116))

## [0.11.0](https://github.com/nambok/mentedb/compare/mentedb-v0.10.8...mentedb-v0.11.0) - 2026-07-06

### Other

- amortize snapshot writes across flushes, reconcile stale snapshots at open ([#113](https://github.com/nambok/mentedb/pull/113))

## [0.10.8](https://github.com/nambok/mentedb/compare/mentedb-v0.10.6...mentedb-v0.10.8) - 2026-07-06

### Fixed

- integration tests carry the new InjectionQuery agent_id field ([#110](https://github.com/nambok/mentedb/pull/110))

### Other

- release v0.10.6 ([#109](https://github.com/nambok/mentedb/pull/109))

## [0.10.7](https://github.com/nambok/mentedb/compare/mentedb-v0.10.6...mentedb-v0.10.7) - 2026-07-06

### Fixed

- integration tests carry the new InjectionQuery agent_id field ([#110](https://github.com/nambok/mentedb/pull/110))

## [0.10.5](https://github.com/nambok/mentedb/compare/mentedb-v0.10.4...mentedb-v0.10.5) - 2026-07-06

### Fixed

- installable on debian and arm linux, publish mentedb-server ([#102](https://github.com/nambok/mentedb/pull/102))

## [0.10.3](https://github.com/nambok/mentedb/compare/mentedb-v0.10.2...mentedb-v0.10.3) - 2026-07-05

### Fixed

- enforce single process ownership with an exclusive lock held from open to close ([#97](https://github.com/nambok/mentedb/pull/97))

## [0.10.2](https://github.com/nambok/mentedb/compare/mentedb-v0.10.1...mentedb-v0.10.2) - 2026-07-04

### Fixed

- correction detection scans only the user message and tags the episodic turn instead of storing a verbatim semantic node ([#96](https://github.com/nambok/mentedb/pull/96))

## [0.10.1](https://github.com/nambok/mentedb/compare/mentedb-v0.10.0...mentedb-v0.10.1) - 2026-07-03

### Other

- lead with Claude Code hooks and connector integration paths, bump benchmark version to 0.10.0 ([#92](https://github.com/nambok/mentedb/pull/92))

## [0.10.0](https://github.com/nambok/mentedb/compare/mentedb-v0.9.2...mentedb-v0.10.0) - 2026-07-03

### Fixed

- dedup write-inference edges, make similarity bands exclusive, run inference in store_batch
- wire embedding provider into server and stop silent zero-vector fallback in process_turn
- crash-durable graph edges via edge log and rebuild graph nodes from storage on open
- make forget durable via WAL-logged page free and update memories in place

### Other

- remove unimplemented vscode setup target from README
- align README and architecture docs with actual engine behavior

## [0.9.2](https://github.com/nambok/mentedb/compare/mentedb-v0.9.1...mentedb-v0.9.2) - 2026-05-13

### Added

- benchmark improvements + engine fixes

### Fixed

- resolve clippy and fmt warnings in engine/SDK

### Other

- update LongMemEval benchmark results to 95.2% (476/500)

## [0.9.0](https://github.com/nambok/mentedb/compare/mentedb-v0.8.2...mentedb-v0.9.0) - 2026-04-27

### Fixed

- WAL durability, page checksums, graph cleanup, salience O(1)

## [0.8.2](https://github.com/nambok/mentedb/compare/mentedb-v0.8.1...mentedb-v0.8.2) - 2026-04-26

### Added

- add enrichment orchestrator module to engine

### Other

- Merge pull request #76 from nambok/feat/enrichment-phase3-4
- fix cargo fmt in enrichment.rs
- add sleeptime enrichment to READMEs

## [0.8.1](https://github.com/nambok/mentedb/compare/mentedb-v0.8.0...mentedb-v0.8.1) - 2026-04-26

### Added

- add community detection and user model generation (Phase 3 & 4)

### Fixed

- address code review findings for Phase 3 & 4

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
