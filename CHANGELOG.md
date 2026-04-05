# Changelog

## [0.2.2](https://github.com/nambok/mentedb/compare/mentedb-v0.2.1...mentedb-v0.2.2) (2026-04-05)


### Features

* add Anthropic support to quality benchmarks ([7d3e305](https://github.com/nambok/mentedb/commit/7d3e305c0f33a673631e8b6edbd8fc8a8b05ba69))
* add CLAUDE.md, copilot-instructions.md, and copilot-setup-steps.yml for AI agent support ([1f9b814](https://github.com/nambok/mentedb/commit/1f9b814956e76ae7ed6b39aec0ab33b16aa60c57))
* add cognitive engine (7 modules), database facade, and server binary ([38ba235](https://github.com/nambok/mentedb/commit/38ba2355b921b049233f122de262091126d9f401))
* add comprehensive docs.rs documentation across all crates ([a1d61a5](https://github.com/nambok/mentedb/commit/a1d61a5e76ae5c57bf2b2bcbcbdea4f32265f4bb))
* add config system, benchmarks, docs and make all heuristics configurable ([9e863a5](https://github.com/nambok/mentedb/commit/9e863a51d709724f7c5d4895a456ad81c80cbf6c))
* add context assembly engine and MQL query parser with planner ([241e4ee](https://github.com/nambok/mentedb/commit/241e4eebd5cf526cbd7ad1ed817338bac5a09c02))
* add criterion performance benchmarks ([cee517a](https://github.com/nambok/mentedb/commit/cee517aa14602a48d8b015b378cc190245eb3b05))
* add embedding crate, WAL crash recovery, entity API, and clippy fixes ([0e90a80](https://github.com/nambok/mentedb/commit/0e90a80d4910e43d65bbea4d383ac823c44896aa))
* add gRPC streaming, dashboard UI, Dockerfile, PyPI trusted publisher, README badges, and npm OIDC publish ([aa97fdf](https://github.com/nambok/mentedb/commit/aa97fdfe87c07704eda3d1842692ae555ed18f8a))
* add index crate (HNSW, bitmap, temporal, salience) and graph crate (CSR, traversal, belief propagation) ([bd921aa](https://github.com/nambok/mentedb/commit/bd921aafa5c5a6d81b38b31e59a5730ad2c1813b))
* add mentedb-extraction crate for LLM-powered memory extraction ([217a396](https://github.com/nambok/mentedb/commit/217a3964ddca4fdf4cb6eb384754d973d3602e0e))
* add multi agent collaboration (MVCC, spaces, events) and consolidation pipeline (decay, archival, GDPR) ([9962ffe](https://github.com/nambok/mentedb/commit/9962ffe46645d0d63d6118f7b5b98606d72a8bd3))
* add Python/TypeScript SDKs, LangChain/CrewAI integrations, backup/restore, resource limits, metrics, and release pipeline ([e8722f0](https://github.com/nambok/mentedb/commit/e8722f051512093760a3c407fc42683f9eadd527))
* add quality benchmark suite (stale belief, delta savings, attention, noise) ([8f68173](https://github.com/nambok/mentedb/commit/8f68173ed6da75c0dbf12e5ad1c53f81b9f7fa11))
* add release-please for automated versioning and releases ([cfb494e](https://github.com/nambok/mentedb/commit/cfb494ecb8317256359c12fc69ba5e9f65f96b6f))
* add REST API server, index/graph persistence, 8 scenario tests, docs, examples, and CI ([a1f0da3](https://github.com/nambok/mentedb/commit/a1f0da3aa14c7c783a168c956690cc4cdf4dc25d))
* add server integration tests, realistic scenario tests, and crates.io publish metadata ([57ddad3](https://github.com/nambok/mentedb/commit/57ddad30e01754a6119acd0ac5ff32ac00d6be82))
* add storage engine with page manager, WAL, buffer pool and engine facade ([a1bd5ff](https://github.com/nambok/mentedb/commit/a1bd5ffba3681828df8dbe84566239b9ed4fccc2))
* scaffold rust workspace with 8 crates ([76a1ab1](https://github.com/nambok/mentedb/commit/76a1ab1480834c0e02be1818a2d2d5d3d2570a8c))
* upgrade server to axum with JWT auth, rate limiting, and WebSocket ([4d033eb](https://github.com/nambok/mentedb/commit/4d033eb6102f39b37c5d173b5f4f01de14e227ed))
* wire extraction into REST API, auto-extract mode, LLM provider config ([99ef014](https://github.com/nambok/mentedb/commit/99ef014a63deb378aea7a270f3a15f1ee8b4f830))


### Bug Fixes

* add NODE_AUTH_TOKEN for npm publish ([e0d5d24](https://github.com/nambok/mentedb/commit/e0d5d24669c3dcfbb107465b8666be3853a3163c))
* address audit findings (partial, server in progress) ([cf9549d](https://github.com/nambok/mentedb/commit/cf9549dc38ba40b9a9ca6c8f97fc1caab5e16667))
* bump pyproject.toml to v0.2.0 ([c8fd37d](https://github.com/nambok/mentedb/commit/c8fd37d06dfb6f5aefd53700a049980309cebbbc))
* cargo fmt on benchmarks ([566d5a4](https://github.com/nambok/mentedb/commit/566d5a4b0e2bb9c067fb22b7de552143f127a852))
* clean up newtype migration, all 363 tests passing ([34585fb](https://github.com/nambok/mentedb/commit/34585fb813fdf90d2a5227a7a0e2f4af11dfb660))
* clean up unused imports and variables in tests ([9285adf](https://github.com/nambok/mentedb/commit/9285adf605520b7ffbe32fab23b9a0696f471382))
* complete audit remediation, 427 tests passing ([f6794a2](https://github.com/nambok/mentedb/commit/f6794a2616485f62d509852279cc3d5603e458ee))
* install protobuf-compiler in CI, release, Dockerfile, and copilot setup ([ea79877](https://github.com/nambok/mentedb/commit/ea79877653e7a2b660469cfb52b098fd3bac7a41))
* resolve clippy warnings in replication and salience crates ([cf8bbc8](https://github.com/nambok/mentedb/commit/cf8bbc86e4dd2871b184ca20830b3afcb6342088))
* space ACLs, embedding validation, cleanup unused imports ([ddddbd3](https://github.com/nambok/mentedb/commit/ddddbd31740878b6628aa122b6765ef0f12cd164))
* switch release-please to simple type for workspace versioning ([e35ade5](https://github.com/nambok/mentedb/commit/e35ade50af0935170a89ea3ab76a086b04cebbb1))
* update Python SDK for newtype IDs ([9f3b13b](https://github.com/nambok/mentedb/commit/9f3b13b6e43aa451c0fe0adb3eb64a4cccbd5381))
* use static license badge and mentedb-core for crates.io badges ([9e517b4](https://github.com/nambok/mentedb/commit/9e517b442c309d1130e50e0fd4775ddb2ad9b13d))
