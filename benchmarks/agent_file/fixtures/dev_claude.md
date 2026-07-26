# Nimbus Development Instructions

Nimbus is a Rust workspace for a real time collaboration server.

## Project overview

The server handles document sync over WebSocket, persistence on Postgres,
and presence broadcasting. Latency budget is 50ms end to end.

## Commands

Always run all three before committing changes, CI runs the same gates:
```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

## Conventions

- Commit messages: conventional style (feat:, fix:, chore:), single line, no emojis
- PR descriptions use Summary and Verification sections, bullets not prose walls
- Use thiserror for error types, never anyhow in library code
- No unwrap() in library code, tests may unwrap
- All heuristics and thresholds must live in Config structs, never hardcoded magic numbers
- No emojis or em dashes in prose anywhere, docs, comments, commit messages included
- Rust edition 2024, Apache 2.0 license

## Architecture notes

- Storage is a write ahead log with 64KB pages and CRC32 checks
- Presence uses a gossip protocol with a 3 second heartbeat
- The sync engine resolves conflicts with CRDTs, last writer wins is forbidden

## Do not

- Never force push to main
- Never commit secrets or .env files
- Never disable a failing test to make CI pass
