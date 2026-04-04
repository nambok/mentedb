# Contributing to MenteDB

Thank you for your interest in contributing to MenteDB. This guide covers
everything you need to get started.

## Prerequisites

- Rust stable toolchain (1.85+, edition 2024)
- `rustfmt` and `clippy` components installed
- Git

Install the toolchain and components:

```bash
rustup toolchain install stable
rustup component add rustfmt clippy
```

## Building

Clone the repository and build the workspace:

```bash
git clone https://github.com/nambok/mentedb.git
cd mentedb
cargo build --workspace
```

## Running Tests

Run all tests across the workspace:

```bash
cargo test --workspace
```

Run tests for a specific crate:

```bash
cargo test -p mentedb-core
cargo test -p mentedb-storage
cargo test -p mentedb-cognitive
```

## Running Benchmarks

Benchmarks use the `criterion` framework:

```bash
cargo bench --workspace
```

Run a specific benchmark:

```bash
cargo bench -p mentedb --bench storage_bench
```

## Code Style

### Formatting

All code must be formatted with `rustfmt`:

```bash
cargo fmt --all
```

Check formatting without modifying files:

```bash
cargo fmt --all -- --check
```

### Linting

All code must pass `clippy` with no warnings:

```bash
cargo clippy --workspace -- -D warnings
```

### Documentation

All public types, methods, and modules must have `///` doc comments.
Module level docs use `//!`. Run `cargo doc --no-deps` to verify
documentation builds cleanly.

### General Style

- Use `snake_case` for functions, methods, and variables.
- Use `CamelCase` for types, traits, and enum variants.
- Prefer returning `MenteResult<T>` over panicking.
- Keep functions focused. If a function exceeds 50 lines, consider
  splitting it.
- Write comments only when the code is not self explanatory.

## Commit Conventions

Use conventional commit messages:

```
feat: add phantom memory priority scoring
fix: correct HNSW distance calculation for zero vectors
chore: update uuid dependency to 1.12
docs: add context assembly section to ARCHITECTURE.md
test: add integration tests for WAL crash recovery
refactor: extract common traversal logic into helper
```

Rules:

- One line summary, 72 characters or fewer.
- Use a conventional prefix: `feat:`, `fix:`, `chore:`, `docs:`,
  `test:`, `refactor:`, `perf:`, `ci:`.
- No emojis.
- No dashes in the summary line.
- Use the imperative mood ("add", not "added" or "adds").
- If a longer explanation is needed, add a blank line after the summary
  followed by body paragraphs.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes in small, focused commits.
3. Ensure `cargo fmt --all -- --check` passes.
4. Ensure `cargo clippy --workspace -- -D warnings` passes.
5. Ensure `cargo test --workspace` passes.
6. Ensure `cargo doc --no-deps` builds without warnings.
7. Open a pull request against `main`.
8. Describe what the PR does and why.
9. Link any related issues.

Pull requests are reviewed for correctness, performance, and adherence
to the project's architectural principles. Be prepared for feedback and
iteration.

## Architecture Overview

Read [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed explanation of the
system design, crate structure, data flow, and design decisions.

The key architectural principle: each crate owns a single responsibility
and depends only on `mentedb-core` for shared types. The `mentedb` facade
crate wires everything together. Changes to one subsystem should not
require changes to another.

## How to Add a New Cognitive Feature

MenteDB's cognitive features live in `crates/mentedb-cognitive/`. To add
a new feature:

### 1. Create the Module

Add a new file in `crates/mentedb-cognitive/src/`, for example
`my_feature.rs`.

### 2. Define the Types

Follow the existing pattern: a config struct, a main struct, and result
types:

```rust
pub struct MyFeatureConfig {
    pub threshold: f32,
}

pub struct MyFeature {
    config: MyFeatureConfig,
}

impl MyFeature {
    pub fn new(config: MyFeatureConfig) -> Self {
        Self { config }
    }

    pub fn analyze(&self, memories: &[MemoryNode]) -> Vec<MyResult> {
        // implementation
    }
}
```

### 3. Register in lib.rs

Add the module declaration and re-exports in
`crates/mentedb-cognitive/src/lib.rs`:

```rust
pub mod my_feature;
pub use my_feature::{MyFeature, MyFeatureConfig, MyResult};
```

### 4. Add Configuration

If the feature has configurable parameters, add them to
`CognitiveConfig` in `crates/mentedb-core/src/config.rs`.

### 5. Write Tests

Every module must have:

- Unit tests in a `#[cfg(test)]` block at the bottom of the file
- At least one integration test in the crate's `tests/` directory

Test the happy path, edge cases (empty input, single element, maximum
size), and error conditions.

### 6. Document

Add `///` doc comments to every public type and method. Add an entry
in ARCHITECTURE.md under the Cognitive Features section explaining
what the feature does and why it matters.

## Testing Requirements

### Unit Tests

Every module must have a `#[cfg(test)]` block with tests covering:

- Normal operation (happy path)
- Edge cases (empty input, boundary values, maximum capacity)
- Error conditions (invalid input, resource exhaustion)

### Integration Tests

Each crate should have integration tests in its `tests/` directory
that exercise the public API end to end. Integration tests should not
depend on internal implementation details.

### Test Naming

Use descriptive test names that explain what is being tested:

```rust
#[test]
fn store_and_recall_returns_stored_memory() { ... }

#[test]
fn forget_nonexistent_memory_returns_not_found() { ... }

#[test]
fn hnsw_search_returns_k_nearest_neighbors() { ... }
```

### Running Specific Tests

```bash
cargo test -p mentedb-core test_name
cargo test -p mentedb-storage -- --nocapture
```

## Questions

Open an issue on GitHub if you have questions about contributing.
