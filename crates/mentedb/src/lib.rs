//! # MenteDB — The Mind Database for AI Agents
//!
//! MenteDB is a purpose-built database engine for AI agent memory.
//! It's a cognition preparation engine that pre-digests knowledge
//! for single-pass transformer consumption.
//!
//! ## Core Concepts
//!
//! - **MemoryNode**: The atomic unit of knowledge (embeddings + graph + temporal + attributes)
//! - **MemoryEdge**: Typed, weighted relationships between memories
//! - **MemoryTier**: Cognitive-inspired storage hierarchy (working → episodic → semantic → procedural → archival)
//! - **Context Assembly**: Token-budget-aware context building that respects attention patterns
//! - **MQL**: Mente Query Language for memory retrieval and manipulation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mentedb::prelude::*;
//!
//! // Coming soon...
//! ```

pub use mentedb_core as core;

/// Commonly used types, re-exported for convenience.
pub mod prelude {
    pub use mentedb_core::edge::EdgeType;
    pub use mentedb_core::memory::MemoryType;
    pub use mentedb_core::types::*;
    pub use mentedb_core::{MenteError, MemoryEdge, MemoryNode, MemoryTier};
}
