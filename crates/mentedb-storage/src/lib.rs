//! MenteDB Storage Engine: page-based storage with WAL and buffer pool.
//!
//! This crate implements the low-level storage engine:
//! - **Page manager**: file-backed 64KB pages with free-list allocation
//! - **Write-ahead log (WAL)**: append-only, CRC-checked, LZ4-compressed entries
//! - **Buffer pool**: fixed-capacity page cache with CLOCK eviction
//! - **Storage engine**: unified facade for memory node persistence
//!
//! # Concurrency
//!
//! Multiple processes can open the same database directory simultaneously.
//! Writes are serialized via `flock(2)` on the WAL file; reads are lock-free.
//! In-memory state (page count, LSN counter) is refreshed from disk under the
//! flock so no process acts on stale data.
//!
//! # Example
//!
//! ```no_run
//! use mentedb_storage::StorageEngine;
//! use mentedb_core::{MemoryNode, memory::MemoryType, types::AgentId};
//!
//! let engine = StorageEngine::open("/tmp/mentedb-data".as_ref())?;
//!
//! let node = MemoryNode::new(
//!     AgentId::new(),
//!     MemoryType::Episodic,
//!     "The user prefers Rust".to_string(),
//!     vec![0.1, 0.2, 0.3],
//! );
//!
//! let page_id = engine.store_memory(&node)?;
//! let loaded = engine.load_memory(page_id)?;
//! assert_eq!(loaded.content, "The user prefers Rust");
//!
//! engine.checkpoint()?;
//! engine.close()?;
//! # Ok::<(), mentedb_core::error::MenteError>(())
//! ```

/// Snapshot backup utilities.
pub mod backup;
/// Fixed capacity page cache with CLOCK eviction.
pub mod buffer;
/// Unified storage facade for memory persistence.
pub mod engine;
/// File backed 16KB page manager with free list allocation.
pub mod page;
/// Append only write ahead log with CRC checks and LZ4 compression.
pub mod wal;

// Re-export key types at crate root for convenience.
pub use buffer::BufferPool;
pub use engine::StorageEngine;
pub use page::{PAGE_DATA_SIZE, PAGE_SIZE, Page, PageHeader, PageId, PageType};
pub use wal::{Lsn, Wal, WalEntry, WalEntryType};
