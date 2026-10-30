//! MenteDB Storage Engine: compact record storage with WAL and record cache.
//!
//! This crate implements the low-level storage engine:
//! - **Record file**: variable length memory records packed into 512 byte
//!   blocks, so the file is about the size of the data
//! - **Locator table**: stable record ids resolved in memory, persisted at
//!   checkpoints and replayed forward from the WAL on open
//! - **Write-ahead log (WAL)**: append-only, CRC-checked, LZ4-compressed
//!   entries that redo every record write byte for byte
//! - **Record cache**: decoded records bounded by a configurable byte budget
//! - **Migration**: databases in the old 64 KB page layout are compacted into
//!   the record layout on open, atomically
//!
//! # Concurrency
//!
//! One process per database directory, enforced by an exclusive lock on a
//! LOCK file (fcntl OFD lock, honored across hosts on NFSv4). Writes are
//! serialized inside the process; reads run concurrently.
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
/// Byte bounded cache of decoded records with CLOCK eviction.
pub mod cache;
/// Unified storage facade for memory persistence.
pub mod engine;
/// Migration from the 64 KB page file layout.
pub mod legacy;
/// Cross-host single-writer lock (fcntl OFD locks on Linux/EFS).
mod lock;
/// Persisted locator table, replayed forward from the WAL on open.
pub mod memory_map;
/// Record file format: extents of 512 byte blocks, payload codec, allocator.
pub mod record;
/// Append only write ahead log with CRC checks and LZ4 compression.
pub mod wal;

// Re-export key types at crate root for convenience.
pub use cache::{BufferStats, RecordCache};
pub use engine::StorageEngine;
pub use legacy::MigrationReport;
pub use memory_map::MappedMemory;
pub use record::{BLOCK_SIZE, PageId};
pub use wal::{Lsn, Wal, WalEntry, WalEntryType};
