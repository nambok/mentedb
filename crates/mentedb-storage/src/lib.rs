//! MenteDB Storage Engine — page-based storage with WAL and buffer pool.
//!
//! This crate implements the low-level storage engine:
//! - **Page manager** — file-backed 16KB pages with free-list allocation
//! - **Write-ahead log (WAL)** — append-only, CRC-checked, LZ4-compressed entries
//! - **Buffer pool** — fixed-capacity page cache with CLOCK eviction
//! - **Storage engine** — unified facade for memory node persistence

pub mod backup;
pub mod buffer;
pub mod engine;
pub mod page;
pub mod wal;

// Re-export key types at crate root for convenience.
pub use buffer::BufferPool;
pub use engine::StorageEngine;
pub use page::{PAGE_DATA_SIZE, PAGE_SIZE, Page, PageHeader, PageId, PageType};
pub use wal::{Lsn, Wal, WalEntry, WalEntryType};
