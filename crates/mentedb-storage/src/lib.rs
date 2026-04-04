//! MenteDB Storage Engine — page-based storage with WAL and buffer pool.
//!
//! This crate implements the low-level storage engine:
//! - Page manager with buddy allocation
//! - Write-ahead log (WAL) with group commit
//! - Buffer pool with CLOCK eviction
//! - Direct I/O integration
