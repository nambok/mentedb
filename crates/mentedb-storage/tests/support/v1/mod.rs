//! The page file storage engine exactly as released in mentedb-storage
//! 0.44.0 (format 1: one 64 KB page per memory), used only to build old
//! format databases for the migration tests and the layout benchmark.
//!
//! The files are the released sources with two mechanical changes: module
//! paths rewritten from `crate::` to `super::`, and their unit tests removed.
//! Do not edit them; they are the fixture generator, so they must keep
//! writing exactly what the released engine wrote.

pub mod buffer;
pub mod engine;
mod lock;
pub mod memory_map;
pub mod page;
pub mod wal;

pub use engine::StorageEngine;
pub use page::{PAGE_SIZE, PageId};
