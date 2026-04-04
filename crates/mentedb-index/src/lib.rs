//! MenteDB Index — high-performance index structures for memory retrieval.
//!
//! This crate provides:
//! - HNSW vector index for approximate nearest neighbor search
//! - Roaring bitmap indexes for tag and attribute filtering
//! - Temporal index for timestamp range queries
//! - Salience index for top-k retrieval by importance
//! - Composite index manager for hybrid search

pub mod bitmap;
pub mod hnsw;
pub mod manager;
pub mod salience;
pub mod temporal;

pub use bitmap::BitmapIndex;
pub use hnsw::{DistanceMetric, HnswIndex};
pub use manager::IndexManager;
pub use salience::SalienceIndex;
pub use temporal::TemporalIndex;
