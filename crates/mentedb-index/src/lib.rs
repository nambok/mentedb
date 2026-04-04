//! MenteDB Index: high performance index structures for memory retrieval.
//!
//! This crate provides:
//! - HNSW vector index for approximate nearest neighbor search
//! - Roaring bitmap indexes for tag and attribute filtering
//! - Temporal index for timestamp range queries
//! - Salience index for top-k retrieval by importance
//! - Composite index manager for hybrid search

/// Roaring bitmap indexes for tag and attribute filtering.
pub mod bitmap;
/// HNSW vector index for approximate nearest neighbor search.
pub mod hnsw;
/// Composite index manager for hybrid search across all index types.
pub mod manager;
/// Salience index for top k retrieval by importance score.
pub mod salience;
/// Temporal index for timestamp range queries.
pub mod temporal;

pub use bitmap::BitmapIndex;
pub use hnsw::{DistanceMetric, HnswIndex};
pub use manager::IndexManager;
pub use salience::SalienceIndex;
pub use temporal::TemporalIndex;
