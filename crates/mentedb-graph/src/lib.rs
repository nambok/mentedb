//! MenteDB Graph — knowledge graph engine with CSR/CSC storage.
//!
//! This crate provides:
//! - Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) graph storage
//! - Typed, weighted edge traversal
//! - Belief propagation and cascade updates
//! - Subgraph extraction for context assembly

pub mod belief;
pub mod contradiction;
pub mod csr;
pub mod manager;
pub mod traversal;

pub use belief::{PropagationConfig, propagate_update, propagate_update_with_config};
pub use contradiction::{detect_cycles, find_contradictions, find_superseded};
pub use csr::{CsrGraph, StoredEdge};
pub use manager::GraphManager;
pub use traversal::{bfs, bfs_filtered, dfs, extract_subgraph, shortest_path};
