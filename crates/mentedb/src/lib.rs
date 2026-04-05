//! # MenteDB: The Mind Database for AI Agents
//!
//! MenteDB is a purpose-built database engine for AI agent memory.
//! It's a cognition preparation engine that pre-digests knowledge
//! for single-pass transformer consumption.
//!
//! ## Core Concepts
//!
//! - **MemoryNode**: The atomic unit of knowledge (embeddings, graph, temporal, attributes)
//! - **MemoryEdge**: Typed, weighted relationships between memories
//! - **MemoryTier**: Cognitive inspired storage hierarchy (working, episodic, semantic, procedural, archival)
//! - **Context Assembly**: Token budget aware context building that respects attention patterns
//! - **MQL**: Mente Query Language for memory retrieval and manipulation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mentedb::prelude::*;
//! use mentedb::MenteDb;
//! use std::path::Path;
//!
//! let mut db = MenteDb::open(Path::new("./my-agent-memory")).unwrap();
//! // store, recall, relate, forget...
//! db.close().unwrap();
//! ```
//!
//! ## Feature Highlights
//!
//! - Seven cognitive features: interference detection, pain signals, phantom tracking,
//!   speculative caching, stream monitoring, trajectory tracking, write inference
//! - HNSW vector index with hybrid search (vector + tags + temporal + salience)
//! - CSR/CSC knowledge graph with belief propagation
//! - Token budget aware context assembly with attention curve optimization
//! - MQL query language with vector, tag, temporal, and graph traversal support
//! - WAL based crash recovery with LZ4 compressed pages
//!
//! ## Repository
//!
//! Source code: <https://github.com/nambok/mentedb>

use std::path::{Path, PathBuf};

use mentedb_context::{AssemblyConfig, ContextAssembler, ContextWindow, ScoredMemory};
use mentedb_core::error::MenteResult;
use mentedb_core::types::MemoryId;
use mentedb_core::{MemoryEdge, MemoryNode, MenteError};
use mentedb_graph::GraphManager;
use mentedb_index::IndexManager;
use mentedb_query::{Mql, QueryPlan};
use mentedb_storage::StorageEngine;
use tracing::{debug, info};

// Re-export sub-crates for direct access.
/// Context assembly engine.
pub use mentedb_context as context;
/// Core types: MemoryNode, MemoryEdge, errors, config.
pub use mentedb_core as core;
/// Knowledge graph engine.
pub use mentedb_graph as graph;
/// Index structures for vector, tag, temporal, and salience search.
pub use mentedb_index as index;
/// MQL parser and query planner.
pub use mentedb_query as query;
/// Page based storage engine with WAL and buffer pool.
pub use mentedb_storage as storage;

/// Commonly used types, re-exported for convenience.
pub mod prelude {
    pub use mentedb_core::edge::EdgeType;
    pub use mentedb_core::error::MenteResult;
    pub use mentedb_core::memory::MemoryType;
    pub use mentedb_core::types::*;
    pub use mentedb_core::{MemoryEdge, MemoryNode, MemoryTier, MenteError};

    pub use crate::MenteDb;
}

use mentedb_storage::PageId;
/// Mapping from MemoryId to the storage PageId where it lives.
use std::collections::HashMap;

/// The unified database facade for MenteDB.
///
/// `MenteDb` coordinates storage, indexing, graph relationships, query parsing,
/// and context assembly into a single coherent API.
pub struct MenteDb {
    storage: StorageEngine,
    index: IndexManager,
    graph: GraphManager,
    /// Maps memory IDs to their storage page IDs for retrieval.
    page_map: HashMap<MemoryId, PageId>,
    /// Expected embedding dimension (0 = no validation).
    embedding_dim: usize,
    /// Database directory path for persistence.
    path: PathBuf,
}

impl MenteDb {
    /// Opens (or creates) a MenteDB instance at the given path.
    pub fn open(path: &Path) -> MenteResult<Self> {
        info!("Opening MenteDB at {}", path.display());
        let storage = StorageEngine::open(path)?;

        let index_dir = path.join("indexes");
        let graph_dir = path.join("graph");

        let index = if index_dir.join("hnsw.json").exists() {
            debug!("Loading indexes from {}", index_dir.display());
            IndexManager::load(&index_dir)?
        } else {
            IndexManager::default()
        };

        let graph = if graph_dir.join("graph.json").exists() {
            debug!("Loading graph from {}", graph_dir.display());
            GraphManager::load(&graph_dir)?
        } else {
            GraphManager::new()
        };

        let page_map = HashMap::new();

        Ok(Self {
            storage,
            index,
            graph,
            page_map,
            embedding_dim: 0,
            path: path.to_path_buf(),
        })
    }

    /// Stores a memory node into the database.
    ///
    /// The node is persisted to storage, added to all indexes, and registered
    /// in the graph for relationship traversal.
    pub fn store(&mut self, node: MemoryNode) -> MenteResult<()> {
        let id = node.id;
        debug!("Storing memory {}", id);

        // Validate embedding dimension when configured.
        if self.embedding_dim > 0
            && !node.embedding.is_empty()
            && node.embedding.len() != self.embedding_dim
        {
            return Err(MenteError::EmbeddingDimensionMismatch {
                got: node.embedding.len(),
                expected: self.embedding_dim,
            });
        }

        let page_id = self.storage.store_memory(&node)?;
        self.page_map.insert(id, page_id);
        self.index.index_memory(&node);
        self.graph.add_memory(id);

        Ok(())
    }

    /// Recalls memories using an MQL query string.
    ///
    /// Parses the query, builds an execution plan, runs it against the
    /// appropriate indexes/graph, and assembles the results into a
    /// token-budget-aware context window.
    pub fn recall(&mut self, query: &str) -> MenteResult<ContextWindow> {
        debug!("Recalling with query: {}", query);
        let plan = Mql::parse(query)?;

        let scored = self.execute_plan(&plan)?;
        let config = AssemblyConfig::default();
        let window = ContextAssembler::assemble(scored, vec![], &config);
        Ok(window)
    }

    /// Shortcut for vector similarity search.
    ///
    /// Returns the top-k most similar memory IDs with their scores.
    pub fn recall_similar(
        &mut self,
        embedding: &[f32],
        k: usize,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        debug!("Recall similar, k={}", k);
        let results = self.index.hybrid_search(embedding, None, None, k);
        Ok(results)
    }

    /// Adds a typed, weighted edge between two memories in the graph.
    pub fn relate(&mut self, edge: MemoryEdge) -> MenteResult<()> {
        debug!("Relating {} -> {}", edge.source, edge.target);
        self.graph.add_relationship(&edge)?;
        Ok(())
    }

    /// Retrieves a single memory by its ID.
    pub fn get_memory(&mut self, id: MemoryId) -> MenteResult<MemoryNode> {
        let page_id = self
            .page_map
            .get(&id)
            .copied()
            .ok_or(MenteError::MemoryNotFound(id))?;
        self.storage.load_memory(page_id)
    }

    /// Removes a memory from storage, indexes, and the graph.
    pub fn forget(&mut self, id: MemoryId) -> MenteResult<()> {
        debug!("Forgetting memory {}", id);

        // Load the node so we can clean up indexes properly.
        if let Some(&page_id) = self.page_map.get(&id)
            && let Ok(node) = self.storage.load_memory(page_id)
        {
            self.index.remove_memory(id, &node);
        }

        self.graph.remove_memory(id);
        self.page_map.remove(&id);
        Ok(())
    }

    /// Returns a reference to the underlying graph manager.
    pub fn graph(&self) -> &GraphManager {
        &self.graph
    }

    /// Flushes all data and closes the database.
    pub fn close(&mut self) -> MenteResult<()> {
        info!("Closing MenteDB");
        self.index.save(&self.path.join("indexes"))?;
        self.graph.save(&self.path.join("graph"))?;
        self.storage.checkpoint()?;
        self.storage.close()?;
        Ok(())
    }

    /// Executes a query plan against the indexes and graph, returning scored memories.
    fn execute_plan(&mut self, plan: &QueryPlan) -> MenteResult<Vec<ScoredMemory>> {
        match plan {
            QueryPlan::VectorSearch { query, k, .. } => {
                let hits = self.index.hybrid_search(query, None, None, *k);
                self.load_scored_memories(&hits)
            }
            QueryPlan::TagScan { tags, limit, .. } => {
                let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                let k = limit.unwrap_or(10);
                // Use a zero-vector for tag-only search; salience+bitmap still apply.
                let hits = self.index.hybrid_search(&[], Some(&tag_refs), None, k);
                self.load_scored_memories(&hits)
            }
            QueryPlan::TemporalScan { start, end, .. } => {
                let hits = self
                    .index
                    .hybrid_search(&[], None, Some((*start, *end)), 100);
                self.load_scored_memories(&hits)
            }
            QueryPlan::GraphTraversal { start, depth, .. } => {
                let (ids, _edges) = self.graph.get_context_subgraph(*start, *depth);
                let scored: Vec<ScoredMemory> = ids
                    .iter()
                    .filter_map(|id| {
                        self.page_map.get(id).and_then(|&pid| {
                            self.storage.load_memory(pid).ok().map(|node| ScoredMemory {
                                memory: node,
                                score: 1.0,
                            })
                        })
                    })
                    .collect();
                Ok(scored)
            }
            QueryPlan::PointLookup { id } => {
                let page_id = self
                    .page_map
                    .get(id)
                    .ok_or(MenteError::MemoryNotFound(*id))?;
                let node = self.storage.load_memory(*page_id)?;
                Ok(vec![ScoredMemory {
                    memory: node,
                    score: 1.0,
                }])
            }
            _ => Ok(vec![]),
        }
    }

    /// Loads MemoryNodes from storage and pairs them with their search scores.
    fn load_scored_memories(&mut self, hits: &[(MemoryId, f32)]) -> MenteResult<Vec<ScoredMemory>> {
        let mut scored = Vec::with_capacity(hits.len());
        for &(id, score) in hits {
            if let Some(&page_id) = self.page_map.get(&id)
                && let Ok(node) = self.storage.load_memory(page_id)
            {
                scored.push(ScoredMemory {
                    memory: node,
                    score,
                });
            }
        }
        Ok(scored)
    }
}
