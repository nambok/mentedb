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

use mentedb_cognitive::write_inference::{
    InferredAction, WriteInferenceConfig, WriteInferenceEngine,
};
use mentedb_consolidation::consolidation::{ConsolidationCandidate, ConsolidationEngine};
use mentedb_consolidation::decay::{DecayConfig, DecayEngine};
use mentedb_context::{AssemblyConfig, ContextAssembler, ContextWindow, ScoredMemory};
use mentedb_core::edge::EdgeType;
use mentedb_core::error::MenteResult;
use mentedb_core::types::{MemoryId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode, MenteError};
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_graph::GraphManager;
use mentedb_index::IndexManager;
use mentedb_query::{Mql, QueryPlan};
use mentedb_storage::StorageEngine;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

// Re-export sub-crates for direct access.

/// Engine version, derived from Cargo.toml at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Cognitive pipeline: speculative caching, trajectory tracking, inference.
pub use mentedb_cognitive as cognitive;
/// Consolidation, decay, and memory lifecycle management.
pub use mentedb_consolidation as consolidation;
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

/// Configuration for the cognitive engine subsystems.
#[derive(Debug, Clone)]
pub struct CognitiveConfig {
    /// Whether write inference (auto-edges, contradiction detection) is enabled on store.
    pub write_inference: bool,
    /// Whether salience decay is applied during retrieval.
    pub decay_on_recall: bool,
    /// Configuration for the write inference engine.
    pub inference_config: WriteInferenceConfig,
    /// Configuration for the decay engine.
    pub decay_config: DecayConfig,
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            write_inference: true,
            decay_on_recall: true,
            inference_config: WriteInferenceConfig::default(),
            decay_config: DecayConfig::default(),
        }
    }
}

/// The unified database facade for MenteDB.
///
/// `MenteDb` coordinates storage, indexing, graph relationships, query parsing,
/// context assembly, and cognitive subsystems into a single coherent API.
///
/// All internal state is protected by fine-grained locks, so every public method
/// takes `&self`. This allows `Arc<MenteDb>` to be shared across threads without
/// an external `RwLock`.
pub struct MenteDb {
    storage: StorageEngine,
    index: IndexManager,
    graph: GraphManager,
    /// Maps memory IDs to their storage page IDs for retrieval.
    page_map: RwLock<HashMap<MemoryId, PageId>>,
    /// Expected embedding dimension (0 = no validation).
    embedding_dim: usize,
    /// Database directory path for persistence.
    path: PathBuf,
    /// Optional embedding provider for auto-embedding on store and search.
    embedder: Option<Box<dyn EmbeddingProvider>>,
    /// Cognitive engine configuration.
    cognitive_config: CognitiveConfig,
    /// Write inference engine for auto-edge creation and contradiction detection.
    write_inference: WriteInferenceEngine,
    /// Decay engine for salience management.
    decay: DecayEngine,
    /// Consolidation engine for memory merging.
    consolidation: ConsolidationEngine,
}

impl MenteDb {
    /// Opens (or creates) a MenteDB instance at the given path.
    pub fn open(path: &Path) -> MenteResult<Self> {
        Self::open_with_config(path, CognitiveConfig::default())
    }

    /// Opens a MenteDB instance with custom cognitive configuration.
    pub fn open_with_config(path: &Path, cognitive_config: CognitiveConfig) -> MenteResult<Self> {
        info!("Opening MenteDB at {}", path.display());
        let storage = StorageEngine::open(path)?;

        let index_dir = path.join("indexes");
        let graph_dir = path.join("graph");

        let index = if index_dir.join("hnsw.bin").exists() || index_dir.join("hnsw.json").exists() {
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

        // Rebuild page map by scanning all pages
        let entries = storage.scan_all_memories();
        let mut page_map = HashMap::new();
        for (memory_id, page_id) in &entries {
            page_map.insert(*memory_id, *page_id);
        }
        if !page_map.is_empty() {
            info!(memories = page_map.len(), "rebuilt page map from storage");
        }

        let write_inference =
            WriteInferenceEngine::with_config(cognitive_config.inference_config.clone());
        let decay = DecayEngine::new(cognitive_config.decay_config.clone());
        let consolidation = ConsolidationEngine::new();

        Ok(Self {
            storage,
            index,
            graph,
            page_map: RwLock::new(page_map),
            embedding_dim: 0,
            path: path.to_path_buf(),
            embedder: None,
            cognitive_config,
            write_inference,
            decay,
            consolidation,
        })
    }

    /// Opens a MenteDB instance with a configured embedding provider.
    pub fn open_with_embedder(
        path: &Path,
        embedder: Box<dyn EmbeddingProvider>,
    ) -> MenteResult<Self> {
        let mut db = Self::open(path)?;
        db.embedding_dim = embedder.dimensions();
        db.embedder = Some(embedder);
        Ok(db)
    }

    /// Opens a MenteDB instance with both embedder and cognitive config.
    pub fn open_with_embedder_and_config(
        path: &Path,
        embedder: Box<dyn EmbeddingProvider>,
        cognitive_config: CognitiveConfig,
    ) -> MenteResult<Self> {
        let mut db = Self::open_with_config(path, cognitive_config)?;
        db.embedding_dim = embedder.dimensions();
        db.embedder = Some(embedder);
        Ok(db)
    }

    /// Set the embedding provider after construction.
    pub fn set_embedder(&mut self, embedder: Box<dyn EmbeddingProvider>) {
        self.embedding_dim = embedder.dimensions();
        self.embedder = Some(embedder);
    }

    /// Generate an embedding for the given text using the configured provider.
    /// Returns None if no provider is configured.
    pub fn embed_text(&self, text: &str) -> MenteResult<Option<Vec<f32>>> {
        match &self.embedder {
            Some(e) => Ok(Some(e.embed(text)?)),
            None => Ok(None),
        }
    }

    /// Stores a memory node into the database.
    ///
    /// The node is persisted to storage, added to all indexes, and registered
    /// in the graph for relationship traversal.
    ///
    /// When cognitive features are enabled (the default), write inference
    /// automatically runs to:
    /// - Detect contradictions with existing memories
    /// - Create relationship edges (Related, Supersedes, Contradicts)
    /// - Invalidate superseded memories
    /// - Propagate confidence changes through the graph
    pub fn store(&self, node: MemoryNode) -> MenteResult<()> {
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
        self.page_map.write().insert(id, page_id);
        self.index.index_memory(&node);
        self.graph.add_memory(id);

        // Run write inference to auto-create edges and detect contradictions.
        if self.cognitive_config.write_inference {
            self.run_write_inference(&node);
        }

        Ok(())
    }

    /// Recalls memories using an MQL query string.
    ///
    /// Parses the query, builds an execution plan, runs it against the
    /// appropriate indexes/graph, and assembles the results into a
    /// token-budget-aware context window.
    pub fn recall(&self, query: &str) -> MenteResult<ContextWindow> {
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
    /// Memories that have been superseded, contradicted, or temporally
    /// invalidated are automatically excluded from results.
    pub fn recall_similar(&self, embedding: &[f32], k: usize) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_similar_filtered(embedding, k, None, None)
    }

    /// Vector similarity search with optional tag and time range filters.
    pub fn recall_similar_filtered(
        &self,
        embedding: &[f32],
        k: usize,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.recall_similar_filtered_at(embedding, k, now, tags, time_range)
    }

    /// Vector similarity search at a specific point in time.
    ///
    /// Only returns memories that were temporally valid at the given timestamp.
    /// Superseded/contradicted memories are excluded unless the edge itself
    /// was not yet valid at that time.
    pub fn recall_similar_at(
        &self,
        embedding: &[f32],
        k: usize,
        at: Timestamp,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_similar_filtered_at(embedding, k, at, None, None)
    }

    /// Vector similarity search at a specific point in time with optional filters.
    ///
    /// Only returns memories that were temporally valid at the given timestamp.
    /// Superseded/contradicted memories are excluded unless the edge itself
    /// was not yet valid at that time. Optionally filters by tags and time range.
    pub fn recall_similar_filtered_at(
        &self,
        embedding: &[f32],
        k: usize,
        at: Timestamp,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_hybrid_at(embedding, None, k, at, tags, time_range)
    }

    /// Hybrid search combining vector similarity and BM25 keyword matching.
    ///
    /// When `query_text` is provided, BM25 results are fused with vector
    /// results via Reciprocal Rank Fusion (RRF) for better recall on
    /// exact entity names, dates, and specific terms.
    pub fn recall_hybrid_at(
        &self,
        embedding: &[f32],
        query_text: Option<&str>,
        k: usize,
        at: Timestamp,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        debug!(
            "Recall hybrid, k={}, at={}, bm25={}",
            k,
            at,
            query_text.is_some()
        );
        // Over-fetch to account for filtered-out results
        let results =
            self.index
                .hybrid_search_with_query(embedding, query_text, tags, time_range, k * 3);
        let graph = self.graph.graph();
        let pm = self.page_map.read();
        let filtered: Vec<(MemoryId, f32)> = results
            .into_iter()
            .filter(|(id, _)| {
                let incoming = graph.incoming(*id);
                let has_active_supersede = incoming.iter().any(|(_, e)| {
                    (e.edge_type == EdgeType::Supersedes || e.edge_type == EdgeType::Contradicts)
                        && e.is_valid_at(at)
                });
                !has_active_supersede
            })
            .filter(|(id, _)| {
                if let Some(&page_id) = pm.get(id)
                    && let Ok(node) = self.storage.load_memory(page_id)
                {
                    node.is_valid_at(at)
                } else {
                    true
                }
            })
            .take(k)
            .collect();
        Ok(filtered)
    }

    /// Multi-query search with Reciprocal Rank Fusion (RRF).
    ///
    /// Runs multiple vector searches (one per embedding) and merges results
    /// using RRF: score = Σ 1/(k + rank_i). This improves recall by matching
    /// on different semantic aspects of a query.
    /// When `query_texts` is provided, each search also runs BM25 matching.
    pub fn recall_similar_multi(
        &self,
        embeddings: &[Vec<f32>],
        k: usize,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_hybrid_multi(embeddings, None, k, tags, time_range)
    }

    /// Multi-query hybrid search with BM25 + vector fusion.
    ///
    /// Each query text is searched via both BM25 and vector, then all results
    /// are merged via RRF.
    pub fn recall_hybrid_multi(
        &self,
        embeddings: &[Vec<f32>],
        query_texts: Option<&[String]>,
        k: usize,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        use std::collections::HashMap;

        let rrf_k: f32 = 60.0;
        let mut rrf_scores: HashMap<MemoryId, f32> = HashMap::new();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for (i, emb) in embeddings.iter().enumerate() {
            let qt = query_texts.and_then(|texts| texts.get(i).map(|s| s.as_str()));
            let results = self.recall_hybrid_at(emb, qt, k, now, tags, time_range)?;
            for (rank, (id, _score)) in results.iter().enumerate() {
                *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
            }
        }

        let mut merged: Vec<(MemoryId, f32)> = rrf_scores.into_iter().collect();
        merged.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(k);
        Ok(merged)
    }

    /// Invalidate a memory by setting its valid_until timestamp.
    ///
    /// The memory remains in storage for historical queries but is excluded
    /// from current recall results.
    pub fn invalidate_memory(&self, id: MemoryId, at: Timestamp) -> MenteResult<()> {
        debug!("Invalidating memory {} at {}", id, at);
        let page_id = self
            .page_map
            .read()
            .get(&id)
            .copied()
            .ok_or(MenteError::MemoryNotFound(id))?;
        let mut node = self.storage.load_memory(page_id)?;
        node.invalidate(at);
        let new_page_id = self.storage.store_memory(&node)?;
        self.page_map.write().insert(id, new_page_id);
        Ok(())
    }

    /// Adds a typed, weighted edge between two memories in the graph.
    pub fn relate(&self, edge: MemoryEdge) -> MenteResult<()> {
        debug!("Relating {} -> {}", edge.source, edge.target);
        self.graph.add_relationship(&edge)?;
        Ok(())
    }

    /// Retrieves a single memory by its ID.
    pub fn get_memory(&self, id: MemoryId) -> MenteResult<MemoryNode> {
        let page_id = self
            .page_map
            .read()
            .get(&id)
            .copied()
            .ok_or(MenteError::MemoryNotFound(id))?;
        self.storage.load_memory(page_id)
    }

    /// Returns all memory IDs currently stored in the database.
    pub fn memory_ids(&self) -> Vec<MemoryId> {
        self.page_map.read().keys().copied().collect()
    }

    /// Returns the number of memories currently stored.
    pub fn memory_count(&self) -> usize {
        self.page_map.read().len()
    }

    /// Removes a memory from storage, indexes, and the graph.
    pub fn forget(&self, id: MemoryId) -> MenteResult<()> {
        debug!("Forgetting memory {}", id);

        if let Some(&page_id) = self.page_map.read().get(&id)
            && let Ok(node) = self.storage.load_memory(page_id)
        {
            self.index.remove_memory(id, &node);
        }

        self.graph.remove_memory(id);
        self.page_map.write().remove(&id);
        Ok(())
    }

    /// Returns a reference to the underlying graph manager.
    pub fn graph(&self) -> &GraphManager {
        &self.graph
    }

    /// Returns a mutable reference to the underlying graph manager.
    #[deprecated(note = "GraphManager now uses interior mutability; use graph() instead")]
    pub fn graph_mut(&mut self) -> &mut GraphManager {
        &mut self.graph
    }

    /// Returns a reference to the cognitive configuration.
    pub fn cognitive_config(&self) -> &CognitiveConfig {
        &self.cognitive_config
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Write Inference
    // -----------------------------------------------------------------------

    /// Run write inference on a newly stored memory.
    ///
    /// Finds semantically similar existing memories, runs the inference engine
    /// to detect contradictions and relationships, then applies the actions
    /// (creating edges, invalidating superseded memories, etc.).
    fn run_write_inference(&self, new_memory: &MemoryNode) {
        // Find candidate memories to compare against via vector search.
        // We load a small set of the most similar memories.
        let candidates = if !new_memory.embedding.is_empty() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            self.recall_hybrid_at(&new_memory.embedding, None, 20, now, None, None)
                .unwrap_or_default()
        } else {
            vec![]
        };

        if candidates.is_empty() {
            return;
        }

        // Load the actual MemoryNode data for each candidate.
        let pm = self.page_map.read();
        let existing: Vec<MemoryNode> = candidates
            .iter()
            .filter(|(id, _)| *id != new_memory.id)
            .filter_map(|(id, _)| {
                pm.get(id)
                    .and_then(|&pid| self.storage.load_memory(pid).ok())
            })
            .collect();
        drop(pm);

        if existing.is_empty() {
            return;
        }

        let actions = self
            .write_inference
            .infer_on_write(new_memory, &existing, &[]);

        let action_count = actions.len();
        for action in actions {
            if let Err(e) = self.apply_inferred_action(action) {
                warn!("Failed to apply inferred action: {}", e);
            }
        }
        if action_count > 0 {
            debug!(
                "Write inference for {} produced {} actions",
                new_memory.id, action_count
            );
        }
    }

    /// Apply a single inferred action from the write inference engine.
    fn apply_inferred_action(&self, action: InferredAction) -> MenteResult<()> {
        match action {
            InferredAction::CreateEdge {
                source,
                target,
                edge_type,
                weight,
            } => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let edge = MemoryEdge {
                    source,
                    target,
                    edge_type,
                    weight,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: None,
                };
                debug!(
                    "Auto-creating {:?} edge {} -> {}",
                    edge_type, source, target
                );
                self.graph.add_relationship(&edge)?;
            }
            InferredAction::InvalidateMemory {
                memory,
                superseded_by,
                valid_until,
            } => {
                debug!(
                    "Invalidating memory {} (superseded by {})",
                    memory, superseded_by
                );
                self.invalidate_memory(memory, valid_until)?;
                // Also create the Supersedes edge.
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let edge = MemoryEdge {
                    source: superseded_by,
                    target: memory,
                    edge_type: EdgeType::Supersedes,
                    weight: 1.0,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: None,
                };
                self.graph.add_relationship(&edge)?;
            }
            InferredAction::MarkObsolete {
                memory,
                superseded_by,
            } => {
                debug!(
                    "Marking {} obsolete (superseded by {})",
                    memory, superseded_by
                );
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                self.invalidate_memory(memory, now)?;
                let edge = MemoryEdge {
                    source: superseded_by,
                    target: memory,
                    edge_type: EdgeType::Supersedes,
                    weight: 1.0,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: None,
                };
                self.graph.add_relationship(&edge)?;
            }
            InferredAction::FlagContradiction {
                existing,
                new,
                reason,
            } => {
                debug!(
                    "Contradiction detected: {} vs {} — {}",
                    existing, new, reason
                );
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let edge = MemoryEdge {
                    source: new,
                    target: existing,
                    edge_type: EdgeType::Contradicts,
                    weight: 1.0,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: Some(reason),
                };
                self.graph.add_relationship(&edge)?;
            }
            InferredAction::UpdateConfidence {
                memory,
                new_confidence,
            } => {
                debug!("Updating confidence for {} to {}", memory, new_confidence);
                if let Ok(mut node) = self.get_memory(memory) {
                    node.confidence = new_confidence;
                    let new_page_id = self.storage.store_memory(&node)?;
                    self.page_map.write().insert(memory, new_page_id);
                }
            }
            InferredAction::PropagateBeliefChange { root, delta } => {
                debug!("Propagating belief change from {} (delta={})", root, delta);
                if let Ok(node) = self.get_memory(root) {
                    let new_confidence = (node.confidence + delta).clamp(0.0, 1.0);
                    let affected = self.graph.propagate_belief_change(root, new_confidence);
                    for (affected_id, new_conf) in affected {
                        if let Ok(mut affected_node) = self.get_memory(affected_id) {
                            affected_node.confidence = new_conf;
                            if let Ok(pid) = self.storage.store_memory(&affected_node) {
                                self.page_map.write().insert(affected_id, pid);
                            }
                        }
                    }
                }
            }
            InferredAction::UpdateContent {
                memory,
                new_content,
                reason,
            } => {
                debug!("Updating content of {}: {}", memory, reason);
                if let Ok(mut node) = self.get_memory(memory) {
                    node.content = new_content;
                    let new_page_id = self.storage.store_memory(&node)?;
                    self.page_map.write().insert(memory, new_page_id);
                    self.index.remove_memory(memory, &node);
                    self.index.index_memory(&node);
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Salience Decay
    // -----------------------------------------------------------------------

    /// Apply salience decay to a batch of memories in-place.
    ///
    /// Call this during retrieval to ensure scores reflect temporal relevance,
    /// or periodically to maintain salience accuracy across the database.
    pub fn apply_decay(&self, memories: &mut [MemoryNode]) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.decay.apply_decay_batch(memories, now);
    }

    /// Compute the decayed salience for a single memory at the current time.
    pub fn compute_decayed_salience(&self, memory: &MemoryNode) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.decay.compute_decay(
            memory.salience,
            memory.created_at,
            memory.accessed_at,
            memory.access_count,
            now,
        )
    }

    /// Apply decay globally: recompute salience for all memories and persist.
    ///
    /// This is an expensive operation intended for periodic maintenance.
    /// For real-time use, prefer `apply_decay` on retrieved memories.
    pub fn apply_decay_global(&self) -> MenteResult<usize> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let ids: Vec<(MemoryId, PageId)> = self
            .page_map
            .read()
            .iter()
            .map(|(mid, pid)| (*mid, *pid))
            .collect();

        let mut updated = 0;
        for (mid, pid) in &ids {
            if let Ok(mut node) = self.storage.load_memory(*pid) {
                let new_salience = self.decay.compute_decay(
                    node.salience,
                    node.created_at,
                    node.accessed_at,
                    node.access_count,
                    now,
                );
                if (new_salience - node.salience).abs() > 0.001 {
                    node.salience = new_salience;
                    let new_pid = self.storage.store_memory(&node)?;
                    self.page_map.write().insert(*mid, new_pid);
                    updated += 1;
                }
            }
        }
        if updated > 0 {
            info!("Decay pass updated {} memories", updated);
        }
        Ok(updated)
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Consolidation
    // -----------------------------------------------------------------------

    /// Find groups of similar memories that are candidates for consolidation.
    ///
    /// Returns clusters of memories that share high semantic similarity and
    /// could be merged into unified knowledge.
    pub fn find_consolidation_candidates(
        &self,
        min_cluster_size: usize,
        similarity_threshold: f32,
    ) -> MenteResult<Vec<ConsolidationCandidate>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Load all memories eligible for consolidation.
        let pm = self.page_map.read();
        let eligible: Vec<MemoryNode> = pm
            .values()
            .filter_map(|pid| self.storage.load_memory(*pid).ok())
            .filter(|node| ConsolidationEngine::should_consolidate(node, now))
            .collect();
        drop(pm);

        if eligible.is_empty() {
            return Ok(vec![]);
        }

        Ok(self
            .consolidation
            .find_candidates(&eligible, min_cluster_size, similarity_threshold))
    }

    /// Consolidate a cluster of memories into a single merged memory.
    ///
    /// The source memories are invalidated (not deleted) and a new consolidated
    /// semantic memory is stored with Derived edges back to the sources.
    pub fn consolidate_cluster(&self, memory_ids: &[MemoryId]) -> MenteResult<MemoryId> {
        let pm = self.page_map.read();
        let cluster: Vec<MemoryNode> = memory_ids
            .iter()
            .filter_map(|id| {
                pm.get(id)
                    .and_then(|&pid| self.storage.load_memory(pid).ok())
            })
            .collect();
        drop(pm);

        if cluster.len() < 2 {
            return Err(MenteError::Query(
                "consolidation requires at least 2 memories".into(),
            ));
        }

        let result = self.consolidation.consolidate(&cluster);

        // Create the consolidated memory node.
        let agent_id = cluster[0].agent_id;
        let mut consolidated = MemoryNode::new(
            agent_id,
            result.new_type,
            result.summary,
            result.combined_embedding,
        );
        consolidated.confidence = result.combined_confidence;

        let consolidated_id = consolidated.id;
        self.store(consolidated)?;

        // Invalidate source memories and create Derived edges.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        for source_id in &result.source_memories {
            let _ = self.invalidate_memory(*source_id, now);
            let edge = MemoryEdge {
                source: consolidated_id,
                target: *source_id,
                edge_type: EdgeType::Derived,
                weight: 1.0,
                created_at: now,
                valid_from: None,
                valid_until: None,
                label: None,
            };
            let _ = self.graph.add_relationship(&edge);
        }

        info!(
            "Consolidated {} memories into {}",
            result.source_memories.len(),
            consolidated_id
        );
        Ok(consolidated_id)
    }

    /// Flushes all data and closes the database.
    pub fn close(&self) -> MenteResult<()> {
        info!("Closing MenteDB");
        self.flush()?;
        self.storage.close()?;
        Ok(())
    }

    /// Flush indexes, graph, and storage to disk without closing.
    ///
    /// Call this periodically to ensure cross-session persistence.
    /// Unlike `close()`, the database remains usable after flushing.
    pub fn flush(&self) -> MenteResult<()> {
        debug!("Flushing MenteDB to disk");
        self.index.save(&self.path.join("indexes"))?;
        self.graph.save(&self.path.join("graph"))?;
        self.storage.checkpoint()?;
        Ok(())
    }

    /// Executes a query plan against the indexes and graph, returning scored memories.
    fn execute_plan(&self, plan: &QueryPlan) -> MenteResult<Vec<ScoredMemory>> {
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
                let pm = self.page_map.read();
                let scored: Vec<ScoredMemory> = ids
                    .iter()
                    .filter_map(|id| {
                        pm.get(id).and_then(|&pid| {
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
                    .read()
                    .get(id)
                    .copied()
                    .ok_or(MenteError::MemoryNotFound(*id))?;
                let node = self.storage.load_memory(page_id)?;
                Ok(vec![ScoredMemory {
                    memory: node,
                    score: 1.0,
                }])
            }
            _ => Ok(vec![]),
        }
    }

    /// Loads MemoryNodes from storage and pairs them with their search scores.
    ///
    /// When decay is enabled, salience is recomputed and factored into the
    /// final score to prioritize temporally relevant memories.
    fn load_scored_memories(&self, hits: &[(MemoryId, f32)]) -> MenteResult<Vec<ScoredMemory>> {
        let pm = self.page_map.read();
        let now = if self.cognitive_config.decay_on_recall {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64
        } else {
            0
        };

        let mut scored = Vec::with_capacity(hits.len());
        for &(id, score) in hits {
            if let Some(&page_id) = pm.get(&id)
                && let Ok(node) = self.storage.load_memory(page_id)
            {
                let final_score = if self.cognitive_config.decay_on_recall {
                    let decayed_salience = self.decay.compute_decay(
                        node.salience,
                        node.created_at,
                        node.accessed_at,
                        node.access_count,
                        now,
                    );
                    // Blend search similarity with decayed salience.
                    // 70% similarity, 30% salience — keeps search relevance
                    // primary but rewards recently active memories.
                    score * 0.7 + decayed_salience * 0.3
                } else {
                    score
                };
                scored.push(ScoredMemory {
                    memory: node,
                    score: final_score,
                });
            }
        }
        // Re-sort by blended score when decay is applied.
        if self.cognitive_config.decay_on_recall {
            scored.sort_unstable_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        Ok(scored)
    }
}
