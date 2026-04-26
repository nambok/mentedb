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
//! - **Unified `process_turn`** pipeline: single call handles context retrieval,
//!   pain signals, episodic storage, write inference, action detection, sentiment,
//!   phantom tracking, trajectory, speculative caching, fact extraction, and
//!   auto-maintenance (decay / archival / consolidation)
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

use mentedb_cognitive::EntityResolver;
use mentedb_cognitive::interference::{InterferenceDetector, InterferencePair};
use mentedb_cognitive::pain::{PainRegistry, PainSignal};
use mentedb_cognitive::phantom::{PhantomConfig, PhantomMemory, PhantomTracker};
use mentedb_cognitive::speculative::{CacheEntry, CacheStats, SpeculativeCache};
use mentedb_cognitive::stream::{CognitionStream, StreamAlert, StreamConfig};
use mentedb_cognitive::trajectory::{TrajectoryNode, TrajectoryTracker};
use mentedb_cognitive::write_inference::{
    InferredAction, WriteInferenceConfig, WriteInferenceEngine,
};
use mentedb_consolidation::archival::{ArchivalConfig, ArchivalDecision, ArchivalPipeline};
use mentedb_consolidation::compression::{CompressedMemory, MemoryCompressor};
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

/// Unified process_turn orchestration.
pub mod process_turn;

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

/// Configuration for sleeptime enrichment pipeline.
///
/// Enrichment runs BETWEEN conversations, never in the hot path.
/// The engine tracks state and provides candidates; callers invoke
/// the async LLM pipeline when ready.
#[derive(Debug, Clone)]
pub struct EnrichmentConfig {
    /// Whether enrichment is enabled. Default: false (opt-in).
    pub enabled: bool,
    /// Run enrichment after this many process_turn calls. Default: 50.
    pub trigger_interval: u64,
    /// Minimum confidence for extracted memories to be stored. Default: 0.6.
    pub min_confidence: f32,
    /// Maximum confidence for enrichment-generated memories. Default: 0.7.
    pub max_enrichment_confidence: f32,
    /// Whether to generate a user model summary. Default: false.
    pub enable_user_model: bool,
    /// Embedding similarity threshold to merge entities. Default: 0.7.
    pub entity_merge_threshold: f32,
    /// Embedding similarity below which entities are kept separate. Default: 0.4.
    pub entity_separate_threshold: f32,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            trigger_interval: 50,
            min_confidence: 0.6,
            max_enrichment_confidence: 0.7,
            enable_user_model: false,
            entity_merge_threshold: 0.7,
            entity_separate_threshold: 0.4,
        }
    }
}

/// Result of running the enrichment pipeline.
#[derive(Debug, Clone, Default)]
pub struct EnrichmentResult {
    /// Number of new memories stored from extraction.
    pub memories_stored: usize,
    /// Number of entity nodes created or updated.
    pub entities_processed: usize,
    /// Number of edges created (Derived, Related, PartOf).
    pub edges_created: usize,
    /// Number of memories skipped as duplicates.
    pub duplicates_skipped: usize,
    /// Number of contradictions detected.
    pub contradictions_found: usize,
    /// Turn ID at which enrichment was completed.
    pub completed_at_turn: u64,
    /// Number of entity links created (Related edges between same-name entities).
    pub entities_linked: usize,
    /// Number of entity pairs left ambiguous (below merge threshold).
    pub entities_ambiguous: usize,
}

/// Result of a single entity linking run.
#[derive(Debug, Clone, Default)]
pub struct EntityLinkResult {
    /// Number of entity pairs linked with Related edges.
    pub linked: usize,
    /// Number of entity pairs tagged as ambiguous (MaybeRelated).
    pub ambiguous: usize,
    /// Number of edges created.
    pub edges_created: usize,
}

/// Configuration for the cognitive engine subsystems.
#[derive(Debug, Clone)]
pub struct CognitiveConfig {
    /// Whether write inference (auto-edges, contradiction detection) is enabled on store.
    pub write_inference: bool,
    /// Whether salience decay is applied during retrieval.
    pub decay_on_recall: bool,
    /// Whether pain tracking is enabled.
    pub pain_tracking: bool,
    /// Whether interference detection is available.
    pub interference_detection: bool,
    /// Whether phantom tracking is enabled.
    pub phantom_tracking: bool,
    /// Whether speculative caching is enabled.
    pub speculative_cache: bool,
    /// Whether archival evaluation is available.
    pub archival_evaluation: bool,
    /// Configuration for the write inference engine.
    pub inference_config: WriteInferenceConfig,
    /// Configuration for the decay engine.
    pub decay_config: DecayConfig,
    /// Configuration for phantom tracking.
    pub phantom_config: PhantomConfig,
    /// Configuration for the archival pipeline.
    pub archival_config: ArchivalConfig,
    /// Configuration for the cognition stream.
    pub stream_config: StreamConfig,
    /// Configuration for sleeptime enrichment.
    pub enrichment_config: EnrichmentConfig,
    /// Similarity threshold for interference detection.
    pub interference_threshold: f32,
    /// Maximum trajectory turns to track.
    pub trajectory_max_turns: usize,
    /// Maximum speculative cache entries.
    pub speculative_cache_size: usize,
    /// Maximum pain signals to retain.
    pub pain_max_warnings: usize,
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            write_inference: true,
            decay_on_recall: true,
            pain_tracking: true,
            interference_detection: true,
            phantom_tracking: true,
            speculative_cache: true,
            archival_evaluation: true,
            inference_config: WriteInferenceConfig::default(),
            decay_config: DecayConfig::default(),
            phantom_config: PhantomConfig::default(),
            archival_config: ArchivalConfig::default(),
            stream_config: StreamConfig::default(),
            enrichment_config: EnrichmentConfig::default(),
            interference_threshold: 0.8,
            trajectory_max_turns: 100,
            speculative_cache_size: 10,
            pain_max_warnings: 5,
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
    /// Pain registry for tracking recurring failures.
    pain: RwLock<PainRegistry>,
    /// Trajectory tracker for conversation patterns.
    trajectory: RwLock<TrajectoryTracker>,
    /// Cognition stream for token-level monitoring.
    stream: CognitionStream,
    /// Phantom tracker for detecting referenced-but-missing knowledge.
    phantom: RwLock<PhantomTracker>,
    /// Speculative cache for pre-fetching likely-needed memories.
    speculative: RwLock<SpeculativeCache>,
    /// Interference detector for finding confusable memories.
    interference: InterferenceDetector,
    /// Entity resolver for canonical name resolution.
    entity_resolver: RwLock<EntityResolver>,
    /// Memory compressor for content summarization.
    compressor: MemoryCompressor,
    /// Archival pipeline for lifecycle evaluation.
    archival: ArchivalPipeline,
    /// Turn ID of the last completed enrichment cycle.
    last_enrichment_turn: RwLock<u64>,
    /// Whether enrichment is currently pending (set by maintenance trigger).
    enrichment_pending: RwLock<bool>,
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
        let pain = RwLock::new(PainRegistry::new(cognitive_config.pain_max_warnings));
        let trajectory = RwLock::new(TrajectoryTracker::new(
            cognitive_config.trajectory_max_turns,
        ));
        let stream = CognitionStream::with_config(cognitive_config.stream_config.clone());
        let phantom = RwLock::new(PhantomTracker::new(cognitive_config.phantom_config.clone()));
        let speculative = RwLock::new(SpeculativeCache::new(
            cognitive_config.speculative_cache_size,
            0.5,
            0.4,
        ));
        let interference = InterferenceDetector::new(cognitive_config.interference_threshold);
        let entity_resolver = RwLock::new(EntityResolver::new());
        let compressor = MemoryCompressor::new();
        let archival = ArchivalPipeline::new(cognitive_config.archival_config.clone());

        // Load persisted state for subsystems that support it.
        let cognitive_dir = path.join("cognitive");
        if cognitive_dir.exists() {
            let _ = trajectory
                .write()
                .transitions
                .load(&cognitive_dir.join("transitions.json"));
            let _ = speculative
                .write()
                .load(&cognitive_dir.join("speculative.json"));
            let _ = entity_resolver
                .write()
                .load(&cognitive_dir.join("entities.json"));
        }

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
            pain,
            trajectory,
            stream,
            phantom,
            speculative,
            interference,
            entity_resolver,
            compressor,
            archival,
            last_enrichment_turn: RwLock::new(0),
            enrichment_pending: RwLock::new(false),
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

        // Persist cognitive subsystem state.
        let cognitive_dir = self.path.join("cognitive");
        if std::fs::create_dir_all(&cognitive_dir).is_ok() {
            let _ = self
                .trajectory
                .read()
                .transitions
                .save(&cognitive_dir.join("transitions.json"), 1);
            let _ = self
                .speculative
                .read()
                .save(&cognitive_dir.join("speculative.json"), 0);
            let _ = self
                .entity_resolver
                .read()
                .save(&cognitive_dir.join("entities.json"));
        }
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

    // -----------------------------------------------------------------------
    // Cognitive Engine: Pain Registry
    // -----------------------------------------------------------------------

    /// Record a pain signal — a recurring failure or frustration pattern.
    ///
    /// Pain signals are tracked by keywords and surfaced as warnings when
    /// similar contexts arise in future queries.
    pub fn record_pain(&self, signal: PainSignal) {
        if self.cognitive_config.pain_tracking {
            self.pain.write().record_pain(signal);
        }
    }

    /// Get pain warnings relevant to the given context keywords.
    ///
    /// Returns formatted warning text if any pain signals match the keywords.
    /// Use this before answering to warn about past failures.
    pub fn get_pain_warnings(&self, context_keywords: &[String]) -> Vec<PainSignal> {
        if !self.cognitive_config.pain_tracking {
            return vec![];
        }
        let registry = self.pain.read();
        registry
            .get_pain_for_context(context_keywords)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Format pain warnings as a human-readable string.
    pub fn format_pain_warnings(&self, signals: &[&PainSignal]) -> String {
        self.pain.read().format_pain_warnings(signals)
    }

    /// Decay all pain signals to reduce intensity over time.
    pub fn decay_pain(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.pain.write().decay_all(now);
    }

    /// Get all recorded pain signals.
    pub fn all_pain_signals(&self) -> Vec<PainSignal> {
        self.pain.read().all_signals().to_vec()
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Trajectory Tracking
    // -----------------------------------------------------------------------

    /// Record a conversation turn in the trajectory tracker.
    ///
    /// Tracks the evolution of topics, decisions, and open questions across
    /// a conversation. Used for resume context and topic prediction.
    pub fn record_trajectory_turn(&self, turn: TrajectoryNode) {
        self.trajectory.write().record_turn(turn);
    }

    /// Get a resume context string summarizing the conversation so far.
    ///
    /// Returns None if no trajectory has been recorded.
    pub fn get_resume_context(&self) -> Option<String> {
        self.trajectory.read().get_resume_context()
    }

    /// Predict the next likely topics based on conversation trajectory.
    ///
    /// Returns up to 3 predicted topic strings based on transition patterns.
    pub fn predict_next_topics(&self) -> Vec<String> {
        self.trajectory.read().predict_next_topics()
    }

    /// Get the full trajectory of recorded turns.
    pub fn get_trajectory(&self) -> Vec<TrajectoryNode> {
        self.trajectory.read().get_trajectory().to_vec()
    }

    /// Reinforce a transition that led to a speculative cache hit.
    pub fn reinforce_transition(&self, hit_topic: &str) {
        self.trajectory.write().reinforce_transition(hit_topic);
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Cognition Stream
    // -----------------------------------------------------------------------

    /// Feed a token to the cognition stream for real-time monitoring.
    ///
    /// Tokens are buffered and analyzed for contradictions with known facts
    /// when `check_stream_alerts()` is called.
    pub fn feed_stream_token(&self, token: &str) {
        self.stream.feed_token(token);
    }

    /// Check for stream alerts against known facts.
    ///
    /// Compares the buffered token stream against the provided known facts
    /// to detect contradictions, corrections, and reinforcements.
    pub fn check_stream_alerts(&self, known_facts: &[(MemoryId, String)]) -> Vec<StreamAlert> {
        self.stream.check_alerts(known_facts)
    }

    /// Drain the token buffer, returning accumulated text.
    pub fn drain_stream_buffer(&self) -> String {
        self.stream.drain_buffer()
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Phantom Tracking
    // -----------------------------------------------------------------------

    /// Detect phantom memories — entities referenced in content but not stored.
    ///
    /// Scans content for entity mentions that don't exist in the known entities
    /// list, flagging them as knowledge gaps that should be filled.
    pub fn detect_phantoms(
        &self,
        content: &str,
        known_entities: &[String],
        turn_id: u64,
    ) -> Vec<PhantomMemory> {
        if !self.cognitive_config.phantom_tracking {
            return vec![];
        }
        self.phantom
            .write()
            .detect_gaps(content, known_entities, turn_id)
    }

    /// Resolve a phantom memory (mark it as no longer a gap).
    pub fn resolve_phantom(&self, phantom_id: MemoryId) {
        self.phantom.write().resolve(phantom_id.into());
    }

    /// Get all active (unresolved) phantom memories, sorted by priority.
    pub fn get_active_phantoms(&self) -> Vec<PhantomMemory> {
        self.phantom
            .read()
            .get_active_phantoms()
            .into_iter()
            .cloned()
            .collect()
    }

    /// Format phantom warnings as a human-readable string.
    pub fn format_phantom_warnings(&self) -> String {
        self.phantom.read().format_phantom_warnings()
    }

    /// Register an entity so the phantom tracker knows it exists.
    pub fn register_entity(&self, entity: &str) {
        self.phantom.write().register_entity(entity);
    }

    /// Register multiple entities at once.
    pub fn register_entities(&self, entities: &[&str]) {
        self.phantom.write().register_entities(entities);
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Speculative Cache
    // -----------------------------------------------------------------------

    /// Try to hit the speculative cache for a query.
    ///
    /// If a previous prediction matches the current query (by keyword overlap
    /// or embedding similarity), returns the pre-assembled context.
    pub fn try_speculative_hit(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
    ) -> Option<CacheEntry> {
        if !self.cognitive_config.speculative_cache {
            return None;
        }
        self.speculative.write().try_hit(query, query_embedding)
    }

    /// Pre-assemble speculative cache entries for predicted topics.
    ///
    /// The builder function should return `(context_text, memory_ids, optional_embedding)`
    /// for each topic prediction.
    pub fn pre_assemble_speculative<F>(&self, predictions: Vec<String>, builder: F)
    where
        F: Fn(&str) -> Option<(String, Vec<MemoryId>, Option<Vec<f32>>)>,
    {
        if self.cognitive_config.speculative_cache {
            self.speculative.write().pre_assemble(predictions, builder);
        }
    }

    /// Evict stale entries from the speculative cache.
    pub fn evict_stale_speculative(&self, max_age_us: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.speculative.write().evict_stale(max_age_us, now);
    }

    /// Get speculative cache statistics.
    pub fn speculative_cache_stats(&self) -> CacheStats {
        self.speculative.read().stats()
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Interference Detection
    // -----------------------------------------------------------------------

    /// Detect interference between a set of memories.
    ///
    /// Returns pairs of memories that are similar enough to cause confusion,
    /// along with disambiguation hints. Use this during context assembly to
    /// add disambiguation notes or separate confusable memories.
    pub fn detect_interference(&self, memories: &[MemoryNode]) -> Vec<InterferencePair> {
        if !self.cognitive_config.interference_detection {
            return vec![];
        }
        self.interference.detect_interference(memories)
    }

    /// Generate a disambiguation hint for two confusable memories.
    pub fn generate_disambiguation(&self, a: &MemoryNode, b: &MemoryNode) -> String {
        self.interference.generate_disambiguation(a, b)
    }

    /// Arrange memory IDs to maximize separation between interfering pairs.
    pub fn arrange_with_separation(
        memories: Vec<MemoryId>,
        pairs: &[InterferencePair],
    ) -> Vec<MemoryId> {
        InterferenceDetector::arrange_with_separation(memories, pairs)
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Entity Resolution
    // -----------------------------------------------------------------------

    /// Resolve an entity name to its canonical form.
    ///
    /// Uses cached aliases and rule-based matching (no LLM).
    pub fn resolve_entity(&self, name: &str) -> mentedb_cognitive::ResolvedEntity {
        self.entity_resolver.read().resolve(name)
    }

    /// Add an alias mapping for entity resolution.
    pub fn add_entity_alias(&self, alias: &str, canonical: &str, confidence: f32) {
        self.entity_resolver
            .write()
            .add_alias(alias, canonical, confidence);
    }

    /// Get the canonical name for an entity, if known.
    pub fn get_canonical_entity(&self, name: &str) -> Option<String> {
        self.entity_resolver.read().get_canonical(name).cloned()
    }

    /// List all known entities in the resolver.
    pub fn known_entities(&self) -> Vec<String> {
        self.entity_resolver.read().known_entities()
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Memory Compression
    // -----------------------------------------------------------------------

    /// Compress a memory's content, extracting key facts and removing filler.
    ///
    /// Returns a compressed representation with the original ID, compressed text,
    /// compression ratio, and extracted key facts.
    pub fn compress_memory(&self, memory: &MemoryNode) -> CompressedMemory {
        self.compressor.compress(memory)
    }

    /// Compress a batch of memories.
    pub fn compress_memories(&self, memories: &[MemoryNode]) -> Vec<CompressedMemory> {
        self.compressor.compress_batch(memories)
    }

    /// Estimate token count for a text string.
    pub fn estimate_tokens(text: &str) -> usize {
        MemoryCompressor::estimate_tokens(text)
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Archival Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate whether a memory should be kept, archived, or deleted.
    ///
    /// Uses age, salience, and access patterns to make lifecycle decisions.
    pub fn evaluate_archival(&self, memory: &MemoryNode) -> ArchivalDecision {
        if !self.cognitive_config.archival_evaluation {
            return ArchivalDecision::Keep;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.archival.evaluate(memory, now)
    }

    /// Evaluate archival decisions for a batch of memories.
    pub fn evaluate_archival_batch(
        &self,
        memories: &[MemoryNode],
    ) -> Vec<(MemoryId, ArchivalDecision)> {
        if !self.cognitive_config.archival_evaluation {
            return memories
                .iter()
                .map(|m| (m.id, ArchivalDecision::Keep))
                .collect();
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.archival.evaluate_batch(memories, now)
    }

    /// Run archival evaluation on all memories in the database.
    ///
    /// Returns decisions for each memory. Does NOT apply them — call
    /// `invalidate_memory` or `forget` to act on the decisions.
    pub fn evaluate_archival_global(&self) -> MenteResult<Vec<(MemoryId, ArchivalDecision)>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let pm = self.page_map.read();
        let memories: Vec<MemoryNode> = pm
            .values()
            .filter_map(|pid| self.storage.load_memory(*pid).ok())
            .collect();
        drop(pm);
        Ok(self.archival.evaluate_batch(&memories, now))
    }

    // -----------------------------------------------------------------------
    // Sleeptime Enrichment Pipeline
    // -----------------------------------------------------------------------

    /// Check whether enrichment is pending (triggered by turn count or manual).
    pub fn needs_enrichment(&self) -> bool {
        if !self.cognitive_config.enrichment_config.enabled {
            return false;
        }
        *self.enrichment_pending.read()
    }

    /// Get the turn ID when enrichment last completed.
    pub fn last_enrichment_turn(&self) -> u64 {
        *self.last_enrichment_turn.read()
    }

    /// Manually trigger enrichment on the next check.
    pub fn request_enrichment(&self) {
        *self.enrichment_pending.write() = true;
    }

    /// Get episodic memories that haven't been enriched yet.
    ///
    /// Returns all Episodic memories created after the last enrichment turn,
    /// sorted by creation time. These are the candidates for LLM extraction.
    pub fn enrichment_candidates(&self) -> Vec<MemoryNode> {
        let last_turn = *self.last_enrichment_turn.read();
        let pm = self.page_map.read();
        let mut candidates: Vec<MemoryNode> = pm
            .values()
            .filter_map(|pid| self.storage.load_memory(*pid).ok())
            .filter(|m| {
                m.memory_type == mentedb_core::memory::MemoryType::Episodic
                    && !m.tags.contains(&"source:enrichment".to_string())
                    && m.created_at > last_turn
            })
            .collect();
        candidates.sort_by_key(|m| m.created_at);
        candidates
    }

    /// Store enrichment results: extracted memories with provenance tracking.
    ///
    /// Each stored memory gets:
    /// - `source:enrichment` tag for identification
    /// - Confidence capped at `max_enrichment_confidence`
    /// - `Derived` edges back to source episodic memories
    ///
    /// Returns (memories_stored, edges_created).
    pub fn store_enrichment_memories(
        &self,
        memories: Vec<MemoryNode>,
        source_ids: &[MemoryId],
    ) -> MenteResult<(usize, usize)> {
        let max_conf = self
            .cognitive_config
            .enrichment_config
            .max_enrichment_confidence;
        let mut stored = 0usize;
        let mut edges = 0usize;

        for mut mem in memories {
            // Tag as enrichment-generated
            if !mem.tags.contains(&"source:enrichment".to_string()) {
                mem.tags.push("source:enrichment".to_string());
            }
            // Cap confidence
            if mem.confidence > max_conf {
                mem.confidence = max_conf;
            }

            let mem_id = mem.id;
            self.store(mem)?;
            stored += 1;

            // Create Derived edges back to source episodics
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            for src_id in source_ids {
                let edge = MemoryEdge {
                    source: mem_id,
                    target: *src_id,
                    edge_type: EdgeType::Derived,
                    weight: 0.8,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: Some("enrichment".to_string()),
                };
                if self.relate(edge).is_ok() {
                    edges += 1;
                }
            }
        }

        debug!(stored, edges, "enrichment memories stored");
        Ok((stored, edges))
    }

    /// Mark enrichment as complete for the given turn.
    pub fn mark_enrichment_complete(&self, turn_id: u64) {
        *self.last_enrichment_turn.write() = turn_id;
        *self.enrichment_pending.write() = false;
        debug!(turn_id, "enrichment cycle complete");
    }

    /// Get the enrichment configuration.
    pub fn enrichment_config(&self) -> &EnrichmentConfig {
        &self.cognitive_config.enrichment_config
    }

    /// Link entities across sessions by name + embedding similarity.
    ///
    /// Scans all semantic memories tagged with `entity:{name}`, groups by
    /// normalized entity name, then compares embedding similarity within
    /// each group to decide:
    /// - **Merge** (sim ≥ merge_threshold): create `Related` edge (weight = similarity)
    /// - **Ambiguous** (separate_threshold ≤ sim < merge_threshold): tag `maybe_related:{other_id}`
    /// - **Separate** (sim < separate_threshold): no action
    ///
    /// Conservative: when in doubt, don't merge. Two separate nodes > one wrong merge.
    pub fn link_entities(&self) -> MenteResult<EntityLinkResult> {
        let merge_thresh = self
            .cognitive_config
            .enrichment_config
            .entity_merge_threshold;
        let separate_thresh = self
            .cognitive_config
            .enrichment_config
            .entity_separate_threshold;
        let pm = self.page_map.read();

        // Collect all entity memories grouped by normalized name
        let mut entity_groups: HashMap<String, Vec<MemoryNode>> = HashMap::new();
        for pid in pm.values() {
            if let Ok(mem) = self.storage.load_memory(*pid) {
                for tag in &mem.tags {
                    if let Some(name) = tag.strip_prefix("entity:") {
                        let normalized = name.to_lowercase().trim().to_string();
                        entity_groups
                            .entry(normalized)
                            .or_default()
                            .push(mem.clone());
                        break;
                    }
                }
            }
        }

        let mut result = EntityLinkResult::default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for (entity_name, members) in &entity_groups {
            if members.len() < 2 {
                continue;
            }

            // Compare all pairs within the group
            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    let a = &members[i];
                    let b = &members[j];

                    if a.embedding.is_empty() || b.embedding.is_empty() {
                        continue;
                    }

                    let sim = mentedb_consolidation::cosine_similarity(&a.embedding, &b.embedding);

                    if sim >= merge_thresh {
                        // High confidence: create Related edge
                        let edge = MemoryEdge {
                            source: a.id,
                            target: b.id,
                            edge_type: EdgeType::Related,
                            weight: sim,
                            created_at: now,
                            valid_from: None,
                            valid_until: None,
                            label: Some(format!("entity_link:{}", entity_name)),
                        };
                        if self.relate(edge).is_ok() {
                            result.edges_created += 1;
                        }
                        result.linked += 1;
                        debug!(
                            entity = entity_name,
                            a = %a.id, b = %b.id, sim,
                            "entity link: merged"
                        );
                    } else if sim >= separate_thresh {
                        // Ambiguous: tag for later resolution
                        result.ambiguous += 1;
                        debug!(
                            entity = entity_name,
                            a = %a.id, b = %b.id, sim,
                            "entity link: ambiguous"
                        );
                    }
                    // Below separate_threshold: different entities, no action
                }
            }
        }

        debug!(
            linked = result.linked,
            ambiguous = result.ambiguous,
            edges = result.edges_created,
            groups = entity_groups.len(),
            "entity linking complete"
        );
        Ok(result)
    }

    /// Get all entity memory nodes (memories tagged with `entity:{name}`).
    pub fn entity_memories(&self) -> Vec<MemoryNode> {
        let pm = self.page_map.read();
        pm.values()
            .filter_map(|pid| self.storage.load_memory(*pid).ok())
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity:")))
            .collect()
    }
}
