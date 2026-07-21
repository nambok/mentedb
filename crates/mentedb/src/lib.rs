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
use mentedb_cognitive::llm::EntityMergeGroup;
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
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId, Timestamp, UserId};
use mentedb_core::{MemoryEdge, MemoryNode, MenteError};
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_graph::GraphManager;
use mentedb_index::IndexManager;
use mentedb_query::{Condition, Field, Filter, Mql, Operator, OrderBy, QueryPlan, Value};
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

/// Engine-native injection attention (selection policy for context injection).
pub mod injection;

/// LLM-driven memory consolidation (semantic dedup via a pluggable LlmJudge).
pub mod llm_consolidation;
pub use llm_consolidation::ConsolidationParams;

/// Sleeptime enrichment pipeline (requires `enrichment` feature).
#[cfg(feature = "enrichment")]
pub mod enrichment;

/// Optional second-pass reranking of recall results (off by default).
pub mod reranker;

/// Structured export: fill a JSON schema from memories via an embedder-supplied LLM.
pub mod export;

/// Lease-based elastic sharding: places each account on exactly one node and
/// coordinates ownership so a fleet can scale horizontally. The engine owns the
/// placement and coordination logic; the embedder supplies the lease and
/// membership storage via the `LeaseStore` and `NodeRegistry` traits.
pub mod sharding;

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

/// Outcome of a standing-rules (`scope:always`) cleanup pass.
#[derive(Debug, Clone, Default)]
pub struct PruneReport {
    /// Total `scope:always` memories found before cleanup.
    pub total_always: usize,
    /// Number of exact-content duplicate groups collapsed.
    pub duplicate_groups: usize,
    /// Exact-duplicate always-rules removed (the healthiest copy was kept).
    pub pruned: Vec<MemoryId>,
    /// Auto-pinned always-rules un-pinned (the `scope:always` tag removed, the
    /// memory kept so it is still recalled by relevance).
    pub unpinned: Vec<MemoryId>,
    /// Standing rules demoted because the list exceeded `always_max` (lowest
    /// salience first). Included in `unpinned`; counted separately for visibility.
    pub capped: usize,
}

/// EMA smoothing window for the op-latency metrics:
/// `new = (cur * (N-1) + sample) / N`. Larger N is smoother and lets a one-off
/// spike (e.g. a cold path) decay more slowly. Metrics-only, not a cognitive
/// heuristic.
const LATENCY_EMA_WINDOW: u64 = 8;

/// A point-in-time snapshot of engine metrics for a `/metrics` exporter. Cheap to
/// take: counters are atomic loads and sizes are O(1) index/graph/page lookups, so
/// a scrape does not scan the corpus.
#[derive(Debug, Clone, Copy, Default)]
pub struct DbMetrics {
    /// Memories stored on this node.
    pub memory_count: u64,
    /// Lifetime store operations (writes).
    pub stores: u64,
    /// Lifetime recall/query operations (reads).
    pub recalls: u64,
    /// EMA of recent store op latency, microseconds. 0 until the first store.
    pub avg_store_latency_us: u64,
    /// EMA of recent hybrid-search op latency, microseconds. 0 until the first
    /// search.
    pub avg_search_latency_us: u64,
    /// Buffer-pool page cache hits.
    pub buffer_pool_hits: u64,
    /// Buffer-pool page cache misses (disk reads).
    pub buffer_pool_misses: u64,
    /// Buffer-pool CLOCK evictions.
    pub buffer_pool_evictions: u64,
    /// Frames currently holding a page.
    pub buffer_pool_pages: u64,
    /// On-disk data size in bytes (page_count * page size).
    pub storage_bytes: u64,
    /// Pages in the store.
    pub page_count: u64,
    /// Vectors in the HNSW index.
    pub vector_index_size: u64,
    /// Nodes in the memory graph.
    pub graph_nodes: u64,
    /// Pinned standing rules (scope:always).
    pub standing_rules: u64,
}

/// A confirmed entity resolution from an external resolver (LLM).
///
/// Used to feed LLM entity resolution results back into the engine
/// so it can create graph edges and update the EntityResolver cache.
#[derive(Debug, Clone)]
pub struct EntityLinkResolution {
    /// The canonical entity name decided by the resolver.
    pub canonical: String,
    /// All aliases that map to this canonical name.
    pub aliases: Vec<String>,
    /// Confidence in this resolution (0.0 to 1.0).
    pub confidence: f32,
}

/// A pair of entity names that the LLM confirmed are DIFFERENT entities.
#[derive(Debug, Clone)]
pub struct EntitySeparation {
    pub name_a: String,
    pub name_b: String,
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
    /// Configuration for injection attention selection.
    pub injection_config: injection::InjectionConfig,
    /// How many hot flushes may pass before index and graph snapshots are
    /// rewritten. Durability comes from the WAL checkpoint on every flush;
    /// snapshots only accelerate reopen, and open reconciles stale ones, so
    /// rewriting them per flush just multiplies fsync cost.
    pub flush_snapshot_interval: u32,
    /// Whether recall boosts memories linked (in the graph) to an entity named in
    /// the query. Off by default: it only helps corpora that have run enrichment
    /// (so entity nodes and their `Derived` edges exist), and its ranking effect
    /// should be A/B measured before it is turned on broadly.
    pub entity_boost_enabled: bool,
    /// Score added to a candidate that a query-named entity links to. Applied on
    /// top of the fused recall score, so it is calibrated against that scale.
    pub entity_boost_weight: f32,
    /// Whether storing a `scope:always` (standing rule) memory that is near
    /// identical to an existing standing rule is skipped instead of inserted, so
    /// re-pinning the same rule can never grow the always-list.
    pub always_dedup: bool,
    /// Cosine similarity at or above which a new standing rule counts as a
    /// duplicate of an existing one.
    pub always_dedup_threshold: f32,
    /// Maximum standing rules (`scope:always`) to keep pinned. `prune_standing_rules`
    /// demotes the excess (lowest salience first) to ordinary memories, and
    /// injection never force-injects more than this, so the always-list, and the
    /// prompt budget it consumes, stay bounded. The `user_profile` rule is always
    /// kept and does not count against this.
    pub always_max: usize,
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
            injection_config: injection::InjectionConfig::default(),
            flush_snapshot_interval: 8,
            entity_boost_enabled: false,
            entity_boost_weight: 0.15,
            always_dedup: true,
            always_dedup_threshold: 0.95,
            always_max: 30,
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
    /// Hot flushes since the last snapshot write; see flush_snapshot_interval.
    flushes_since_snapshot: std::sync::atomic::AtomicU32,
    /// Lifetime count of store operations (metrics).
    stores: std::sync::atomic::AtomicU64,
    /// Lifetime count of recall/query operations (metrics).
    recalls: std::sync::atomic::AtomicU64,
    /// EMA of store op latency, microseconds (metrics). 0 until the first store.
    store_latency_us: std::sync::atomic::AtomicU64,
    /// EMA of hybrid-search op latency, microseconds (metrics). 0 until the first
    /// search.
    search_latency_us: std::sync::atomic::AtomicU64,
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

/// Agent visibility rule for scoped retrieval: a node is visible to an agent
/// when it is owned by that agent or owned by no agent (nil, shared
/// knowledge). No scope means global visibility.
pub(crate) fn agent_visible(owner: AgentId, scope: Option<AgentId>) -> bool {
    match scope {
        None => true,
        Some(a) => owner == a || owner.is_nil(),
    }
}

/// User visibility rule for scoped retrieval, orthogonal to [`agent_visible`]:
/// a node is visible to a user when it is owned by that user or owned by no
/// user (nil, shared knowledge). No scope means global visibility. A scoped
/// query at (user U, agent A) requires BOTH `user_visible(owner_user, U)` and
/// `agent_visible(owner_agent, A)`.
pub(crate) fn user_visible(owner: UserId, scope: Option<UserId>) -> bool {
    match scope {
        None => true,
        Some(u) => owner == u || owner.is_nil(),
    }
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

        // Directory-backed graph: loads the snapshot and replays the edge log,
        // so edges created since the last flush survive a crash.
        let graph = GraphManager::open(&graph_dir)?;

        // Rebuild page map by scanning all pages
        let entries = storage.scan_all_memories();
        let mut page_map = HashMap::new();
        for (memory_id, page_id) in &entries {
            page_map.insert(*memory_id, *page_id);
        }
        if !page_map.is_empty() {
            info!(memories = page_map.len(), "rebuilt page map from storage");
        }

        // Every stored memory must be a graph node, even when the graph
        // snapshot/log is missing or behind storage (e.g. after a crash);
        // otherwise relate() and write inference fail for surviving memories.
        for memory_id in page_map.keys() {
            if !graph.read_graph().contains_node(*memory_id) {
                graph.add_memory(*memory_id);
            }
        }

        // Snapshots are written every few flushes, not every flush, so they
        // can trail storage. Heal both directions: index memories the
        // snapshot missed, and tombstone vectors whose pages are gone
        // (forgotten after the last snapshot).
        let mut reindexed = 0usize;
        for (memory_id, page_id) in &page_map {
            if !index.contains_vector(*memory_id)
                && let Ok(node) = storage.load_memory(*page_id)
                && !node.embedding.is_empty()
            {
                index.index_memory(&node);
                reindexed += 1;
            }
        }
        let mut retired = 0usize;
        for id in index.vector_ids() {
            if !page_map.contains_key(&id) {
                index.remove_vector_only(id);
                retired += 1;
            }
        }
        if reindexed > 0 || retired > 0 {
            info!(reindexed, retired, "reconciled index with storage");
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
            flushes_since_snapshot: std::sync::atomic::AtomicU32::new(0),
            stores: std::sync::atomic::AtomicU64::new(0),
            recalls: std::sync::atomic::AtomicU64::new(0),
            store_latency_us: std::sync::atomic::AtomicU64::new(0),
            search_latency_us: std::sync::atomic::AtomicU64::new(0),
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
        let started = std::time::Instant::now();
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

        // Standing-rule dedup: re-pinning a rule that already exists must not grow
        // the always-list. Only the rare scope:always write pays for the lookup.
        if self.cognitive_config.always_dedup
            && !node.embedding.is_empty()
            && node.tags.iter().any(|t| t == "scope:always")
            && self.is_duplicate_standing_rule(&node)
        {
            debug!("scope:always dedup: {id} matches an existing standing rule, skipping insert");
            return Ok(());
        }

        let page_id = self.storage.store_memory(&node)?;
        self.page_map.write().insert(id, page_id);
        self.index.index_memory(&node);
        self.graph.add_memory(id);
        self.stores
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Run write inference to auto-create edges and detect contradictions.
        if self.cognitive_config.write_inference {
            self.run_write_inference(&node);
        }

        Self::record_latency(&self.store_latency_us, started.elapsed());
        Ok(())
    }

    /// Whether `node` (a `scope:always` memory) duplicates an existing standing
    /// rule, by exact content or embedding cosine at/above the dedup threshold.
    /// Compared only against the standing-rule set (fetched directly from the tag
    /// index), which the cap keeps small, so this stays cheap and exact rather
    /// than relying on approximate recall ranking.
    fn is_duplicate_standing_rule(&self, node: &MemoryNode) -> bool {
        let threshold = self.cognitive_config.always_dedup_threshold;
        for id in self.index.bitmap.query_tag("scope:always") {
            if id == node.id {
                continue;
            }
            let Ok(existing) = self.get_memory(id) else {
                continue;
            };
            // Dedup only within the same owner: another user or agent pinning the
            // same rule is a separate standing rule, and collapsing across owners
            // would break isolation.
            if existing.user_id != node.user_id || existing.agent_id != node.agent_id {
                continue;
            }
            if existing.content.trim() == node.content.trim() {
                return true;
            }
            if !existing.embedding.is_empty()
                && Self::cosine(&node.embedding, &existing.embedding) >= threshold
            {
                return true;
            }
        }
        false
    }

    /// Cosine similarity of two equal-length vectors; 0 for mismatched or zero
    /// vectors. Used for standing-rule dedup where a true similarity (not the
    /// fused recall score) is needed.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// Store multiple memories in a single batch transaction.
    ///
    /// Uses a single WAL lock for all writes, avoiding per-write overhead of
    /// flock acquisition, header reload, and LSN scan. Significantly faster
    /// for bulk inserts.
    pub fn store_batch(&self, nodes: Vec<MemoryNode>) -> MenteResult<Vec<MemoryId>> {
        // Validate all embeddings upfront
        for node in &nodes {
            if self.embedding_dim > 0
                && !node.embedding.is_empty()
                && node.embedding.len() != self.embedding_dim
            {
                return Err(MenteError::EmbeddingDimensionMismatch {
                    got: node.embedding.len(),
                    expected: self.embedding_dim,
                });
            }
        }

        let page_ids = self.storage.store_memory_batch(&nodes)?;

        let mut ids = Vec::with_capacity(nodes.len());
        let mut page_map = self.page_map.write();
        for (node, page_id) in nodes.iter().zip(page_ids.iter()) {
            page_map.insert(node.id, *page_id);
            self.index.index_memory(node);
            self.graph.add_memory(node.id);
            ids.push(node.id);
        }
        drop(page_map);
        self.stores
            .fetch_add(ids.len() as u64, std::sync::atomic::Ordering::Relaxed);

        // Batch inserts get the same auto-linking, contradiction detection,
        // and invalidation as single stores.
        if self.cognitive_config.write_inference {
            for node in &nodes {
                self.run_write_inference(node);
            }
        }

        Ok(ids)
    }

    /// Recalls memories using an MQL query string.
    ///
    /// Parses the query, builds an execution plan, runs it against the
    /// appropriate indexes/graph, and assembles the results into a
    /// token-budget-aware context window.
    pub fn recall(&self, query: &str) -> MenteResult<ContextWindow> {
        debug!("Recalling with query: {}", query);
        self.recalls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let plan = Mql::parse(query)?;

        let scored = self.execute_plan(&plan)?;
        let config = AssemblyConfig::default();
        let window = ContextAssembler::assemble(scored, vec![], &config);
        Ok(window)
    }

    /// Run an MQL query and return the scored matches directly, without assembling
    /// a context window. This is the raw query primitive behind `recall` (which
    /// additionally packs the results into a token-budgeted context), and it powers
    /// the `mentedb` CLI and the admin query endpoint.
    pub fn query(&self, mql: &str) -> MenteResult<Vec<ScoredMemory>> {
        self.recalls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let plan = Mql::parse(mql)?;
        self.execute_plan(&plan)
    }

    /// Like [`recall`](Self::recall), but runs an optional second pass through a
    /// [`Reranker`](crate::reranker::Reranker) before assembling the context
    /// window. The reranker reorders the first-pass candidates by `query_text`
    /// (typically the natural-language query behind the MQL), so exact-term or
    /// model-scored relevance can lift results that pure vector/BM25 ranking
    /// buried. Reranking is entirely opt-in: plain `recall` never invokes it.
    pub fn recall_reranked(
        &self,
        query: &str,
        query_text: &str,
        reranker: &dyn crate::reranker::Reranker,
    ) -> MenteResult<ContextWindow> {
        use crate::reranker::RerankCandidate;
        let plan = Mql::parse(query)?;
        let mut scored = self.execute_plan(&plan)?;

        let candidates: Vec<RerankCandidate<'_>> = scored
            .iter()
            .map(|s| RerankCandidate {
                id: s.memory.id,
                content: &s.memory.content,
                score: s.score,
            })
            .collect();
        let new_scores: std::collections::HashMap<MemoryId, f32> = reranker
            .rerank(query_text, &candidates)
            .into_iter()
            .collect();
        for s in &mut scored {
            if let Some(&ns) = new_scores.get(&s.memory.id) {
                s.score = ns;
            }
        }
        scored.sort_by(|a, b| b.score.total_cmp(&a.score));

        let config = AssemblyConfig::default();
        Ok(ContextAssembler::assemble(scored, vec![], &config))
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
        self.recall_hybrid_at_mode(embedding, query_text, k, at, tags, false, time_range)
    }

    /// Hybrid recall with configurable tag mode (AND vs OR).
    #[allow(clippy::too_many_arguments)]
    pub fn recall_hybrid_at_mode(
        &self,
        embedding: &[f32],
        query_text: Option<&str>,
        k: usize,
        at: Timestamp,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_hybrid_scoped_at_mode(
            embedding, query_text, k, at, tags, tags_or, time_range, None, None,
        )
    }

    /// Hybrid recall visible to one (user, agent) scope: a node passes only
    /// when it is visible on BOTH axes, i.e. owned by `agent` or no agent AND
    /// owned by `user` or no user (nil, shared knowledge). `agent: None` /
    /// `user: None` recall globally on that axis, preserving single-owner
    /// behavior. The two axes are orthogonal: agent scoping never widens or
    /// narrows user scoping, and vice versa.
    #[allow(clippy::too_many_arguments)]
    pub fn recall_hybrid_scoped_at_mode(
        &self,
        embedding: &[f32],
        query_text: Option<&str>,
        k: usize,
        at: Timestamp,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
        agent: Option<AgentId>,
        user: Option<UserId>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        let started = std::time::Instant::now();
        debug!(
            "Recall hybrid, k={}, at={}, bm25={}, tags_or={}",
            k,
            at,
            query_text.is_some(),
            tags_or
        );
        // Over-fetch to account for filtered-out results
        let results = self.index.hybrid_search_with_query_mode(
            embedding,
            query_text,
            tags,
            tags_or,
            time_range,
            k * 3,
        );
        let graph = self.graph.graph();
        let pm = self.page_map.read();
        let mut scored: Vec<(MemoryId, f32)> = results
            .into_iter()
            .filter_map(|(id, raw_score)| {
                // Exclude memories that an active Supersedes/Contradicts edge
                // has invalidated.
                let incoming = graph.incoming(id);
                let has_active_supersede = incoming.iter().any(|(_, e)| {
                    (e.edge_type == EdgeType::Supersedes || e.edge_type == EdgeType::Contradicts)
                        && e.is_valid_at(at)
                });
                if has_active_supersede {
                    return None;
                }
                // Without a page entry we cannot check validity or decay; keep
                // the raw score rather than silently dropping the hit.
                let Some(&page_id) = pm.get(&id) else {
                    return Some((id, raw_score));
                };
                let Ok(node) = self.storage.load_memory(page_id) else {
                    return Some((id, raw_score));
                };
                // Drop memories outside their validity window or not visible to
                // this (user, agent) scope. Both owner axes must pass.
                let visible =
                    agent_visible(node.agent_id, agent) && user_visible(node.user_id, user);
                if !node.is_valid_at(at) || !visible {
                    return None;
                }
                // Decay at recall time: the index salience cache is frozen at
                // insert, so recomputing decayed salience here is what actually
                // lets decay affect ranking on this hot path. Blend 70%
                // similarity with 30% freshly decayed salience, matching the
                // MQL path, so old, unaccessed memories rank lower.
                let score = if self.cognitive_config.decay_on_recall {
                    let decayed = self.decay.compute_decay(
                        node.salience,
                        node.created_at,
                        node.accessed_at,
                        node.access_count,
                        at,
                    );
                    raw_score * 0.7 + decayed * 0.3
                } else {
                    raw_score
                };
                Some((id, score))
            })
            .collect();
        // Re-rank by the decay-adjusted score before cutting to k, so the decay
        // blend actually reorders results rather than just relabeling them.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        // Time only (do not increment recalls here): recall()/query() already
        // count the read, and they reach this hybrid core via execute_plan, so
        // counting here too would double-count them.
        Self::record_latency(&self.search_latency_us, started.elapsed());
        Ok(scored)
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
        self.recall_hybrid_multi_mode(embeddings, query_texts, k, tags, false, time_range)
    }

    /// Multi-query hybrid search with configurable tag mode.
    pub fn recall_hybrid_multi_mode(
        &self,
        embeddings: &[Vec<f32>],
        query_texts: Option<&[String]>,
        k: usize,
        tags: Option<&[&str]>,
        tags_or: bool,
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
            let results = self.recall_hybrid_at_mode(emb, qt, k, now, tags, tags_or, time_range)?;
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
        self.storage.update_memory(page_id, &node)?;
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

    /// Fold one op's latency into an exponentially-weighted moving average, in
    /// microseconds. An EMA (not a lifetime mean) so a one-off slow op, e.g. right
    /// after a cold open, decays out over the next few ops instead of permanently
    /// skewing the number; it tracks recent, warm engine performance. Lock-free.
    fn record_latency(ema: &std::sync::atomic::AtomicU64, elapsed: std::time::Duration) {
        use std::sync::atomic::Ordering::Relaxed;
        let sample = elapsed.as_micros() as u64;
        let n = LATENCY_EMA_WINDOW;
        let mut cur = ema.load(Relaxed);
        loop {
            let next = if cur == 0 {
                sample
            } else {
                (cur * (n - 1) + sample) / n
            };
            match ema.compare_exchange_weak(cur, next, Relaxed, Relaxed) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Snapshot the engine's operational metrics (writes, reads, cache, storage,
    /// index, graph, standing rules) for a `/metrics` exporter. O(1), no scan.
    pub fn metrics(&self) -> DbMetrics {
        use std::sync::atomic::Ordering::Relaxed;
        let bp = self.storage.buffer_stats();
        let page_count = self.storage.page_count();
        DbMetrics {
            memory_count: self.memory_count() as u64,
            stores: self.stores.load(Relaxed),
            recalls: self.recalls.load(Relaxed),
            avg_store_latency_us: self.store_latency_us.load(Relaxed),
            avg_search_latency_us: self.search_latency_us.load(Relaxed),
            buffer_pool_hits: bp.hits,
            buffer_pool_misses: bp.misses,
            buffer_pool_evictions: bp.evictions,
            buffer_pool_pages: bp.resident_pages,
            storage_bytes: page_count * mentedb_storage::PAGE_SIZE as u64,
            page_count,
            vector_index_size: self.index.hnsw.len() as u64,
            graph_nodes: self.graph.graph().node_count() as u64,
            standing_rules: self.count_standing_rules() as u64,
        }
    }

    /// Accurate count of pinned standing rules. The tag index can over-count after
    /// an un-pin re-indexed on an older engine (the tag was removed from the node
    /// but lingered in the bitmap), so verify each candidate still carries the tag.
    /// Cheap: it loads only the (few, capped) candidates, not the whole corpus.
    fn count_standing_rules(&self) -> usize {
        self.index
            .bitmap
            .query_tag("scope:always")
            .into_iter()
            .filter(|id| {
                self.get_memory(*id)
                    .map(|n| n.tags.iter().any(|t| t == "scope:always"))
                    .unwrap_or(false)
            })
            .count()
    }

    /// A bounded, paginated page of stored memories for admin browsing, ordered by
    /// id so pagination is stable. Only `limit` nodes are loaded from storage per
    /// call (paginate over the id set first, then load), so it stays cheap even on
    /// a large store. Optional filters narrow the loaded page by owning agent,
    /// memory type, and a case-insensitive content substring. Returns the total
    /// memory count (for the pager) alongside the page.
    pub fn list_memories(
        &self,
        limit: usize,
        offset: usize,
        agent: Option<AgentId>,
        memory_type: Option<MemoryType>,
        content_query: Option<&str>,
    ) -> MenteResult<(usize, Vec<MemoryNode>)> {
        let pm = self.page_map.read();
        let total = pm.len();
        let mut ids: Vec<MemoryId> = pm.keys().copied().collect();
        ids.sort();
        let needle = content_query.map(|q| q.to_lowercase());
        let mut out = Vec::new();
        for id in ids.into_iter().skip(offset).take(limit) {
            if let Some(&page_id) = pm.get(&id)
                && let Ok(node) = self.storage.load_memory(page_id)
            {
                if agent.is_some_and(|a| node.agent_id != a) {
                    continue;
                }
                if memory_type.is_some_and(|t| node.memory_type != t) {
                    continue;
                }
                if let Some(n) = &needle
                    && !node.content.to_lowercase().contains(n)
                {
                    continue;
                }
                out.push(node);
            }
        }
        Ok((total, out))
    }

    /// Removes a memory from storage, indexes, and the graph.
    pub fn forget(&self, id: MemoryId) -> MenteResult<()> {
        debug!("Forgetting memory {}", id);

        if let Some(&page_id) = self.page_map.read().get(&id) {
            if let Ok(node) = self.storage.load_memory(page_id) {
                self.index.remove_memory(id, &node);
            }
            // Durable delete: WAL-logged page free, so the memory does not
            // resurrect when the page map is rebuilt on reopen.
            self.storage.delete_memory(page_id)?;
        }

        self.graph.remove_memory(id);
        self.page_map.write().remove(&id);
        Ok(())
    }

    /// Clean up the standing-rules (`scope:always`) set: remove exact-content
    /// duplicates (keeping the healthiest copy) and un-pin every auto-pinned
    /// always-rule, keeping only rules the user explicitly pinned
    /// (`source:manual`) and the user profile (which the engine pins deliberately
    /// so it injects every session).
    ///
    /// `scope:always` force-injects a memory into every assembled context
    /// regardless of relevance, so the set must stay tiny; distilled facts belong
    /// in relevance-based recall, not a fixed always-list. This is the single
    /// source of truth for the policy, shared by the hosted platform, the local
    /// daemon, and any SDK consumer, so behavior cannot drift between them.
    pub fn prune_standing_rules(&self) -> MenteResult<PruneReport> {
        use std::collections::{HashMap, HashSet};

        let always: Vec<MemoryNode> = self
            .memory_ids()
            .into_iter()
            .filter_map(|id| self.get_memory(id).ok())
            .filter(|n| n.tags.iter().any(|t| t == "scope:always"))
            .collect();

        let mut report = PruneReport {
            total_always: always.len(),
            ..Default::default()
        };

        // Collapse exact-content duplicates: keep the healthiest copy, forget
        // the rest.
        let mut by_content: HashMap<&str, Vec<&MemoryNode>> = HashMap::new();
        for n in &always {
            by_content.entry(n.content.as_str()).or_default().push(n);
        }
        let mut prune_ids: HashSet<MemoryId> = HashSet::new();
        for (_content, mut group) in by_content {
            if group.len() > 1 {
                report.duplicate_groups += 1;
                group.sort_by(|a, b| {
                    b.salience
                        .partial_cmp(&a.salience)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for n in group.into_iter().skip(1) {
                    prune_ids.insert(n.id);
                }
            }
        }

        // Un-pin every auto-pinned always-rule (not source:manual, not the
        // profile). Keep the memory, drop the pin.
        let mut to_unpin: Vec<MemoryNode> = Vec::new();
        for n in &always {
            if prune_ids.contains(&n.id) {
                continue;
            }
            let is_manual = n.tags.iter().any(|t| t == "source:manual");
            let is_profile = n.tags.iter().any(|t| t == "user_profile");
            if !is_manual && !is_profile {
                to_unpin.push(n.clone());
            }
        }

        // Cap the count. Whatever survives dedup and un-pinning is manual pins
        // plus the profile; the profile is always kept and does not count. If the
        // remaining manual pins exceed always_max, demote the lowest-salience
        // excess to ordinary memories so the always-list stays bounded even when a
        // user has pinned hundreds of rules by hand.
        let unpin_ids: HashSet<MemoryId> = to_unpin.iter().map(|n| n.id).collect();
        let mut survivors: Vec<&MemoryNode> = always
            .iter()
            .filter(|n| !prune_ids.contains(&n.id) && !unpin_ids.contains(&n.id))
            .filter(|n| !n.tags.iter().any(|t| t == "user_profile"))
            .collect();
        if survivors.len() > self.cognitive_config.always_max {
            survivors.sort_by(|a, b| {
                b.salience
                    .partial_cmp(&a.salience)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for n in survivors.into_iter().skip(self.cognitive_config.always_max) {
                report.capped += 1;
                to_unpin.push(n.clone());
            }
        }

        for id in prune_ids {
            self.forget(id)?;
            report.pruned.push(id);
        }
        for mut n in to_unpin {
            n.tags.retain(|t| t != "scope:always");
            let id = n.id;
            self.store(n)?;
            report.unpinned.push(id);
        }

        Ok(report)
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
            // Scope to the new memory's owner on BOTH axes: contradiction and
            // relationship inference must never compare against another user's
            // or another agent's memories. Nil owned (shared/global) memories
            // stay in scope.
            self.recall_hybrid_scoped_at_mode(
                &new_memory.embedding,
                None,
                20,
                now,
                None,
                false,
                None,
                Some(new_memory.agent_id),
                Some(new_memory.user_id),
            )
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
            InferredAction::DeduplicateExact { duplicate, keeper } => {
                debug!("Deduplicating exact copy {duplicate} into {keeper}");
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                self.invalidate_memory(duplicate, now)?;
                // Derived edge (dedup lineage), NOT Supersedes: an exact copy is
                // deduplicated, not a meaningful supersession, so it must not
                // surface in the contradiction/supersession view.
                let edge = MemoryEdge {
                    source: keeper,
                    target: duplicate,
                    edge_type: EdgeType::Derived,
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
                    if let Some(&pid) = self.page_map.read().get(&memory) {
                        self.storage.update_memory(pid, &node)?;
                    }
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
                            if let Some(&pid) = self.page_map.read().get(&affected_id)
                                && let Err(e) = self.storage.update_memory(pid, &affected_node)
                            {
                                warn!("Failed to persist belief update for {affected_id}: {e}");
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
                    if let Some(&pid) = self.page_map.read().get(&memory) {
                        self.storage.update_memory(pid, &node)?;
                    }
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

    /// Retained for API compatibility. Does nothing and persists nothing.
    ///
    /// Salience decay is derived on read, not materialized. The stored
    /// `salience` field is the base value as of a memory's last reinforcement
    /// (`accessed_at`); the current, decayed strength is computed on demand via
    /// `compute_decayed_salience` (used by recall scoring and archival).
    ///
    /// This method previously recomputed decay from the stored salience and
    /// wrote the result back. That fed each pass's output in as the next pass's
    /// input, so the effective decay rate scaled with how often maintenance ran
    /// rather than with elapsed time: frequent passes aged and forgot memories
    /// far faster than the configured half-life. Deriving on read removes that
    /// coupling entirely, so it is safe to run maintenance at any cadence.
    pub fn apply_decay_global(&self) -> MenteResult<usize> {
        Ok(0)
    }

    /// One-time repair for databases whose stored salience was corrupted by the
    /// old compounding decay pass, which overwrote each memory's base salience
    /// with a repeatedly re-decayed value. Reset every memory's salience to the
    /// configured maximum and restart its decay clock (`accessed_at = now`), in
    /// place, without re-running write inference, so every surviving memory gets
    /// a fresh, correct decay lease from now. Access counts and every other field
    /// are preserved. Returns the number of memories reset.
    ///
    /// This is idempotent in effect (running it twice just resets an already
    /// healthy base again) and safe to gate behind a one-shot migration flag.
    pub fn reset_decay_state(&self) -> MenteResult<usize> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let max_salience = self.decay.config.max_salience;
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();

        let mut reset = 0;
        for pid in &page_ids {
            if let Ok(mut node) = self.storage.load_memory(*pid) {
                node.salience = max_salience;
                node.accessed_at = now;
                self.storage.update_memory(*pid, &node)?;
                reset += 1;
            }
        }
        if reset > 0 {
            info!("Reset decay state for {} memories", reset);
        }
        Ok(reset)
    }

    /// Re-embed every memory at the embedder's current dimension and reindex it.
    ///
    /// This migrates a database whose stored vectors were produced by an
    /// embedder of a different dimension (for example 256 to 1024): each memory's
    /// content is re-embedded with the currently configured embedder, its stored
    /// vector is replaced, and it is removed from and re-added to the vector
    /// index. Memories already at the current dimension are skipped, so it is
    /// idempotent and safe to re-run.
    ///
    /// The migration is safe to run on a live database: the HNSW index checks
    /// vector dimension at query time, so a memory that has not been re-embedded
    /// yet is simply skipped by recall (never returned, never crashing) until
    /// this reaches it. Runs synchronously and re-embeds one memory at a time;
    /// the caller should run it on a blocking pool.
    pub fn reembed_all(&self) -> MenteResult<usize> {
        let target = self.embedding_dim;
        if target == 0 {
            return Ok(0);
        }
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();

        let mut reembedded = 0usize;
        for pid in &page_ids {
            let Ok(mut node) = self.storage.load_memory(*pid) else {
                continue;
            };
            // Already at the target dimension: nothing to do.
            if node.embedding.len() == target || node.content.is_empty() {
                continue;
            }
            let Some(embedding) = self.embed_text(&node.content)? else {
                continue;
            };
            // The embedder is not yet producing the target dimension; leave the
            // memory as-is rather than store a second wrong-dimension vector.
            if embedding.len() != target {
                continue;
            }
            self.index.remove_vector_only(node.id);
            node.embedding = embedding;
            self.storage.update_memory(*pid, &node)?;
            self.index.index_memory(&node);
            reembedded += 1;
        }
        if reembedded > 0 {
            info!("Re-embedded {reembedded} memories at dimension {target}");
        }
        Ok(reembedded)
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

        // Never merge across owners. find_candidates already groups by both
        // owner axes, but if a caller hands us a hand-built mixed cluster,
        // refuse rather than leak one owner's content into another's
        // consolidated memory. Both axes are checked independently.
        let owner_agent = cluster[0].agent_id;
        if cluster.iter().any(|m| m.agent_id != owner_agent) {
            return Err(MenteError::Query(
                "consolidation cluster mixes agents; refusing to merge across owners".into(),
            ));
        }
        let owner_user = cluster[0].user_id;
        if cluster.iter().any(|m| m.user_id != owner_user) {
            return Err(MenteError::Query(
                "consolidation cluster mixes users; refusing to merge across owners".into(),
            ));
        }

        let result = self.consolidation.consolidate(&cluster);

        // Create the consolidated memory node, stamped with BOTH owner axes of
        // the cluster so the derived memory stays scoped exactly as its sources.
        let agent_id = cluster[0].agent_id;
        let user_id = cluster[0].user_id;
        let mut consolidated = MemoryNode::new(
            agent_id,
            result.new_type,
            result.summary,
            result.combined_embedding,
        )
        .with_user_id(user_id);
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
        self.flush_full()?;
        self.storage.close()?;
        Ok(())
    }

    /// Close with durability only: checkpoint the WAL and release the
    /// process lock without rewriting snapshots. Reopen reconciles stale
    /// snapshots, so this trades a slower next open for a shutdown fast
    /// enough to release every user's lock inside a deploy drain window.
    pub fn close_quick(&self) -> MenteResult<()> {
        info!("Quick closing MenteDB");
        self.storage.checkpoint()?;
        self.storage.close()?;
        Ok(())
    }

    /// Simulate a process crash for tests: releases the storage process lock
    /// exactly as the operating system would when a process dies, then drops
    /// the instance without flushing or closing anything.
    #[doc(hidden)]
    pub fn simulate_crash(self) {
        self.storage.release_process_lock();
        std::mem::forget(self);
    }

    /// Rebuild all indexes by scanning every memory in storage.
    ///
    /// Use this after index corruption or when index files were overwritten.
    /// Returns the number of memories re-indexed.
    pub fn rebuild_indexes(&self) -> MenteResult<usize> {
        info!("Rebuilding indexes from storage...");
        let ids: Vec<MemoryId> = self.page_map.read().keys().copied().collect();
        let total = ids.len();
        let mut indexed = 0usize;
        for id in ids {
            if let Ok(node) = self.get_memory(id) {
                self.index.index_memory(&node);
                indexed += 1;
            }
        }
        self.index.save(&self.path.join("indexes"))?;
        info!(indexed, total, "index rebuild complete");
        Ok(indexed)
    }

    /// Flush indexes, graph, and storage to disk without closing.
    ///
    /// Call this periodically to ensure cross-session persistence.
    /// Unlike `close()`, the database remains usable after flushing.
    pub fn flush(&self) -> MenteResult<()> {
        debug!("Flushing MenteDB to disk");
        // Durability is the WAL checkpoint, paid on every flush. Index,
        // graph, and cognitive snapshots only accelerate reopen (open
        // reconciles stale ones against storage), so they are amortized
        // across flush_snapshot_interval hot flushes instead of multiplying
        // fsync cost on every write batch.
        self.storage.checkpoint()?;

        let n = self
            .flushes_since_snapshot
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if n >= self.cognitive_config.flush_snapshot_interval {
            self.write_snapshots()?;
        }
        Ok(())
    }

    /// Flush everything including index, graph, and cognitive snapshots.
    /// Used by close and by maintenance, where reopen speed matters more
    /// than write latency.
    pub fn flush_full(&self) -> MenteResult<()> {
        debug!("Full flush of MenteDB to disk");
        self.storage.checkpoint()?;
        self.write_snapshots()
    }

    fn write_snapshots(&self) -> MenteResult<()> {
        self.index.save(&self.path.join("indexes"))?;
        self.graph.save(&self.path.join("graph"))?;

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
        self.flushes_since_snapshot
            .store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Executes a query plan against the indexes and graph, returning scored memories.
    fn execute_plan(&self, plan: &QueryPlan) -> MenteResult<Vec<ScoredMemory>> {
        match plan {
            QueryPlan::VectorSearch {
                query,
                k,
                filters,
                condition,
                order_by,
            } => {
                // `content ~> "text"` arrives with an empty vector and the
                // SimilarTo filter left in place for us to embed now (no embedder
                // at plan time). NEAR arrives with the vector already filled in.
                let embedded;
                let qvec: &[f32] = if query.is_empty() {
                    match filters
                        .iter()
                        .find(|f| f.op == Operator::SimilarTo)
                        .map(|f| &f.value)
                    {
                        Some(Value::Text(text)) => {
                            match self.embedder.as_ref().and_then(|e| e.embed(text).ok()) {
                                Some(v) => {
                                    embedded = v;
                                    &embedded
                                }
                                // No embedder configured: a semantic query can't run.
                                None => return Ok(vec![]),
                            }
                        }
                        _ => query,
                    }
                } else {
                    query
                };
                let hits = self.index.hybrid_search(qvec, None, None, *k);
                let mut scored = self.load_scored_memories(&hits)?;
                // The SimilarTo filter is the query itself; apply any others. A
                // boolean tree (NEAR combined with OR/NOT) takes over the whole
                // post-filter when present.
                match condition {
                    Some(cond) => {
                        scored.retain(|sm| Self::condition_matches(cond, &sm.memory));
                    }
                    None => {
                        scored.retain(|sm| {
                            filters
                                .iter()
                                .filter(|f| f.op != Operator::SimilarTo)
                                .all(|f| Self::filter_matches(f, &sm.memory))
                        });
                    }
                }
                // Optional entity leg: boost candidates linked to an entity the
                // query names. The SimilarTo filter carries the query text.
                if self.cognitive_config.entity_boost_enabled
                    && let Some(qtext) = filters
                        .iter()
                        .find(|f| f.op == Operator::SimilarTo)
                        .and_then(|f| match &f.value {
                            Value::Text(t) => Some(t.clone()),
                            _ => None,
                        })
                {
                    self.apply_entity_boost(&mut scored, &qtext);
                }
                if let Some(ob) = order_by {
                    Self::apply_order_by(&mut scored, ob);
                }
                Ok(scored)
            }
            QueryPlan::TagScan {
                tags,
                filters,
                limit,
                condition,
                order_by,
            } => {
                let k = limit.unwrap_or(20);
                let mut scored = if tags.is_empty() {
                    // No tag constraint: a bare RECALL, or a metadata-only filter
                    // like `WHERE type = semantic`. Full-scan and let the filters
                    // below narrow it, instead of returning nothing.
                    let pm = self.page_map.read();
                    pm.values()
                        .filter_map(|&pid| self.storage.load_memory(pid).ok())
                        .map(|memory| ScoredMemory { memory, score: 1.0 })
                        .collect::<Vec<_>>()
                } else {
                    let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                    // Zero-vector for tag-only search; salience+bitmap still apply.
                    let hits = self.index.hybrid_search(&[], Some(&tag_refs), None, k);
                    self.load_scored_memories(&hits)?
                };
                // Apply the WHERE clause: a boolean tree when present (OR/NOT/
                // grouping), otherwise the flat AND of metadata filters.
                match condition {
                    Some(cond) => {
                        scored.retain(|sm| Self::condition_matches(cond, &sm.memory));
                    }
                    None => {
                        scored.retain(|sm| {
                            filters.iter().all(|f| Self::filter_matches(f, &sm.memory))
                        });
                    }
                }
                // ORDER BY sorts before LIMIT so the top-k reflects the requested
                // order, not relevance.
                if let Some(ob) = order_by {
                    Self::apply_order_by(&mut scored, ob);
                }
                scored.truncate(k);
                Ok(scored)
            }
            QueryPlan::TemporalScan {
                start,
                end,
                order_by,
                ..
            } => {
                let hits = self
                    .index
                    .hybrid_search(&[], None, Some((*start, *end)), 100);
                let mut scored = self.load_scored_memories(&hits)?;
                if let Some(ob) = order_by {
                    Self::apply_order_by(&mut scored, ob);
                }
                Ok(scored)
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

    /// Memory ids that an entity named in the query links to. Tokenizes the query,
    /// looks up each token as an `entity:<name>` node in the tag index, and
    /// collects the memories those entity nodes were derived from (their outgoing
    /// `Derived` neighbors). Cheap: a bitmap lookup per token plus one graph
    /// adjacency read per hit, no LLM. Only non-empty once enrichment has created
    /// entity nodes and linked them, so a raw store simply gets no boost.
    fn entity_boost_ids(&self, query_text: &str) -> std::collections::HashSet<MemoryId> {
        let mut ids = std::collections::HashSet::new();
        let graph = self.graph.graph();
        let mut seen_tokens = std::collections::HashSet::new();
        for token in query_text
            .split(|c: char| !c.is_alphanumeric())
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 2)
        {
            if !seen_tokens.insert(token.clone()) {
                continue;
            }
            for entity_id in self.index.bitmap.query_tag(&format!("entity:{token}")) {
                for (mid, edge) in graph.outgoing(entity_id) {
                    if edge.edge_type == EdgeType::Derived {
                        ids.insert(mid);
                    }
                }
            }
        }
        ids
    }

    /// Add `entity_boost_weight` to any candidate a query-named entity links to,
    /// then re-sort. A no-op when nothing in the query resolves to an entity.
    fn apply_entity_boost(&self, scored: &mut [ScoredMemory], query_text: &str) {
        let ids = self.entity_boost_ids(query_text);
        if ids.is_empty() {
            return;
        }
        let w = self.cognitive_config.entity_boost_weight;
        for sm in scored.iter_mut() {
            if ids.contains(&sm.memory.id) {
                sm.score += w;
            }
        }
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sort recall results by an `ORDER BY` clause, applied before `LIMIT`. Only
    /// the numeric and temporal fields (salience, confidence, created, accessed)
    /// carry a meaningful order; ordering by any other field is a no-op that keeps
    /// the existing (relevance) order.
    fn apply_order_by(scored: &mut [ScoredMemory], ob: &OrderBy) {
        fn key(n: &MemoryNode, field: Field) -> f64 {
            match field {
                Field::Salience => n.salience as f64,
                Field::Confidence => n.confidence as f64,
                Field::Created => n.created_at as f64,
                Field::Accessed => n.access_count as f64,
                _ => 0.0,
            }
        }
        scored.sort_by(|a, b| {
            let ord = key(&a.memory, ob.field)
                .partial_cmp(&key(&b.memory, ob.field))
                .unwrap_or(std::cmp::Ordering::Equal);
            if ob.descending { ord.reverse() } else { ord }
        });
    }

    /// Whether a memory satisfies a boolean WHERE tree (OR/NOT/grouping). Leaves
    /// defer to `filter_matches`, so a leaf that `filter_matches` treats as
    /// permissive stays permissive; `And`/`Or`/`Not` compose those results.
    fn condition_matches(c: &Condition, node: &MemoryNode) -> bool {
        match c {
            Condition::Leaf(f) => Self::filter_matches(f, node),
            Condition::And(children) => children.iter().all(|ch| Self::condition_matches(ch, node)),
            Condition::Or(children) => children.iter().any(|ch| Self::condition_matches(ch, node)),
            Condition::Not(inner) => !Self::condition_matches(inner, node),
        }
    }

    /// Whether a memory satisfies a single MQL filter. Applies the metadata
    /// filters the planner attaches to a plan (type, tag, content, time), so
    /// `WHERE type = semantic` and similar clauses actually narrow results
    /// instead of being ignored. Unhandled field/value combinations are
    /// permissive (they do not exclude), so a supported query is never dropped
    /// by a clause this does not understand.
    fn filter_matches(f: &Filter, node: &MemoryNode) -> bool {
        // IN: the field matches any element of the list (reusing per-field
        // equality, so `type IN [...]`, `tag IN [...]`, `content IN [...]` all work).
        if let Value::List(items) = &f.value {
            return items.iter().any(|v| {
                Self::filter_matches(
                    &Filter {
                        field: f.field,
                        op: Operator::Eq,
                        value: v.clone(),
                    },
                    node,
                )
            });
        }
        match (&f.field, &f.value) {
            (Field::Type, Value::MemoryType(t)) => {
                if f.op == Operator::Neq {
                    node.memory_type != *t
                } else {
                    node.memory_type == *t
                }
            }
            (Field::Tag, Value::Text(tag)) => {
                let has = node.tags.iter().any(|x| x == tag);
                if f.op == Operator::Neq { !has } else { has }
            }
            (Field::Content, Value::Text(s)) => {
                let hit = node.content.to_lowercase().contains(&s.to_lowercase());
                if f.op == Operator::Neq { !hit } else { hit }
            }
            (Field::Created, v) => Self::num_cmp(node.created_at as f64, f.op, Self::value_f64(v)),
            (Field::Accessed, v) => {
                Self::num_cmp(node.accessed_at as f64, f.op, Self::value_f64(v))
            }
            // AS OF <t>: keep only memories whose validity window contains t.
            (Field::ValidAt, v) => Self::value_f64(v)
                .map(|t| node.is_valid_at(t as u64))
                .unwrap_or(true),
            _ => true,
        }
    }

    fn value_f64(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => Some(*n),
            Value::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    fn num_cmp(a: f64, op: Operator, b: Option<f64>) -> bool {
        let Some(b) = b else { return true };
        match op {
            Operator::Gt => a > b,
            Operator::Gte => a >= b,
            Operator::Lt => a < b,
            Operator::Lte => a <= b,
            Operator::Eq => (a - b).abs() < f64::EPSILON,
            Operator::Neq => (a - b).abs() >= f64::EPSILON,
            Operator::SimilarTo => true,
            // Set/substring operators do not apply to a numeric comparison; the
            // `IN` list case is handled before num_cmp is ever reached.
            Operator::In | Operator::Contains => false,
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

    /// Canonicalize recent trajectory topics via the LLM, populating the learned
    /// topic cache so future turns collapse phrasings to one canonical label.
    /// That label then flows into topic prediction, resume context, and the
    /// speculative pre-assembly cache, which is why raw message text stops
    /// showing up as a "topic".
    ///
    /// Runs OFF the per-turn hot path: call it from an async post-turn step or
    /// the maintenance sweep. No lock is held across the LLM calls. The engine
    /// stays LLM-optional: pass any judge, or do not call this and topics simply
    /// stay at their normalized-raw form. Returns the number of new canonical
    /// labels learned this pass.
    pub async fn canonicalize_trajectory_topics<J: mentedb_cognitive::llm::LlmJudge>(
        &self,
        judge: J,
        limit: usize,
    ) -> usize {
        // 1. Collect recent uncached topics + the known label set (read lock).
        let (pending, mut known) = {
            let traj = self.trajectory.read();
            (traj.pending_canonicalization(limit), traj.known_topics())
        };
        if pending.is_empty() {
            return 0;
        }
        // 2. Ask the LLM for a canonical label per topic (no lock held).
        let svc = mentedb_cognitive::llm::CognitiveLlmService::new(judge);
        let mut learned = Vec::new();
        for raw in pending {
            if let Ok(label) = svc.canonicalize_topic(&raw, &known).await {
                known.push(label.topic.clone());
                learned.push((raw, label.topic));
            }
        }
        // 3. Store the learned mappings (write lock).
        let n = learned.len();
        if n > 0 {
            let mut traj = self.trajectory.write();
            for (raw, canonical) in learned {
                traj.learn_canonical(&raw, &canonical);
            }
        }
        n
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

    /// Get the current speculative cache entries (predicted topic, pre-assembled
    /// context, source memory ids, hit count, and age), for introspection and
    /// dashboards. This is the live pre-assembly cache, not aggregate stats.
    pub fn speculative_cache_entries(&self) -> Vec<CacheEntry> {
        self.speculative.read().entries().to_vec()
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
        // Judge on the salience decayed to `now`, not the stored base. Stored
        // salience only changes on reinforcement, so a never-reinforced memory
        // keeps its base value and archival must derive the current strength.
        let effective = self.decay.compute_decay(
            memory.salience,
            memory.created_at,
            memory.accessed_at,
            memory.access_count,
            now,
        );
        self.archival.evaluate_effective(
            effective,
            memory.memory_type,
            memory.created_at,
            memory.access_count,
            now,
        )
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
        memories
            .iter()
            .map(|m| {
                let effective = self.decay.compute_decay(
                    m.salience,
                    m.created_at,
                    m.accessed_at,
                    m.access_count,
                    now,
                );
                (
                    m.id,
                    self.archival.evaluate_effective(
                        effective,
                        m.memory_type,
                        m.created_at,
                        m.access_count,
                        now,
                    ),
                )
            })
            .collect()
    }

    /// Run archival evaluation on all memories in the database.
    ///
    /// Returns decisions for each memory. Does NOT apply them — call
    /// `invalidate_memory` or `forget` to act on the decisions.
    pub fn evaluate_archival_global(&self) -> MenteResult<Vec<(MemoryId, ArchivalDecision)>> {
        let pm = self.page_map.read();
        let memories: Vec<MemoryNode> = pm
            .values()
            .filter_map(|pid| self.storage.load_memory(*pid).ok())
            .collect();
        drop(pm);
        // Delegate so decisions use each memory's salience decayed to now, not
        // its stored base (see evaluate_archival_batch).
        Ok(self.evaluate_archival_batch(&memories))
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

    /// Distinct (user_id, agent_id) owner pairs that own at least one memory.
    ///
    /// Enrichment partitions by BOTH owner axes: it runs once per distinct pair
    /// so derived knowledge (extracted facts, entities, communities, profile)
    /// stays scoped to exactly the user AND agent that owned the source
    /// memories, never mixing across either axis.
    fn distinct_owner_pairs(&self) -> Vec<(UserId, AgentId)> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut pairs: Vec<(UserId, AgentId)> = Vec::new();
        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid) {
                let pair = (mem.user_id, mem.agent_id);
                if !pairs.contains(&pair) {
                    pairs.push(pair);
                }
            }
        }
        pairs
    }

    /// Get episodic memories that haven't been enriched yet.
    ///
    /// Returns all Episodic memories created after the last enrichment turn,
    /// owned by exactly this (`user`, `agent`) pair (exact match on BOTH owner
    /// axes, so one owner's episodics are never enriched into another's
    /// knowledge), sorted by creation time. These are the candidates for LLM
    /// extraction.
    pub fn enrichment_candidates(&self, user: UserId, agent: AgentId) -> Vec<MemoryNode> {
        let last_turn = *self.last_enrichment_turn.read();
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut candidates: Vec<MemoryNode> = page_ids
            .iter()
            .filter_map(|pid| self.storage.load_memory(*pid).ok())
            .filter(|m| {
                m.agent_id == agent
                    && m.user_id == user
                    && m.memory_type == mentedb_core::memory::MemoryType::Episodic
                    && !m.tags.contains(&"source:enrichment".to_string())
                    && m.created_at > last_turn
            })
            .collect();
        candidates.sort_by_key(|m| m.created_at);
        candidates
    }

    /// All unenriched episodic candidates across every owner, sorted by creation
    /// time. This is an owner-agnostic introspection view (the enrichment
    /// pipeline itself scopes per owner via [`Self::enrichment_candidates`]); it
    /// exists so SDK callers can list pending work without an agent argument.
    pub fn all_enrichment_candidates(&self) -> Vec<MemoryNode> {
        let mut candidates: Vec<MemoryNode> = self
            .distinct_owner_pairs()
            .into_iter()
            .flat_map(|(user, agent)| self.enrichment_candidates(user, agent))
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

    /// Get all unique entity names from stored entity memories owned by `agent`.
    ///
    /// Returns deduplicated, normalized entity names extracted from
    /// `entity:{name}` tags across this agent's stored memories only, so one
    /// user's entities never bleed into another's resolution.
    pub fn all_entity_names(&self, user: UserId, agent: AgentId) -> Vec<String> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut names = std::collections::HashSet::new();
        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid) {
                if mem.agent_id != agent || mem.user_id != user {
                    continue;
                }
                for tag in &mem.tags {
                    if let Some(name) = tag.strip_prefix("entity:") {
                        names.insert(name.to_lowercase().trim().to_string());
                    }
                }
            }
        }
        let mut sorted: Vec<String> = names.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Get entity names that the EntityResolver hasn't resolved yet.
    ///
    /// These are the entities that need LLM resolution. The EntityResolver
    /// cache handles known entities for free.
    pub fn unresolved_entity_names(&self, user: UserId, agent: AgentId) -> Vec<String> {
        let all_names = self.all_entity_names(user, agent);
        self.entity_resolver.read().unresolved_names(&all_names)
    }

    /// Get entity names with their memory content for LLM context.
    ///
    /// Returns (name, content) pairs for entities that need resolution.
    /// The content helps the LLM disambiguate (e.g., "Python" near
    /// "web framework" vs "Python" near "Monty Python").
    pub fn entity_names_with_context(
        &self,
        user: UserId,
        agent: AgentId,
    ) -> Vec<(String, Option<String>)> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut entity_contexts: HashMap<String, String> = HashMap::new();

        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid) {
                if mem.agent_id != agent || mem.user_id != user {
                    continue;
                }
                for tag in &mem.tags {
                    if let Some(name) = tag.strip_prefix("entity:") {
                        let normalized = name.to_lowercase().trim().to_string();
                        // One entity tag per memory (enrichment creates separate memories per entity)
                        entity_contexts
                            .entry(normalized)
                            .and_modify(|existing| {
                                // Append content from multiple mentions, cap at ~500 chars
                                if existing.len() < 300 {
                                    existing.push_str(" | ");
                                    let remaining = 500usize.saturating_sub(existing.len());
                                    existing.push_str(
                                        mentedb_core::text::truncate_on_char_boundary(
                                            &mem.content,
                                            remaining,
                                        ),
                                    );
                                }
                            })
                            .or_insert_with(|| {
                                mentedb_core::text::truncate_on_char_boundary(&mem.content, 300)
                                    .to_string()
                            });
                        break;
                    }
                }
            }
        }

        entity_contexts
            .into_iter()
            .map(|(name, ctx)| (name, Some(ctx)))
            .collect()
    }

    /// Apply LLM entity resolution results: create graph edges and update cache.
    ///
    /// Takes merge groups from the LLM (via `CognitiveLlmService.resolve_entities()`)
    /// and confirmed-different pairs. Creates `entity_link:` edges between entity
    /// memories that belong to the same group, learns aliases in the EntityResolver,
    /// and negative-caches confirmed-different pairs.
    pub fn apply_entity_link_resolutions(
        &self,
        merge_groups: &[EntityLinkResolution],
        separations: &[EntitySeparation],
        user: UserId,
        agent: AgentId,
    ) -> MenteResult<EntityLinkResult> {
        let mut result = EntityLinkResult::default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Build a map: normalized entity name → list of memory IDs (this
        // (user, agent) pair's entities only, so links never span either owner
        // axis).
        let entity_memory_map = self.build_entity_memory_map(user, agent);

        let mut resolver = self.entity_resolver.write();

        for group in merge_groups {
            // Learn the group in the resolver cache
            let mut aliases: Vec<String> = group.aliases.clone();
            aliases.retain(|a| a.to_lowercase() != group.canonical.to_lowercase());
            resolver.learn_group(&EntityMergeGroup {
                canonical: group.canonical.clone(),
                aliases,
                confidence: group.confidence,
            });

            // Collect all memory IDs for this merge group
            let mut group_memory_ids: Vec<MemoryId> = Vec::new();

            // Add memories for the canonical name
            let canonical_norm = group.canonical.to_lowercase();
            if let Some(ids) = entity_memory_map.get(&canonical_norm) {
                group_memory_ids.extend(ids);
            }

            // Add memories for each alias
            for alias in &group.aliases {
                let alias_norm = alias.to_lowercase();
                if let Some(ids) = entity_memory_map.get(&alias_norm) {
                    group_memory_ids.extend(ids);
                }
            }

            group_memory_ids.sort();
            group_memory_ids.dedup();

            // Create edges between all pairs in the group
            let label = format!("entity_link:{}", canonical_norm);
            for i in 0..group_memory_ids.len() {
                for j in (i + 1)..group_memory_ids.len() {
                    let a_id = group_memory_ids[i];
                    let b_id = group_memory_ids[j];

                    // Check for existing edge
                    let graph = self.graph.read_graph();
                    let already_linked = graph.outgoing(a_id).iter().any(|(tid, e)| {
                        *tid == b_id
                            && e.edge_type == EdgeType::Related
                            && e.label
                                .as_ref()
                                .is_some_and(|l| l.starts_with("entity_link:"))
                    });
                    drop(graph);

                    if already_linked {
                        continue;
                    }

                    let edge = MemoryEdge {
                        source: a_id,
                        target: b_id,
                        edge_type: EdgeType::Related,
                        weight: group.confidence,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: Some(label.clone()),
                    };
                    if self.relate(edge).is_ok() {
                        result.edges_created += 1;
                    }
                    result.linked += 1;
                }
            }

            debug!(
                canonical = group.canonical,
                aliases = ?group.aliases,
                memories = group_memory_ids.len(),
                "entity resolution: merged group"
            );
        }

        // Process negative cache entries
        for sep in separations {
            resolver.mark_different(&sep.name_a, &sep.name_b);
            debug!(
                a = sep.name_a,
                b = sep.name_b,
                "entity resolution: confirmed different"
            );
        }

        // Persist resolver state
        let cognitive_dir = self.path.join("cognitive");
        if cognitive_dir.exists() || std::fs::create_dir_all(&cognitive_dir).is_ok() {
            let _ = resolver.save(&cognitive_dir.join("entities.json"));
        }

        debug!(
            linked = result.linked,
            edges = result.edges_created,
            groups = merge_groups.len(),
            separations = separations.len(),
            "entity link resolutions applied"
        );
        Ok(result)
    }

    /// Link entities using only the sync EntityResolver (cache + rules, no LLM).
    ///
    /// This is the fast path — links entities that are already known to be
    /// the same from previous LLM resolutions. Runs once per owner so entity
    /// links never span users; each owner's entities are linked in isolation
    /// (nil owned / global entities are linked among themselves).
    pub fn link_entities(&self) -> MenteResult<EntityLinkResult> {
        let mut total = EntityLinkResult::default();
        for (user, agent) in self.distinct_owner_pairs() {
            let r = self.link_entities_for(user, agent)?;
            total.linked += r.linked;
            total.edges_created += r.edges_created;
        }
        Ok(total)
    }

    /// Sync entity linking scoped to a single (user, agent) owner pair. See
    /// [`Self::link_entities`].
    pub(crate) fn link_entities_for(
        &self,
        user: UserId,
        agent: AgentId,
    ) -> MenteResult<EntityLinkResult> {
        let entity_memory_map = self.build_entity_memory_map(user, agent);
        let resolver = self.entity_resolver.read();

        // Group entity names by their resolved canonical name
        let mut canonical_groups: HashMap<String, Vec<String>> = HashMap::new();
        for entity_name in entity_memory_map.keys() {
            let resolved = resolver.resolve(entity_name);
            if resolved.source != mentedb_cognitive::ResolutionSource::Identity {
                canonical_groups
                    .entry(resolved.canonical.clone())
                    .or_default()
                    .push(entity_name.clone());
            }
        }

        drop(resolver);

        let mut result = EntityLinkResult::default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for (canonical, names) in &canonical_groups {
            // Collect all memory IDs across all aliases in this group
            let mut group_memory_ids: Vec<MemoryId> = Vec::new();
            for name in names {
                if let Some(ids) = entity_memory_map.get(name) {
                    group_memory_ids.extend(ids);
                }
            }
            // Also include the canonical name itself
            if let Some(ids) = entity_memory_map.get(canonical) {
                group_memory_ids.extend(ids);
            }
            group_memory_ids.sort();
            group_memory_ids.dedup();

            if group_memory_ids.len() < 2 {
                continue;
            }

            let label = format!("entity_link:{}", canonical);
            for i in 0..group_memory_ids.len() {
                for j in (i + 1)..group_memory_ids.len() {
                    let a_id = group_memory_ids[i];
                    let b_id = group_memory_ids[j];

                    let graph = self.graph.read_graph();
                    let already_linked = graph.outgoing(a_id).iter().any(|(tid, e)| {
                        *tid == b_id
                            && e.edge_type == EdgeType::Related
                            && e.label
                                .as_ref()
                                .is_some_and(|l| l.starts_with("entity_link:"))
                    });
                    drop(graph);

                    if already_linked {
                        continue;
                    }

                    let edge = MemoryEdge {
                        source: a_id,
                        target: b_id,
                        edge_type: EdgeType::Related,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: Some(label.clone()),
                    };
                    if self.relate(edge).is_ok() {
                        result.edges_created += 1;
                    }
                    result.linked += 1;
                }
            }
        }

        debug!(
            linked = result.linked,
            edges = result.edges_created,
            groups = canonical_groups.len(),
            "sync entity linking complete"
        );
        Ok(result)
    }

    /// Build a map of normalized entity name → list of MemoryIds, restricted to
    /// entity memories owned by exactly this (`user`, `agent`) pair so links and
    /// community membership never cross either owner axis.
    fn build_entity_memory_map(
        &self,
        user: UserId,
        agent: AgentId,
    ) -> HashMap<String, Vec<MemoryId>> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut map: HashMap<String, Vec<MemoryId>> = HashMap::new();
        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid) {
                if mem.agent_id != agent || mem.user_id != user {
                    continue;
                }
                for tag in &mem.tags {
                    if let Some(name) = tag.strip_prefix("entity:") {
                        let normalized = name.to_lowercase().trim().to_string();
                        // One entity tag per memory (enrichment creates separate memories per entity)
                        map.entry(normalized).or_default().push(mem.id);
                        break;
                    }
                }
            }
        }
        map
    }

    /// Get all entity memory nodes (memories tagged with `entity:{name}`).
    pub fn entity_memories(&self) -> Vec<MemoryNode> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        page_ids
            .iter()
            .filter_map(|pid| self.storage.load_memory(*pid).ok())
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity:")))
            .collect()
    }

    /// Get entity categories with their member entities for community detection.
    ///
    /// Returns a map of category → list of (entity_name, context_snippet).
    /// Categories come from `entity_type:` tags on entity memories.
    pub fn entity_communities(
        &self,
        user: UserId,
        agent: AgentId,
    ) -> HashMap<String, Vec<(String, String)>> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut categories: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid) {
                // Only cluster this (user, agent) pair's entities so communities
                // never mix across either owner axis.
                if mem.agent_id != agent || mem.user_id != user {
                    continue;
                }
                // Skip non-entity memories and existing community summaries
                if mem.tags.iter().any(|t| t == "community_summary") {
                    continue;
                }

                let entity_name = mem
                    .tags
                    .iter()
                    .find_map(|t| t.strip_prefix("entity:"))
                    .map(|n| n.to_string());

                if let Some(name) = entity_name {
                    let entity_type = mem
                        .tags
                        .iter()
                        .find_map(|t| t.strip_prefix("entity_type:"))
                        .unwrap_or("general")
                        .to_lowercase();

                    let context: String = mem.content.chars().take(200).collect();
                    categories
                        .entry(entity_type)
                        .or_default()
                        .push((name, context));
                }
            }
        }

        // Only return categories with 2+ entities (meaningful clusters)
        categories.retain(|_, members| members.len() >= 2);
        categories
    }

    /// Store a community summary memory with edges to member entities.
    ///
    /// Creates a `community_summary` tagged memory and `Derived` edges
    /// from the summary to each member entity in the category.
    pub fn store_community_summary(
        &self,
        category: &str,
        summary: &str,
        member_names: &[String],
        user: UserId,
        agent: AgentId,
    ) -> MenteResult<MemoryId> {
        if category.is_empty() {
            return Err(MenteError::Storage(
                "community category cannot be empty".into(),
            ));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Check if a community summary already exists for this category AND both
        // owner axes, so one owner's summary never overwrites another's.
        let community_tag = format!("community:{}", category);
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut existing_id = None;
        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid)
                && mem.agent_id == agent
                && mem.user_id == user
                && mem.tags.iter().any(|t| t == &community_tag)
            {
                // Update existing summary content
                let mut updated = mem.clone();
                updated.content = summary.to_string();
                if let Some(ref embedder) = self.embedder {
                    updated.embedding = embedder
                        .embed(summary)
                        .unwrap_or_else(|_| updated.embedding.clone());
                }
                self.storage.update_memory(*pid, &updated)?;
                existing_id = Some(updated.id);
                break;
            }
        }

        let node_id = if let Some(id) = existing_id {
            id
        } else {
            // Create new community summary
            let embedding = self
                .embedder
                .as_ref()
                .and_then(|e| e.embed(summary).ok())
                .unwrap_or_default();

            let mut node =
                MemoryNode::new(agent, MemoryType::Semantic, summary.to_string(), embedding)
                    .with_user_id(user);
            node.tags = vec![
                "community_summary".to_string(),
                community_tag,
                "source:enrichment".to_string(),
            ];
            node.confidence = 0.7;
            let id = node.id;
            self.store(node)?;
            id
        };

        // (Re)create Derived edges from summary to member entity memories.
        // On update this refreshes edges to reflect current membership.
        let entity_map = self.build_entity_memory_map(user, agent);
        for name in member_names {
            let normalized = name.to_lowercase();
            if let Some(member_ids) = entity_map.get(&normalized) {
                for member_id in member_ids {
                    self.relate(MemoryEdge {
                        source: node_id,
                        target: *member_id,
                        edge_type: EdgeType::Derived,
                        weight: 0.8,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: Some(format!("community_member:{}", category)),
                    })?;
                }
            }
        }

        Ok(node_id)
    }

    /// Get existing community summaries owned by exactly this (`user`, `agent`)
    /// pair, so each owner only sees its own community summaries.
    pub fn community_summaries(&self, user: UserId, agent: AgentId) -> Vec<MemoryNode> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        page_ids
            .iter()
            .filter_map(|pid| self.storage.load_memory(*pid).ok())
            .filter(|m| {
                m.agent_id == agent
                    && m.user_id == user
                    && m.tags.iter().any(|t| t == "community_summary")
            })
            .collect()
    }

    /// Collect all semantic/procedural facts for user profile generation.
    ///
    /// Returns high-confidence memories suitable for profile building.
    pub fn profile_facts(&self, user: UserId, agent: AgentId) -> Vec<String> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut facts = Vec::new();

        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid) {
                // Only this (user, agent) pair's facts, so a profile is built
                // from one owner's knowledge and never mixes in another owner's
                // memories on either axis.
                if mem.agent_id != agent || mem.user_id != user {
                    continue;
                }
                // Only semantic and procedural memories with decent confidence
                if mem.confidence < 0.5 {
                    continue;
                }
                match mem.memory_type {
                    MemoryType::Semantic | MemoryType::Procedural => {
                        // Skip community summaries and entity nodes
                        if mem
                            .tags
                            .iter()
                            .any(|t| t == "community_summary" || t.starts_with("entity:"))
                        {
                            continue;
                        }
                        facts.push(mem.content.chars().take(300).collect());
                    }
                    _ => {}
                }
            }
        }

        // Cap at 100 most relevant facts to fit in LLM context
        facts.truncate(100);
        facts
    }

    /// Store or update the user profile as an always-scoped memory owned by
    /// exactly this (`user`, `agent`) pair.
    ///
    /// There is exactly one user profile memory (tagged `user_profile`) per
    /// owner pair. If one already exists for this pair, it's replaced entirely;
    /// one owner's profile never overwrites another's.
    pub fn store_user_profile(
        &self,
        profile: &str,
        user: UserId,
        agent: AgentId,
    ) -> MenteResult<MemoryId> {
        // Find existing profile for this owner pair
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid)
                && mem.agent_id == agent
                && mem.user_id == user
                && mem.tags.iter().any(|t| t == "user_profile")
            {
                // Update in place, bumping created_at so it reflects when the
                // profile was last rebuilt. `profile_updated_at` reads created_at;
                // without this the "updated" time stayed frozen at first creation
                // (a regenerated profile still showed "updated N days ago").
                let mut updated = mem.clone();
                updated.content = profile.to_string();
                updated.created_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                if let Some(ref embedder) = self.embedder {
                    updated.embedding = embedder
                        .embed(profile)
                        .unwrap_or_else(|_| updated.embedding.clone());
                }
                self.storage.update_memory(*pid, &updated)?;
                return Ok(updated.id);
            }
        }

        // Create new profile
        let embedding = self
            .embedder
            .as_ref()
            .and_then(|e| e.embed(profile).ok())
            .unwrap_or_default();

        let mut node = MemoryNode::new(agent, MemoryType::Semantic, profile.to_string(), embedding)
            .with_user_id(user);
        node.tags = vec![
            "user_profile".to_string(),
            "scope:always".to_string(),
            "source:enrichment".to_string(),
        ];
        node.confidence = 0.8;
        let node_id = node.id;
        self.store(node)?;

        Ok(node_id)
    }

    /// Get the account-level user profile: the one owned by neither a user nor
    /// an agent (both owner axes nil).
    ///
    /// Deterministic. Since enrichment now builds one profile per (user, agent)
    /// owner pair, an account may hold many; this returns only the shared,
    /// unowned one, never an arbitrary owner's. A multi-user app that keeps only
    /// per-owner profiles gets `None` here; use [`user_profile_for`] to read a
    /// specific owner's, and [`profile_owners`] to list which owners have one.
    pub fn user_profile(&self) -> Option<MemoryNode> {
        self.user_profile_for(UserId::nil(), AgentId::nil())
    }

    /// Get the user profile owned by exactly this (`user`, `agent`) pair, if one
    /// exists. One owner's profile is never returned for another, so a dashboard
    /// or a per-turn context primer can show the right profile for a scope.
    pub fn user_profile_for(&self, user: UserId, agent: AgentId) -> Option<MemoryNode> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid)
                && mem.user_id == user
                && mem.agent_id == agent
                && mem.tags.iter().any(|t| t == "user_profile")
            {
                return Some(mem);
            }
        }
        None
    }

    /// List the owner pairs that have a user profile, most recently built first.
    /// Lets a dashboard enumerate the profiles it can show without scanning
    /// memory bodies itself.
    pub fn profile_owners(&self) -> Vec<(UserId, AgentId)> {
        let page_ids: Vec<PageId> = self.page_map.read().values().copied().collect();
        let mut owners: Vec<(UserId, AgentId, u64)> = Vec::new();
        for pid in &page_ids {
            if let Ok(mem) = self.storage.load_memory(*pid)
                && mem.tags.iter().any(|t| t == "user_profile")
            {
                owners.push((mem.user_id, mem.agent_id, mem.created_at));
            }
        }
        owners.sort_by_key(|(_, _, ts)| std::cmp::Reverse(*ts));
        owners.into_iter().map(|(u, a, _)| (u, a)).collect()
    }
}

#[cfg(test)]
mod mql_execution_tests {
    use super::*;

    /// Regression: a bare `RECALL` and a `WHERE type = ...` filter must return
    /// the matching memories. Before the executor fix these produced an empty
    /// TagScan (empty tags, filters ignored) and returned nothing.
    #[test]
    fn recall_bare_and_by_type_return_matches() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            "the sky is blue".into(),
            vec![0.1_f32; 8],
        ))
        .unwrap();
        db.store(MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            "water is wet".into(),
            vec![0.2_f32; 8],
        ))
        .unwrap();
        db.store(MemoryNode::new(
            AgentId::nil(),
            MemoryType::Procedural,
            "how to tie a knot".into(),
            vec![0.3_f32; 8],
        ))
        .unwrap();

        // Bare RECALL returns all memories (previously empty).
        let all = db.recall("RECALL memories LIMIT 50").unwrap();
        let all_n: usize = all.blocks.iter().map(|b| b.memories.len()).sum();
        assert_eq!(all_n, 3, "bare RECALL should return all three memories");

        // A type filter narrows to just that type (previously empty).
        let sem = db
            .recall("RECALL memories WHERE type = semantic LIMIT 50")
            .unwrap();
        let sem_mems: Vec<&ScoredMemory> = sem.blocks.iter().flat_map(|b| &b.memories).collect();
        assert_eq!(
            sem_mems.len(),
            2,
            "type = semantic should return two memories"
        );
        assert!(
            sem_mems
                .iter()
                .all(|sm| sm.memory.memory_type == MemoryType::Semantic),
            "only semantic memories should come back"
        );
    }

    /// Build a small fixture: three types, two of them tagged "keep", so boolean
    /// WHERE clauses have something to include and exclude.
    fn boolean_fixture() -> (tempfile::TempDir, MenteDb) {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        let mut sem_keep = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            "semantic kept".into(),
            vec![0.1_f32; 8],
        );
        sem_keep.tags = vec!["keep".into()];
        let sem_plain = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            "semantic plain".into(),
            vec![0.2_f32; 8],
        );
        let mut proc_keep = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Procedural,
            "procedural kept".into(),
            vec![0.3_f32; 8],
        );
        proc_keep.tags = vec!["keep".into()];
        let anti = MemoryNode::new(
            AgentId::nil(),
            MemoryType::AntiPattern,
            "antipattern plain".into(),
            vec![0.4_f32; 8],
        );
        for n in [sem_keep, sem_plain, proc_keep, anti] {
            db.store(n).unwrap();
        }
        (dir, db)
    }

    fn contents(db: &MenteDb, mql: &str) -> Vec<String> {
        let mut v: Vec<String> = db
            .query(mql)
            .unwrap()
            .into_iter()
            .map(|s| s.memory.content)
            .collect();
        v.sort();
        v
    }

    /// OR must union the branches: `type = semantic OR type = procedural` returns
    /// exactly the semantic and procedural rows, never the antipattern.
    #[test]
    fn or_unions_branches() {
        let (_d, db) = boolean_fixture();
        let got = contents(
            &db,
            "RECALL WHERE type = semantic OR type = procedural LIMIT 50",
        );
        assert_eq!(
            got,
            vec![
                "procedural kept".to_string(),
                "semantic kept".to_string(),
                "semantic plain".to_string()
            ],
            "OR should return semantic and procedural, excluding antipattern"
        );
    }

    /// NOT must exclude: `NOT type = semantic` returns everything that is not
    /// semantic.
    #[test]
    fn not_excludes() {
        let (_d, db) = boolean_fixture();
        let got = contents(&db, "RECALL WHERE NOT type = semantic LIMIT 50");
        assert_eq!(
            got,
            vec![
                "antipattern plain".to_string(),
                "procedural kept".to_string()
            ],
            "NOT type = semantic should drop both semantic rows"
        );
    }

    /// AND with NOT: `type = semantic AND NOT tag = keep` isolates the untagged
    /// semantic row, proving NOT composes under AND.
    #[test]
    fn and_not_composes() {
        let (_d, db) = boolean_fixture();
        let got = contents(
            &db,
            r#"RECALL WHERE type = semantic AND NOT tag = "keep" LIMIT 50"#,
        );
        assert_eq!(
            got,
            vec!["semantic plain".to_string()],
            "only the untagged semantic row should remain"
        );
    }

    /// Grouping changes meaning: `(type = semantic OR type = procedural) AND tag =
    /// keep` must return only the kept rows of those two types, not the plain
    /// semantic one. This is the precedence case a flat AND list cannot express.
    #[test]
    fn grouping_binds_before_and() {
        let (_d, db) = boolean_fixture();
        let got = contents(
            &db,
            r#"RECALL WHERE (type = semantic OR type = procedural) AND tag = "keep" LIMIT 50"#,
        );
        assert_eq!(
            got,
            vec!["procedural kept".to_string(), "semantic kept".to_string()],
            "grouping must apply the tag filter to both OR branches"
        );
    }

    /// Regression: a pure AND clause still returns the intersection (and stays on
    /// the flat-filter fast path, which this exercises end to end).
    #[test]
    fn pure_and_still_intersects() {
        let (_d, db) = boolean_fixture();
        let got = contents(
            &db,
            r#"RECALL WHERE type = semantic AND tag = "keep" LIMIT 50"#,
        );
        assert_eq!(got, vec!["semantic kept".to_string()]);
    }
}

#[cfg(test)]
mod order_by_tests {
    use super::*;

    fn contents(db: &MenteDb, mql: &str) -> Vec<String> {
        db.query(mql)
            .unwrap()
            .into_iter()
            .map(|s| s.memory.content)
            .collect()
    }

    /// ORDER BY sorts by the field and applies before LIMIT, both directions.
    #[test]
    fn order_by_salience_sorts_before_limit() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        for (content, sal) in [("low", 0.2_f32), ("high", 0.9), ("mid", 0.5)] {
            let mut n = MemoryNode::new(
                AgentId::nil(),
                MemoryType::Semantic,
                content.into(),
                vec![0.1_f32; 8],
            );
            n.salience = sal;
            db.store(n).unwrap();
        }

        // DESC: highest salience first, LIMIT keeps the top 2.
        assert_eq!(
            contents(
                &db,
                "RECALL WHERE type = semantic ORDER BY salience DESC LIMIT 2"
            ),
            vec!["high".to_string(), "mid".to_string()]
        );
        // ASC: lowest first.
        assert_eq!(
            contents(
                &db,
                "RECALL WHERE type = semantic ORDER BY salience ASC LIMIT 2"
            ),
            vec!["low".to_string(), "mid".to_string()]
        );
        // No direction defaults to ASC.
        assert_eq!(
            contents(
                &db,
                "RECALL WHERE type = semantic ORDER BY salience LIMIT 1"
            ),
            vec!["low".to_string()]
        );
    }
}

#[cfg(test)]
mod metrics_snapshot_tests {
    use super::*;

    /// The metrics snapshot counts writes and reads, sizes the store and index,
    /// and reflects buffer-pool activity.
    #[test]
    fn metrics_track_writes_reads_and_cache() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        for i in 0..3 {
            db.store(MemoryNode::new(
                AgentId::nil(),
                MemoryType::Semantic,
                format!("fact {i}"),
                vec![0.1_f32 + i as f32 * 0.1; 8],
            ))
            .unwrap();
        }
        // Read paths: recall (assembles a window) and query (raw).
        let _ = db.recall("RECALL memories LIMIT 10").unwrap();
        let _ = db.query("RECALL LIMIT 10").unwrap();
        // Exercise the hybrid vector-search path so search latency is recorded
        // (this path is not counted in recalls, only timed).
        let _ = db.recall_similar(&[0.15_f32; 8], 3).unwrap();

        let m = db.metrics();
        assert_eq!(m.stores, 3, "three writes counted");
        assert_eq!(m.recalls, 2, "recall + query counted as reads");
        assert!(
            m.avg_store_latency_us > 0,
            "store latency EMA populated after writes"
        );
        assert!(
            m.avg_search_latency_us > 0,
            "search latency EMA populated after a hybrid search"
        );
        assert_eq!(m.memory_count, 3);
        assert_eq!(m.vector_index_size, 3, "three vectors indexed");
        assert_eq!(m.graph_nodes, 3, "three graph nodes");
        assert!(m.storage_bytes > 0, "pages written to disk");
        assert!(
            m.buffer_pool_hits + m.buffer_pool_misses > 0,
            "reads exercised the buffer pool"
        );
        assert_eq!(m.standing_rules, 0, "no pins yet");
    }
}

#[cfg(test)]
mod entity_boost_tests {
    use super::*;

    /// With a query-named entity linked (via a `Derived` edge) to a lower-ranked
    /// memory, the boost lifts that memory above an unrelated higher-scored one.
    /// This exercises the whole leg: `entity:<name>` tag lookup, graph adjacency,
    /// score bump, and re-sort, on a synthetic corpus with no LLM.
    #[test]
    fn boost_lifts_entity_linked_memory() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();

        let kafka_mem = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Episodic,
            "we chose kafka for the pipeline".into(),
            vec![0.1_f32; 8],
        );
        let kafka_id = kafka_mem.id;
        let other_mem = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Episodic,
            "the sky is blue".into(),
            vec![0.2_f32; 8],
        );
        let other_id = other_mem.id;
        let mut entity = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            "Kafka, a message broker".into(),
            vec![0.3_f32; 8],
        );
        entity.tags.push("entity:kafka".into());
        let entity_id = entity.id;

        db.store(kafka_mem.clone()).unwrap();
        db.store(other_mem.clone()).unwrap();
        db.store(entity).unwrap();
        // Entity node -> Derived -> the episodic it was extracted from, exactly as
        // enrichment records it.
        db.relate(MemoryEdge {
            source: entity_id,
            target: kafka_id,
            edge_type: EdgeType::Derived,
            weight: 1.0,
            created_at: 0,
            valid_from: None,
            valid_until: None,
            label: Some("enrichment".into()),
        })
        .unwrap();

        // Baseline ranking puts the unrelated memory ahead.
        let mut scored = vec![
            ScoredMemory {
                memory: other_mem,
                score: 0.50,
            },
            ScoredMemory {
                memory: kafka_mem,
                score: 0.40,
            },
        ];
        db.apply_entity_boost(&mut scored, "did we pick kafka");

        assert_eq!(
            scored[0].memory.id, kafka_id,
            "the kafka-linked memory should rank first after the entity boost"
        );
        let boosted = scored.iter().find(|s| s.memory.id == kafka_id).unwrap();
        assert!(
            (boosted.score - 0.55).abs() < 1e-6,
            "the linked memory should gain exactly the boost weight"
        );
        assert!(
            scored.iter().any(|s| s.memory.id == other_id),
            "the unrelated memory is kept, just reordered"
        );
    }

    /// A query that names no known entity leaves ranking untouched.
    #[test]
    fn no_entity_in_query_is_a_noop() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        let a = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Episodic,
            "alpha".into(),
            vec![0.1_f32; 8],
        );
        let b = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Episodic,
            "beta".into(),
            vec![0.2_f32; 8],
        );
        let (aid, bid) = (a.id, b.id);
        let mut scored = vec![
            ScoredMemory {
                memory: a,
                score: 0.50,
            },
            ScoredMemory {
                memory: b,
                score: 0.40,
            },
        ];
        db.apply_entity_boost(&mut scored, "nothing here resolves to an entity");
        assert_eq!(scored[0].memory.id, aid);
        assert_eq!(scored[1].memory.id, bid);
    }
}

#[cfg(test)]
mod standing_rule_policy_tests {
    use super::*;

    fn always_rule(content: &str, emb: Vec<f32>, manual: bool) -> MemoryNode {
        let mut n = MemoryNode::new(AgentId::nil(), MemoryType::Semantic, content.into(), emb);
        n.tags.push("scope:always".into());
        if manual {
            n.tags.push("source:manual".into());
        }
        n
    }

    fn always_count(db: &MenteDb) -> usize {
        db.memory_ids()
            .iter()
            .filter_map(|id| db.get_memory(*id).ok())
            .filter(|n| n.tags.iter().any(|t| t == "scope:always"))
            .count()
    }

    /// Regression: un-pinning a rule (removing scope:always and re-storing) must
    /// drop the standing-rule count. The tag index used to only add tags, so the
    /// count stayed stale (the admin dashboard showed 261 when the account had 0).
    #[test]
    fn standing_rule_count_drops_after_unpin() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        let rule = always_rule("always use tabs", vec![0.2_f32; 8], false);
        let id = rule.id;
        db.store(rule).unwrap();
        assert_eq!(db.metrics().standing_rules, 1);

        // Un-pin exactly as the prune does: strip the tag, re-store the same id.
        let mut m = db.get_memory(id).unwrap();
        m.tags.retain(|t| t != "scope:always");
        db.store(m).unwrap();

        assert_eq!(
            db.metrics().standing_rules,
            0,
            "un-pinned rule must not be counted"
        );
        assert_eq!(
            always_count(&db),
            0,
            "the node itself no longer carries the tag"
        );
    }

    /// Re-pinning a rule that already exists (same embedding, reworded) does not
    /// grow the always-list.
    #[test]
    fn dedup_skips_near_identical_standing_rule() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(always_rule("always use tabs", vec![0.5_f32; 8], true))
            .unwrap();
        db.store(always_rule("tabs, always", vec![0.5_f32; 8], true))
            .unwrap();
        assert_eq!(
            always_count(&db),
            1,
            "a near-identical standing rule must not be inserted twice"
        );
    }

    /// A genuinely different standing rule is still pinned.
    #[test]
    fn dedup_allows_a_distinct_standing_rule() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        let mut e1 = vec![0.0_f32; 8];
        e1[0] = 1.0;
        let mut e2 = vec![0.0_f32; 8];
        e2[1] = 1.0;
        db.store(always_rule("always use tabs", e1, true)).unwrap();
        db.store(always_rule("never deploy on friday", e2, true))
            .unwrap();
        assert_eq!(
            always_count(&db),
            2,
            "a distinct standing rule is still stored"
        );
    }

    /// A hand-pinned always-list beyond the cap is demoted down to always_max.
    #[test]
    fn prune_caps_the_always_list() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = CognitiveConfig {
            always_max: 3,
            always_dedup: false, // isolate the cap from dedup
            ..Default::default()
        };
        let db = MenteDb::open_with_config(dir.path(), cfg).unwrap();
        for i in 0..5 {
            let mut e = vec![0.0_f32; 8];
            e[i] = 1.0;
            db.store(always_rule(&format!("manual rule {i}"), e, true))
                .unwrap();
        }
        assert_eq!(always_count(&db), 5);
        let report = db.prune_standing_rules().unwrap();
        assert_eq!(report.capped, 2, "two rules over the cap are demoted");
        assert_eq!(
            always_count(&db),
            3,
            "the always-list is bounded at always_max"
        );
    }
}

#[cfg(test)]
mod distinct_owner_tests {
    use super::*;

    /// `distinct_owner_pairs` must enumerate every (user, agent) pair that holds
    /// a memory, since enrichment loops over its result to scope derived
    /// knowledge per owner pair. This is a unit test (not an integration test)
    /// because the method is private.
    #[test]
    fn distinct_owner_pairs_returns_all_owner_agents() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        let alice = AgentId::new();
        let bob = AgentId::new();

        db.store(MemoryNode::new(
            alice,
            MemoryType::Semantic,
            "alice likes falcons".into(),
            vec![0.1_f32; 8],
        ))
        .unwrap();
        db.store(MemoryNode::new(
            bob,
            MemoryType::Semantic,
            "bob likes turtles".into(),
            vec![0.2_f32; 8],
        ))
        .unwrap();

        let pairs = db.distinct_owner_pairs();
        let agents: Vec<AgentId> = pairs.iter().map(|(_, a)| *a).collect();
        assert!(
            agents.contains(&alice),
            "distinct owners must include alice"
        );
        assert!(agents.contains(&bob), "distinct owners must include bob");
    }

    /// Owner pairs are distinguished on BOTH axes: the same agent with two
    /// different users yields two distinct pairs, so enrichment partitions per
    /// user even under a shared agent.
    #[test]
    fn distinct_owner_pairs_splits_by_user_under_same_agent() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        let agent = AgentId::new();
        let ua = UserId::new();
        let ub = UserId::new();

        db.store(
            MemoryNode::new(
                agent,
                MemoryType::Semantic,
                "user a fact".into(),
                vec![0.1_f32; 8],
            )
            .with_user_id(ua),
        )
        .unwrap();
        db.store(
            MemoryNode::new(
                agent,
                MemoryType::Semantic,
                "user b fact".into(),
                vec![0.2_f32; 8],
            )
            .with_user_id(ub),
        )
        .unwrap();

        let pairs = db.distinct_owner_pairs();
        assert!(
            pairs.contains(&(ua, agent)),
            "must include (user a, agent) pair"
        );
        assert!(
            pairs.contains(&(ub, agent)),
            "must include (user b, agent) pair"
        );
        assert_ne!(
            (ua, agent),
            (ub, agent),
            "same agent, different users are distinct owner pairs"
        );
    }

    /// A user profile belongs to exactly one (user, agent) owner. Reading one
    /// owner's profile must never return another's, the account-level
    /// `user_profile()` must return only the unowned/global one (never an
    /// arbitrary owner's), and `profile_owners` must enumerate them all. This is
    /// the fix for the owner-blind `user_profile()` that returned whichever
    /// profile happened to be scanned first once several owners had one.
    #[test]
    fn profile_is_owner_scoped_and_enumerable() {
        let dir = tempfile::tempdir().unwrap();
        let db = MenteDb::open(dir.path()).unwrap();
        let agent = AgentId::new();
        let alice = UserId::new();
        let bob = UserId::new();

        db.store_user_profile("alice's profile", alice, agent)
            .unwrap();
        db.store_user_profile("bob's profile", bob, agent).unwrap();
        db.store_user_profile("the shared profile", UserId::nil(), AgentId::nil())
            .unwrap();

        // Each owner reads exactly its own, never the other's.
        assert_eq!(
            db.user_profile_for(alice, agent).unwrap().content,
            "alice's profile"
        );
        assert_eq!(
            db.user_profile_for(bob, agent).unwrap().content,
            "bob's profile"
        );

        // The account-level accessor returns only the unowned/global profile,
        // deterministically, not alice's or bob's.
        assert_eq!(db.user_profile().unwrap().content, "the shared profile");

        // An owner with no profile gets None, never a fallback to someone else's.
        assert!(db.user_profile_for(UserId::new(), agent).is_none());

        // Every owner with a profile is enumerable.
        let owners = db.profile_owners();
        assert_eq!(owners.len(), 3);
        assert!(owners.contains(&(alice, agent)));
        assert!(owners.contains(&(bob, agent)));
        assert!(owners.contains(&(UserId::nil(), AgentId::nil())));
    }
}
