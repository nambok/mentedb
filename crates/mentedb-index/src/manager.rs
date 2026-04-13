//! Composite index manager that owns and coordinates all index types.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mentedb_core::MemoryNode;
use mentedb_core::error::MenteResult;
use mentedb_core::types::{MemoryId, Timestamp};

use crate::bitmap::BitmapIndex;
use crate::bm25::Bm25Index;
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::salience::SalienceIndex;
use crate::temporal::TemporalIndex;

/// Configuration for the composite index manager.
#[derive(Default)]
pub struct IndexManagerConfig {
    /// HNSW configuration parameters.
    pub hnsw: HnswConfig,
}

/// Owns all index types and provides unified indexing and hybrid search.
pub struct IndexManager {
    /// Vector similarity index.
    pub hnsw: HnswIndex,
    /// BM25 full-text index for keyword search.
    pub bm25: Bm25Index,
    /// Tag and attribute bitmap index.
    pub bitmap: BitmapIndex,
    /// Timestamp range index.
    pub temporal: TemporalIndex,
    /// Importance score index.
    pub salience: SalienceIndex,
}

impl IndexManager {
    /// Create a new index manager with the given configuration.
    pub fn new(config: IndexManagerConfig) -> Self {
        Self {
            hnsw: HnswIndex::new(config.hnsw),
            bm25: Bm25Index::new(),
            bitmap: BitmapIndex::new(),
            temporal: TemporalIndex::new(),
            salience: SalienceIndex::new(),
        }
    }

    /// Save all indexes to the given directory.
    pub fn save(&self, dir: &Path) -> MenteResult<()> {
        std::fs::create_dir_all(dir)?;
        self.hnsw.save(&dir.join("hnsw.json"))?;
        self.bm25.save(&dir.join("bm25.json"))?;
        self.bitmap.save(&dir.join("bitmap.json"))?;
        self.temporal.save(&dir.join("temporal.json"))?;
        self.salience.save(&dir.join("salience.json"))?;
        Ok(())
    }

    /// Load all indexes from the given directory.
    pub fn load(dir: &Path) -> MenteResult<Self> {
        let hnsw = HnswIndex::load(&dir.join("hnsw.json"), HnswConfig::default().ef_search)?;
        let bm25_path = dir.join("bm25.json");
        let bm25 = if bm25_path.exists() {
            Bm25Index::load(&bm25_path)?
        } else {
            Bm25Index::new()
        };
        let bitmap = BitmapIndex::load(&dir.join("bitmap.json"))?;
        let temporal = TemporalIndex::load(&dir.join("temporal.json"))?;
        let salience = SalienceIndex::load(&dir.join("salience.json"))?;
        Ok(Self {
            hnsw,
            bm25,
            bitmap,
            temporal,
            salience,
        })
    }

    /// Index a memory node across all indexes.
    pub fn index_memory(&self, node: &MemoryNode) {
        // Vector index
        if !node.embedding.is_empty() {
            let _ = self.hnsw.insert(node.id, &node.embedding);
        }

        // BM25 full-text index
        if !node.content.is_empty() {
            self.bm25.insert(node.id, &node.content);
        }

        // Tag bitmap index
        for tag in &node.tags {
            self.bitmap.add_tag(node.id, tag);
        }

        // Temporal index
        self.temporal.insert(node.id, node.created_at);

        // Salience index
        self.salience.insert(node.id, node.salience);
    }

    /// Remove a memory from all indexes.
    pub fn remove_memory(&self, id: MemoryId, node: &MemoryNode) {
        let _ = self.hnsw.remove(id);
        self.bm25.remove(id);
        self.bitmap.remove_all(id);
        self.temporal.remove(id, node.created_at);
        self.salience.remove(id, node.salience);
    }

    /// Hybrid search combining vector similarity, BM25 keyword matching,
    /// tag filtering, time range, and salience.
    ///
    /// Strategy:
    /// 1. Vector search (HNSW) for top candidates
    /// 2. BM25 keyword search for top candidates
    /// 3. Merge via Reciprocal Rank Fusion (RRF)
    /// 4. Filter by tags and time range
    /// 5. Boost by salience and recency
    /// 6. Return top k results
    pub fn hybrid_search(
        &self,
        query_embedding: &[f32],
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        self.hybrid_search_with_query(query_embedding, None, tags, time_range, k)
    }

    /// Hybrid search with an optional text query for BM25 matching.
    ///
    /// When `query_text` is provided, BM25 results are merged with vector
    /// results via RRF. When None, behaves like vector-only search.
    pub fn hybrid_search_with_query(
        &self,
        query_embedding: &[f32],
        query_text: Option<&str>,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let fetch_k = k * 4;
        let rrf_k: f32 = 60.0;

        // Step 1: Vector search candidates
        let vector_candidates = self.hnsw.search(query_embedding, fetch_k);

        // Step 2: BM25 search candidates (if query text provided and index has docs)
        let bm25_candidates = match query_text {
            Some(qt) if !self.bm25.is_empty() => self.bm25.search(qt, fetch_k),
            _ => Vec::new(),
        };

        if vector_candidates.is_empty() && bm25_candidates.is_empty() {
            return Vec::new();
        }

        // Step 3: Merge via RRF
        let mut rrf_scores: HashMap<MemoryId, f32> = HashMap::new();

        for (rank, (id, _)) in vector_candidates.iter().enumerate() {
            *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }
        for (rank, (id, _)) in bm25_candidates.iter().enumerate() {
            *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }

        // Build set of tag-filtered ids (if tags are specified)
        let tag_filter: Option<HashSet<MemoryId>> = tags.map(|t| {
            if t.is_empty() {
                HashSet::new()
            } else {
                self.bitmap.query_tags_and(t).into_iter().collect()
            }
        });

        // Build set of time-range-filtered ids (if time range is specified)
        let time_filter: Option<HashSet<MemoryId>> =
            time_range.map(|(start, end)| self.temporal.range(start, end).into_iter().collect());

        // Step 4: Filter and boost with salience/recency
        let max_ts = rrf_scores
            .keys()
            .filter_map(|id| self.temporal.get_timestamp(*id))
            .max()
            .unwrap_or(1) as f64;

        let mut scored: Vec<(MemoryId, f32)> = rrf_scores
            .into_iter()
            .filter(|(id, _)| {
                if let Some(ref tf) = tag_filter
                    && !tf.contains(id)
                {
                    return false;
                }
                if let Some(ref trf) = time_filter
                    && !trf.contains(id)
                {
                    return false;
                }
                true
            })
            .map(|(id, rrf_score)| {
                let salience = self.salience.get_salience(id).unwrap_or(0.5);
                let ts = self.temporal.get_timestamp(id).unwrap_or(0) as f64;
                let recency = if max_ts > 0.0 {
                    (ts / max_ts) as f32
                } else {
                    0.0
                };

                // RRF is the primary signal, salience and recency are light boosts
                let combined = rrf_score * 0.7 + salience * 0.05 + recency * 0.02;
                (id, combined)
            })
            .collect();

        // Sort descending by combined score
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

impl Default for IndexManager {
    fn default() -> Self {
        Self::new(IndexManagerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::AgentId;

    fn make_node(
        embedding: Vec<f32>,
        tags: Vec<String>,
        salience: f32,
        created_at: u64,
    ) -> MemoryNode {
        let mut node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Episodic,
            "test".into(),
            embedding,
        );
        node.tags = tags;
        node.salience = salience;
        node.created_at = created_at;
        node
    }

    #[test]
    fn test_index_and_search() {
        let mgr = IndexManager::default();
        let node = make_node(vec![1.0, 0.0, 0.0, 0.0], vec!["test".into()], 0.8, 1000);
        mgr.index_memory(&node);

        let results = mgr.hybrid_search(&[1.0, 0.0, 0.0, 0.0], None, None, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, node.id);
    }

    #[test]
    fn test_tag_filter() {
        let mgr = IndexManager::default();
        let a = make_node(vec![1.0, 0.0, 0.0, 0.0], vec!["alpha".into()], 0.8, 1000);
        let b = make_node(vec![0.9, 0.1, 0.0, 0.0], vec!["beta".into()], 0.8, 1000);
        mgr.index_memory(&a);
        mgr.index_memory(&b);

        let results = mgr.hybrid_search(&[1.0, 0.0, 0.0, 0.0], Some(&["alpha"]), None, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, a.id);
    }

    #[test]
    fn test_time_filter() {
        let mgr = IndexManager::default();
        let a = make_node(vec![1.0, 0.0, 0.0, 0.0], vec![], 0.8, 100);
        let b = make_node(vec![0.9, 0.1, 0.0, 0.0], vec![], 0.8, 500);
        mgr.index_memory(&a);
        mgr.index_memory(&b);

        let results = mgr.hybrid_search(&[1.0, 0.0, 0.0, 0.0], None, Some((400, 600)), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, b.id);
    }

    #[test]
    fn test_remove_memory() {
        let mgr = IndexManager::default();
        let node = make_node(vec![1.0, 0.0, 0.0, 0.0], vec!["t".into()], 0.5, 100);
        let id = node.id;
        mgr.index_memory(&node);
        mgr.remove_memory(id, &node);

        let results = mgr.hybrid_search(&[1.0, 0.0, 0.0, 0.0], None, None, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_search() {
        let mgr = IndexManager::default();
        let results = mgr.hybrid_search(&[1.0, 0.0], None, None, 5);
        assert!(results.is_empty());
    }
}
