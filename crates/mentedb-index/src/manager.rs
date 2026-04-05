//! Composite index manager that owns and coordinates all index types.

use std::collections::HashSet;
use std::path::Path;

use mentedb_core::MemoryNode;
use mentedb_core::error::MenteResult;
use mentedb_core::types::{MemoryId, Timestamp};

use crate::bitmap::BitmapIndex;
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
            bitmap: BitmapIndex::new(),
            temporal: TemporalIndex::new(),
            salience: SalienceIndex::new(),
        }
    }

    /// Save all indexes to the given directory.
    pub fn save(&self, dir: &Path) -> MenteResult<()> {
        std::fs::create_dir_all(dir)?;
        self.hnsw.save(&dir.join("hnsw.json"))?;
        self.bitmap.save(&dir.join("bitmap.json"))?;
        self.temporal.save(&dir.join("temporal.json"))?;
        self.salience.save(&dir.join("salience.json"))?;
        Ok(())
    }

    /// Load all indexes from the given directory.
    pub fn load(dir: &Path) -> MenteResult<Self> {
        let hnsw = HnswIndex::load(&dir.join("hnsw.json"), HnswConfig::default().ef_search)?;
        let bitmap = BitmapIndex::load(&dir.join("bitmap.json"))?;
        let temporal = TemporalIndex::load(&dir.join("temporal.json"))?;
        let salience = SalienceIndex::load(&dir.join("salience.json"))?;
        Ok(Self {
            hnsw,
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
        self.bitmap.remove_all(id);
        self.temporal.remove(id, node.created_at);
        self.salience.remove(id, node.salience);
    }

    /// Hybrid search combining vector similarity, tag filtering, time range, and salience.
    ///
    /// Strategy:
    /// 1. Vector search to get top k*4 candidates
    /// 2. Filter by tags (if provided) and time range (if provided)
    /// 3. Re-rank by combined score: vector_sim * 0.6 + salience * 0.3 + recency * 0.1
    /// 4. Return top k results
    pub fn hybrid_search(
        &self,
        query_embedding: &[f32],
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        if k == 0 {
            return Vec::new();
        }

        // Step 1: vector search for top k*4 candidates
        let vector_candidates = self.hnsw.search(query_embedding, k * 4);

        if vector_candidates.is_empty() {
            return Vec::new();
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

        // Find the max distance for normalization
        let max_dist = vector_candidates
            .iter()
            .map(|(_, d)| *d)
            .fold(f32::NEG_INFINITY, f32::max)
            .max(f32::EPSILON);

        // Find the latest timestamp among candidates for recency normalization
        let max_ts = vector_candidates
            .iter()
            .filter_map(|(id, _)| self.temporal.get_timestamp(*id))
            .max()
            .unwrap_or(1) as f64;

        // Step 2 & 3: filter and re-rank
        let mut scored: Vec<(MemoryId, f32)> = vector_candidates
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
            .map(|(id, dist)| {
                // Vector similarity: 1.0 - normalized distance
                let vector_sim = 1.0 - (dist / max_dist);

                // Salience score (default 0.5 if not found)
                let salience = self.salience.get_salience(id).unwrap_or(0.5);

                // Recency: normalize timestamp to [0, 1]
                let ts = self.temporal.get_timestamp(id).unwrap_or(0) as f64;
                let recency = if max_ts > 0.0 {
                    (ts / max_ts) as f32
                } else {
                    0.0
                };

                let combined = vector_sim * 0.6 + salience * 0.3 + recency * 0.1;
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
