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

    /// Save all indexes to the given directory (bincode format).
    pub fn save(&self, dir: &Path) -> MenteResult<()> {
        std::fs::create_dir_all(dir)?;
        self.hnsw.save(&dir.join("hnsw.bin"))?;
        self.bm25.save(&dir.join("bm25.bin"))?;
        self.bitmap.save(&dir.join("bitmap.bin"))?;
        self.temporal.save(&dir.join("temporal.bin"))?;
        self.salience.save(&dir.join("salience.bin"))?;
        Ok(())
    }

    /// Load all indexes from the given directory. Tries `.bin` first, falls back to `.json`.
    pub fn load(dir: &Path) -> MenteResult<Self> {
        let hnsw_path = Self::resolve_path(dir, "hnsw");
        let hnsw = HnswIndex::load(&hnsw_path, HnswConfig::default().ef_search)?;

        let bm25_bin = dir.join("bm25.bin");
        let bm25_json = dir.join("bm25.json");
        let bm25 = if bm25_bin.exists() {
            Bm25Index::load(&bm25_bin)?
        } else if bm25_json.exists() {
            Bm25Index::load(&bm25_json)?
        } else {
            Bm25Index::new()
        };

        let bitmap = BitmapIndex::load(&Self::resolve_path(dir, "bitmap"))?;
        let temporal = TemporalIndex::load(&Self::resolve_path(dir, "temporal"))?;
        let salience = SalienceIndex::load(&Self::resolve_path(dir, "salience"))?;
        Ok(Self {
            hnsw,
            bm25,
            bitmap,
            temporal,
            salience,
        })
    }

    /// Resolve index file path: prefer `.bin`, fall back to `.json`.
    fn resolve_path(dir: &Path, name: &str) -> std::path::PathBuf {
        let bin = dir.join(format!("{name}.bin"));
        if bin.exists() {
            bin
        } else {
            dir.join(format!("{name}.json"))
        }
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
        self.hybrid_search_with_query_mode(query_embedding, query_text, tags, false, time_range, k)
    }

    /// Hybrid search with configurable tag mode (AND vs OR).
    pub fn hybrid_search_with_query_mode(
        &self,
        query_embedding: &[f32],
        query_text: Option<&str>,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        if k == 0 {
            return Vec::new();
        }

        // Build tag filter set (if tags are specified)
        let tag_filter: Option<HashSet<MemoryId>> = tags.map(|t| {
            if t.is_empty() {
                HashSet::new()
            } else if tags_or {
                self.bitmap.query_tags_or(t).into_iter().collect()
            } else {
                self.bitmap.query_tags_and(t).into_iter().collect()
            }
        });

        // Build time-range filter set
        let time_filter: Option<HashSet<MemoryId>> =
            time_range.map(|(start, end)| self.temporal.range(start, end).into_iter().collect());

        // Combine filters into a single candidate set
        let candidate_set: Option<HashSet<MemoryId>> = match (&tag_filter, &time_filter) {
            (Some(tf), Some(trf)) => Some(tf.intersection(trf).copied().collect()),
            (Some(tf), None) => Some(tf.clone()),
            (None, Some(trf)) => Some(trf.clone()),
            (None, None) => None,
        };

        // Pre-filtered path: when we have a candidate set and it's reasonably sized,
        // do brute-force search directly over the candidates instead of global search + post-filter.
        // This is critical for OR-tag queries with many tags where global top-k misses most matches.
        let use_prefilter = candidate_set.as_ref().is_some_and(|cs| {
            let cs_len = cs.len();
            // Use pre-filter when candidate set is non-trivial but manageable for brute-force
            // (up to 500K is fine — brute-force cosine on 384-dim vectors is fast)
            cs_len > 0 && cs_len <= 500_000
        });

        let fetch_k = k * 4;
        let rrf_k: f32 = 60.0;

        let (vector_candidates, bm25_candidates) = if use_prefilter {
            let cs = candidate_set.as_ref().unwrap();
            let vc = self.hnsw.search_filtered(query_embedding, cs, fetch_k);
            let bc = match query_text {
                Some(qt) if !self.bm25.is_empty() => self.bm25.search_filtered(qt, fetch_k, cs),
                _ => Vec::new(),
            };
            (vc, bc)
        } else {
            let vc = self.hnsw.search(query_embedding, fetch_k);
            let bc = match query_text {
                Some(qt) if !self.bm25.is_empty() => self.bm25.search(qt, fetch_k),
                _ => Vec::new(),
            };
            (vc, bc)
        };

        if vector_candidates.is_empty() && bm25_candidates.is_empty() {
            return Vec::new();
        }

        // Merge via RRF
        let mut rrf_scores: HashMap<MemoryId, f32> = HashMap::new();

        for (rank, (id, _)) in vector_candidates.iter().enumerate() {
            *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }
        for (rank, (id, _)) in bm25_candidates.iter().enumerate() {
            *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }

        // Post-filter only needed when NOT using pre-filter path
        let mut scored: Vec<(MemoryId, f32)> = rrf_scores
            .into_iter()
            .filter(|(id, _)| {
                if !use_prefilter {
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
                }
                true
            })
            .map(|(id, rrf_score)| {
                let salience = self.salience.get_salience(id).unwrap_or(0.5);
                let recency = 0.5f32;

                let combined = rrf_score * 0.7 + salience * 0.05 + recency * 0.02;
                (id, combined)
            })
            .collect();

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
