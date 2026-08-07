//! Composite index manager that owns and coordinates all index types.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mentedb_core::MemoryNode;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId, Timestamp, UserId};

use crate::bitmap::BitmapIndex;
use crate::bm25::Bm25Index;
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::salience::SalienceIndex;
use crate::temporal::TemporalIndex;

/// Canonical browse string for a memory type: the lowercased debug name with
/// `antipattern` normalized to `anti_pattern`. Kept identical to the platform's
/// `type_str` and the engine's scope-count type name so an index lookup returns
/// exactly the set a per-node type filter would.
pub fn type_name(t: MemoryType) -> String {
    format!("{t:?}")
        .to_lowercase()
        .replace("antipattern", "anti_pattern")
}

/// Reserved bitmap key for a memory type. The NUL prefix cannot occur in a user
/// tag, so these keys never collide with tags and never surface in tag listings.
pub fn type_index_key(t: MemoryType) -> String {
    format!("\0type:{}", type_name(t))
}

/// Reserved bitmap key for a memory type given its already-canonical string,
/// so callers holding the browse string (not the enum) build the same key.
pub fn type_index_key_str(type_name: &str) -> String {
    format!("\0type:{type_name}")
}

/// Reserved bitmap key for an owner user.
pub fn user_index_key(user: UserId) -> String {
    format!("\0user:{user}")
}

/// Reserved bitmap key for an owner agent.
pub fn agent_index_key(agent: AgentId) -> String {
    format!("\0agent:{agent}")
}

/// How the hybrid-search legs (vector, keyword) are fused into one ranking.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// When true, fuse the legs by normalized score magnitude (min-max per leg)
    /// instead of rank-only Reciprocal Rank Fusion. RRF keeps only the rank, so
    /// a 0.95 and a 0.55 cosine hit collapse to adjacent ranks and the small
    /// salience term can reorder them. The magnitude-aware path keeps that spread
    /// so a clearly stronger match cannot fall below a weaker one. Off by default,
    /// enable to A/B against RRF before making it the default.
    pub magnitude_aware: bool,
    /// Weight on the normalized vector-similarity leg (magnitude-aware path).
    pub w_vector: f32,
    /// Weight on the normalized BM25 keyword leg (magnitude-aware path).
    pub w_bm25: f32,
    /// Weight on salience (magnitude-aware path): a nudge for ties, not a driver.
    pub w_salience: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            magnitude_aware: false,
            w_vector: 1.0,
            w_bm25: 1.0,
            w_salience: 0.1,
        }
    }
}

/// Configuration for the composite index manager.
#[derive(Default)]
pub struct IndexManagerConfig {
    /// HNSW configuration parameters.
    pub hnsw: HnswConfig,
    /// Hybrid-search fusion strategy.
    pub fusion: FusionConfig,
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
    /// Hybrid-search fusion strategy.
    fusion: FusionConfig,
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
            fusion: config.fusion,
        }
    }

    /// Set the hybrid-search fusion strategy. Lets an embedder flip the
    /// magnitude-aware path on after `load`/`default` (both of which start it
    /// off) so it can be toggled from the host's config without a rebuild.
    pub fn set_fusion(&mut self, fusion: FusionConfig) {
        self.fusion = fusion;
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
            fusion: FusionConfig::default(),
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

        // BM25 full-text index, over the context-prefixed text so contextual
        // retrieval also matches the caller's situating blurb.
        let indexed = node.indexed_text();
        if !indexed.is_empty() {
            self.bm25.insert(node.id, &indexed);
        }

        // Tag bitmap index. Clear this id's existing tags first, so a re-index
        // after an edit (e.g. un-pinning, which removes scope:always and re-stores
        // the node) reflects the node's CURRENT tags exactly, not the union across
        // every stored version. Without this, a removed tag lingers in the index
        // and over-counts on query_tag. Harmless no-op on a first insert.
        self.bitmap.remove_all(node.id);
        for tag in &node.tags {
            self.bitmap.add_tag(node.id, tag);
        }
        // Structured-browse dimensions (memory type, owner user, owner agent) are
        // indexed as reserved bitmap keys so the dashboard can narrow by them in
        // O(matches), the same way it narrows by a tag, instead of scanning every
        // memory. The NUL prefix keeps them from ever colliding with a user tag
        // and out of every tag-enumeration path (none of which use a NUL prefix).
        // remove_all above clears these too, so a re-index reflects current state.
        self.bitmap
            .add_tag(node.id, &type_index_key(node.memory_type));
        self.bitmap.add_tag(node.id, &user_index_key(node.user_id));
        self.bitmap
            .add_tag(node.id, &agent_index_key(node.agent_id));

        // Temporal index
        self.temporal.insert(node.id, node.created_at);

        // Salience index
        self.salience.insert(node.id, node.salience);
    }

    /// True when the memory's vector is indexed and live.
    pub fn contains_vector(&self, id: MemoryId) -> bool {
        self.hnsw.contains(id)
    }

    /// All live vector indexed memory ids.
    pub fn vector_ids(&self) -> Vec<MemoryId> {
        self.hnsw.ids()
    }

    /// Tombstone a vector whose backing page no longer exists; used by open
    /// time reconciliation when a snapshot is newer than the last forget.
    pub fn remove_vector_only(&self, id: MemoryId) {
        let _ = self.hnsw.remove(id);
        self.bm25.remove(id);
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

        // A candidate survives filtering unless it falls outside the tag/time
        // sets. When the pre-filter path ran, candidates are already constrained
        // to those sets, so this is a no-op there (matches the prior behavior).
        let keep = |id: &MemoryId| -> bool {
            if use_prefilter {
                return true;
            }
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
        };

        let mut scored: Vec<(MemoryId, f32)> = if self.fusion.magnitude_aware {
            // Magnitude-aware fusion: normalize each leg to [0, 1] within its own
            // candidate set (nearest vector = 1, top BM25 = 1) and combine by a
            // weighted sum. Unlike RRF this preserves the score spread, so a
            // clearly stronger vector match stays ahead of a weaker one instead
            // of collapsing to an adjacent rank that salience can then flip.
            let vec_norm = normalize_closeness(&vector_candidates);
            let bm25_norm = normalize_scores(&bm25_candidates);

            let mut ids: HashSet<MemoryId> =
                HashSet::with_capacity(vec_norm.len() + bm25_norm.len());
            ids.extend(vec_norm.keys().copied());
            ids.extend(bm25_norm.keys().copied());

            ids.into_iter()
                .filter(|id| keep(id))
                .map(|id| {
                    let v = vec_norm.get(&id).copied().unwrap_or(0.0);
                    let b = bm25_norm.get(&id).copied().unwrap_or(0.0);
                    let salience = self.salience.get_salience(id).unwrap_or(0.5);
                    let combined = self.fusion.w_vector * v
                        + self.fusion.w_bm25 * b
                        + self.fusion.w_salience * salience;
                    (id, combined)
                })
                .collect()
        } else {
            // Reciprocal Rank Fusion (rank-only): robust to differing leg scales
            // but discards score magnitude. The default until the magnitude-aware
            // path is validated by an A/B eval.
            let mut rrf_scores: HashMap<MemoryId, f32> = HashMap::new();

            for (rank, (id, _)) in vector_candidates.iter().enumerate() {
                *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
            }
            for (rank, (id, _)) in bm25_candidates.iter().enumerate() {
                *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
            }

            rrf_scores
                .into_iter()
                .filter(|(id, _)| keep(id))
                .map(|(id, rrf_score)| {
                    let salience = self.salience.get_salience(id).unwrap_or(0.5);
                    let recency = 0.5f32;

                    let combined = rrf_score * 0.7 + salience * 0.05 + recency * 0.02;
                    (id, combined)
                })
                .collect()
        };

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Normalize vector-search distances to a [0, 1] closeness within the candidate
/// set: the nearest candidate maps to 1.0, the farthest to 0.0. Metric-agnostic,
/// it only needs smaller to mean nearer. A degenerate set (empty, single, or all
/// equal) invents no spread: every entry maps to 1.0.
fn normalize_closeness(candidates: &[(MemoryId, f32)]) -> HashMap<MemoryId, f32> {
    let mut out = HashMap::with_capacity(candidates.len());
    if candidates.is_empty() {
        return out;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for (_, d) in candidates {
        lo = lo.min(*d);
        hi = hi.max(*d);
    }
    let range = hi - lo;
    for (id, d) in candidates {
        let closeness = if range <= f32::EPSILON {
            1.0
        } else {
            (hi - *d) / range
        };
        out.entry(*id).or_insert(closeness);
    }
    out
}

/// Normalize keyword (BM25) scores to [0, 1] within the candidate set: the top
/// score maps to 1.0, the lowest to 0.0. BM25 magnitudes are unbounded and
/// corpus-dependent, so normalizing within the set keeps the leg comparable to
/// the vector leg without a fixed scale. A degenerate set maps every entry to 1.0.
fn normalize_scores(candidates: &[(MemoryId, f32)]) -> HashMap<MemoryId, f32> {
    let mut out = HashMap::with_capacity(candidates.len());
    if candidates.is_empty() {
        return out;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for (_, s) in candidates {
        lo = lo.min(*s);
        hi = hi.max(*s);
    }
    let range = hi - lo;
    for (id, s) in candidates {
        let norm = if range <= f32::EPSILON {
            1.0
        } else {
            (*s - lo) / range
        };
        out.entry(*id).or_insert(norm);
    }
    out
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

    fn magnitude_aware_manager() -> IndexManager {
        IndexManager::new(IndexManagerConfig {
            hnsw: HnswConfig::default(),
            fusion: FusionConfig {
                magnitude_aware: true,
                ..Default::default()
            },
        })
    }

    // set_fusion is how an embedder turns v2 on after load()/default(), which
    // both start it off. Flipping it must change the ranking exactly as the
    // constructed-on path does.
    #[test]
    fn set_fusion_toggles_the_ranking() {
        let query = [1.0f32, 0.0];
        let strong = make_node(vec![0.95, 0.312_25], vec![], 0.0, 100);
        let weak = make_node(vec![0.55, 0.835_16], vec![], 1.0, 100);

        let mut mgr = IndexManager::default();
        mgr.index_memory(&strong);
        mgr.index_memory(&weak);
        // Off by default: salience flips the weaker match up.
        assert_eq!(mgr.hybrid_search(&query, None, None, 10)[0].0, weak.id);

        mgr.set_fusion(FusionConfig {
            magnitude_aware: true,
            ..Default::default()
        });
        // Flipped on: the stronger match is restored to the top.
        assert_eq!(mgr.hybrid_search(&query, None, None, 10)[0].0, strong.id);
    }

    // The core claim of the flag: rank-only RRF flattens the vector score gap so
    // the small salience term can flip a clearly stronger match below a weaker,
    // more salient one; the magnitude-aware fusion keeps the stronger match on
    // top. Both nodes are unit vectors so the cosine to the query is their first
    // component: 0.95 (strong) vs 0.55 (weak).
    #[test]
    fn magnitude_aware_keeps_the_stronger_match_above_a_salient_weaker_one() {
        let query = [1.0f32, 0.0];
        let strong = make_node(vec![0.95, 0.312_25], vec![], 0.0, 100);
        let weak = make_node(vec![0.55, 0.835_16], vec![], 1.0, 100);

        // v1 default (rank-only RRF): the salient but weaker match wins.
        let v1 = IndexManager::default();
        v1.index_memory(&strong);
        v1.index_memory(&weak);
        let r1 = v1.hybrid_search(&query, None, None, 10);
        assert_eq!(r1.len(), 2);
        assert_eq!(
            r1[0].0, weak.id,
            "RRF lets salience flip a clearly weaker match to the top"
        );

        // v2 magnitude-aware: the stronger match stays on top.
        let v2 = magnitude_aware_manager();
        v2.index_memory(&strong);
        v2.index_memory(&weak);
        let r2 = v2.hybrid_search(&query, None, None, 10);
        assert_eq!(r2.len(), 2);
        assert_eq!(
            r2[0].0, strong.id,
            "magnitude-aware fusion keeps the stronger match on top"
        );
    }

    // The refactor pulled the tag/time filter into a shared closure used by both
    // paths. Prove the magnitude-aware path still filters exactly like v1 does.
    #[test]
    fn magnitude_aware_respects_tag_and_time_filters() {
        let by_tag = magnitude_aware_manager();
        let a = make_node(vec![1.0, 0.0, 0.0, 0.0], vec!["alpha".into()], 0.8, 1000);
        let b = make_node(vec![0.9, 0.1, 0.0, 0.0], vec!["beta".into()], 0.8, 1000);
        by_tag.index_memory(&a);
        by_tag.index_memory(&b);
        let tagged = by_tag.hybrid_search(&[1.0, 0.0, 0.0, 0.0], Some(&["alpha"]), None, 10);
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].0, a.id);

        let by_time = magnitude_aware_manager();
        let c = make_node(vec![1.0, 0.0, 0.0, 0.0], vec![], 0.8, 100);
        let d = make_node(vec![0.9, 0.1, 0.0, 0.0], vec![], 0.8, 500);
        by_time.index_memory(&c);
        by_time.index_memory(&d);
        let ranged = by_time.hybrid_search(&[1.0, 0.0, 0.0, 0.0], None, Some((400, 600)), 10);
        assert_eq!(ranged.len(), 1);
        assert_eq!(ranged[0].0, d.id);
    }

    // Magnitude-aware fusion must never return more than the pre-change default,
    // and with the flag off the default path is unchanged (guarded by the suite
    // above). Here: a keyword-only hit still surfaces under the magnitude path.
    #[test]
    fn magnitude_aware_matches_keyword_only_hits() {
        let mgr = magnitude_aware_manager();
        let mut hit = make_node(vec![0.0, 1.0, 0.0, 0.0], vec![], 0.5, 1);
        hit.content = "quarterly revenue projection".into();
        let mut other = make_node(vec![1.0, 0.0, 0.0, 0.0], vec![], 0.5, 2);
        other.content = "unrelated note".into();
        mgr.index_memory(&hit);
        mgr.index_memory(&other);

        // Query vector points at `other`, but the text only matches `hit`.
        let results = mgr.hybrid_search_with_query(
            &[1.0, 0.0, 0.0, 0.0],
            Some("quarterly revenue"),
            None,
            None,
            10,
        );
        assert!(
            results.iter().any(|(id, _)| *id == hit.id),
            "keyword-only match must survive magnitude-aware fusion"
        );
    }

    #[test]
    fn normalizers_handle_degenerate_sets() {
        assert!(normalize_closeness(&[]).is_empty());
        assert!(normalize_scores(&[]).is_empty());

        let n1 = make_node(vec![1.0, 0.0], vec![], 0.5, 1);
        let n2 = make_node(vec![1.0, 0.0], vec![], 0.5, 2);

        // A single candidate normalizes to 1.0 in both legs.
        assert_eq!(
            normalize_closeness(&[(n1.id, 0.42)]).get(&n1.id),
            Some(&1.0)
        );
        assert_eq!(normalize_scores(&[(n1.id, 7.0)]).get(&n1.id), Some(&1.0));

        // All-equal inputs invent no spread: every entry maps to 1.0.
        let equal = [(n1.id, 0.3), (n2.id, 0.3)];
        let cl = normalize_closeness(&equal);
        assert_eq!(cl.get(&n1.id), Some(&1.0));
        assert_eq!(cl.get(&n2.id), Some(&1.0));

        // A real spread ranks nearest = 1.0, farthest = 0.0.
        let spread = [(n1.id, 0.1), (n2.id, 0.9)];
        let cl2 = normalize_closeness(&spread);
        assert_eq!(cl2.get(&n1.id), Some(&1.0));
        assert_eq!(cl2.get(&n2.id), Some(&0.0));
    }
}
