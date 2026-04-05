//! HNSW (Hierarchical Navigable Small World) vector index.
//!
//! A full from-scratch implementation supporting cosine, euclidean, and dot-product
//! distance metrics with configurable construction and search parameters.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use ahash::HashMap;
use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};

use mentedb_core::MenteError;
use mentedb_core::error::MenteResult;
use mentedb_core::types::MemoryId;

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Supported distance metrics for the HNSW index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// Compute distance between two vectors using a 4-accumulator loop-unrolled pattern.
fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::DotProduct => dot_product_distance(a, b),
    }
}

/// 1 - cosine_similarity. Lower is more similar.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot0, mut dot1, mut dot2, mut dot3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut na0, mut na1, mut na2, mut na3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut nb0, mut nb1, mut nb2, mut nb3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        let (a0, a1, a2, a3) = (a[base], a[base + 1], a[base + 2], a[base + 3]);
        let (b0, b1, b2, b3) = (b[base], b[base + 1], b[base + 2], b[base + 3]);
        dot0 += a0 * b0;
        dot1 += a1 * b1;
        dot2 += a2 * b2;
        dot3 += a3 * b3;
        na0 += a0 * a0;
        na1 += a1 * a1;
        na2 += a2 * a2;
        na3 += a3 * a3;
        nb0 += b0 * b0;
        nb1 += b1 * b1;
        nb2 += b2 * b2;
        nb3 += b3 * b3;
    }
    let mut dot = dot0 + dot1 + dot2 + dot3;
    let mut norm_a = na0 + na1 + na2 + na3;
    let mut norm_b = nb0 + nb1 + nb2 + nb3;

    for i in (chunks * 4)..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / denom)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    let mut sum = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

/// Negative dot product so that lower = more similar.
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }
    let mut sum = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..a.len() {
        sum += a[i] * b[i];
    }
    -sum
}

// ---------------------------------------------------------------------------
// HNSW internals
// ---------------------------------------------------------------------------

/// A neighbour entry in the priority queue: (distance, node_index).
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    dist: f32,
    idx: usize,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // natural ordering (min-heap when used with Reverse)
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// A single node in the HNSW graph.
#[derive(Clone, Serialize, Deserialize)]
struct HnswNode {
    id: MemoryId,
    vector: Vec<f32>,
    /// Connections per layer. Layer 0 may have up to M*2 neighbours.
    layers: Vec<Vec<usize>>,
}

/// Internal HNSW data, protected by RwLock.
#[derive(Serialize, Deserialize)]
struct HnswInner {
    nodes: Vec<HnswNode>,
    /// Map from MemoryId → index in `nodes`.
    id_to_idx: HashMap<MemoryId, usize>,
    /// Indices of deleted/tombstoned nodes (available for reuse is not implemented here).
    deleted: HashSet<usize>,
    entry_point: Option<usize>,
    max_level: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    level_mult: f64,
    metric: DistanceMetric,
}

/// Thread-safe HNSW index.
pub struct HnswIndex {
    inner: RwLock<HnswInner>,
    ef_search: usize,
}

/// Configuration for constructing an HNSW index.
pub struct HnswConfig {
    /// Maximum number of connections per node per layer.
    pub m: usize,
    /// Size of the dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search.
    pub ef_search: usize,
    /// Distance metric used for similarity computation.
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::Cosine,
        }
    }
}

impl HnswIndex {
    /// Create a new empty HNSW index with the given configuration.
    pub fn new(config: HnswConfig) -> Self {
        let level_mult = 1.0 / (config.m as f64).ln();
        Self {
            ef_search: config.ef_search,
            inner: RwLock::new(HnswInner {
                nodes: Vec::new(),
                id_to_idx: HashMap::default(),
                deleted: HashSet::new(),
                entry_point: None,
                max_level: 0,
                m: config.m,
                m_max0: config.m * 2,
                ef_construction: config.ef_construction,
                level_mult,
                metric: config.metric,
            }),
        }
    }

    /// Insert a vector with the given MemoryId.
    pub fn insert(&self, id: MemoryId, vector: &[f32]) -> MenteResult<()> {
        let mut inner = self.inner.write();

        if inner.id_to_idx.contains_key(&id) {
            return Err(MenteError::Index(format!("duplicate id: {id}")));
        }

        let node_level = random_level(inner.level_mult, inner.m);
        let node_idx = inner.nodes.len();

        let mut layers = Vec::with_capacity(node_level + 1);
        for _ in 0..=node_level {
            layers.push(Vec::new());
        }

        inner.nodes.push(HnswNode {
            id,
            vector: vector.to_vec(),
            layers,
        });
        inner.id_to_idx.insert(id, node_idx);

        // First node becomes entry point
        if inner.entry_point.is_none() {
            inner.entry_point = Some(node_idx);
            inner.max_level = node_level;
            return Ok(());
        }

        let ep = inner.entry_point.unwrap();
        let metric = inner.metric;
        let ef_construction = inner.ef_construction;
        let m = inner.m;
        let m_max0 = inner.m_max0;

        // Greedily descend from max_level to node_level+1
        let mut current_ep = ep;
        let query = &inner.nodes[node_idx].vector.clone();

        for level in (node_level + 1..=inner.max_level).rev() {
            current_ep = greedy_closest(
                &inner.nodes,
                &inner.deleted,
                current_ep,
                query,
                level,
                metric,
            );
        }

        // For each level from min(node_level, max_level) down to 0, do ef-search and connect
        let top = node_level.min(inner.max_level);
        for level in (0..=top).rev() {
            let max_conn = if level == 0 { m_max0 } else { m };

            let candidates = search_layer(
                &inner.nodes,
                &inner.deleted,
                current_ep,
                query,
                ef_construction,
                level,
                metric,
            );

            // Select M nearest neighbours
            let neighbours = select_neighbours(&candidates, max_conn);

            // Add bidirectional connections
            inner.nodes[node_idx].layers[level] = neighbours.iter().map(|c| c.idx).collect();

            for &cand in &neighbours {
                let neighbour_idx = cand.idx;
                inner.nodes[neighbour_idx].layers[level].push(node_idx);

                // Prune if over capacity
                if inner.nodes[neighbour_idx].layers[level].len() > max_conn {
                    let nv = inner.nodes[neighbour_idx].vector.clone();
                    let mut scored: Vec<Candidate> = inner.nodes[neighbour_idx].layers[level]
                        .iter()
                        .map(|&ni| Candidate {
                            dist: compute_distance(&nv, &inner.nodes[ni].vector, metric),
                            idx: ni,
                        })
                        .collect();
                    scored.sort_unstable_by(|a, b| {
                        a.dist
                            .partial_cmp(&b.dist)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    scored.truncate(max_conn);
                    inner.nodes[neighbour_idx].layers[level] =
                        scored.iter().map(|c| c.idx).collect();
                }
            }

            if !candidates.is_empty() {
                current_ep = candidates[0].idx;
            }
        }

        // Promote entry point if new node is higher level
        if node_level > inner.max_level {
            inner.entry_point = Some(node_idx);
            inner.max_level = node_level;
        }

        Ok(())
    }

    /// Search for the k nearest neighbours to the query vector.
    /// Returns `(MemoryId, distance)` pairs sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(MemoryId, f32)> {
        let inner = self.inner.read();

        let ep = match inner.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        // Dimension safety: query must match stored vector dimensions
        if query.len() != inner.nodes[ep].vector.len() {
            return Vec::new();
        }

        let metric = inner.metric;
        let ef = self.ef_search.max(k);

        // Greedy descent to layer 0
        let mut current_ep = ep;
        for level in (1..=inner.max_level).rev() {
            current_ep = greedy_closest(
                &inner.nodes,
                &inner.deleted,
                current_ep,
                query,
                level,
                metric,
            );
        }

        let candidates = search_layer(
            &inner.nodes,
            &inner.deleted,
            current_ep,
            query,
            ef,
            0,
            metric,
        );

        candidates
            .into_iter()
            .filter(|c| !inner.deleted.contains(&c.idx))
            .take(k)
            .map(|c| (inner.nodes[c.idx].id, c.dist))
            .collect()
    }

    /// Mark a node as deleted (tombstone). Does not reclaim memory.
    pub fn remove(&self, id: MemoryId) -> MenteResult<()> {
        let mut inner = self.inner.write();
        let idx = inner
            .id_to_idx
            .get(&id)
            .copied()
            .ok_or(MenteError::MemoryNotFound(id))?;
        inner.deleted.insert(idx);
        Ok(())
    }

    /// Number of active (non-deleted) vectors in the index.
    pub fn len(&self) -> usize {
        let inner = self.inner.read();
        inner.nodes.len() - inner.deleted.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize the index to bytes for persistence.
    pub fn serialize(&self) -> MenteResult<Vec<u8>> {
        let inner = self.inner.read();
        serde_json::to_vec(&*inner).map_err(|e| MenteError::Serialization(e.to_string()))
    }

    /// Deserialize an index from bytes.
    pub fn deserialize(data: &[u8], ef_search: usize) -> MenteResult<Self> {
        let inner: HnswInner =
            serde_json::from_slice(data).map_err(|e| MenteError::Serialization(e.to_string()))?;
        Ok(Self {
            ef_search,
            inner: RwLock::new(inner),
        })
    }

    /// Save the index to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> MenteResult<()> {
        let data = self.serialize()?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load the index from a JSON file.
    pub fn load(path: &std::path::Path, ef_search: usize) -> MenteResult<Self> {
        let data = std::fs::read(path)?;
        Self::deserialize(&data, ef_search)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Random level assignment: floor(-ln(uniform) * level_mult).
fn random_level(level_mult: f64, _m: usize) -> usize {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.r#gen::<f64>();
    // Avoid log(0)
    let r = r.max(f64::EPSILON);

    (-r.ln() * level_mult).floor() as usize
}

/// Greedily walk neighbours at `level` to find the node closest to `query`.
fn greedy_closest(
    nodes: &[HnswNode],
    deleted: &HashSet<usize>,
    mut current: usize,
    query: &[f32],
    level: usize,
    metric: DistanceMetric,
) -> usize {
    let mut best_dist = compute_distance(&nodes[current].vector, query, metric);
    loop {
        let mut changed = false;
        if level < nodes[current].layers.len() {
            for &neighbour in &nodes[current].layers[level] {
                if deleted.contains(&neighbour) {
                    continue;
                }
                if level >= nodes[neighbour].layers.len() {
                    continue;
                }
                let d = compute_distance(&nodes[neighbour].vector, query, metric);
                if d < best_dist {
                    best_dist = d;
                    current = neighbour;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    current
}

/// ef-bounded search at a single layer. Returns candidates sorted by ascending distance.
fn search_layer(
    nodes: &[HnswNode],
    deleted: &HashSet<usize>,
    entry: usize,
    query: &[f32],
    ef: usize,
    level: usize,
    metric: DistanceMetric,
) -> Vec<Candidate> {
    let entry_dist = compute_distance(&nodes[entry].vector, query, metric);
    let entry_cand = Candidate {
        dist: entry_dist,
        idx: entry,
    };

    // Min-heap of candidates to explore
    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
    // Max-heap of current best results
    let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut visited: HashSet<usize> = HashSet::new();

    candidates.push(Reverse(entry_cand));
    results.push(entry_cand);
    visited.insert(entry);

    while let Some(Reverse(current)) = candidates.pop() {
        let worst_result = results.peek().map(|c| c.dist).unwrap_or(f32::MAX);
        if current.dist > worst_result && results.len() >= ef {
            break;
        }

        if level < nodes[current.idx].layers.len() {
            for &neighbour in &nodes[current.idx].layers[level] {
                if !visited.insert(neighbour) {
                    continue;
                }
                if level >= nodes[neighbour].layers.len() {
                    continue;
                }

                let d = compute_distance(&nodes[neighbour].vector, query, metric);
                let worst_result = results.peek().map(|c| c.dist).unwrap_or(f32::MAX);

                if d < worst_result || results.len() < ef {
                    let cand = Candidate {
                        dist: d,
                        idx: neighbour,
                    };
                    candidates.push(Reverse(cand));
                    if !deleted.contains(&neighbour) {
                        results.push(cand);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
    }

    let mut res: Vec<Candidate> = results.into_vec();
    res.sort_unstable_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    res
}

/// Select the nearest `max_count` neighbours from sorted candidates.
fn select_neighbours(candidates: &[Candidate], max_count: usize) -> Vec<Candidate> {
    candidates.iter().copied().take(max_count).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(dim: usize, val: f32) -> Vec<f32> {
        vec![val; dim]
    }

    fn random_vec(dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.r#gen::<f32>()).collect()
    }

    #[test]
    fn test_insert_and_search_single() {
        let idx = HnswIndex::new(HnswConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        });
        let id = MemoryId::new();
        idx.insert(id, &make_vec(8, 1.0)).unwrap();

        let results = idx.search(&make_vec(8, 1.0), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
        assert!(results[0].1 < 0.001);
    }

    #[test]
    fn test_search_nearest() {
        let idx = HnswIndex::new(HnswConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        });
        let id_a = MemoryId::new();
        let id_b = MemoryId::new();
        idx.insert(id_a, &make_vec(8, 0.0)).unwrap();
        idx.insert(id_b, &make_vec(8, 10.0)).unwrap();

        // Query close to id_a
        let results = idx.search(&make_vec(8, 0.1), 1);
        assert_eq!(results[0].0, id_a);
    }

    #[test]
    fn test_remove() {
        let idx = HnswIndex::new(HnswConfig::default());
        let id = MemoryId::new();
        idx.insert(id, &make_vec(4, 1.0)).unwrap();
        assert_eq!(idx.len(), 1);

        idx.remove(id).unwrap();
        assert_eq!(idx.len(), 0);

        let results = idx.search(&make_vec(4, 1.0), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_duplicate_insert() {
        let idx = HnswIndex::new(HnswConfig::default());
        let id = MemoryId::new();
        idx.insert(id, &make_vec(4, 1.0)).unwrap();
        assert!(idx.insert(id, &make_vec(4, 2.0)).is_err());
    }

    #[test]
    fn test_many_vectors() {
        let idx = HnswIndex::new(HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 30,
            metric: DistanceMetric::Euclidean,
        });

        let dim = 16;
        let mut ids = Vec::new();
        for _ in 0..100 {
            let id = MemoryId::new();
            ids.push(id);
            idx.insert(id, &random_vec(dim)).unwrap();
        }

        assert_eq!(idx.len(), 100);
        let results = idx.search(&random_vec(dim), 10);
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_serialize_deserialize() {
        let idx = HnswIndex::new(HnswConfig {
            metric: DistanceMetric::Cosine,
            ..Default::default()
        });
        let id = MemoryId::new();
        let vec = random_vec(8);
        idx.insert(id, &vec).unwrap();

        let data = idx.serialize().unwrap();
        let idx2 = HnswIndex::deserialize(&data, 50).unwrap();
        let results = idx2.search(&vec, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id);
    }
}
