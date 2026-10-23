//! HNSW (Hierarchical Navigable Small World) vector index.
//!
//! A full from-scratch implementation supporting cosine, euclidean, and dot-product
//! distance metrics with configurable construction and search parameters.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use ahash::HashMap;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

/// Compute distance between two vectors, dispatching to SIMD when available.
///
/// Vectors of different dimensions (an index mid way through an embedder
/// change) have no meaningful distance, so they are treated as infinitely far
/// apart and never become neighbours or results of each other.
fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::DotProduct => dot_product_distance(a, b),
    }
}

// ---------------------------------------------------------------------------
// Platform dispatch — picks the best available SIMD at runtime
// ---------------------------------------------------------------------------

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::cosine_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { simd_neon::cosine_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::cosine_scalar(a, b)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::euclidean_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { simd_neon::euclidean_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::euclidean_scalar(a, b)
}

fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_x86::dot_product_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { simd_neon::dot_product_neon(a, b) };
    }
    #[allow(unreachable_code)]
    scalar::dot_product_scalar(a, b)
}

// ---------------------------------------------------------------------------
// Scalar fallback (4-accumulator loop-unrolled)
// ---------------------------------------------------------------------------

mod scalar {
    #[inline]
    pub fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot0, mut dot1, mut dot2, mut dot3) = (0.0f32, 0.0, 0.0, 0.0);
        let (mut na0, mut na1, mut na2, mut na3) = (0.0f32, 0.0, 0.0, 0.0);
        let (mut nb0, mut nb1, mut nb2, mut nb3) = (0.0f32, 0.0, 0.0, 0.0);
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

    #[inline]
    pub fn euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0, 0.0, 0.0);
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

    #[inline]
    pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0, 0.0, 0.0);
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
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 + FMA (8-wide, 256-bit)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps(v, 1);
            let lo = _mm256_castps256_ps128(v);
            let sum128 = _mm_add_ps(hi, lo);
            let hi64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, hi64);
            let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
            let sum32 = _mm_add_ss(sum64, hi32);
            _mm_cvtss_f32(sum32)
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let chunks = n / 8;
            let mut dot = _mm256_setzero_ps();
            let mut norm_a = _mm256_setzero_ps();
            let mut norm_b = _mm256_setzero_ps();
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                dot = _mm256_fmadd_ps(va, vb, dot);
                norm_a = _mm256_fmadd_ps(va, va, norm_a);
                norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
            }
            let mut dot_sum = hsum256(dot);
            let mut na_sum = hsum256(norm_a);
            let mut nb_sum = hsum256(norm_b);
            for i in (chunks * 8)..n {
                let ai = *a.get_unchecked(i);
                let bi = *b.get_unchecked(i);
                dot_sum += ai * bi;
                na_sum += ai * ai;
                nb_sum += bi * bi;
            }
            let denom = (na_sum * nb_sum).sqrt();
            if denom < f32::EPSILON {
                1.0
            } else {
                1.0 - (dot_sum / denom)
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let chunks = n / 8;
            let mut acc = _mm256_setzero_ps();
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                let diff = _mm256_sub_ps(va, vb);
                acc = _mm256_fmadd_ps(diff, diff, acc);
            }
            let mut sum = hsum256(acc);
            for i in (chunks * 8)..n {
                let d = *a.get_unchecked(i) - *b.get_unchecked(i);
                sum += d * d;
            }
            sum.sqrt()
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let chunks = n / 8;
            let mut acc = _mm256_setzero_ps();
            for i in 0..chunks {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                acc = _mm256_fmadd_ps(va, vb, acc);
            }
            let mut sum = hsum256(acc);
            for i in (chunks * 8)..n {
                sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            }
            -sum
        }
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON (4-wide, 128-bit) — always available on Apple Silicon / Graviton
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod simd_neon {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
        unsafe {
            let pair = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
            vget_lane_f32(vpadd_f32(pair, pair), 0)
        }
    }

    pub unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let chunks = n / 4;
            let mut dot = vdupq_n_f32(0.0);
            let mut norm_a = vdupq_n_f32(0.0);
            let mut norm_b = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                dot = vfmaq_f32(dot, va, vb);
                norm_a = vfmaq_f32(norm_a, va, va);
                norm_b = vfmaq_f32(norm_b, vb, vb);
            }
            let mut dot_sum = hsum_f32x4(dot);
            let mut na_sum = hsum_f32x4(norm_a);
            let mut nb_sum = hsum_f32x4(norm_b);
            for i in (chunks * 4)..n {
                let ai = *a.get_unchecked(i);
                let bi = *b.get_unchecked(i);
                dot_sum += ai * bi;
                na_sum += ai * ai;
                nb_sum += bi * bi;
            }
            let denom = (na_sum * nb_sum).sqrt();
            if denom < f32::EPSILON {
                1.0
            } else {
                1.0 - (dot_sum / denom)
            }
        }
    }

    pub unsafe fn euclidean_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let chunks = n / 4;
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                let diff = vsubq_f32(va, vb);
                acc = vfmaq_f32(acc, diff, diff);
            }
            let mut sum = hsum_f32x4(acc);
            for i in (chunks * 4)..n {
                let d = *a.get_unchecked(i) - *b.get_unchecked(i);
                sum += d * d;
            }
            sum.sqrt()
        }
    }

    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len().min(b.len());
            let chunks = n / 4;
            let mut acc = vdupq_n_f32(0.0);
            for i in 0..chunks {
                let va = vld1q_f32(a.as_ptr().add(i * 4));
                let vb = vld1q_f32(b.as_ptr().add(i * 4));
                acc = vfmaq_f32(acc, va, vb);
            }
            let mut sum = hsum_f32x4(acc);
            for i in (chunks * 4)..n {
                sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            }
            -sum
        }
    }
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
    /// Fill each neighbour list back to capacity with the closest links the
    /// diversity heuristic set aside (`keepPrunedConnections` in the HNSW
    /// paper). Not persisted, like `ef_search`, so the file format is
    /// unchanged and a loaded index uses the default.
    #[serde(skip, default = "default_keep_pruned")]
    keep_pruned_connections: bool,
    /// Number of links pointing at each node, per layer, from any node
    /// (tombstones included, since search routes through them). Derived
    /// from `nodes`, so it is not persisted and is rebuilt on load.
    #[serde(skip)]
    in_degree: Vec<Vec<u32>>,
    /// Source of node levels. Not persisted: a loaded index draws fresh
    /// entropy, a new one uses `HnswConfig::seed` when set.
    #[serde(skip, default = "StdRng::from_os_rng")]
    level_rng: StdRng,
}

fn default_keep_pruned() -> bool {
    true
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
    /// After the diversity heuristic picks a node's neighbours, fill the
    /// remaining slots with the closest candidates it set aside, so every
    /// node keeps a full neighbour list (`keepPrunedConnections` in the HNSW
    /// paper). Turning it off keeps only the links the heuristic chose.
    pub keep_pruned_connections: bool,
    /// Seed for the random level of each inserted node. `None` draws from
    /// the operating system. Set it to build the same graph from the same
    /// inserts, for tests and reproductions.
    pub seed: Option<u64>,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::Cosine,
            keep_pruned_connections: true,
            seed: None,
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
                keep_pruned_connections: config.keep_pruned_connections,
                in_degree: Vec::new(),
                level_rng: match config.seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => StdRng::from_os_rng(),
                },
            }),
        }
    }

    /// Insert a vector with the given MemoryId.
    ///
    /// Inserting an id that is already live in the index is an error, because
    /// silently keeping either vector would hide a caller bug. Use
    /// [`Self::upsert`] to replace the vector of a live id on purpose.
    ///
    /// Inserting an id that was removed earlier is allowed: the id gets a
    /// fresh node with the new vector and new neighbour links, and the old
    /// tombstoned node stays in the graph only as a routing hop, never as a
    /// result. Before this, a removed id stayed mapped to its tombstone, so
    /// storing it again failed with a duplicate id error and the memory was
    /// never found by vector search again.
    pub fn insert(&self, id: MemoryId, vector: &[f32]) -> MenteResult<()> {
        let mut inner = self.inner.write();

        if let Some(idx) = inner.id_to_idx.get(&id)
            && !inner.deleted.contains(idx)
        {
            return Err(MenteError::Index(format!("duplicate id: {id}")));
        }

        Self::insert_node(&mut inner, id, vector);
        Ok(())
    }

    /// Insert a vector, replacing the current vector when the id is already
    /// live.
    ///
    /// This is the explicit update form of [`Self::insert`]: a live id is
    /// tombstoned and the new vector is linked in as a fresh node, all under
    /// one write lock, so concurrent updates of the same id never observe a
    /// gap or race into a duplicate id error. When the live vector already
    /// equals `vector`, nothing changes, so re indexing an unchanged memory
    /// does not grow the graph.
    pub fn upsert(&self, id: MemoryId, vector: &[f32]) {
        let mut inner = self.inner.write();

        if let Some(&idx) = inner.id_to_idx.get(&id)
            && !inner.deleted.contains(&idx)
        {
            if inner.nodes[idx].vector == vector {
                return;
            }
            inner.deleted.insert(idx);
        }

        Self::insert_node(&mut inner, id, vector);
    }

    /// Link a new node for `id` into the graph and point the id at it. The
    /// caller has already checked that `id` is not live.
    fn insert_node(inner: &mut HnswInner, id: MemoryId, vector: &[f32]) {
        let node_level = random_level(&mut inner.level_rng, inner.level_mult);
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
        inner.in_degree.push(vec![0; node_level + 1]);
        inner.id_to_idx.insert(id, node_idx);

        // First node becomes entry point
        if inner.entry_point.is_none() {
            inner.entry_point = Some(node_idx);
            inner.max_level = node_level;
            return;
        }

        let Some(ep) = inner.entry_point else {
            return;
        };
        let metric = inner.metric;
        let ef_construction = inner.ef_construction;
        let m = inner.m;
        let m_max0 = inner.m_max0;

        // Where to start the descent. Normally the entry point, but a vector
        // whose dimension differs from the entry point's (re embedding after
        // an embedder change) cannot be routed through it, so start from the
        // highest live node of the same dimension instead. With none, the
        // node starts its own graph and search reaches it by a scan, see
        // `search`.
        let ep_matches = inner.nodes[ep].vector.len() == vector.len();
        let same_dim_start = if ep_matches {
            None
        } else {
            inner
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, n)| {
                    *i != node_idx && n.vector.len() == vector.len() && !inner.deleted.contains(i)
                })
                .max_by_key(|(_, n)| n.layers.len())
                .map(|(i, n)| (i, n.layers.len() - 1))
        };
        let start = if ep_matches {
            Some((ep, inner.max_level))
        } else {
            same_dim_start
        };

        // A tombstoned entry point of the old dimension is handed over to the
        // new dimension, so searches at the new dimension route through the
        // graph again instead of scanning.
        if !ep_matches && inner.deleted.contains(&ep) {
            let (new_ep, new_level) = match same_dim_start {
                Some((s, sl)) if sl > node_level => (s, sl),
                _ => (node_idx, node_level),
            };
            inner.entry_point = Some(new_ep);
            inner.max_level = new_level;
        }

        let Some((start_idx, start_level)) = start else {
            return;
        };

        // Greedily descend from start_level to node_level+1
        let mut current_ep = start_idx;
        let query = &inner.nodes[node_idx].vector.clone();

        for level in (node_level + 1..=start_level).rev() {
            current_ep = greedy_closest(
                &inner.nodes,
                &inner.deleted,
                current_ep,
                query,
                level,
                metric,
            );
        }

        // For each level from min(node_level, start_level) down to 0, do
        // ef-search and connect.
        let keep_pruned = inner.keep_pruned_connections;
        let top = node_level.min(start_level);
        for level in (0..=top).rev() {
            let max_conn = if level == 0 { m_max0 } else { m };

            let mut candidates = search_layer(
                &inner.nodes,
                &inner.deleted,
                current_ep,
                query,
                ef_construction,
                level,
                metric,
            );
            // Only nodes of the same dimension can ever be neighbours.
            candidates.retain(|c| inner.nodes[c.idx].vector.len() == query.len());

            let neighbours =
                select_neighbours(&inner.nodes, &candidates, max_conn, keep_pruned, metric);

            // Add bidirectional connections
            inner.nodes[node_idx].layers[level] = neighbours.iter().map(|c| c.idx).collect();
            for c in &neighbours {
                inner.in_degree[c.idx][level] += 1;
            }

            for c in &neighbours {
                inner.nodes[c.idx].layers[level].push(node_idx);
                inner.in_degree[node_idx][level] += 1;
                if inner.nodes[c.idx].layers[level].len() > max_conn {
                    shrink_links(inner, c.idx, level, max_conn, node_idx);
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
    }

    /// Search for the k nearest neighbours to the query vector.
    /// Returns `(MemoryId, distance)` pairs sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(MemoryId, f32)> {
        let inner = self.inner.read();

        let ep = match inner.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let metric = inner.metric;

        // A query whose dimension differs from the entry point's cannot be
        // routed through the graph. That only happens while vectors of two
        // dimensions coexist (an embedder change in progress), so scan the
        // live vectors of the query's dimension instead of returning nothing.
        if query.len() != inner.nodes[ep].vector.len() {
            let mut results: Vec<(MemoryId, f32)> = inner
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, n)| n.vector.len() == query.len() && !inner.deleted.contains(i))
                .map(|(_, n)| (n.id, compute_distance(query, &n.vector, metric)))
                .collect();
            results.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
            return results;
        }

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
            .filter(|c| {
                !inner.deleted.contains(&c.idx) && inner.nodes[c.idx].vector.len() == query.len()
            })
            .take(k)
            .map(|c| (inner.nodes[c.idx].id, c.dist))
            .collect()
    }

    /// Brute-force search over a specific subset of memory IDs.
    ///
    /// Used for pre-filtered search when a tag/bitmap filter has already
    /// identified the candidate set. Returns up to `k` results sorted by
    /// distance (ascending = most similar first for cosine).
    pub fn search_filtered(
        &self,
        query: &[f32],
        candidates: &HashSet<MemoryId>,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        if k == 0 || candidates.is_empty() {
            return Vec::new();
        }

        let inner = self.inner.read();
        let metric = inner.metric;

        let mut results: Vec<(MemoryId, f32)> = candidates
            .iter()
            .filter_map(|id| {
                let idx = inner.id_to_idx.get(id)?;
                if inner.deleted.contains(idx) {
                    return None;
                }
                let node = &inner.nodes[*idx];
                if node.vector.len() != query.len() {
                    return None;
                }
                let dist = compute_distance(query, &node.vector, metric);
                Some((*id, dist))
            })
            .collect();

        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// True when the id is indexed and not tombstoned.
    pub fn contains(&self, id: MemoryId) -> bool {
        let inner = self.inner.read();
        match inner.id_to_idx.get(&id) {
            Some(idx) => !inner.deleted.contains(idx),
            None => false,
        }
    }

    /// All live (non tombstoned) memory ids in the index.
    pub fn ids(&self) -> Vec<MemoryId> {
        let inner = self.inner.read();
        inner
            .id_to_idx
            .iter()
            .filter(|(_, idx)| !inner.deleted.contains(*idx))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Mark a node as deleted (tombstone). Does not reclaim memory. The id
    /// can be inserted again afterwards, see [`Self::insert`].
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
        bincode::serialize(&*inner).map_err(|e| MenteError::Serialization(e.to_string()))
    }

    /// Deserialize an index from bytes (bincode, with JSON fallback for migration).
    pub fn deserialize(data: &[u8], ef_search: usize) -> MenteResult<Self> {
        let mut inner: HnswInner = bincode::deserialize(data)
            .or_else(|_| serde_json::from_slice(data))
            .map_err(|e| MenteError::Serialization(e.to_string()))?;
        inner.rebuild_in_degree();
        Ok(Self {
            ef_search,
            inner: RwLock::new(inner),
        })
    }

    /// Save the index to a binary file.
    pub fn save(&self, path: &std::path::Path) -> MenteResult<()> {
        let data = self.serialize()?;
        crate::write_atomic(path, &data)?;
        Ok(())
    }

    /// Load the index from a file (bincode or JSON).
    pub fn load(path: &std::path::Path, ef_search: usize) -> MenteResult<Self> {
        let data = std::fs::read(path)?;
        Self::deserialize(&data, ef_search)
    }
}

impl HnswInner {
    /// Recount the links pointing at every node from the adjacency lists.
    fn rebuild_in_degree(&mut self) {
        let mut in_degree: Vec<Vec<u32>> =
            self.nodes.iter().map(|n| vec![0; n.layers.len()]).collect();
        for node in &self.nodes {
            for (level, links) in node.layers.iter().enumerate() {
                for &target in links {
                    if let Some(count) = in_degree
                        .get_mut(target)
                        .and_then(|layers| layers.get_mut(level))
                    {
                        *count += 1;
                    }
                }
            }
        }
        self.in_degree = in_degree;
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Random level assignment: floor(-ln(uniform) * level_mult).
fn random_level(rng: &mut StdRng, level_mult: f64) -> usize {
    let r: f64 = rng.random::<f64>();
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

/// Pick up to `max_count` neighbours for a node from `candidates`, sorted by
/// ascending distance to that node, with the diversity heuristic of the HNSW
/// paper (algorithm 4): a candidate is taken only when it is closer to the
/// node than to every neighbour already taken. Keeping only the nearest
/// candidates instead clusters the links inside one dense region and leaves
/// points between regions with no incoming link at all, so greedy search can
/// never reach them. With `keep_pruned`, the slots left over are filled with
/// the closest candidates the heuristic set aside.
fn select_neighbours(
    nodes: &[HnswNode],
    candidates: &[Candidate],
    max_count: usize,
    keep_pruned: bool,
    metric: DistanceMetric,
) -> Vec<Candidate> {
    let mut selected: Vec<Candidate> = Vec::with_capacity(max_count);
    let mut set_aside: Vec<Candidate> = Vec::new();
    for &cand in candidates {
        if selected.len() >= max_count {
            break;
        }
        let diverse = selected.iter().all(|s| {
            compute_distance(&nodes[cand.idx].vector, &nodes[s.idx].vector, metric) > cand.dist
        });
        if diverse {
            selected.push(cand);
        } else {
            set_aside.push(cand);
        }
    }
    if keep_pruned {
        let room = max_count - selected.len();
        selected.extend(set_aside.into_iter().take(room));
    }
    selected
}

/// Bring the neighbour list of `node` at `level` back to `max_conn` links
/// after the link to `added` pushed it over.
///
/// The links to drop follow the diversity rule of [`select_neighbours`]: a
/// link is redundant when its target is closer to another neighbour than to
/// `node`, because search reaches it through that neighbour anyway. With
/// `keep_pruned_connections` the list keeps every slot, so only the excess
/// goes. The list already held its links before `added` joined, so only
/// redundancy involving `added` is new: links farther than `added` that it
/// covers go first, then `added` itself when a nearer link covers it, then
/// the farthest link. That is one pass over the list, where rerunning the
/// heuristic from scratch would be quadratic in it on every insert. Without
/// `keep_pruned_connections`, the heuristic picks the links to keep.
///
/// In every case a link is never dropped when it is the last one pointing at
/// its target: that would leave the target with no incoming link at this
/// layer, unreachable by any search however large `ef` is. A droppable link
/// goes in its place, and when every link is the last one into its target
/// the list stays over capacity rather than cut a node off.
fn shrink_links(inner: &mut HnswInner, node: usize, level: usize, max_conn: usize, added: usize) {
    let metric = inner.metric;
    let nodes = &inner.nodes;
    let base = &nodes[node].vector;
    let mut scored: Vec<Candidate> = nodes[node].layers[level]
        .iter()
        .map(|&ni| Candidate {
            dist: compute_distance(base, &nodes[ni].vector, metric),
            idx: ni,
        })
        .collect();
    scored.sort_unstable_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let excess = scored.len().saturating_sub(max_conn);
    let droppable = |c: &Candidate| inner.in_degree[c.idx][level] > 1;
    let mut drop: Vec<usize> = Vec::with_capacity(excess);

    if inner.keep_pruned_connections {
        if let Some(pos) = scored.iter().position(|c| c.idx == added) {
            let added_vec = &nodes[added].vector;
            for far in scored[pos + 1..].iter().rev() {
                if drop.len() == excess {
                    break;
                }
                if droppable(far)
                    && compute_distance(&nodes[far.idx].vector, added_vec, metric) < far.dist
                {
                    drop.push(far.idx);
                }
            }
            let added_cand = &scored[pos];
            if drop.len() < excess
                && droppable(added_cand)
                && scored[..pos].iter().any(|near| {
                    compute_distance(added_vec, &nodes[near.idx].vector, metric) < added_cand.dist
                })
            {
                drop.push(added);
            }
        }
        for c in scored.iter().rev() {
            if drop.len() == excess {
                break;
            }
            if droppable(c) && !drop.contains(&c.idx) {
                drop.push(c.idx);
            }
        }
    } else {
        let keep = select_neighbours(nodes, &scored, max_conn, false, metric);
        let kept: HashSet<usize> = keep.iter().map(|c| c.idx).collect();
        drop = scored
            .iter()
            .filter(|c| !kept.contains(&c.idx) && droppable(c))
            .map(|c| c.idx)
            .collect();
    }

    if drop.is_empty() {
        return;
    }
    inner.nodes[node].layers[level].retain(|i| !drop.contains(i));
    for target in drop {
        inner.in_degree[target][level] -= 1;
    }
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
        let mut rng = rand::rng();
        (0..dim).map(|_| rng.random::<f32>()).collect()
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
    fn rejected_live_duplicate_keeps_the_original_vector() {
        let idx = HnswIndex::new(HnswConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        });
        let id = MemoryId::new();
        idx.insert(id, &make_vec(4, 1.0)).unwrap();
        let err = idx.insert(id, &make_vec(4, 9.0)).unwrap_err();
        assert!(err.to_string().contains("duplicate id"), "{err}");
        assert_eq!(idx.len(), 1);
        let hit = idx.search(&make_vec(4, 1.0), 1);
        assert_eq!(hit[0].0, id);
        assert!(hit[0].1 < 0.001, "original vector still indexed");
    }

    /// Fills an index with random vectors so a re insert has to be linked
    /// into a real graph, not just become the only node.
    fn populated(n: usize, dim: usize) -> HnswIndex {
        let idx = HnswIndex::new(HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 50,
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        });
        for _ in 0..n {
            idx.insert(MemoryId::new(), &random_vec(dim)).unwrap();
        }
        idx
    }

    #[test]
    fn insert_after_remove_is_found_by_search() {
        let dim = 16;
        let idx = populated(200, dim);
        let id = MemoryId::new();
        let old = random_vec(dim);
        idx.insert(id, &old).unwrap();
        idx.remove(id).unwrap();
        assert!(!idx.contains(id));

        // Same id again, with a different vector.
        let fresh = random_vec(dim);
        idx.insert(id, &fresh).unwrap();
        assert!(idx.contains(id));
        assert_eq!(idx.len(), 201);
        assert!(idx.ids().contains(&id));

        let hits = idx.search(&fresh, 1);
        assert_eq!(hits.first().map(|h| h.0), Some(id));
        assert!(hits[0].1 < 0.001, "found by its new vector");

        // The old vector no longer ranks the id at distance zero, and the id
        // is never returned twice.
        let near_old = idx.search(&old, 50);
        assert!(near_old.iter().filter(|h| h.0 == id).count() <= 1);
        assert!(near_old.iter().all(|h| h.0 != id || h.1 > 0.001));
    }

    #[test]
    fn insert_after_remove_works_for_the_only_node() {
        let idx = HnswIndex::new(HnswConfig::default());
        let id = MemoryId::new();
        idx.insert(id, &make_vec(4, 1.0)).unwrap();
        idx.remove(id).unwrap();
        idx.insert(id, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        let hits = idx.search(&[1.0, 0.0, 0.0, 0.0], 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id);
    }

    #[test]
    fn remove_after_reinsert_removes_the_new_vector() {
        let idx = populated(50, 8);
        let id = MemoryId::new();
        let v = random_vec(8);
        idx.insert(id, &random_vec(8)).unwrap();
        idx.remove(id).unwrap();
        idx.insert(id, &v).unwrap();
        idx.remove(id).unwrap();
        assert!(!idx.contains(id));
        assert_eq!(idx.len(), 50);
        assert!(idx.search(&v, 60).iter().all(|h| h.0 != id));
        // And it can come back a third time.
        idx.insert(id, &v).unwrap();
        assert_eq!(idx.search(&v, 1)[0].0, id);
    }

    #[test]
    fn upsert_replaces_a_live_vector() {
        let dim = 16;
        let idx = populated(200, dim);
        let id = MemoryId::new();
        let (old, new) = (random_vec(dim), random_vec(dim));
        idx.insert(id, &old).unwrap();
        idx.upsert(id, &new);
        assert_eq!(idx.len(), 201, "replaced, not added");
        let hits = idx.search(&new, 1);
        assert_eq!(hits[0].0, id);
        assert!(hits[0].1 < 0.001);
        let near_old = idx.search(&old, 50);
        assert!(near_old.iter().filter(|h| h.0 == id).count() <= 1);
    }

    #[test]
    fn upsert_inserts_new_and_removed_ids() {
        let idx = HnswIndex::new(HnswConfig::default());
        let new_id = MemoryId::new();
        idx.upsert(new_id, &[1.0, 0.0]);
        assert!(idx.contains(new_id));
        idx.remove(new_id).unwrap();
        idx.upsert(new_id, &[0.0, 1.0]);
        assert!(idx.contains(new_id));
        assert_eq!(idx.search(&[0.0, 1.0], 1)[0].0, new_id);
    }

    #[test]
    fn upsert_with_the_same_vector_does_not_grow_the_graph() {
        let idx = HnswIndex::new(HnswConfig::default());
        let id = MemoryId::new();
        idx.insert(id, &make_vec(4, 1.0)).unwrap();
        for _ in 0..5 {
            idx.upsert(id, &make_vec(4, 1.0));
        }
        assert_eq!(idx.inner.read().nodes.len(), 1);
        assert!(idx.inner.read().deleted.is_empty());
    }

    #[test]
    fn upsert_to_a_new_dimension_keeps_both_dimensions_searchable() {
        // Models a re embed after an embedder change: vectors move from 8 to
        // 16 dimensions one at a time, and every memory stays findable at
        // whichever dimension it currently has.
        let old_dim = 8;
        let new_dim = 16;
        let idx = HnswIndex::new(HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 50,
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        });
        let ids: Vec<MemoryId> = (0..60).map(|_| MemoryId::new()).collect();
        let old: Vec<Vec<f32>> = ids.iter().map(|_| random_vec(old_dim)).collect();
        let new: Vec<Vec<f32>> = ids.iter().map(|_| random_vec(new_dim)).collect();
        for (id, v) in ids.iter().zip(&old) {
            idx.insert(*id, v).unwrap();
        }

        for i in 0..ids.len() {
            idx.upsert(ids[i], &new[i]);
            assert_eq!(idx.len(), ids.len());
            // Every migrated memory is found at the new dimension.
            for j in 0..=i {
                assert_eq!(
                    idx.search(&new[j], 1)[0].0,
                    ids[j],
                    "migrated {j} at step {i}"
                );
            }
            // Every memory not yet migrated is found at the old dimension.
            for j in i + 1..ids.len() {
                assert_eq!(
                    idx.search(&old[j], 1)[0].0,
                    ids[j],
                    "pending {j} at step {i}"
                );
            }
            // Results never mix dimensions.
            assert!(idx.search(&new[0], 100).len() == i + 1);
        }
    }

    #[test]
    fn distance_between_different_dimensions_is_infinite() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
        ] {
            assert_eq!(compute_distance(&[1.0; 4], &[1.0; 8], metric), f32::MAX);
        }
    }

    #[test]
    fn reinsert_after_remove_survives_serialization() {
        let dim = 8;
        let idx = populated(30, dim);
        let id = MemoryId::new();
        let v = random_vec(dim);
        idx.insert(id, &random_vec(dim)).unwrap();
        idx.remove(id).unwrap();
        idx.insert(id, &v).unwrap();

        let back = HnswIndex::deserialize(&idx.serialize().unwrap(), 50).unwrap();
        assert!(back.contains(id));
        assert_eq!(back.len(), 31);
        assert_eq!(back.search(&v, 1)[0].0, id);
        // A live duplicate is still rejected after the round trip.
        assert!(back.insert(id, &make_vec(dim, 1.0)).is_err());
    }

    #[test]
    fn test_many_vectors() {
        let idx = HnswIndex::new(HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 30,
            metric: DistanceMetric::Euclidean,
            ..Default::default()
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

    /// Live nodes of the entry point's dimension that a search can never
    /// reach: walks every layer 0 link from the entry point, through
    /// tombstones as search does. Also checks the in degree bookkeeping
    /// against a recount of the links.
    fn unreachable_at_layer_zero(idx: &HnswIndex) -> Vec<usize> {
        let inner = idx.inner.read();
        let mut recount: Vec<Vec<u32>> = inner
            .nodes
            .iter()
            .map(|n| vec![0; n.layers.len()])
            .collect();
        for node in &inner.nodes {
            for (level, links) in node.layers.iter().enumerate() {
                for &t in links {
                    recount[t][level] += 1;
                }
            }
        }
        assert_eq!(recount, inner.in_degree, "in degree bookkeeping drifted");

        let Some(ep) = inner.entry_point else {
            return Vec::new();
        };
        let dim = inner.nodes[ep].vector.len();
        let mut seen = HashSet::from([ep]);
        let mut stack = vec![ep];
        while let Some(current) = stack.pop() {
            for &next in &inner.nodes[current].layers[0] {
                if seen.insert(next) {
                    stack.push(next);
                }
            }
        }
        (0..inner.nodes.len())
            .filter(|i| {
                !inner.deleted.contains(i)
                    && inner.nodes[*i].vector.len() == dim
                    && !seen.contains(i)
            })
            .collect()
    }

    fn seeded_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.random::<f32>()).collect()
    }

    fn seeded_config(seed: u64) -> HnswConfig {
        HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 50,
            metric: DistanceMetric::Euclidean,
            seed: Some(seed),
            ..Default::default()
        }
    }

    #[test]
    fn every_live_node_stays_reachable_at_layer_zero() {
        // Pruning neighbour lists by distance alone dropped the last link
        // into some nodes, so no search could reach them at any ef: about a
        // third of graphs this size had at least one such node.
        for seed in 0..20 {
            let idx = HnswIndex::new(seeded_config(seed));
            let mut rng = StdRng::seed_from_u64(1_000 + seed);
            for _ in 0..300 {
                idx.insert(MemoryId::new(), &seeded_vec(&mut rng, 16))
                    .unwrap();
            }
            assert_eq!(
                unreachable_at_layer_zero(&idx),
                Vec::<usize>::new(),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn every_live_node_stays_reachable_through_upserts_and_removes() {
        let dim = 16;
        let idx = HnswIndex::new(seeded_config(42));
        let mut rng = StdRng::seed_from_u64(42);
        let ids: Vec<MemoryId> = (0..200).map(|_| MemoryId::new()).collect();
        for id in &ids {
            idx.insert(*id, &seeded_vec(&mut rng, dim)).unwrap();
        }
        for (i, id) in ids.iter().enumerate() {
            match i % 3 {
                0 => idx.upsert(*id, &seeded_vec(&mut rng, dim)),
                1 => idx.remove(*id).unwrap(),
                _ => {}
            }
            idx.insert(MemoryId::new(), &seeded_vec(&mut rng, dim))
                .unwrap();
        }
        assert_eq!(unreachable_at_layer_zero(&idx), Vec::<usize>::new());
    }

    #[test]
    fn in_degree_is_rebuilt_after_load() {
        let idx = HnswIndex::new(seeded_config(3));
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..100 {
            idx.insert(MemoryId::new(), &seeded_vec(&mut rng, 8))
                .unwrap();
        }
        let back = HnswIndex::deserialize(&idx.serialize().unwrap(), 50).unwrap();
        for _ in 0..100 {
            back.insert(MemoryId::new(), &seeded_vec(&mut rng, 8))
                .unwrap();
        }
        assert_eq!(unreachable_at_layer_zero(&back), Vec::<usize>::new());
    }

    #[test]
    fn same_seed_and_inserts_build_the_same_graph() {
        let build = || {
            let idx = HnswIndex::new(seeded_config(9));
            let mut rng = StdRng::seed_from_u64(9);
            for i in 0..80u128 {
                idx.insert(
                    MemoryId::from(uuid::Uuid::from_u128(i + 1)),
                    &seeded_vec(&mut rng, 8),
                )
                .unwrap();
            }
            let inner = idx.inner.read();
            inner
                .nodes
                .iter()
                .map(|n| n.layers.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(build(), build());
    }
}
