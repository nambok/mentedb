//! Compressed Sparse Row/Column graph storage with delta log for incremental updates.

use ahash::HashMap;
use mentedb_core::edge::{EdgeType, MemoryEdge};
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::{MemoryId, Timestamp};
use serde::{Deserialize, Serialize};

/// Compact edge data stored in CSR/CSC arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredEdge {
    /// The relationship type.
    pub edge_type: EdgeType,
    /// Edge weight (0.0 to 1.0).
    pub weight: f32,
    /// When this edge was created.
    pub created_at: Timestamp,
    /// When this relationship became valid. None = since creation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<Timestamp>,
    /// When this relationship stopped being valid. None = still valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<Timestamp>,
    /// Semantic label for the relationship (e.g. "owns", "attends").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl StoredEdge {
    /// Converts a [`MemoryEdge`] into a compact stored representation.
    pub fn from_memory_edge(edge: &MemoryEdge) -> Self {
        Self {
            edge_type: edge.edge_type,
            weight: edge.weight,
            created_at: edge.created_at,
            valid_from: edge.valid_from,
            valid_until: edge.valid_until,
            label: edge.label.clone(),
        }
    }

    /// Returns true if this edge is temporally valid at the given timestamp.
    pub fn is_valid_at(&self, at: Timestamp) -> bool {
        let from = self.valid_from.unwrap_or(0);
        match self.valid_until {
            Some(until) => at >= from && at < until,
            None => at >= from,
        }
    }

    /// Returns true if this edge has been invalidated.
    pub fn is_invalidated(&self) -> bool {
        self.valid_until.is_some()
    }
}

/// A pending edge in the delta log before compaction into CSR.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeltaEdge {
    source_idx: u32,
    target_idx: u32,
    data: StoredEdge,
}

/// Compressed Sparse Row storage for one direction (outgoing or incoming).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CompressedStorage {
    /// Length = num_nodes + 1. row_offsets[i]..row_offsets[i+1] gives the range in col_indices/edge_data.
    row_offsets: Vec<u32>,
    /// Column indices (target node for CSR, source node for CSC).
    col_indices: Vec<u32>,
    /// Edge metadata parallel to col_indices.
    edge_data: Vec<StoredEdge>,
}

impl CompressedStorage {
    #[allow(dead_code)]
    fn new(num_nodes: usize) -> Self {
        Self {
            row_offsets: vec![0; num_nodes + 1],
            col_indices: Vec::new(),
            edge_data: Vec::new(),
        }
    }

    /// Get neighbors and edge data for a given row index.
    fn neighbors(&self, row: u32) -> &[u32] {
        let row = row as usize;
        if row + 1 >= self.row_offsets.len() {
            return &[];
        }
        let start = self.row_offsets[row] as usize;
        let end = self.row_offsets[row + 1] as usize;
        &self.col_indices[start..end]
    }

    fn edge_data_for(&self, row: u32) -> &[StoredEdge] {
        let row = row as usize;
        if row + 1 >= self.row_offsets.len() {
            return &[];
        }
        let start = self.row_offsets[row] as usize;
        let end = self.row_offsets[row + 1] as usize;
        &self.edge_data[start..end]
    }
}

/// Bidirectional graph with CSR (outgoing) and CSC (incoming) plus a delta log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsrGraph {
    /// Maps MemoryId -> internal u32 index.
    id_to_idx: HashMap<MemoryId, u32>,
    /// Maps internal u32 index -> MemoryId.
    idx_to_id: Vec<MemoryId>,

    /// CSR for outgoing edges.
    csr: CompressedStorage,
    /// CSC for incoming edges.
    csc: CompressedStorage,

    /// Recent edges not yet merged into the compressed storage.
    delta_edges: Vec<DeltaEdge>,
    /// Edges marked for removal (source_idx, target_idx).
    removed_edges: Vec<(u32, u32)>,

    /// Live edge count per type, maintained incrementally so counts (e.g. the
    /// dashboard conflict badge) are O(1) instead of a walk over every node ever
    /// created. The detectors store one directed edge per conflict pair, so a
    /// directed count equals the pair count. Grounded authoritatively on
    /// `compact` and after load; adjusted on add and per-node removal.
    #[serde(default)]
    edge_type_counts: HashMap<EdgeType, usize>,

    /// Live conflict-class edges (contradicts, supersedes) with their weight,
    /// maintained incrementally so the dashboard conflict view enumerates the
    /// actual pairs in O(conflicts) instead of walking every node's out-edges
    /// (which rescans the delta log per node, O(nodes x delta)). Keyed by
    /// `(source, target, type)` so a pair carrying two different edge types does
    /// not collide. Skipped on serialize (a tuple key is not a JSON map key) and
    /// reground from the live edges by `recompute_edge_counts` on load, so it
    /// costs nothing on disk and is always correct after a restart.
    #[serde(skip)]
    conflict_edges: HashMap<(MemoryId, MemoryId, EdgeType), f32>,
}

/// The edge types the dashboard treats as conflicts, maintained as a live pair
/// list for O(conflicts) enumeration.
fn is_conflict_type(t: EdgeType) -> bool {
    matches!(t, EdgeType::Contradicts | EdgeType::Supersedes)
}

impl CsrGraph {
    /// Creates a new empty CSR graph.
    pub fn new() -> Self {
        Self {
            id_to_idx: HashMap::default(),
            idx_to_id: Vec::new(),
            csr: CompressedStorage::default(),
            csc: CompressedStorage::default(),
            delta_edges: Vec::new(),
            removed_edges: Vec::new(),
            edge_type_counts: HashMap::default(),
            conflict_edges: HashMap::default(),
        }
    }

    /// Register a node. Returns its internal index.
    pub fn add_node(&mut self, id: MemoryId) -> u32 {
        if let Some(&idx) = self.id_to_idx.get(&id) {
            return idx;
        }
        let idx = self.idx_to_id.len() as u32;
        self.id_to_idx.insert(id, idx);
        self.idx_to_id.push(id);
        idx
    }

    /// Remove a node and all its edges.
    pub fn remove_node(&mut self, id: MemoryId) {
        let Some(&idx) = self.id_to_idx.get(&id) else {
            return;
        };

        // Collect the types of this node's still-live edges so the running
        // per-type counts can be decremented. `is_removed` guards against
        // decrementing an edge twice when both of its endpoints are removed.
        let mut decrements: Vec<EdgeType> = Vec::new();
        {
            let out_neighbors = self.csr.neighbors(idx).to_vec();
            let out_types: Vec<EdgeType> = self
                .csr
                .edge_data_for(idx)
                .iter()
                .map(|e| e.edge_type)
                .collect();
            for (i, &neighbor) in out_neighbors.iter().enumerate() {
                if !self.is_removed(idx, neighbor) {
                    decrements.push(out_types[i]);
                }
            }
            let in_neighbors = self.csc.neighbors(idx).to_vec();
            let in_types: Vec<EdgeType> = self
                .csc
                .edge_data_for(idx)
                .iter()
                .map(|e| e.edge_type)
                .collect();
            for (i, &neighbor) in in_neighbors.iter().enumerate() {
                if !self.is_removed(neighbor, idx) {
                    decrements.push(in_types[i]);
                }
            }
            for e in &self.delta_edges {
                if e.source_idx == idx || e.target_idx == idx {
                    decrements.push(e.data.edge_type);
                }
            }
        }
        for et in decrements {
            if let Some(c) = self.edge_type_counts.get_mut(&et) {
                *c = c.saturating_sub(1);
            }
        }

        // Mark all outgoing and incoming edges for removal
        for &neighbor in self.csr.neighbors(idx) {
            self.removed_edges.push((idx, neighbor));
        }
        for &neighbor in self.csc.neighbors(idx) {
            self.removed_edges.push((neighbor, idx));
        }
        // Also remove from delta
        self.delta_edges
            .retain(|e| e.source_idx != idx && e.target_idx != idx);
        // Drop any conflict edges touching this node.
        self.conflict_edges
            .retain(|(s, t, _), _| *s != id && *t != id);
        self.id_to_idx.remove(&id);
    }

    /// Add an edge to the delta log.
    ///
    /// Deduplicates: if a currently-valid edge with the same source, target,
    /// and type already exists (in CSR or the delta log), the graph is left
    /// unchanged. Write inference and pipeline passes may infer the same
    /// relationship more than once; parallel duplicates carry no information.
    pub fn add_edge(&mut self, edge: &MemoryEdge) {
        let source_idx = self.add_node(edge.source);
        let target_idx = self.add_node(edge.target);

        let duplicate = self.outgoing_by_idx(source_idx).into_iter().any(|(t, e)| {
            t == edge.target && e.edge_type == edge.edge_type && e.valid_until.is_none()
        });
        if duplicate {
            return;
        }

        self.delta_edges.push(DeltaEdge {
            source_idx,
            target_idx,
            data: StoredEdge::from_memory_edge(edge),
        });
        *self.edge_type_counts.entry(edge.edge_type).or_default() += 1;
        if is_conflict_type(edge.edge_type) {
            self.conflict_edges
                .insert((edge.source, edge.target, edge.edge_type), edge.weight);
        }
    }

    /// Strengthen an edge by incrementing its weight (Hebbian learning).
    ///
    /// Updates the existing edge rather than appending a parallel duplicate:
    /// a delta edge is bumped in place; a compressed (CSR) edge is marked
    /// removed and replaced with a delta override, so compaction keeps
    /// exactly one copy.
    pub fn strengthen_edge(&mut self, source: MemoryId, target: MemoryId, delta: f32) {
        let (Some(&source_idx), Some(&target_idx)) =
            (self.id_to_idx.get(&source), self.id_to_idx.get(&target))
        else {
            return;
        };

        // Keep the conflict-edge weights in step with the live edge (a pair may
        // carry either conflict type).
        for t in [EdgeType::Contradicts, EdgeType::Supersedes] {
            if let Some(w) = self.conflict_edges.get_mut(&(source, target, t)) {
                *w = (*w + delta).min(1.0);
            }
        }

        // Bump an existing delta edge in place.
        if let Some(existing) = self
            .delta_edges
            .iter_mut()
            .find(|e| e.source_idx == source_idx && e.target_idx == target_idx)
        {
            existing.data.weight = (existing.data.weight + delta).min(1.0);
            return;
        }

        // Edge lives in compressed storage: suppress the CSR copy and push a
        // delta override. Removed-edge filtering only applies to compressed
        // edges, so the override remains visible.
        if let Some((_, stored)) = self
            .outgoing_by_idx(source_idx)
            .into_iter()
            .find(|(id, _)| *id == target)
        {
            let new_weight = (stored.weight + delta).min(1.0);
            self.removed_edges.push((source_idx, target_idx));
            self.delta_edges.push(DeltaEdge {
                source_idx,
                target_idx,
                data: StoredEdge {
                    weight: new_weight,
                    ..stored
                },
            });
        }
    }

    /// Mark an edge for removal.
    pub fn remove_edge(&mut self, source: MemoryId, target: MemoryId) {
        let (Some(&src_idx), Some(&tgt_idx)) =
            (self.id_to_idx.get(&source), self.id_to_idx.get(&target))
        else {
            return;
        };
        self.removed_edges.push((src_idx, tgt_idx));
        self.delta_edges
            .retain(|e| !(e.source_idx == src_idx && e.target_idx == tgt_idx));
        // Pair-granular removal drops every edge between the pair.
        self.conflict_edges
            .retain(|(s, t, _), _| !(*s == source && *t == target));
    }

    /// Remove every edge whose type is in `types`, preserving all other edges,
    /// including edges of a different type between the same pair. Returns the
    /// number of edges removed.
    ///
    /// The removal model suppresses a compressed edge at pair granularity, so
    /// when a suppressed pair also carries a non-matching edge, that sibling is
    /// re-added to the delta log to keep it. Callers should `compact()` after to
    /// materialize the result into both CSR and CSC. This is a one-time cleanup
    /// primitive (used to purge conflict edges the old write-time heuristic
    /// created at ~0% precision), not a hot path.
    pub fn remove_edges_of_types(&mut self, types: &[EdgeType]) -> usize {
        let matches = |et: EdgeType| types.contains(&et);
        let mut removed = 0usize;

        // Delta edges: drop matching ones directly.
        let before = self.delta_edges.len();
        self.delta_edges.retain(|e| !matches(e.data.edge_type));
        removed += before - self.delta_edges.len();

        // Compressed edges: suppress matching ones. Suppression is pair level,
        // so restore any non-matching sibling of a suppressed pair via delta.
        let mut restore: Vec<DeltaEdge> = Vec::new();
        let n = self.idx_to_id.len() as u32;
        for src_idx in 0..n {
            let row: Vec<(u32, StoredEdge)> = {
                let neighbors = self.csr.neighbors(src_idx);
                let edges = self.csr.edge_data_for(src_idx);
                let mut v = Vec::new();
                for (i, &t) in neighbors.iter().enumerate() {
                    if !self.is_removed(src_idx, t) {
                        v.push((t, edges[i].clone()));
                    }
                }
                v
            };
            let matched_pairs: std::collections::HashSet<u32> = row
                .iter()
                .filter(|(_, e)| matches(e.edge_type))
                .map(|(t, _)| *t)
                .collect();
            if matched_pairs.is_empty() {
                continue;
            }
            for &tgt_idx in &matched_pairs {
                self.removed_edges.push((src_idx, tgt_idx));
            }
            for (tgt_idx, e) in row {
                if !matched_pairs.contains(&tgt_idx) {
                    continue;
                }
                if matches(e.edge_type) {
                    removed += 1;
                } else {
                    restore.push(DeltaEdge {
                        source_idx: src_idx,
                        target_idx: tgt_idx,
                        data: e,
                    });
                }
            }
        }
        self.delta_edges.extend(restore);
        // Bulk type removal touches many edges via suppression and restore; just
        // recount from the resulting live set rather than track each delta.
        self.recompute_edge_counts();
        removed
    }

    /// Get all outgoing edges from a node (CSR + delta, minus removed).
    pub fn outgoing(&self, id: MemoryId) -> Vec<(MemoryId, StoredEdge)> {
        let Some(&idx) = self.id_to_idx.get(&id) else {
            return Vec::new();
        };
        self.outgoing_by_idx(idx)
    }

    /// Get outgoing edges that are temporally valid at the given timestamp.
    pub fn outgoing_valid_at(&self, id: MemoryId, at: Timestamp) -> Vec<(MemoryId, StoredEdge)> {
        self.outgoing(id)
            .into_iter()
            .filter(|(_, e)| e.is_valid_at(at))
            .collect()
    }

    pub(crate) fn outgoing_by_idx(&self, idx: u32) -> Vec<(MemoryId, StoredEdge)> {
        let mut results = Vec::new();

        // From compressed storage
        let neighbors = self.csr.neighbors(idx);
        let edges = self.csr.edge_data_for(idx);
        for (i, &neighbor) in neighbors.iter().enumerate() {
            if !self.is_removed(idx, neighbor)
                && let Some(&id) = self.idx_to_id.get(neighbor as usize)
            {
                results.push((id, edges[i].clone()));
            }
        }

        // From delta
        for delta in &self.delta_edges {
            if delta.source_idx == idx
                && let Some(&id) = self.idx_to_id.get(delta.target_idx as usize)
            {
                results.push((id, delta.data.clone()));
            }
        }

        results
    }

    /// `(contradicts, supersedes)` live conflict-edge counts, read O(1) from the
    /// running per-type counts. The detectors store one directed edge per
    /// conflict pair, so a directed count is the pair count.
    pub fn conflict_edge_counts(&self) -> (usize, usize) {
        (
            self.edge_type_counts
                .get(&EdgeType::Contradicts)
                .copied()
                .unwrap_or(0),
            self.edge_type_counts
                .get(&EdgeType::Supersedes)
                .copied()
                .unwrap_or(0),
        )
    }

    /// Every live conflict-class edge as `(source, target, type, weight)`, read
    /// in O(conflicts) from the maintained pair list, so the dashboard conflict
    /// view never walks the whole graph. Grounded on compact/load, so it stays
    /// correct across a restart.
    pub fn conflict_edges(&self) -> Vec<(MemoryId, MemoryId, EdgeType, f32)> {
        self.conflict_edges
            .iter()
            .map(|(&(s, t, et), &w)| (s, t, et, w))
            .collect()
    }

    /// Recount live edges by type from scratch. Authoritative but O(nodes +
    /// edges); called only when the graph is already being walked wholesale
    /// (`compact`, bulk type removal) or grounded after load, never on a read.
    pub fn recompute_edge_counts(&mut self) {
        let mut counts: HashMap<EdgeType, usize> = HashMap::default();
        let mut conflicts: HashMap<(MemoryId, MemoryId, EdgeType), f32> = HashMap::default();
        let node_count = self.idx_to_id.len() as u32;
        for idx in 0..node_count {
            let neighbors = self.csr.neighbors(idx);
            let data = self.csr.edge_data_for(idx);
            for (i, &nbr) in neighbors.iter().enumerate() {
                if !self.is_removed(idx, nbr) {
                    let et = data[i].edge_type;
                    *counts.entry(et).or_default() += 1;
                    if is_conflict_type(et)
                        && let (Some(&s), Some(&t)) = (
                            self.idx_to_id.get(idx as usize),
                            self.idx_to_id.get(nbr as usize),
                        )
                    {
                        conflicts.insert((s, t, et), data[i].weight);
                    }
                }
            }
        }
        for d in &self.delta_edges {
            if !self.is_removed(d.source_idx, d.target_idx) {
                let et = d.data.edge_type;
                *counts.entry(et).or_default() += 1;
                if is_conflict_type(et)
                    && let (Some(&s), Some(&t)) = (
                        self.idx_to_id.get(d.source_idx as usize),
                        self.idx_to_id.get(d.target_idx as usize),
                    )
                {
                    conflicts.insert((s, t, et), d.data.weight);
                }
            }
        }
        self.edge_type_counts = counts;
        self.conflict_edges = conflicts;
    }

    /// Get all incoming edges to a node (CSC + delta, minus removed).
    pub fn incoming(&self, id: MemoryId) -> Vec<(MemoryId, StoredEdge)> {
        let Some(&idx) = self.id_to_idx.get(&id) else {
            return Vec::new();
        };
        self.incoming_by_idx(idx)
    }

    /// Get incoming edges that are temporally valid at the given timestamp.
    pub fn incoming_valid_at(&self, id: MemoryId, at: Timestamp) -> Vec<(MemoryId, StoredEdge)> {
        self.incoming(id)
            .into_iter()
            .filter(|(_, e)| e.is_valid_at(at))
            .collect()
    }

    pub(crate) fn incoming_by_idx(&self, idx: u32) -> Vec<(MemoryId, StoredEdge)> {
        let mut results = Vec::new();

        // From compressed storage (CSC)
        let neighbors = self.csc.neighbors(idx);
        let edges = self.csc.edge_data_for(idx);
        for (i, &neighbor) in neighbors.iter().enumerate() {
            if !self.is_removed(neighbor, idx)
                && let Some(&id) = self.idx_to_id.get(neighbor as usize)
            {
                results.push((id, edges[i].clone()));
            }
        }

        // From delta
        for delta in &self.delta_edges {
            if delta.target_idx == idx
                && let Some(&id) = self.idx_to_id.get(delta.source_idx as usize)
            {
                results.push((id, delta.data.clone()));
            }
        }

        results
    }

    /// Check if a node exists in the graph.
    pub fn contains_node(&self, id: MemoryId) -> bool {
        self.id_to_idx.contains_key(&id)
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.idx_to_id.len()
    }

    /// Resolve a MemoryId to its internal index.
    pub(crate) fn get_idx(&self, id: MemoryId) -> Option<u32> {
        self.id_to_idx.get(&id).copied()
    }

    /// Resolve an internal index to its MemoryId.
    #[allow(dead_code)]
    pub(crate) fn get_id(&self, idx: u32) -> Option<MemoryId> {
        self.idx_to_id.get(idx as usize).copied()
    }

    /// All registered node IDs.
    pub fn node_ids(&self) -> &[MemoryId] {
        &self.idx_to_id
    }

    fn is_removed(&self, source: u32, target: u32) -> bool {
        self.removed_edges
            .iter()
            .any(|&(s, t)| s == source && t == target)
    }

    /// Merge all delta edges and removals into the compressed CSR/CSC storage.
    pub fn compact(&mut self) {
        let num_nodes = self.idx_to_id.len();

        // Collect all edges: existing (minus removed) + delta
        let mut all_edges: Vec<(u32, u32, StoredEdge)> = Vec::new();

        // Existing CSR edges
        for row in 0..num_nodes {
            let row = row as u32;
            let neighbors = self.csr.neighbors(row);
            let edges = self.csr.edge_data_for(row);
            for (i, &col) in neighbors.iter().enumerate() {
                if !self.is_removed(row, col) {
                    all_edges.push((row, col, edges[i].clone()));
                }
            }
        }

        // Delta edges
        for delta in &self.delta_edges {
            all_edges.push((delta.source_idx, delta.target_idx, delta.data.clone()));
        }

        // Build CSR (sorted by source)
        self.csr = Self::build_compressed(&all_edges, num_nodes, false);

        // Build CSC (sorted by target)
        self.csc = Self::build_compressed(&all_edges, num_nodes, true);

        self.delta_edges.clear();
        self.removed_edges.clear();

        // Ground the running per-type counts against the freshly compacted
        // edge set, so any incremental drift since the last compaction is reset.
        self.recompute_edge_counts();
    }

    fn build_compressed(
        edges: &[(u32, u32, StoredEdge)],
        num_nodes: usize,
        transpose: bool,
    ) -> CompressedStorage {
        // Count edges per row
        let mut counts = vec![0u32; num_nodes];
        for &(src, tgt, ref _data) in edges {
            let row = if transpose { tgt } else { src };
            if (row as usize) < num_nodes {
                counts[row as usize] += 1;
            }
        }

        // Build offsets via prefix sum
        let mut row_offsets = vec![0u32; num_nodes + 1];
        for i in 0..num_nodes {
            row_offsets[i + 1] = row_offsets[i] + counts[i];
        }

        let total = row_offsets[num_nodes] as usize;
        let mut col_indices = vec![0u32; total];
        let mut edge_data = vec![
            StoredEdge {
                edge_type: EdgeType::Related,
                weight: 0.0,
                created_at: 0,
                valid_from: None,
                valid_until: None,
                label: None,
            };
            total
        ];

        // Fill using write cursors
        let mut cursors = row_offsets[..num_nodes].to_vec();
        for &(src, tgt, ref data) in edges {
            let (row, col) = if transpose { (tgt, src) } else { (src, tgt) };
            if (row as usize) < num_nodes {
                let pos = cursors[row as usize] as usize;
                col_indices[pos] = col;
                edge_data[pos] = data.clone();
                cursors[row as usize] += 1;
            }
        }

        CompressedStorage {
            row_offsets,
            col_indices,
            edge_data,
        }
    }
    /// Save the graph snapshot to a file.
    pub fn save(&self, path: &std::path::Path) -> MenteResult<()> {
        let data =
            serde_json::to_vec(self).map_err(|e| MenteError::Serialization(e.to_string()))?;
        // Atomic snapshot: write to a temp file, fsync, then rename over the
        // old snapshot so a crash mid-save never leaves a truncated graph.
        let tmp_path = path.with_extension("json.tmp");
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&tmp_path)?;
            file.write_all(&data)?;
            file.sync_data()?;
        }
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Load the graph from a file.
    pub fn load(path: &std::path::Path) -> MenteResult<Self> {
        let data = std::fs::read(path)?;
        let mut graph: Self =
            serde_json::from_slice(&data).map_err(|e| MenteError::Serialization(e.to_string()))?;
        // Ground the per-type counts from the loaded edges. Snapshots written
        // before this field existed carry an empty map; recomputing makes the
        // counts correct regardless, and the edge-log replay in
        // `GraphManager::open` then adjusts for changes since the snapshot.
        graph.recompute_edge_counts();
        Ok(graph)
    }
}

impl Default for CsrGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edge(src: MemoryId, tgt: MemoryId, etype: EdgeType) -> MemoryEdge {
        MemoryEdge {
            source: src,
            target: tgt,
            edge_type: etype,
            weight: 0.8,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        }
    }

    #[test]
    fn test_add_node_idempotent() {
        let mut g = CsrGraph::new();
        let id = MemoryId::new();
        let idx1 = g.add_node(id);
        let idx2 = g.add_node(id);
        assert_eq!(idx1, idx2);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn edge_type_counts_track_add_compact_and_remove() {
        let mut g = CsrGraph::new();
        let (a, b, c, d) = (
            MemoryId::new(),
            MemoryId::new(),
            MemoryId::new(),
            MemoryId::new(),
        );

        // Adds land in the delta log and bump the running counts. Related edges
        // are not conflicts and must not be counted.
        g.add_edge(&make_edge(a, b, EdgeType::Contradicts));
        g.add_edge(&make_edge(c, d, EdgeType::Supersedes));
        g.add_edge(&make_edge(a, c, EdgeType::Related));
        assert_eq!(g.conflict_edge_counts(), (1, 1));
        // An exact-duplicate add is a no-op and must not double count.
        g.add_edge(&make_edge(a, b, EdgeType::Contradicts));
        assert_eq!(g.conflict_edge_counts(), (1, 1));

        // Counts survive compaction (the CSR is rebuilt and the counts are
        // reground from the compacted edge set).
        g.compact();
        assert_eq!(g.conflict_edge_counts(), (1, 1));

        // A new edge after compaction lands in the delta log again.
        g.add_edge(&make_edge(a, d, EdgeType::Supersedes));
        assert_eq!(g.conflict_edge_counts(), (1, 2));

        // Removing a node drops its edges from the counts: a->b (contradicts,
        // now compacted) and a->d (supersedes, in delta). c->d supersedes stays.
        g.remove_node(a);
        assert_eq!(g.conflict_edge_counts(), (0, 1));
    }

    #[test]
    fn conflict_edges_enumerated_maintained_and_persist() {
        let mut g = CsrGraph::new();
        let (a, b, c, d) = (
            MemoryId::new(),
            MemoryId::new(),
            MemoryId::new(),
            MemoryId::new(),
        );
        g.add_edge(&make_edge(a, b, EdgeType::Contradicts));
        g.add_edge(&make_edge(c, d, EdgeType::Supersedes));
        g.add_edge(&make_edge(a, c, EdgeType::Related)); // not a conflict

        // Enumeration returns only the two conflict pairs, not the related edge.
        let pairs = g.conflict_edges();
        assert_eq!(pairs.len(), 2);
        assert!(
            pairs
                .iter()
                .any(|(s, t, k, _)| *s == a && *t == b && *k == EdgeType::Contradicts)
        );
        assert!(
            pairs
                .iter()
                .any(|(s, t, k, _)| *s == c && *t == d && *k == EdgeType::Supersedes)
        );

        // Survives compaction (reground from the compacted set).
        g.compact();
        assert_eq!(g.conflict_edges().len(), 2);

        // Persisted and reground on load, so a restart keeps the view O(conflicts).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.json");
        g.save(&path).unwrap();
        let reloaded = CsrGraph::load(&path).unwrap();
        assert_eq!(reloaded.conflict_edges().len(), 2);

        // Removing a node drops the conflict edge that touched it.
        g.remove_node(a);
        let pairs = g.conflict_edges();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, c);
        assert_eq!(pairs[0].2, EdgeType::Supersedes);
    }

    #[test]
    fn test_add_and_query_edges() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Caused));
        g.add_edge(&make_edge(a, c, EdgeType::Related));

        let out = g.outgoing(a);
        assert_eq!(out.len(), 2);

        let inc_b = g.incoming(b);
        assert_eq!(inc_b.len(), 1);
        assert_eq!(inc_b[0].0, a);
    }

    #[test]
    fn test_remove_edge() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Caused));
        assert_eq!(g.outgoing(a).len(), 1);

        g.remove_edge(a, b);
        assert_eq!(g.outgoing(a).len(), 0);
    }

    #[test]
    fn test_remove_edges_of_types_preserves_siblings() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        // a->b carries BOTH a Supersedes and a Related edge (the collateral
        // case: pair-level suppression must not drop the Related sibling).
        g.add_edge(&make_edge(a, b, EdgeType::Supersedes));
        g.add_edge(&make_edge(a, b, EdgeType::Related));
        // a->c is a lone Contradicts edge.
        g.add_edge(&make_edge(a, c, EdgeType::Contradicts));
        // b->c is unrelated and must survive.
        g.add_edge(&make_edge(b, c, EdgeType::Caused));

        // Compact first so the targets live in compressed storage, exercising
        // the suppress-pair + restore-sibling path rather than delta retain.
        g.compact();

        let removed = g.remove_edges_of_types(&[EdgeType::Contradicts, EdgeType::Supersedes]);
        g.compact();
        assert_eq!(removed, 2, "one Supersedes + one Contradicts");

        // a->b: Supersedes gone, Related preserved.
        let ab: Vec<_> = g.outgoing(a).into_iter().filter(|(t, _)| *t == b).collect();
        assert_eq!(ab.len(), 1);
        assert_eq!(ab[0].1.edge_type, EdgeType::Related);

        // a->c contradiction gone entirely.
        assert!(g.outgoing(a).into_iter().all(|(t, _)| t != c));

        // Unrelated edge intact; incoming reflects the cleanup (only b->c left).
        assert_eq!(g.outgoing(b).len(), 1);
        let inc_c = g.incoming(c);
        assert_eq!(inc_c.len(), 1);
        assert_eq!(inc_c[0].0, b);
    }

    #[test]
    fn test_compact() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Caused));
        g.add_edge(&make_edge(b, c, EdgeType::Before));
        g.compact();

        let out_a = g.outgoing(a);
        assert_eq!(out_a.len(), 1);
        assert_eq!(out_a[0].0, b);

        let inc_c = g.incoming(c);
        assert_eq!(inc_c.len(), 1);
        assert_eq!(inc_c[0].0, b);
    }

    #[test]
    fn test_compact_with_removals() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Caused));
        g.add_edge(&make_edge(a, c, EdgeType::Related));
        g.compact();

        g.remove_edge(a, b);
        g.compact();

        let out = g.outgoing(a);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, c);
    }

    #[test]
    fn test_remove_node_cleans_id_to_idx() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Caused));
        assert!(g.contains_node(a));
        assert!(g.contains_node(b));

        g.remove_node(a);
        assert!(
            !g.contains_node(a),
            "removed node should not be in id_to_idx"
        );
        assert!(g.contains_node(b), "unrelated node should still exist");

        // Edges involving the removed node should be gone
        assert!(g.outgoing(a).is_empty());
        assert!(g.incoming(b).is_empty());
    }

    #[test]
    fn test_remove_node_then_readd() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Caused));
        g.remove_node(a);

        // Re-adding the same ID should get a fresh index
        g.add_edge(&make_edge(a, c, EdgeType::Related));
        assert!(g.contains_node(a));
        let out = g.outgoing(a);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, c);
    }
}
