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
    }

    /// Add an edge to the delta log.
    pub fn add_edge(&mut self, edge: &MemoryEdge) {
        let source_idx = self.add_node(edge.source);
        let target_idx = self.add_node(edge.target);
        self.delta_edges.push(DeltaEdge {
            source_idx,
            target_idx,
            data: StoredEdge::from_memory_edge(edge),
        });
    }

    /// Strengthen an edge by incrementing its weight (Hebbian learning).
    /// Adds a delta edge with the new weight; compaction will merge it.
    pub fn strengthen_edge(&mut self, source: MemoryId, target: MemoryId, delta: f32) {
        // Find the existing edge to get its current data
        if let Some(existing) = self
            .outgoing(source)
            .into_iter()
            .find(|(id, _)| *id == target)
        {
            let (_, stored) = existing;
            let new_weight = (stored.weight + delta).min(1.0);
            let source_idx = self.add_node(source);
            let target_idx = self.add_node(target);
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
    /// Save the graph to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> MenteResult<()> {
        let data =
            serde_json::to_vec(self).map_err(|e| MenteError::Serialization(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load the graph from a JSON file.
    pub fn load(path: &std::path::Path) -> MenteResult<Self> {
        let data = std::fs::read(path)?;
        let graph: Self =
            serde_json::from_slice(&data).map_err(|e| MenteError::Serialization(e.to_string()))?;
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
}
