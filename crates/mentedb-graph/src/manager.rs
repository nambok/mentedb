//! High-level knowledge graph manager.

use std::path::Path;

use mentedb_core::edge::MemoryEdge;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;
use parking_lot::RwLock;

use crate::belief::propagate_update;
use crate::contradiction::find_contradictions;
use crate::csr::CsrGraph;
use crate::traversal::extract_subgraph;

/// Owns a `CsrGraph` and provides high-level graph operations.
///
/// All methods take `&self` — internal `RwLock` handles concurrency.
pub struct GraphManager {
    graph: RwLock<CsrGraph>,
}

impl GraphManager {
    /// Creates a new graph manager with an empty graph.
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(CsrGraph::new()),
        }
    }

    /// Save the graph to the given directory.
    pub fn save(&self, dir: &Path) -> MenteResult<()> {
        std::fs::create_dir_all(dir)?;
        self.graph.read().save(&dir.join("graph.json"))
    }

    /// Load the graph from the given directory.
    pub fn load(dir: &Path) -> MenteResult<Self> {
        let graph = CsrGraph::load(&dir.join("graph.json"))?;
        Ok(Self {
            graph: RwLock::new(graph),
        })
    }

    /// Register a memory node in the graph.
    pub fn add_memory(&self, id: MemoryId) {
        self.graph.write().add_node(id);
    }

    /// Remove a memory node and all its edges.
    pub fn remove_memory(&self, id: MemoryId) {
        self.graph.write().remove_node(id);
    }

    /// Add a relationship (edge) between two memory nodes.
    pub fn add_relationship(&self, edge: &MemoryEdge) -> MenteResult<()> {
        let mut g = self.graph.write();
        if !g.contains_node(edge.source) {
            return Err(MenteError::MemoryNotFound(edge.source));
        }
        if !g.contains_node(edge.target) {
            return Err(MenteError::MemoryNotFound(edge.target));
        }
        g.add_edge(edge);
        Ok(())
    }

    /// Extract a context subgraph around a center node.
    pub fn get_context_subgraph(
        &self,
        center: MemoryId,
        depth: usize,
    ) -> (Vec<MemoryId>, Vec<MemoryEdge>) {
        extract_subgraph(&self.graph.read(), center, depth)
    }

    /// Propagate a confidence change through the graph.
    pub fn propagate_belief_change(
        &self,
        id: MemoryId,
        new_confidence: f32,
    ) -> Vec<(MemoryId, f32)> {
        propagate_update(&self.graph.read(), id, new_confidence)
    }

    /// Find all nodes that contradict the given node.
    pub fn find_all_contradictions(&self, id: MemoryId) -> Vec<MemoryId> {
        find_contradictions(&self.graph.read(), id)
    }

    /// Merge the delta log into CSR/CSC compressed storage.
    pub fn compact(&self) {
        self.graph.write().compact();
    }

    /// Strengthen an edge weight (Hebbian learning: neurons that fire together wire together).
    pub fn strengthen_edge(&self, source: MemoryId, target: MemoryId, delta: f32) {
        self.graph.write().strengthen_edge(source, target, delta);
    }

    /// Access the underlying graph for read-only traversals.
    ///
    /// Returns a read guard — hold it only briefly to avoid blocking writers.
    pub fn read_graph(&self) -> parking_lot::RwLockReadGuard<'_, CsrGraph> {
        self.graph.read()
    }

    /// Alias for `read_graph()` — backward compatible access to the CsrGraph.
    pub fn graph(&self) -> parking_lot::RwLockReadGuard<'_, CsrGraph> {
        self.graph.read()
    }
}

impl Default for GraphManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::edge::EdgeType;

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
    fn test_add_memory_and_relationship() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        assert!(
            mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
                .is_ok()
        );
    }

    #[test]
    fn test_relationship_missing_node() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        mgr.add_memory(a);
        // b not added
        assert!(
            mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
                .is_err()
        );
    }

    #[test]
    fn test_context_subgraph() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_memory(c);
        mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
            .unwrap();
        mgr.add_relationship(&make_edge(b, c, EdgeType::Related))
            .unwrap();

        let (nodes, edges) = mgr.get_context_subgraph(a, 2);
        assert_eq!(nodes.len(), 3);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_compact() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
            .unwrap();
        mgr.compact();

        let out = mgr.graph().outgoing(a);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_belief_propagation() {
        let mgr = GraphManager::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_relationship(&MemoryEdge {
            source: a,
            target: b,
            edge_type: EdgeType::Caused,
            weight: 1.0,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        })
        .unwrap();

        let results = mgr.propagate_belief_change(a, 0.5);
        assert!(results.len() >= 2);
    }
}
