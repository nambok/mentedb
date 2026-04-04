//! High-level knowledge graph manager.

use std::path::Path;

use mentedb_core::edge::MemoryEdge;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;

use crate::belief::propagate_update;
use crate::contradiction::find_contradictions;
use crate::csr::CsrGraph;
use crate::traversal::extract_subgraph;

/// Owns a `CsrGraph` and provides high-level graph operations.
pub struct GraphManager {
    graph: CsrGraph,
}

impl GraphManager {
    /// Creates a new graph manager with an empty graph.
    pub fn new() -> Self {
        Self {
            graph: CsrGraph::new(),
        }
    }

    /// Save the graph to the given directory.
    pub fn save(&self, dir: &Path) -> MenteResult<()> {
        std::fs::create_dir_all(dir)?;
        self.graph.save(&dir.join("graph.json"))
    }

    /// Load the graph from the given directory.
    pub fn load(dir: &Path) -> MenteResult<Self> {
        let graph = CsrGraph::load(&dir.join("graph.json"))?;
        Ok(Self { graph })
    }

    /// Access the underlying graph (for traversals, etc.).
    pub fn graph(&self) -> &CsrGraph {
        &self.graph
    }

    /// Register a memory node in the graph.
    pub fn add_memory(&mut self, id: MemoryId) {
        self.graph.add_node(id);
    }

    /// Remove a memory node and all its edges.
    pub fn remove_memory(&mut self, id: MemoryId) {
        self.graph.remove_node(id);
    }

    /// Add a relationship (edge) between two memory nodes.
    pub fn add_relationship(&mut self, edge: &MemoryEdge) -> MenteResult<()> {
        if !self.graph.contains_node(edge.source) {
            return Err(MenteError::MemoryNotFound(edge.source));
        }
        if !self.graph.contains_node(edge.target) {
            return Err(MenteError::MemoryNotFound(edge.target));
        }
        self.graph.add_edge(edge);
        Ok(())
    }

    /// Extract a context subgraph around a center node.
    pub fn get_context_subgraph(
        &self,
        center: MemoryId,
        depth: usize,
    ) -> (Vec<MemoryId>, Vec<MemoryEdge>) {
        extract_subgraph(&self.graph, center, depth)
    }

    /// Propagate a confidence change through the graph.
    pub fn propagate_belief_change(
        &self,
        id: MemoryId,
        new_confidence: f32,
    ) -> Vec<(MemoryId, f32)> {
        propagate_update(&self.graph, id, new_confidence)
    }

    /// Find all nodes that contradict the given node.
    pub fn find_all_contradictions(&self, id: MemoryId) -> Vec<MemoryId> {
        find_contradictions(&self.graph, id)
    }

    /// Merge the delta log into CSR/CSC compressed storage.
    pub fn compact(&mut self) {
        self.graph.compact();
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
    use uuid::Uuid;

    fn make_edge(src: MemoryId, tgt: MemoryId, etype: EdgeType) -> MemoryEdge {
        MemoryEdge {
            source: src,
            target: tgt,
            edge_type: etype,
            weight: 0.8,
            created_at: 1000,
        }
    }

    #[test]
    fn test_add_memory_and_relationship() {
        let mut mgr = GraphManager::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        mgr.add_memory(a);
        mgr.add_memory(b);
        assert!(
            mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
                .is_ok()
        );
    }

    #[test]
    fn test_relationship_missing_node() {
        let mut mgr = GraphManager::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        mgr.add_memory(a);
        // b not added
        assert!(
            mgr.add_relationship(&make_edge(a, b, EdgeType::Caused))
                .is_err()
        );
    }

    #[test]
    fn test_context_subgraph() {
        let mut mgr = GraphManager::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
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
        let mut mgr = GraphManager::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
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
        let mut mgr = GraphManager::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_relationship(&MemoryEdge {
            source: a,
            target: b,
            edge_type: EdgeType::Caused,
            weight: 1.0,
            created_at: 1000,
        })
        .unwrap();

        let results = mgr.propagate_belief_change(a, 0.5);
        assert!(results.len() >= 2);
    }
}
