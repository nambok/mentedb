//! Belief propagation through causal and supporting edges.

use std::collections::VecDeque;

use ahash::{HashMap, HashSet};
use mentedb_core::edge::EdgeType;
use mentedb_core::types::MemoryId;

use crate::csr::CsrGraph;

/// Configuration for belief propagation.
#[derive(Debug, Clone)]
pub struct PropagationConfig {
    /// Maximum BFS depth for propagation (default: 5).
    pub max_depth: usize,
    /// Dampening factor for Caused edges (default: 0.9).
    pub caused_dampening: f32,
    /// Factor for Supports edges (default: 0.5).
    pub supports_factor: f32,
    /// Factor for Contradicts edges (default: 0.7).
    pub contradicts_factor: f32,
    /// Floor multiplier for Supersedes edges (default: 0.1).
    pub supersedes_floor: f32,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            max_depth: 5,
            caused_dampening: 0.9,
            supports_factor: 0.5,
            contradicts_factor: 0.7,
            supersedes_floor: 0.1,
        }
    }
}

/// Propagate a confidence change through the graph.
///
/// Returns a list of (affected_node, new_confidence) pairs. The initial node
/// is included with `new_confidence` as its updated value.
pub fn propagate_update(
    graph: &CsrGraph,
    changed_id: MemoryId,
    new_confidence: f32,
) -> Vec<(MemoryId, f32)> {
    propagate_update_with_config(graph, changed_id, new_confidence, &PropagationConfig::default())
}

/// Propagate a confidence change through the graph with custom configuration.
pub fn propagate_update_with_config(
    graph: &CsrGraph,
    changed_id: MemoryId,
    new_confidence: f32,
    config: &PropagationConfig,
) -> Vec<(MemoryId, f32)> {
    let Some(_) = graph.get_idx(changed_id) else {
        return Vec::new();
    };

    // Track computed confidences for each affected node
    let mut confidences: HashMap<MemoryId, f32> = HashMap::default();
    confidences.insert(changed_id, new_confidence);

    // BFS queue: (node_id, confidence_at_node, current_depth)
    let mut queue: VecDeque<(MemoryId, f32, usize)> = VecDeque::new();
    queue.push_back((changed_id, new_confidence, 0));

    let mut visited = HashSet::default();
    visited.insert(changed_id);

    while let Some((node, node_confidence, depth)) = queue.pop_front() {
        if depth >= config.max_depth {
            continue;
        }

        for (neighbor, edge) in graph.outgoing(node) {
            let new_conf = match edge.edge_type {
                EdgeType::Caused => {
                    // child confidence = parent confidence * edge weight * dampening
                    node_confidence * edge.weight * config.caused_dampening
                }
                EdgeType::Supports => {
                    // supported node confidence += delta * edge weight * factor
                    let current = confidences.get(&neighbor).copied().unwrap_or(1.0);
                    let delta = node_confidence - 1.0; // change from baseline
                    (current + delta * edge.weight * config.supports_factor).clamp(0.0, 1.0)
                }
                EdgeType::Contradicts => {
                    // contradicted node confidence -= delta * edge weight * factor
                    let current = confidences.get(&neighbor).copied().unwrap_or(1.0);
                    (current - node_confidence * edge.weight * config.contradicts_factor).clamp(0.0, 1.0)
                }
                EdgeType::Supersedes => {
                    // superseded node confidence = min(current, new * floor)
                    let current = confidences.get(&neighbor).copied().unwrap_or(1.0);
                    current.min(node_confidence * config.supersedes_floor)
                }
                // Other edge types don't propagate belief
                _ => continue,
            };

            confidences.insert(neighbor, new_conf);
            if visited.insert(neighbor) {
                queue.push_back((neighbor, new_conf, depth + 1));
            }
        }
    }

    confidences.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::edge::MemoryEdge;
    use uuid::Uuid;

    fn make_edge(src: MemoryId, tgt: MemoryId, etype: EdgeType, weight: f32) -> MemoryEdge {
        MemoryEdge {
            source: src,
            target: tgt,
            edge_type: etype,
            weight,
            created_at: 1000,
        }
    }

    #[test]
    fn test_caused_propagation() {
        let mut g = CsrGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        g.add_edge(&make_edge(a, b, EdgeType::Caused, 1.0));

        let result = propagate_update(&g, a, 0.5);
        let map: HashMap<MemoryId, f32> = result.into_iter().collect();

        assert!((map[&a] - 0.5).abs() < 0.001);
        // b = 0.5 * 1.0 * 0.9 = 0.45
        assert!((map[&b] - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_contradicts_propagation() {
        let mut g = CsrGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        g.add_edge(&make_edge(a, b, EdgeType::Contradicts, 1.0));

        let result = propagate_update(&g, a, 0.8);
        let map: HashMap<MemoryId, f32> = result.into_iter().collect();

        // b = max(0, 1.0 - 0.8 * 1.0 * 0.7) = 0.44
        assert!((map[&b] - 0.44).abs() < 0.001);
    }

    #[test]
    fn test_supersedes_propagation() {
        let mut g = CsrGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        g.add_edge(&make_edge(a, b, EdgeType::Supersedes, 1.0));

        let result = propagate_update(&g, a, 0.9);
        let map: HashMap<MemoryId, f32> = result.into_iter().collect();

        // b = min(1.0, 0.9 * 0.1) = 0.09
        assert!((map[&b] - 0.09).abs() < 0.001);
    }

    #[test]
    fn test_max_depth_limit() {
        // Build a chain of 10 Caused edges; only first 5 should propagate
        let mut g = CsrGraph::new();
        let ids: Vec<MemoryId> = (0..10).map(|_| Uuid::new_v4()).collect();
        for i in 0..9 {
            g.add_edge(&make_edge(ids[i], ids[i + 1], EdgeType::Caused, 1.0));
        }

        let result = propagate_update(&g, ids[0], 1.0);
        let map: HashMap<MemoryId, f32> = result.into_iter().collect();

        // Should have at most 6 nodes (start + 5 depth levels)
        assert!(map.len() <= 6);
    }

    #[test]
    fn test_unrelated_edges_dont_propagate() {
        let mut g = CsrGraph::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        g.add_edge(&make_edge(a, b, EdgeType::Related, 1.0));

        let result = propagate_update(&g, a, 0.5);
        // Only the changed node itself
        assert_eq!(result.len(), 1);
    }
}
