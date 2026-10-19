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
    /// Outgoing edge types the walk follows (default: `Caused` and
    /// `Supports`, the dependency edges). Any other edge type is ignored.
    ///
    /// The dependency edges are the default because they are the only ones
    /// whose meaning survives a change in the source's confidence: an effect
    /// or a supported claim is only as believable as what it rests on.
    /// `Contradicts` and `Supersedes` record a conflict that was already
    /// resolved, and walking them from a memory that just lost a conflict
    /// would move the memories it once contradicted or replaced on no new
    /// evidence. They can be added back here explicitly.
    pub edge_types: Vec<EdgeType>,
    /// Smallest per memory change [`propagate_delta`] records or walks past
    /// (default: 0.01). A damped change below this stops that branch.
    pub min_delta: f32,
    /// Most memories one [`propagate_delta`] call may change, root excluded
    /// (default: 64). The walk stops once it has this many.
    pub max_affected: usize,
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
            edge_types: vec![EdgeType::Caused, EdgeType::Supports],
            min_delta: 0.01,
            max_affected: 64,
            caused_dampening: 0.9,
            supports_factor: 0.5,
            contradicts_factor: 0.7,
            supersedes_floor: 0.1,
        }
    }
}

impl PropagationConfig {
    /// The defaults with every belief carrying edge type followed (`Caused`,
    /// `Supports`, `Contradicts`, `Supersedes`), the edge set of
    /// [`propagate_update`].
    pub fn all_edges() -> Self {
        Self {
            edge_types: vec![
                EdgeType::Caused,
                EdgeType::Supports,
                EdgeType::Contradicts,
                EdgeType::Supersedes,
            ],
            ..Self::default()
        }
    }
}

/// Propagate a confidence change through the graph, following every belief
/// carrying edge type ([`PropagationConfig::all_edges`]).
///
/// Returns a list of (affected_node, new_confidence) pairs. The initial node
/// is included with `new_confidence` as its updated value. The values are
/// absolute and computed from an assumed baseline of 1.0 for each neighbor,
/// not from the neighbors' stored confidences; the engine's write path uses
/// [`propagate_delta`] instead.
pub fn propagate_update(
    graph: &CsrGraph,
    changed_id: MemoryId,
    new_confidence: f32,
) -> Vec<(MemoryId, f32)> {
    propagate_update_with_config(
        graph,
        changed_id,
        new_confidence,
        &PropagationConfig::all_edges(),
    )
}

/// Propagate a change of `delta` in `root`'s confidence to the memories that
/// depend on it, as damped relative changes.
///
/// Returns `(memory, delta)` pairs for the affected memories, root excluded:
/// the caller owns the root's own confidence and has already set it. A
/// neighbor reached over an edge of weight `w` moves by the source's delta
/// times `w` (clamped to `[0, 1]`) times the edge type's factor:
/// `caused_dampening` for `Caused`, `supports_factor` for `Supports`, and the
/// negated `contradicts_factor` for `Contradicts` (less belief in a claim is
/// more belief in its contradiction). `Supersedes` is never walked here: a
/// superseded memory is closed by its validity window, and its confidence
/// belongs to the supersession that closed it.
///
/// Only edge types in `config.edge_types` are followed. Each memory is
/// changed once, by the first (shortest) path that reaches it; a change
/// smaller than `config.min_delta` is dropped and its branch stops; the walk
/// stops at `config.max_depth` hops and after `config.max_affected` memories.
/// Pure graph work: the caller applies the deltas to stored confidences
/// after releasing the graph lock.
pub fn propagate_delta(
    graph: &CsrGraph,
    root: MemoryId,
    delta: f32,
    config: &PropagationConfig,
) -> Vec<(MemoryId, f32)> {
    let mut affected = Vec::new();
    if graph.get_idx(root).is_none()
        || !delta.is_finite()
        || delta.abs() < config.min_delta
        || config.max_affected == 0
    {
        return affected;
    }

    let mut visited = HashSet::default();
    visited.insert(root);
    let mut queue: VecDeque<(MemoryId, f32, usize)> = VecDeque::new();
    queue.push_back((root, delta, 0));

    while let Some((node, node_delta, depth)) = queue.pop_front() {
        if depth >= config.max_depth {
            continue;
        }
        for (neighbor, edge) in graph.outgoing(node) {
            if visited.contains(&neighbor) || !config.edge_types.contains(&edge.edge_type) {
                continue;
            }
            let factor = match edge.edge_type {
                EdgeType::Caused => config.caused_dampening,
                EdgeType::Supports => config.supports_factor,
                EdgeType::Contradicts => -config.contradicts_factor,
                _ => continue,
            };
            let child_delta = node_delta * edge.weight.clamp(0.0, 1.0) * factor;
            if !child_delta.is_finite() || child_delta.abs() < config.min_delta {
                continue;
            }
            visited.insert(neighbor);
            affected.push((neighbor, child_delta));
            if affected.len() >= config.max_affected {
                return affected;
            }
            queue.push_back((neighbor, child_delta, depth + 1));
        }
    }
    affected
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
            if !config.edge_types.contains(&edge.edge_type) {
                continue;
            }
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
                    (current - node_confidence * edge.weight * config.contradicts_factor)
                        .clamp(0.0, 1.0)
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

    fn make_edge(src: MemoryId, tgt: MemoryId, etype: EdgeType, weight: f32) -> MemoryEdge {
        MemoryEdge {
            source: src,
            target: tgt,
            edge_type: etype,
            weight,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        }
    }

    #[test]
    fn test_caused_propagation() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
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
        let a = MemoryId::new();
        let b = MemoryId::new();
        g.add_edge(&make_edge(a, b, EdgeType::Contradicts, 1.0));

        let result = propagate_update(&g, a, 0.8);
        let map: HashMap<MemoryId, f32> = result.into_iter().collect();

        // b = max(0, 1.0 - 0.8 * 1.0 * 0.7) = 0.44
        assert!((map[&b] - 0.44).abs() < 0.001);
    }

    #[test]
    fn test_supersedes_propagation() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
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
        let ids: Vec<MemoryId> = (0..10).map(|_| MemoryId::new()).collect();
        for i in 0..9 {
            g.add_edge(&make_edge(ids[i], ids[i + 1], EdgeType::Caused, 1.0));
        }

        let result = propagate_update(&g, ids[0], 1.0);
        let map: HashMap<MemoryId, f32> = result.into_iter().collect();

        // Should have at most 6 nodes (start + 5 depth levels)
        assert!(map.len() <= 6);
    }

    fn delta_map(
        g: &CsrGraph,
        root: MemoryId,
        delta: f32,
        cfg: &PropagationConfig,
    ) -> HashMap<MemoryId, f32> {
        propagate_delta(g, root, delta, cfg).into_iter().collect()
    }

    #[test]
    fn delta_damps_along_a_caused_chain_and_excludes_the_root() {
        let mut g = CsrGraph::new();
        let (a, b, c) = (MemoryId::new(), MemoryId::new(), MemoryId::new());
        g.add_edge(&make_edge(a, b, EdgeType::Caused, 1.0));
        g.add_edge(&make_edge(b, c, EdgeType::Caused, 1.0));
        let map = delta_map(&g, a, -0.4, &PropagationConfig::default());
        assert!(!map.contains_key(&a), "the root is the caller's to set");
        assert!((map[&b] + 0.36).abs() < 1e-5);
        assert!((map[&c] + 0.324).abs() < 1e-5);
    }

    #[test]
    fn delta_default_edge_set_skips_conflict_edges() {
        let mut g = CsrGraph::new();
        let (a, x, y, s) = (
            MemoryId::new(),
            MemoryId::new(),
            MemoryId::new(),
            MemoryId::new(),
        );
        g.add_edge(&make_edge(a, x, EdgeType::Contradicts, 1.0));
        g.add_edge(&make_edge(a, y, EdgeType::Supersedes, 1.0));
        g.add_edge(&make_edge(a, s, EdgeType::Supports, 1.0));
        let map = delta_map(&g, a, -0.4, &PropagationConfig::default());
        assert_eq!(map.len(), 1);
        assert!((map[&s] + 0.2).abs() < 1e-5);

        // Opting Contradicts in inverts the change.
        let map = delta_map(&g, a, -0.4, &PropagationConfig::all_edges());
        assert!((map[&x] - 0.28).abs() < 1e-5);
        assert!(
            !map.contains_key(&y),
            "Supersedes is never walked by deltas"
        );
    }

    #[test]
    fn delta_respects_min_delta_and_max_affected() {
        let mut g = CsrGraph::new();
        let root = MemoryId::new();
        for _ in 0..10 {
            g.add_edge(&make_edge(root, MemoryId::new(), EdgeType::Caused, 1.0));
        }
        let cfg = PropagationConfig {
            max_affected: 3,
            ..PropagationConfig::default()
        };
        assert_eq!(propagate_delta(&g, root, -0.5, &cfg).len(), 3);
        let cfg = PropagationConfig {
            min_delta: 0.5,
            ..PropagationConfig::default()
        };
        // 0.5 * 0.9 = 0.45 is below the floor on every branch.
        assert!(propagate_delta(&g, root, -0.5, &cfg).is_empty());
    }

    #[test]
    fn test_unrelated_edges_dont_propagate() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        g.add_edge(&make_edge(a, b, EdgeType::Related, 1.0));

        let result = propagate_update(&g, a, 0.5);
        // Only the changed node itself
        assert_eq!(result.len(), 1);
    }
}
