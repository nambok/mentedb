//! Contradiction and cycle detection in the knowledge graph.

use std::collections::VecDeque;

use ahash::HashSet;
use mentedb_core::edge::EdgeType;
use mentedb_core::types::MemoryId;

use crate::csr::CsrGraph;

/// Find all nodes that contradict `id`, including transitive contradictions
/// through Supports->Contradicts chains.
pub fn find_contradictions(graph: &CsrGraph, id: MemoryId) -> Vec<MemoryId> {
    let mut contradictions = HashSet::default();
    let mut visited = HashSet::default();
    let mut queue = VecDeque::new();

    visited.insert(id);

    // Direct contradictions
    for (neighbor, edge) in graph.outgoing(id) {
        if edge.edge_type == EdgeType::Contradicts {
            contradictions.insert(neighbor);
            visited.insert(neighbor);
        }
    }
    // Also check incoming Contradicts edges
    for (neighbor, edge) in graph.incoming(id) {
        if edge.edge_type == EdgeType::Contradicts {
            contradictions.insert(neighbor);
            visited.insert(neighbor);
        }
    }

    // Transitive: nodes that Support `id` may have Contradicts edges
    // Follow Supports edges to `id` (incoming), then their Contradicts
    queue.push_back(id);
    while let Some(node) = queue.pop_front() {
        // Find nodes that support this node
        for (supporter, edge) in graph.incoming(node) {
            if edge.edge_type == EdgeType::Supports && visited.insert(supporter) {
                // Check if supporter contradicts anything
                for (target, e2) in graph.outgoing(supporter) {
                    if e2.edge_type == EdgeType::Contradicts && target != id {
                        contradictions.insert(target);
                    }
                }
            }
        }
        // Follow outgoing Supports to find Contradicts chains
        for (supported, edge) in graph.outgoing(node) {
            if edge.edge_type == EdgeType::Supports && visited.insert(supported) {
                for (target, e2) in graph.outgoing(supported) {
                    if e2.edge_type == EdgeType::Contradicts {
                        contradictions.insert(target);
                    }
                }
            }
        }
    }

    contradictions.into_iter().collect()
}

/// Find all nodes superseded by `id` (directly or transitively).
pub fn find_superseded(graph: &CsrGraph, id: MemoryId) -> Vec<MemoryId> {
    let mut result = Vec::new();
    let mut visited = HashSet::default();
    let mut queue = VecDeque::new();

    visited.insert(id);
    queue.push_back(id);

    while let Some(node) = queue.pop_front() {
        for (neighbor, edge) in graph.outgoing(node) {
            if edge.edge_type == EdgeType::Supersedes && visited.insert(neighbor) {
                result.push(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    result
}

/// Detect cycles in the graph considering only the specified edge types.
/// Returns a list of cycles, each represented as a vector of node IDs.
pub fn detect_cycles(graph: &CsrGraph, edge_types: &[EdgeType]) -> Vec<Vec<MemoryId>> {
    let filter: HashSet<EdgeType> = edge_types.iter().copied().collect();
    let mut cycles = Vec::new();
    let mut globally_visited = HashSet::default();

    for &start_id in graph.node_ids() {
        if globally_visited.contains(&start_id) {
            continue;
        }

        // DFS with path tracking for cycle detection
        let mut stack: Vec<(MemoryId, Vec<MemoryId>)> = vec![(start_id, vec![start_id])];
        let mut in_stack = HashSet::default();
        in_stack.insert(start_id);
        let mut local_visited = HashSet::default();
        local_visited.insert(start_id);

        while let Some((node, path)) = stack.pop() {
            // Rebuild in_stack from the current path
            in_stack.clear();
            for &p in &path {
                in_stack.insert(p);
            }

            for (neighbor, edge) in graph.outgoing(node) {
                if !filter.contains(&edge.edge_type) {
                    continue;
                }

                if in_stack.contains(&neighbor) {
                    // Found a cycle - extract it
                    if let Some(pos) = path.iter().position(|&n| n == neighbor) {
                        let cycle: Vec<MemoryId> = path[pos..].to_vec();
                        // Only add if we haven't found an equivalent cycle
                        if !cycles.iter().any(|c: &Vec<MemoryId>| {
                            c.len() == cycle.len() && cycle.iter().all(|n| c.contains(n))
                        }) {
                            cycles.push(cycle);
                        }
                    }
                } else if local_visited.insert(neighbor) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor);
                    stack.push((neighbor, new_path));
                }
            }

            globally_visited.insert(node);
        }
    }

    cycles
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
    fn test_direct_contradictions() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Contradicts, 1.0));
        g.add_edge(&make_edge(c, a, EdgeType::Contradicts, 1.0));

        let contras = find_contradictions(&g, a);
        assert!(contras.contains(&b));
        assert!(contras.contains(&c));
    }

    #[test]
    fn test_transitive_contradictions() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        // a is supported by b, and b contradicts c
        g.add_edge(&make_edge(b, a, EdgeType::Supports, 1.0));
        g.add_edge(&make_edge(b, c, EdgeType::Contradicts, 1.0));

        let contras = find_contradictions(&g, a);
        assert!(contras.contains(&c));
    }

    #[test]
    fn test_find_superseded() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Supersedes, 1.0));
        g.add_edge(&make_edge(b, c, EdgeType::Supersedes, 1.0));

        let superseded = find_superseded(&g, a);
        assert_eq!(superseded.len(), 2);
        assert!(superseded.contains(&b));
        assert!(superseded.contains(&c));
    }

    #[test]
    fn test_detect_cycle() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Caused, 1.0));
        g.add_edge(&make_edge(b, c, EdgeType::Caused, 1.0));
        g.add_edge(&make_edge(c, a, EdgeType::Caused, 1.0));

        let cycles = detect_cycles(&g, &[EdgeType::Caused]);
        assert!(!cycles.is_empty());
        assert_eq!(cycles[0].len(), 3);
    }

    #[test]
    fn test_no_cycle() {
        let mut g = CsrGraph::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        g.add_edge(&make_edge(a, b, EdgeType::Caused, 1.0));
        g.add_edge(&make_edge(b, c, EdgeType::Caused, 1.0));

        let cycles = detect_cycles(&g, &[EdgeType::Caused]);
        assert!(cycles.is_empty());
    }
}
