//! Graph traversal algorithms: BFS, DFS, shortest path, subgraph extraction.

use std::collections::VecDeque;

use ahash::{HashMap, HashSet};
use mentedb_core::edge::{EdgeType, MemoryEdge};
use mentedb_core::types::MemoryId;

use crate::csr::CsrGraph;

/// Breadth-first search returning (node, depth) pairs.
pub fn bfs(graph: &CsrGraph, start: MemoryId, max_depth: usize) -> Vec<(MemoryId, usize)> {
    let Some(_) = graph.get_idx(start) else {
        return Vec::new();
    };

    let mut visited = HashSet::default();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    visited.insert(start);
    queue.push_back((start, 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        result.push((node, depth));
        if depth >= max_depth {
            continue;
        }
        for (neighbor, _edge) in graph.outgoing(node) {
            if visited.insert(neighbor) {
                queue.push_back((neighbor, depth + 1));
            }
        }
    }

    result
}

/// Depth-first search returning (node, depth) pairs.
pub fn dfs(graph: &CsrGraph, start: MemoryId, max_depth: usize) -> Vec<(MemoryId, usize)> {
    let Some(_) = graph.get_idx(start) else {
        return Vec::new();
    };

    let mut visited = HashSet::default();
    let mut stack = vec![(start, 0usize)];
    let mut result = Vec::new();

    while let Some((node, depth)) = stack.pop() {
        if !visited.insert(node) {
            continue;
        }
        result.push((node, depth));
        if depth >= max_depth {
            continue;
        }
        for (neighbor, _edge) in graph.outgoing(node) {
            if !visited.contains(&neighbor) {
                stack.push((neighbor, depth + 1));
            }
        }
    }

    result
}

/// BFS that only follows edges matching the given edge types.
pub fn bfs_filtered(
    graph: &CsrGraph,
    start: MemoryId,
    max_depth: usize,
    edge_filter: &[EdgeType],
) -> Vec<(MemoryId, usize)> {
    let Some(_) = graph.get_idx(start) else {
        return Vec::new();
    };

    let filter_set: HashSet<EdgeType> = edge_filter.iter().copied().collect();
    let mut visited = HashSet::default();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    visited.insert(start);
    queue.push_back((start, 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        result.push((node, depth));
        if depth >= max_depth {
            continue;
        }
        for (neighbor, edge) in graph.outgoing(node) {
            if filter_set.contains(&edge.edge_type) && visited.insert(neighbor) {
                queue.push_back((neighbor, depth + 1));
            }
        }
    }

    result
}

/// Extract all nodes and edges within `radius` hops of `center`.
pub fn extract_subgraph(
    graph: &CsrGraph,
    center: MemoryId,
    radius: usize,
) -> (Vec<MemoryId>, Vec<MemoryEdge>) {
    let nodes_with_depth = bfs(graph, center, radius);
    let node_set: HashSet<MemoryId> = nodes_with_depth.iter().map(|&(id, _)| id).collect();

    let nodes: Vec<MemoryId> = nodes_with_depth.into_iter().map(|(id, _)| id).collect();
    let mut edges = Vec::new();

    for &node in &nodes {
        for (neighbor, stored) in graph.outgoing(node) {
            if node_set.contains(&neighbor) {
                edges.push(MemoryEdge {
                    source: node,
                    target: neighbor,
                    edge_type: stored.edge_type,
                    weight: stored.weight,
                    created_at: stored.created_at,
                    valid_from: stored.valid_from,
                    valid_until: stored.valid_until,
                });
            }
        }
    }

    (nodes, edges)
}

/// Find shortest path using BFS. Returns None if no path exists.
pub fn shortest_path(graph: &CsrGraph, from: MemoryId, to: MemoryId) -> Option<Vec<MemoryId>> {
    if from == to {
        return Some(vec![from]);
    }

    let _ = graph.get_idx(from)?;
    let _ = graph.get_idx(to)?;

    let mut visited = HashSet::default();
    let mut parent: HashMap<MemoryId, MemoryId> = HashMap::default();
    let mut queue = VecDeque::new();

    visited.insert(from);
    queue.push_back(from);

    while let Some(node) = queue.pop_front() {
        for (neighbor, _) in graph.outgoing(node) {
            if visited.insert(neighbor) {
                parent.insert(neighbor, node);
                if neighbor == to {
                    // Reconstruct path
                    let mut path = vec![to];
                    let mut cur = to;
                    while let Some(&prev) = parent.get(&cur) {
                        path.push(prev);
                        cur = prev;
                        if cur == from {
                            break;
                        }
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(neighbor);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edge(src: MemoryId, tgt: MemoryId, etype: EdgeType) -> MemoryEdge {
        MemoryEdge {
            source: src,
            target: tgt,
            edge_type: etype,
            weight: 1.0,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
        }
    }

    fn build_chain() -> (CsrGraph, Vec<MemoryId>) {
        // a -> b -> c -> d
        let mut g = CsrGraph::new();
        let ids: Vec<MemoryId> = (0..4).map(|_| MemoryId::new()).collect();
        g.add_edge(&make_edge(ids[0], ids[1], EdgeType::Caused));
        g.add_edge(&make_edge(ids[1], ids[2], EdgeType::Caused));
        g.add_edge(&make_edge(ids[2], ids[3], EdgeType::Related));
        (g, ids)
    }

    #[test]
    fn test_bfs_chain() {
        let (g, ids) = build_chain();
        let result = bfs(&g, ids[0], 10);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], (ids[0], 0));
        assert_eq!(result[1], (ids[1], 1));
    }

    #[test]
    fn test_bfs_max_depth() {
        let (g, ids) = build_chain();
        let result = bfs(&g, ids[0], 1);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_dfs_chain() {
        let (g, ids) = build_chain();
        let result = dfs(&g, ids[0], 10);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].0, ids[0]);
    }

    #[test]
    fn test_bfs_filtered() {
        let (g, ids) = build_chain();
        // Only follow Caused edges, so we stop before the Related edge
        let result = bfs_filtered(&g, ids[0], 10, &[EdgeType::Caused]);
        assert_eq!(result.len(), 3); // a, b, c but not d
    }

    #[test]
    fn test_shortest_path() {
        let (g, ids) = build_chain();
        let path = shortest_path(&g, ids[0], ids[3]);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], ids[0]);
        assert_eq!(path[3], ids[3]);
    }

    #[test]
    fn test_shortest_path_no_path() {
        let (g, ids) = build_chain();
        // No reverse path from d to a in a directed graph
        let path = shortest_path(&g, ids[3], ids[0]);
        assert!(path.is_none());
    }

    #[test]
    fn test_extract_subgraph() {
        let (g, ids) = build_chain();
        let (nodes, edges) = extract_subgraph(&g, ids[0], 2);
        assert_eq!(nodes.len(), 3); // a, b, c
        assert_eq!(edges.len(), 2); // a->b, b->c
    }
}
