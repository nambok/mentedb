//! End-to-end integration tests for mentedb-graph.

use mentedb_core::edge::{EdgeType, MemoryEdge};
use mentedb_core::types::MemoryId;
use mentedb_graph::*;

fn edge(src: MemoryId, tgt: MemoryId, etype: EdgeType, weight: f32) -> MemoryEdge {
    MemoryEdge {
        source: src,
        target: tgt,
        edge_type: etype,
        weight,
        created_at: 1000,
        valid_from: None,
        valid_until: None,
    }
}

fn ids(n: usize) -> Vec<MemoryId> {
    (0..n).map(|_| MemoryId::new()).collect()
}

#[test]
fn test_graph_manager_full_workflow() {
    let mut mgr = GraphManager::new();
    let n = ids(5);

    // Add nodes
    for &id in &n {
        mgr.add_memory(id);
    }

    // Build: n[0] -> n[1] -> n[2], n[0] -> n[3], n[3] -> n[4]
    mgr.add_relationship(&edge(n[0], n[1], EdgeType::Caused, 0.9))
        .unwrap();
    mgr.add_relationship(&edge(n[1], n[2], EdgeType::Caused, 0.8))
        .unwrap();
    mgr.add_relationship(&edge(n[0], n[3], EdgeType::Related, 0.7))
        .unwrap();
    mgr.add_relationship(&edge(n[3], n[4], EdgeType::Supports, 0.6))
        .unwrap();

    // Context subgraph from n[0] with depth 2
    let (nodes, edges) = mgr.get_context_subgraph(n[0], 2);
    assert!(nodes.len() >= 4); // n[0], n[1], n[2], n[3] at least
    assert!(edges.len() >= 3);

    // Compact and verify
    mgr.compact();
    let (nodes2, edges2) = mgr.get_context_subgraph(n[0], 2);
    assert_eq!(nodes.len(), nodes2.len());
    assert_eq!(edges.len(), edges2.len());
}

#[test]
fn test_traversal_algorithms() {
    let mut g = CsrGraph::new();
    let n = ids(5);

    // Diamond: n[0] -> n[1], n[0] -> n[2], n[1] -> n[3], n[2] -> n[3], n[3] -> n[4]
    g.add_edge(&edge(n[0], n[1], EdgeType::Caused, 1.0));
    g.add_edge(&edge(n[0], n[2], EdgeType::Caused, 1.0));
    g.add_edge(&edge(n[1], n[3], EdgeType::Caused, 1.0));
    g.add_edge(&edge(n[2], n[3], EdgeType::Related, 1.0));
    g.add_edge(&edge(n[3], n[4], EdgeType::Caused, 1.0));

    // BFS from n[0]
    let bfs_result = bfs(&g, n[0], 10);
    assert_eq!(bfs_result.len(), 5);

    // DFS from n[0]
    let dfs_result = dfs(&g, n[0], 10);
    assert_eq!(dfs_result.len(), 5);

    // Filtered BFS: only Caused edges, should skip n[2]->n[3] (Related)
    let filtered = bfs_filtered(&g, n[0], 10, &[EdgeType::Caused]);
    // n[0], n[1], n[2] (via Caused), n[3] (via n[1]->n[3] Caused), n[4]
    assert!(filtered.len() >= 4);

    // Shortest path n[0] -> n[4]
    let path = shortest_path(&g, n[0], n[4]).unwrap();
    assert!(path.len() >= 3);
    assert_eq!(*path.first().unwrap(), n[0]);
    assert_eq!(*path.last().unwrap(), n[4]);
}

#[test]
fn test_belief_propagation_chain() {
    let mut g = CsrGraph::new();
    let n = ids(4);

    // Chain: n[0] -Caused-> n[1] -Caused-> n[2] -Contradicts-> n[3]
    g.add_edge(&edge(n[0], n[1], EdgeType::Caused, 1.0));
    g.add_edge(&edge(n[1], n[2], EdgeType::Caused, 1.0));
    g.add_edge(&edge(n[2], n[3], EdgeType::Contradicts, 1.0));

    let results = propagate_update(&g, n[0], 0.8);
    let map: ahash::HashMap<MemoryId, f32> = results.into_iter().collect();

    assert!(map.contains_key(&n[0]));
    assert!(map.contains_key(&n[1]));
    assert!(map.contains_key(&n[2]));
    assert!(map.contains_key(&n[3]));

    // n[1] = 0.8 * 1.0 * 0.9 = 0.72
    assert!((map[&n[1]] - 0.72).abs() < 0.01);
}

#[test]
fn test_contradiction_and_superseded() {
    let mut g = CsrGraph::new();
    let n = ids(4);

    g.add_edge(&edge(n[0], n[1], EdgeType::Contradicts, 1.0));
    g.add_edge(&edge(n[0], n[2], EdgeType::Supersedes, 1.0));
    g.add_edge(&edge(n[2], n[3], EdgeType::Supersedes, 1.0));

    let contras = find_contradictions(&g, n[0]);
    assert!(contras.contains(&n[1]));

    let superseded = find_superseded(&g, n[0]);
    assert_eq!(superseded.len(), 2);
    assert!(superseded.contains(&n[2]));
    assert!(superseded.contains(&n[3]));
}

#[test]
fn test_cycle_detection() {
    let mut g = CsrGraph::new();
    let n = ids(3);

    g.add_edge(&edge(n[0], n[1], EdgeType::Caused, 1.0));
    g.add_edge(&edge(n[1], n[2], EdgeType::Caused, 1.0));
    g.add_edge(&edge(n[2], n[0], EdgeType::Caused, 1.0));

    let cycles = detect_cycles(&g, &[EdgeType::Caused]);
    assert!(!cycles.is_empty());

    // No cycles if we filter for a different edge type
    let no_cycles = detect_cycles(&g, &[EdgeType::Related]);
    assert!(no_cycles.is_empty());
}

#[test]
fn test_remove_and_compact() {
    let mut mgr = GraphManager::new();
    let n = ids(3);
    for &id in &n {
        mgr.add_memory(id);
    }
    mgr.add_relationship(&edge(n[0], n[1], EdgeType::Caused, 1.0))
        .unwrap();
    mgr.add_relationship(&edge(n[1], n[2], EdgeType::Related, 1.0))
        .unwrap();
    mgr.compact();

    // Remove middle node
    mgr.remove_memory(n[1]);
    mgr.compact();

    let out = mgr.graph().outgoing(n[0]);
    assert!(out.is_empty());
}

// ---------------------------------------------------------------------------
// Persistence tests
// ---------------------------------------------------------------------------

#[test]
fn test_graph_manager_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let graph_dir = dir.path().join("graph");

    let a = MemoryId::new();
    let b = MemoryId::new();
    let c = MemoryId::new();

    {
        let mut mgr = GraphManager::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_memory(c);
        mgr.add_relationship(&edge(a, b, EdgeType::Caused, 0.8))
            .unwrap();
        mgr.add_relationship(&edge(b, c, EdgeType::Related, 0.5))
            .unwrap();
        mgr.save(&graph_dir).unwrap();
    }

    let mgr2 = GraphManager::load(&graph_dir).unwrap();

    assert!(mgr2.graph().contains_node(a));
    assert!(mgr2.graph().contains_node(b));
    assert!(mgr2.graph().contains_node(c));
    assert_eq!(mgr2.graph().node_count(), 3);

    let out_a = mgr2.graph().outgoing(a);
    assert_eq!(out_a.len(), 1);
    assert_eq!(out_a[0].0, b);
    assert_eq!(out_a[0].1.edge_type, EdgeType::Caused);

    let (nodes, edges) = mgr2.get_context_subgraph(a, 2);
    assert_eq!(nodes.len(), 3);
    assert_eq!(edges.len(), 2);
}

#[test]
fn test_graph_save_load_after_compact() {
    let dir = tempfile::tempdir().unwrap();
    let graph_dir = dir.path().join("graph");

    let a = MemoryId::new();
    let b = MemoryId::new();

    {
        let mut mgr = GraphManager::new();
        mgr.add_memory(a);
        mgr.add_memory(b);
        mgr.add_relationship(&edge(a, b, EdgeType::Supports, 0.9))
            .unwrap();
        mgr.compact();
        mgr.save(&graph_dir).unwrap();
    }

    let mgr2 = GraphManager::load(&graph_dir).unwrap();
    let out = mgr2.graph().outgoing(a);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, b);
    assert_eq!(out[0].1.edge_type, EdgeType::Supports);
}
