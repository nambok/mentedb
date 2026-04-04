//! Integration tests for mentedb-index: exercises all indexes together via IndexManager.

use mentedb_core::memory::MemoryType;
use mentedb_core::MemoryNode;
use mentedb_index::{IndexManager, HnswIndex, DistanceMetric, BitmapIndex, TemporalIndex, SalienceIndex};
use mentedb_index::manager::IndexManagerConfig;
use mentedb_index::hnsw::HnswConfig;

fn make_node(embedding: Vec<f32>, tags: Vec<&str>, salience: f32, created_at: u64) -> MemoryNode {
    let mut node = MemoryNode::new(
        uuid::Uuid::new_v4(),
        MemoryType::Episodic,
        "integration test memory".into(),
        embedding,
    );
    node.tags = tags.into_iter().map(String::from).collect();
    node.salience = salience;
    node.created_at = created_at;
    node
}

#[test]
fn test_hybrid_search_full_pipeline() {
    let mgr = IndexManager::new(IndexManagerConfig {
        hnsw: HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 30,
            metric: DistanceMetric::Cosine,
        },
    });

    // Insert a variety of memories
    let a = make_node(vec![1.0, 0.0, 0.0, 0.0], vec!["work", "important"], 0.9, 1000);
    let b = make_node(vec![0.0, 1.0, 0.0, 0.0], vec!["personal"], 0.5, 2000);
    let c = make_node(vec![0.9, 0.1, 0.0, 0.0], vec!["work"], 0.7, 3000);
    let d = make_node(vec![0.0, 0.0, 1.0, 0.0], vec!["work", "important"], 0.3, 500);

    let a_id = a.id;
    let c_id = c.id;

    mgr.index_memory(&a);
    mgr.index_memory(&b);
    mgr.index_memory(&c);
    mgr.index_memory(&d);

    // Search with vector close to `a` and `c`, filtered by "work" tag
    let results = mgr.hybrid_search(&[1.0, 0.0, 0.0, 0.0], Some(&["work"]), None, 2);
    assert_eq!(results.len(), 2);
    // a and c should be the top results (closest vectors with "work" tag)
    let ids: Vec<_> = results.iter().map(|r| r.0).collect();
    assert!(ids.contains(&a_id));
    assert!(ids.contains(&c_id));

    // Search with time filter
    let results = mgr.hybrid_search(
        &[1.0, 0.0, 0.0, 0.0],
        Some(&["work"]),
        Some((2000, 4000)),
        10,
    );
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, c_id);
}

#[test]
fn test_hybrid_search_no_filters() {
    let mgr = IndexManager::default();
    let a = make_node(vec![1.0, 0.0, 0.0], vec![], 0.8, 100);
    let b = make_node(vec![0.0, 1.0, 0.0], vec![], 0.2, 200);

    let a_id = a.id;
    mgr.index_memory(&a);
    mgr.index_memory(&b);

    let results = mgr.hybrid_search(&[1.0, 0.0, 0.0], None, None, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, a_id);
}

#[test]
fn test_index_remove_requery() {
    let mgr = IndexManager::default();
    let node = make_node(vec![1.0, 0.0, 0.0, 0.0], vec!["ephemeral"], 0.5, 100);
    let id = node.id;

    mgr.index_memory(&node);

    // Verify present
    let results = mgr.hybrid_search(&[1.0, 0.0, 0.0, 0.0], Some(&["ephemeral"]), None, 10);
    assert_eq!(results.len(), 1);

    // Remove and verify absent
    mgr.remove_memory(id, &node);
    let results = mgr.hybrid_search(&[1.0, 0.0, 0.0, 0.0], Some(&["ephemeral"]), None, 10);
    assert!(results.is_empty());
}

#[test]
fn test_salience_ranking_influence() {
    let mgr = IndexManager::new(IndexManagerConfig {
        hnsw: HnswConfig {
            metric: DistanceMetric::Euclidean,
            ..HnswConfig::default()
        },
    });

    // Two nodes with identical vectors but different salience
    let high = make_node(vec![1.0, 1.0, 1.0, 1.0], vec![], 1.0, 1000);
    let low = make_node(vec![1.0, 1.0, 1.0, 1.0], vec![], 0.0, 1000);

    let high_id = high.id;
    mgr.index_memory(&high);
    mgr.index_memory(&low);

    let results = mgr.hybrid_search(&[1.0, 1.0, 1.0, 1.0], None, None, 2);
    assert_eq!(results.len(), 2);
    // High salience should rank first
    assert_eq!(results[0].0, high_id);
}

#[test]
fn test_many_memories_hybrid() {
    let mgr = IndexManager::new(IndexManagerConfig {
        hnsw: HnswConfig {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
            metric: DistanceMetric::Euclidean,
        },
    });

    // Insert 50 memories
    for i in 0..50 {
        let val = i as f32 / 50.0;
        let node = make_node(
            vec![val, 1.0 - val, 0.0, 0.0],
            vec!["batch"],
            val,
            (i * 100) as u64,
        );
        mgr.index_memory(&node);
    }

    let results = mgr.hybrid_search(&[1.0, 0.0, 0.0, 0.0], Some(&["batch"]), None, 5);
    assert_eq!(results.len(), 5);
    // All results should have positive combined scores
    for (_, score) in &results {
        assert!(*score > 0.0);
    }
}
