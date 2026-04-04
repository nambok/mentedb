//! Integration tests for mentedb-index: exercises all indexes together via IndexManager.

use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_index::hnsw::HnswConfig;
use mentedb_index::manager::IndexManagerConfig;
use mentedb_index::{
    BitmapIndex, DistanceMetric, HnswIndex, IndexManager, SalienceIndex, TemporalIndex,
};
use uuid::Uuid;

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
    let a = make_node(
        vec![1.0, 0.0, 0.0, 0.0],
        vec!["work", "important"],
        0.9,
        1000,
    );
    let b = make_node(vec![0.0, 1.0, 0.0, 0.0], vec!["personal"], 0.5, 2000);
    let c = make_node(vec![0.9, 0.1, 0.0, 0.0], vec!["work"], 0.7, 3000);
    let d = make_node(
        vec![0.0, 0.0, 1.0, 0.0],
        vec!["work", "important"],
        0.3,
        500,
    );

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

// ---------------------------------------------------------------------------
// Persistence tests
// ---------------------------------------------------------------------------

#[test]
fn test_hnsw_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hnsw.json");

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let vec1 = vec![1.0, 0.0, 0.0, 0.0];
    let vec2 = vec![0.0, 1.0, 0.0, 0.0];

    {
        let idx = HnswIndex::new(HnswConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        });
        idx.insert(id1, &vec1).unwrap();
        idx.insert(id2, &vec2).unwrap();
        idx.save(&path).unwrap();
    }

    let idx2 = HnswIndex::load(&path, 50).unwrap();
    assert_eq!(idx2.len(), 2);
    let results = idx2.search(&vec1, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id1);
}

#[test]
fn test_bitmap_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bitmap.json");

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();

    {
        let idx = BitmapIndex::new();
        idx.add_tag(id1, "alpha");
        idx.add_tag(id1, "beta");
        idx.add_tag(id2, "alpha");
        idx.save(&path).unwrap();
    }

    let idx2 = BitmapIndex::load(&path).unwrap();
    let alpha = idx2.query_tag("alpha");
    assert_eq!(alpha.len(), 2);

    let both = idx2.query_tags_and(&["alpha", "beta"]);
    assert_eq!(both.len(), 1);
    assert_eq!(both[0], id1);
}

#[test]
fn test_temporal_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("temporal.json");

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    {
        let idx = TemporalIndex::new();
        idx.insert(id1, 100);
        idx.insert(id2, 200);
        idx.insert(id3, 300);
        idx.save(&path).unwrap();
    }

    let idx2 = TemporalIndex::load(&path).unwrap();
    let range = idx2.range(100, 200);
    assert_eq!(range.len(), 2);
    assert!(range.contains(&id1));
    assert!(range.contains(&id2));

    assert_eq!(idx2.get_timestamp(id3), Some(300));
}

#[test]
fn test_salience_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("salience.json");

    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let id3 = Uuid::new_v4();

    {
        let idx = SalienceIndex::new();
        idx.insert(id1, 0.3);
        idx.insert(id2, 0.9);
        idx.insert(id3, 0.6);
        idx.save(&path).unwrap();
    }

    let idx2 = SalienceIndex::load(&path).unwrap();
    let top = idx2.top_k(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].0, id2);
    assert_eq!(top[1].0, id3);
}

#[test]
fn test_index_manager_save_load() {
    let dir = tempfile::tempdir().unwrap();
    let index_dir = dir.path().join("indexes");

    let node = {
        let mut n = MemoryNode::new(
            Uuid::new_v4(),
            MemoryType::Episodic,
            "test memory".into(),
            vec![1.0, 0.0, 0.0, 0.0],
        );
        n.tags = vec!["important".into()];
        n.salience = 0.8;
        n.created_at = 1000;
        n
    };

    {
        let mgr = IndexManager::default();
        mgr.index_memory(&node);
        mgr.save(&index_dir).unwrap();
    }

    let mgr2 = IndexManager::load(&index_dir).unwrap();

    // Vector search should find the node
    let results = mgr2.hybrid_search(&[1.0, 0.0, 0.0, 0.0], None, None, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, node.id);

    // Tag filter should work
    let results = mgr2.hybrid_search(&[1.0, 0.0, 0.0, 0.0], Some(&["important"]), None, 1);
    assert_eq!(results.len(), 1);
}
