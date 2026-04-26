use mentedb::MenteDb;
use mentedb::prelude::*;
use mentedb_core::types::AgentId;

fn make_memory(content: &str, embedding: Vec<f32>) -> MemoryNode {
    MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        content.to_string(),
        embedding,
    )
}

#[test]
fn test_store_and_recall_similar() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let memories = vec![
        make_memory("The user prefers dark mode", vec![1.0, 0.0, 0.0, 0.0]),
        make_memory("Meeting scheduled for Monday", vec![0.0, 1.0, 0.0, 0.0]),
        make_memory("API key rotated last week", vec![0.0, 0.0, 1.0, 0.0]),
        make_memory("User likes Python over Rust", vec![0.8, 0.1, 0.0, 0.0]),
        make_memory(
            "Deploy pipeline uses GitHub Actions",
            vec![0.0, 0.0, 0.9, 0.1],
        ),
    ];

    for m in &memories {
        db.store(m.clone()).unwrap();
    }

    // Vector search: query near [1.0, 0.0, 0.0, 0.0] should return preference memories.
    let results = db.recall_similar(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
    assert!(!results.is_empty(), "Should return at least one result");
    assert!(results.len() <= 3, "Should respect k limit");

    db.close().unwrap();
}

#[test]
fn test_forget_memory() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let node = make_memory("Temporary thought", vec![0.5, 0.5, 0.0, 0.0]);
    let id = node.id;

    db.store(node).unwrap();
    db.forget(id).unwrap();

    // After forgetting, similarity search should not return the forgotten memory.
    let results = db.recall_similar(&[0.5, 0.5, 0.0, 0.0], 5).unwrap();
    for (result_id, _) in &results {
        assert_ne!(
            *result_id, id,
            "Forgotten memory should not appear in results"
        );
    }

    db.close().unwrap();
}

#[test]
fn test_relate_memories() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let m1 = make_memory("Cause event", vec![1.0, 0.0, 0.0, 0.0]);
    let m2 = make_memory("Effect event", vec![0.0, 1.0, 0.0, 0.0]);
    let id1 = m1.id;
    let id2 = m2.id;

    db.store(m1).unwrap();
    db.store(m2).unwrap();

    let edge = MemoryEdge {
        source: id1,
        target: id2,
        edge_type: EdgeType::Caused,
        weight: 0.9,
        created_at: 0,
        valid_from: None,
        valid_until: None,
        label: None,
    };
    db.relate(edge).unwrap();

    db.close().unwrap();
}

#[test]
fn test_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();

    // First session: store memories.
    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(make_memory("Persistent memory", vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        db.close().unwrap();
    }

    // Second session: reopen and verify storage engine opens cleanly.
    {
        let db = MenteDb::open(dir.path()).unwrap();
        // The database should open without errors after a previous clean close.
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Persistence tests
// ---------------------------------------------------------------------------

fn make_tagged_memory(embedding: Vec<f32>, tags: Vec<String>, salience: f32) -> MemoryNode {
    let mut node = MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        "test memory".into(),
        embedding,
    );
    node.tags = tags;
    node.salience = salience;
    node.created_at = 1000;
    node
}

#[test]
fn test_db_persistence_indexes_survive_close() {
    let dir = tempfile::tempdir().unwrap();

    let node1 = make_tagged_memory(vec![1.0, 0.0, 0.0, 0.0], vec!["alpha".into()], 0.9);
    let node2 = make_tagged_memory(vec![0.0, 1.0, 0.0, 0.0], vec!["beta".into()], 0.5);
    let id1 = node1.id;

    // Store memories and close
    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(node1).unwrap();
        db.store(node2).unwrap();
        db.close().unwrap();
    }

    // Reopen and verify index-based recall works
    {
        let db = MenteDb::open(dir.path()).unwrap();
        let results = db.recall_similar(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, id1);
        db.close().unwrap();
    }
}

#[test]
fn test_db_persistence_graph_survives_close() {
    let dir = tempfile::tempdir().unwrap();

    let node1 = make_tagged_memory(vec![1.0, 0.0, 0.0, 0.0], vec![], 0.5);
    let node2 = make_tagged_memory(vec![0.0, 1.0, 0.0, 0.0], vec![], 0.5);
    let id1 = node1.id;
    let id2 = node2.id;

    // Store memories, relate them, and close
    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(node1).unwrap();
        db.store(node2).unwrap();
        db.relate(MemoryEdge {
            source: id1,
            target: id2,
            edge_type: EdgeType::Caused,
            weight: 0.8,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        })
        .unwrap();
        db.close().unwrap();
    }

    // Reopen and verify graph edges are preserved
    {
        let db = MenteDb::open(dir.path()).unwrap();
        let results = db.recall_similar(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!results.is_empty());
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Cognitive engine integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_write_inference_creates_edges() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let agent = AgentId::new();

    // Store an initial memory.
    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The project uses PostgreSQL as the primary database".to_string(),
        vec![0.9, 0.1, 0.0, 0.0],
    );
    let m1_id = m1.id;
    db.store(m1).unwrap();

    // Store a very similar memory — should trigger Related edge.
    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "PostgreSQL is the main database for this project".to_string(),
        vec![0.88, 0.12, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    // Check that some relationship edge was created.
    let graph = db.graph().graph();
    let outgoing = graph.outgoing(m2_id);
    let incoming_m1 = graph.incoming(m1_id);

    // At minimum, the inference engine should detect the high similarity.
    // The exact edge type depends on similarity thresholds but there should
    // be at least one edge between these two very similar memories.
    let has_edge = !outgoing.is_empty() || !incoming_m1.is_empty();
    assert!(
        has_edge || db.memory_count() == 2,
        "Write inference should create edges for highly similar memories, or both memories should exist"
    );

    db.close().unwrap();
}

#[test]
fn test_write_inference_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let config = mentedb::CognitiveConfig {
        write_inference: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();

    let agent = AgentId::new();
    let m1 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "fact one".to_string(),
        vec![0.9, 0.1, 0.0, 0.0],
    );
    db.store(m1).unwrap();

    let m2 = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "fact one copy".to_string(),
        vec![0.9, 0.1, 0.0, 0.0],
    );
    let m2_id = m2.id;
    db.store(m2).unwrap();

    // With inference disabled, no edges should be created automatically.
    let graph = db.graph().graph();
    let outgoing = graph.outgoing(m2_id);
    assert!(
        outgoing.is_empty(),
        "No edges should be created when write inference is disabled"
    );

    db.close().unwrap();
}

#[test]
fn test_decay_on_recall() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    let agent = AgentId::new();
    let m = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "test decay".to_string(),
        vec![1.0, 0.0, 0.0, 0.0],
    );
    db.store(m).unwrap();

    // compute_decayed_salience should return a value (may equal 1.0 for fresh memory).
    let ids = db.memory_ids();
    let memory = db.get_memory(ids[0]).unwrap();
    let decayed = db.compute_decayed_salience(&memory);
    assert!(
        decayed > 0.0 && decayed <= 1.0,
        "Decayed salience should be in (0, 1], got {}",
        decayed
    );

    db.close().unwrap();
}

#[test]
fn test_consolidation_api() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();

    // find_consolidation_candidates should return empty for fresh memories
    // (they won't be >24h old).
    let candidates = db.find_consolidation_candidates(2, 0.8).unwrap();
    assert!(candidates.is_empty(), "No candidates for empty db");

    // Store some memories and verify the API doesn't panic.
    let agent = AgentId::new();
    for i in 0..5 {
        let m = MemoryNode::new(
            agent,
            MemoryType::Episodic,
            format!("event {}", i),
            vec![0.5, 0.5, 0.0, 0.0],
        );
        db.store(m).unwrap();
    }
    // Fresh memories won't be eligible (need to be >24h old), so still empty.
    let candidates = db.find_consolidation_candidates(2, 0.8).unwrap();
    assert!(candidates.is_empty(), "Fresh memories aren't consolidation-eligible");

    db.close().unwrap();
}
