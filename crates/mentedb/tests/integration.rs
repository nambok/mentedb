use mentedb::prelude::*;
use mentedb::MenteDb;
use uuid::Uuid;

fn make_memory(content: &str, embedding: Vec<f32>) -> MemoryNode {
    MemoryNode::new(
        Uuid::new_v4(),
        MemoryType::Episodic,
        content.to_string(),
        embedding,
    )
}

#[test]
fn test_store_and_recall_similar() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = MenteDb::open(dir.path()).unwrap();

    let memories = vec![
        make_memory("The user prefers dark mode", vec![1.0, 0.0, 0.0, 0.0]),
        make_memory("Meeting scheduled for Monday", vec![0.0, 1.0, 0.0, 0.0]),
        make_memory("API key rotated last week", vec![0.0, 0.0, 1.0, 0.0]),
        make_memory("User likes Python over Rust", vec![0.8, 0.1, 0.0, 0.0]),
        make_memory("Deploy pipeline uses GitHub Actions", vec![0.0, 0.0, 0.9, 0.1]),
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
    let mut db = MenteDb::open(dir.path()).unwrap();

    let node = make_memory("Temporary thought", vec![0.5, 0.5, 0.0, 0.0]);
    let id = node.id;

    db.store(node).unwrap();
    db.forget(id).unwrap();

    // After forgetting, similarity search should not return the forgotten memory.
    let results = db.recall_similar(&[0.5, 0.5, 0.0, 0.0], 5).unwrap();
    for (result_id, _) in &results {
        assert_ne!(*result_id, id, "Forgotten memory should not appear in results");
    }

    db.close().unwrap();
}

#[test]
fn test_relate_memories() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = MenteDb::open(dir.path()).unwrap();

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
    };
    db.relate(edge).unwrap();

    db.close().unwrap();
}

#[test]
fn test_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();

    // First session: store memories.
    {
        let mut db = MenteDb::open(dir.path()).unwrap();
        db.store(make_memory("Persistent memory", vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        db.close().unwrap();
    }

    // Second session: reopen and verify storage engine opens cleanly.
    {
        let mut db = MenteDb::open(dir.path()).unwrap();
        // The database should open without errors after a previous clean close.
        db.close().unwrap();
    }
}
