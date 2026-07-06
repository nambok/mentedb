//! Hot flushes checkpoint the WAL only; index and graph snapshots are
//! rewritten every flush_snapshot_interval flushes (and always on close).
//! Open reconciles stale snapshots against storage in both directions, so a
//! crash between snapshot writes never loses recall and never resurrects
//! forgotten memories into the index.

use mentedb::CognitiveConfig;
use mentedb::MenteDb;
use mentedb::prelude::*;

fn vec_for(axis: usize) -> Vec<f32> {
    let mut v = vec![0.1_f32; 8];
    v[axis] = 1.0;
    v
}

fn store(db: &MenteDb, content: &str, axis: usize) -> MemoryId {
    let node = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        content.to_string(),
        vec_for(axis),
    );
    let id = node.id;
    db.store(node).unwrap();
    id
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

fn recall_ids(db: &MenteDb, axis: usize) -> Vec<MemoryId> {
    db.recall_hybrid_at(&vec_for(axis), None, 10, now_us(), None, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

#[test]
fn recall_survives_crash_between_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let db = MenteDb::open_with_config(dir.path(), CognitiveConfig::default()).unwrap();
        // Establish snapshots once so they exist, then go stale.
        db.flush_full().unwrap();
        id = store(&db, "stored after the last snapshot", 0);
        // Hot flush: WAL checkpoint only, snapshots untouched.
        db.flush().unwrap();
        db.simulate_crash();
    }
    let db = MenteDb::open_with_config(dir.path(), CognitiveConfig::default()).unwrap();
    assert!(
        recall_ids(&db, 0).contains(&id),
        "open must reindex memories the stale snapshot missed"
    );
}

#[test]
fn forgotten_memory_stays_gone_despite_newer_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let db = MenteDb::open_with_config(dir.path(), CognitiveConfig::default()).unwrap();
        id = store(&db, "will be forgotten", 1);
        // Snapshot contains the memory.
        db.flush_full().unwrap();
        // Forget afterwards, hot flush only: the snapshot still has it.
        db.forget(id).unwrap();
        db.flush().unwrap();
        db.simulate_crash();
    }
    let db = MenteDb::open_with_config(dir.path(), CognitiveConfig::default()).unwrap();
    assert!(
        !recall_ids(&db, 1).contains(&id),
        "open must retire vectors whose pages are gone"
    );
    assert!(
        db.get_memory(id).is_err(),
        "the memory itself stays deleted"
    );
}

#[test]
fn snapshots_write_on_the_configured_interval() {
    let dir = tempfile::tempdir().unwrap();
    let config = CognitiveConfig {
        flush_snapshot_interval: 3,
        ..CognitiveConfig::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();
    store(&db, "first", 2);
    db.flush_full().unwrap();

    let hnsw = dir.path().join("indexes").join("hnsw.bin");
    let before = std::fs::metadata(&hnsw).unwrap().modified().unwrap();

    store(&db, "second", 3);
    db.flush().unwrap();
    db.flush().unwrap();
    let mid = std::fs::metadata(&hnsw).unwrap().modified().unwrap();
    assert_eq!(before, mid, "snapshots must not rewrite on hot flushes");

    db.flush().unwrap();
    let after = std::fs::metadata(&hnsw).unwrap().modified().unwrap();
    assert!(after > before, "the Nth flush must rewrite snapshots");
}
