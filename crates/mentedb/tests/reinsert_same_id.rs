//! A memory stored again under an id the vector index has seen before must be
//! found by its new vector: after a forget, after an in place edit, after a
//! re embed, and after a reopen.

use mentedb::MenteDb;
use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId};
use mentedb_embedding::HashEmbeddingProvider;

fn node(agent: AgentId, content: &str, embedding: Vec<f32>) -> MemoryNode {
    MemoryNode::new(agent, MemoryType::Semantic, content.to_string(), embedding)
}

fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0; 8];
    v[i] = 1.0;
    v
}

fn top(db: &MenteDb, q: &[f32]) -> Option<MemoryId> {
    db.recall_similar(q, 1).unwrap().first().map(|h| h.0)
}

/// A few unrelated memories so the index is a graph, not a single node.
fn store_neighbours(db: &MenteDb, agent: AgentId) {
    for (i, text) in ["alpha notes", "bravo notes", "charlie notes"]
        .iter()
        .enumerate()
    {
        db.store(node(agent, text, axis(i + 4))).unwrap();
    }
}

#[test]
fn store_after_forget_with_the_same_id_is_recalled() {
    let dir = tempfile::tempdir().unwrap();
    let agent = AgentId::new();
    let db = MenteDb::open(dir.path()).unwrap();
    store_neighbours(&db, agent);

    let original = node(agent, "the deploy runs on jenkins", axis(0));
    let id = original.id;
    db.store(original).unwrap();
    assert_eq!(top(&db, &axis(0)), Some(id));

    // The edit flow: forget, then store the edited memory under the same id.
    db.forget(id).unwrap();
    assert_ne!(top(&db, &axis(0)), Some(id));
    let mut edited = node(agent, "the deploy runs on github actions", axis(1));
    edited.id = id;
    db.store(edited).unwrap();

    assert_eq!(top(&db, &axis(1)), Some(id), "found by the edited vector");
    assert_eq!(
        db.get_memory(id).unwrap().content,
        "the deploy runs on github actions"
    );
    db.close().unwrap();

    // Still found after a reopen, with the edited content.
    let db = MenteDb::open(dir.path()).unwrap();
    assert_eq!(top(&db, &axis(1)), Some(id), "found after reopen");
    assert_eq!(
        db.get_memory(id).unwrap().content,
        "the deploy runs on github actions"
    );
    db.close().unwrap();
}

#[test]
fn restore_of_a_live_id_replaces_its_vector() {
    let dir = tempfile::tempdir().unwrap();
    let agent = AgentId::new();
    let db = MenteDb::open(dir.path()).unwrap();
    store_neighbours(&db, agent);

    let original = node(agent, "standup is at nine", axis(0));
    let id = original.id;
    db.store(original).unwrap();

    let mut edited = node(agent, "standup moved to ten thirty", axis(2));
    edited.id = id;
    db.store(edited).unwrap();

    assert_eq!(top(&db, &axis(2)), Some(id), "found by the new vector");
    assert_ne!(top(&db, &axis(0)), Some(id), "old vector no longer indexed");
    let hits = db.recall_similar(&axis(2), 10).unwrap();
    assert_eq!(hits.iter().filter(|h| h.0 == id).count(), 1, "listed once");
    db.close().unwrap();

    let db = MenteDb::open(dir.path()).unwrap();
    assert_eq!(top(&db, &axis(2)), Some(id), "found after reopen");
    db.close().unwrap();
}

#[test]
fn reembed_of_an_indexed_memory_keeps_it_searchable() {
    // The memory already has an indexed vector at an old dimension. Re
    // embedding tombstones that vector and indexes the new one under the
    // same id, which used to fail as a duplicate id and leave the memory out
    // of vector search.
    let dir = tempfile::tempdir().unwrap();
    let agent = AgentId::new();
    let m = node(agent, "the deploy runs on github actions", vec![0.5; 4]);
    let id = m.id;
    {
        let db = MenteDb::open(dir.path()).unwrap();
        db.store(m).unwrap();
        assert_eq!(top(&db, &[0.5; 4]), Some(id));
        db.close().unwrap();
    }

    let db =
        MenteDb::open_with_embedder(dir.path(), Box::new(HashEmbeddingProvider::new(16))).unwrap();
    assert_eq!(db.reembed_all().unwrap(), 1);
    assert_eq!(db.get_memory(id).unwrap().embedding.len(), 16);

    let q = db
        .embed_text("the deploy runs on github actions")
        .unwrap()
        .unwrap();
    assert_eq!(top(&db, &q), Some(id), "re embedded memory is recalled");
    db.close().unwrap();

    let db =
        MenteDb::open_with_embedder(dir.path(), Box::new(HashEmbeddingProvider::new(16))).unwrap();
    assert_eq!(top(&db, &q), Some(id), "still recalled after reopen");
    db.close().unwrap();
}

#[test]
fn rebuild_indexes_keeps_every_memory_once() {
    let dir = tempfile::tempdir().unwrap();
    let agent = AgentId::new();
    let db = MenteDb::open(dir.path()).unwrap();
    store_neighbours(&db, agent);
    let m = node(agent, "on call rotation is weekly", axis(0));
    let id = m.id;
    db.store(m).unwrap();

    db.rebuild_indexes().unwrap();
    db.rebuild_indexes().unwrap();
    let hits = db.recall_similar(&axis(0), 10).unwrap();
    assert_eq!(hits.len(), 4, "no memory listed twice");
    assert_eq!(hits[0].0, id);
    db.close().unwrap();
}
