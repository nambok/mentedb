//! Memories stored without a vector are embedded when an embedder is set.

use mentedb::MenteDb;
use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;
use mentedb_embedding::HashEmbeddingProvider;

fn node(content: &str) -> MemoryNode {
    MemoryNode::new(
        AgentId::new(),
        MemoryType::Semantic,
        content.to_string(),
        vec![],
    )
}

#[test]
fn store_without_a_vector_is_searchable() {
    let dir = tempfile::tempdir().unwrap();
    let db =
        MenteDb::open_with_embedder(dir.path(), Box::new(HashEmbeddingProvider::new(64))).unwrap();
    let n = node("the quick brown fox");
    let id = n.id;
    db.store(n).unwrap();
    assert_eq!(db.get_memory(id).unwrap().embedding.len(), 64);
    let q = db.embed_text("the quick brown fox").unwrap().unwrap();
    let hits = db.recall_similar(&q, 1).unwrap();
    assert_eq!(hits.first().map(|h| h.0), Some(id));
}

#[test]
fn store_batch_embeds_only_the_missing_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let db =
        MenteDb::open_with_embedder(dir.path(), Box::new(HashEmbeddingProvider::new(64))).unwrap();
    let mut given = node("already has one");
    given.embedding = vec![0.5; 64];
    let given_id = given.id;
    let missing = node("needs a vector");
    let missing_id = missing.id;
    db.store_batch(vec![given, missing]).unwrap();
    assert_eq!(
        db.get_memory(given_id).unwrap().embedding,
        vec![0.5; 64],
        "caller vector kept"
    );
    assert_eq!(db.get_memory(missing_id).unwrap().embedding.len(), 64);
}

#[test]
fn without_an_embedder_vectors_stay_empty() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let n = node("keyword only");
    let id = n.id;
    db.store(n).unwrap();
    assert!(db.get_memory(id).unwrap().embedding.is_empty());
}
