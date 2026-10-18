//! Store reports when it skips a duplicate, and memories that differ only in
//! their numbers are never treated as duplicates.

use mentedb::{MenteDb, StoreOutcome};
use mentedb_core::MemoryNode;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;
use mentedb_embedding::provider::EmbeddingProvider;

/// Bag of words embedder: word order does not change the vector, so "record
/// 3-4" and "record 4-3" embed identically, as they nearly do with a real
/// semantic model. This isolates the number check from the embedding gate.
struct BagOfWords;

impl EmbeddingProvider for BagOfWords {
    fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
        let mut v = vec![0.0f32; 64];
        for w in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
        {
            let h = w
                .to_lowercase()
                .bytes()
                .fold(7u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
            v[(h % 64) as usize] += 1.0;
        }
        Ok(v)
    }
    fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
    fn dimensions(&self) -> usize {
        64
    }
    fn model_name(&self) -> &str {
        "bag-of-words-test"
    }
}

fn db(dir: &std::path::Path) -> MenteDb {
    MenteDb::open_with_embedder(dir, Box::new(BagOfWords)).unwrap()
}

fn node(agent: AgentId, content: &str) -> MemoryNode {
    MemoryNode::new(agent, MemoryType::Semantic, content.to_string(), vec![])
}

#[test]
fn a_skipped_duplicate_names_the_memory_it_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());
    let agent = AgentId::new();
    let first = node(agent, "The deploy pipeline runs on GitHub Actions");
    let first_id = first.id;
    assert_eq!(db.store_with_outcome(first).unwrap(), StoreOutcome::Stored);
    let again = node(agent, "The deploy pipeline runs on GitHub Actions");
    let again_id = again.id;
    assert_eq!(
        db.store_with_outcome(again).unwrap(),
        StoreOutcome::Duplicate { of: first_id }
    );
    assert!(
        db.get_memory(again_id).is_err(),
        "the skipped id was never written"
    );
    assert!(db.get_memory(first_id).is_ok());
}

#[test]
fn same_words_with_different_numbers_are_different_memories() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());
    let agent = AgentId::new();
    for (a, b) in [
        ("crash test record 3-4", "crash test record 4-3"),
        (
            "Team meeting on May 3 at 4pm",
            "Team meeting on May 4 at 3pm",
        ),
        (
            "Retry failed uploads 3 times",
            "Retry failed uploads 5 times",
        ),
    ] {
        assert_eq!(
            db.store_with_outcome(node(agent, a)).unwrap(),
            StoreOutcome::Stored,
            "{a}"
        );
        assert_eq!(
            db.store_with_outcome(node(agent, b)).unwrap(),
            StoreOutcome::Stored,
            "{b} is not a duplicate of {a}"
        );
    }
}

#[test]
fn a_restatement_with_the_same_numbers_is_still_a_duplicate() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());
    let agent = AgentId::new();
    let first = node(agent, "Retry failed uploads 3 times");
    let first_id = first.id;
    db.store(first).unwrap();
    assert_eq!(
        db.store_with_outcome(node(agent, "Retry failed uploads 3 times"))
            .unwrap(),
        StoreOutcome::Duplicate { of: first_id }
    );
}
