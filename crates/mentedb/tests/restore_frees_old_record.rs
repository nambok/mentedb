//! Storing a memory again under its own id must replace its record, never leave
//! the old one behind: a stale copy outlives a later forget or erase, comes back
//! as the memory on the next open, and keeps the erased bytes on disk.

use std::path::Path;

use mentedb::prelude::*;
use mentedb::{CognitiveConfig, EraseOptions};

fn config() -> CognitiveConfig {
    CognitiveConfig {
        write_inference: false,
        memory_dedup: false,
        ..CognitiveConfig::default()
    }
}

fn open(dir: &Path) -> MenteDb {
    MenteDb::open_with_config(dir, config()).expect("open")
}

fn stored_and_retagged(db: &MenteDb, content: &str) -> MemoryId {
    let node = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        content.to_string(),
        (0..32).map(|i| i as f32 / 32.0 - 0.5).collect(),
    );
    let id = node.id;
    db.store(node).unwrap();
    // What a background tagger does: load, add a tag, store again.
    let mut again = db.get_memory(id).unwrap();
    again.tags.push("category:test".into());
    db.store(again).unwrap();
    id
}

fn records_holding(dir: &Path, needle: &str) -> bool {
    let bytes = std::fs::read(dir.join("records.db")).unwrap();
    bytes.windows(needle.len()).any(|w| w == needle.as_bytes())
}

#[test]
fn a_forgotten_memory_that_was_stored_twice_stays_forgotten_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = stored_and_retagged(&db, "twice-stored-forget-marker-a1 is gone");
    db.forget(id).unwrap();
    db.close().unwrap();
    let db = open(dir.path());
    assert!(db.get_memory(id).is_err(), "the forgotten memory came back");
}

#[test]
fn an_erased_memory_that_was_stored_twice_leaves_no_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = stored_and_retagged(&db, "twice-stored-erase-marker-b2 is gone");
    let receipt = db.erase(id, EraseOptions::default()).unwrap();
    let v = receipt.verification.unwrap();
    assert!(!v.residual_found, "residual: {:?}", v.hits);
    db.close().unwrap();
    assert!(!records_holding(dir.path(), "twice-stored-erase-marker-b2"));
    let db = open(dir.path());
    assert!(db.get_memory(id).is_err(), "the erased memory came back");
}

#[test]
fn storing_again_keeps_the_latest_version_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = stored_and_retagged(&db, "twice-stored-latest-marker-c3");
    db.close().unwrap();
    let db = open(dir.path());
    let node = db.get_memory(id).unwrap();
    assert!(node.tags.iter().any(|t| t == "category:test"));
    assert_eq!(db.memory_count(), 1, "one memory, not two records of it");
}

/// A database an older engine wrote: one memory held by two records, the later
/// one current. Opening it frees the stale copy, so a delete afterwards holds.
#[test]
fn opening_frees_stale_copies_an_older_engine_left() {
    use mentedb::storage::StorageEngine;

    let dir = tempfile::tempdir().unwrap();
    let mut node = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        "stale-copy-marker-d4 old version".to_string(),
        (0..32).map(|i| i as f32 / 32.0 - 0.5).collect(),
    );
    let id = node.id;
    {
        let db = open(dir.path());
        db.store(node.clone()).unwrap();
        db.close().unwrap();
    }
    {
        // Write the second record the way older engines did: a new record for
        // the same id, the first left in place.
        let storage = StorageEngine::open(dir.path()).unwrap();
        node.content = "stale-copy-marker-d4 new version".to_string();
        storage
            .store_memory_batch(std::slice::from_ref(&node))
            .unwrap();
        let held = storage
            .memory_map()
            .unwrap()
            .into_iter()
            .filter(|m| m.id == id)
            .count();
        assert_eq!(held, 2, "the setup has two records for one memory");
        storage.checkpoint().unwrap();
    }

    let db = open(dir.path());
    assert_eq!(
        db.get_memory(id).unwrap().content,
        "stale-copy-marker-d4 new version",
        "the later record is the memory"
    );
    db.forget(id).unwrap();
    db.close().unwrap();
    let db = open(dir.path());
    assert!(db.get_memory(id).is_err(), "the stale copy came back");
}
