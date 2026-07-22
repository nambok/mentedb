//! The value-update supersession loop, end to end: storing a corrected fact
//! with the same sentence frame must hide the outdated one from recall (the
//! live bug: "cortado" and "flat white" favorites coexisted and recall served
//! both).

use mentedb::MenteDb;
use mentedb::prelude::*;

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

#[test]
fn corrected_fact_supersedes_and_recall_serves_only_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let mut old = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The user's favorite coffee order is a cortado".into(),
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    old.created_at = now_us();
    let old_id = old.id;
    db.store(old).unwrap();

    let mut new = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The user's favorite coffee order is a flat white".into(),
        vec![0.99, 0.08, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    new.created_at = now_us() + 1;
    let new_id = new.id;
    db.store(new).unwrap();

    let hits = db
        .recall_hybrid_at(
            &[1.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            Some("favorite coffee order"),
            5,
            now_us() + 10,
            None,
            None,
            None,
        )
        .unwrap();
    let ids: Vec<MemoryId> = hits.iter().map(|(id, _)| *id).collect();
    assert!(
        ids.contains(&new_id),
        "the corrected fact must be recallable"
    );
    assert!(
        !ids.contains(&old_id),
        "the superseded fact must be hidden from recall"
    );
}

#[test]
fn different_subjects_both_survive() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();

    let mut dog = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "My dog is allergic to chicken".into(),
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    dog.created_at = now_us();
    let dog_id = dog.id;
    db.store(dog).unwrap();

    let mut cat = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "My cat is allergic to chicken".into(),
        vec![0.99, 0.08, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    cat.created_at = now_us() + 1;
    let cat_id = cat.id;
    db.store(cat).unwrap();

    let hits = db
        .recall_hybrid_at(
            &[1.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            Some("allergic to chicken"),
            5,
            now_us() + 10,
            None,
            None,
            None,
        )
        .unwrap();
    let ids: Vec<MemoryId> = hits.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&dog_id), "the dog fact must survive");
    assert!(ids.contains(&cat_id), "the cat fact must survive");
}
