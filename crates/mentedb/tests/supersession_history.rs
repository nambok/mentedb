//! Supersession history: the chain of replaced versions behind a fact is
//! reachable through a public API, so "what was my previous X" is answerable
//! downstream (the stored data always had it; recall alone hides it).

use mentedb::MenteDb;
use mentedb::prelude::*;

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

fn fact(agent: AgentId, content: &str, at: u64) -> MemoryNode {
    let mut n = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        content.into(),
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    n.created_at = at;
    n
}

#[test]
fn history_walks_the_supersession_chain() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();
    let t = now_us();

    // Three generations of the same fact; the value-update rule chains the
    // supersessions at store time.
    let v1 = fact(agent, "My phone number is 1111", t);
    let v1_id = v1.id;
    db.store(v1).unwrap();
    let v2 = fact(agent, "My phone number is 2222", t + 1);
    let v2_id = v2.id;
    db.store(v2).unwrap();
    let v3 = fact(agent, "My phone number is 3333", t + 2);
    let v3_id = v3.id;
    db.store(v3).unwrap();

    let history = db.supersession_history(v3_id, 5).unwrap();
    let ids: Vec<MemoryId> = history.iter().map(|n| n.id).collect();
    assert!(
        ids.contains(&v2_id),
        "history must contain the direct predecessor, got {ids:?}"
    );
    // The chain shape depends on which candidate the value-update rule picked
    // per store, but the oldest version must be reachable through the walk.
    let full: Vec<String> = history.iter().map(|n| n.content.clone()).collect();
    assert!(
        ids.contains(&v1_id) || full.iter().any(|c| c.contains("1111")),
        "the oldest version should be reachable in the chain, got {full:?}"
    );

    // A memory with no predecessors has an empty history.
    let fresh = fact(agent, "The office wifi password rotates monthly", t + 3);
    let fresh_id = fresh.id;
    db.store(fresh).unwrap();
    assert!(db.supersession_history(fresh_id, 5).unwrap().is_empty());
}

#[test]
fn history_respects_depth_cap() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();
    let t = now_us();

    let v1 = fact(agent, "My desk is number 11", t);
    db.store(v1).unwrap();
    let v2 = fact(agent, "My desk is number 22", t + 1);
    db.store(v2).unwrap();
    let v3 = fact(agent, "My desk is number 33", t + 2);
    let v3_id = v3.id;
    db.store(v3).unwrap();

    assert!(db.supersession_history(v3_id, 1).unwrap().len() <= 1);
    assert!(db.supersession_history(v3_id, 0).unwrap().is_empty());
}
