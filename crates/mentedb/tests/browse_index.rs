//! The dashboard browse filters (memory type, owner user, owner agent) must be
//! served from the bitmap index in O(matches), returning exactly the set a
//! per-node filter would, and staying correct across a re-index and a reopen.

use mentedb::MenteDb;
use mentedb::prelude::{AgentId, MemoryNode, MemoryType, UserId};

fn node(user: u128, agent: u128, ty: MemoryType, content: &str, tags: &[&str]) -> MemoryNode {
    let mut n = MemoryNode::new(
        AgentId(uuid::Uuid::from_u128(agent)),
        ty,
        content.to_string(),
        Vec::new(),
    );
    n.user_id = UserId(uuid::Uuid::from_u128(user));
    n.tags = tags.iter().map(|t| t.to_string()).collect();
    n
}

#[test]
fn type_and_owner_indexes_match_a_full_filter() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    let a = node(1, 1, MemoryType::Semantic, "sem a", &[]);
    let a_id = a.id;
    let b = node(1, 2, MemoryType::Procedural, "proc b", &[]);
    let b_id = b.id;
    let c = node(2, 1, MemoryType::Semantic, "sem c", &[]);
    let c_id = c.id;
    db.store(a).unwrap();
    db.store(b).unwrap();
    db.store(c).unwrap();

    // Type index: exactly the two semantic memories, not the procedural one.
    let mut sem = db.memory_ids_of_type_str("semantic");
    sem.sort();
    let mut want = vec![a_id, c_id];
    want.sort();
    assert_eq!(sem, want, "semantic type index");
    assert_eq!(db.memory_ids_of_type_str("procedural"), vec![b_id]);
    assert!(db.memory_ids_of_type_str("episodic").is_empty());

    // Owner user index: user 1 owns a and b; user 2 owns c.
    let mut u1 = db.memory_ids_for_user(UserId(uuid::Uuid::from_u128(1)));
    u1.sort();
    let mut want_u1 = vec![a_id, b_id];
    want_u1.sort();
    assert_eq!(u1, want_u1, "user 1 index");
    assert_eq!(
        db.memory_ids_for_user(UserId(uuid::Uuid::from_u128(2))),
        vec![c_id]
    );

    // Owner agent index: agent 1 owns a and c; agent 2 owns b.
    let mut ag1 = db.memory_ids_for_agent(AgentId(uuid::Uuid::from_u128(1)));
    ag1.sort();
    let mut want_ag1 = vec![a_id, c_id];
    want_ag1.sort();
    assert_eq!(ag1, want_ag1, "agent 1 index");
}

#[test]
fn reindex_after_forget_keeps_the_browse_index_exact() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");
    let a = node(1, 1, MemoryType::Semantic, "keep", &[]);
    let a_id = a.id;
    let b = node(1, 1, MemoryType::Semantic, "drop", &[]);
    let b_id = b.id;
    db.store(a).unwrap();
    db.store(b).unwrap();
    db.forget(b_id).unwrap();

    let sem = db.memory_ids_of_type_str("semantic");
    assert_eq!(
        sem,
        vec![a_id],
        "forgotten memory dropped from the type index"
    );
    assert!(
        !db.memory_ids_for_user(UserId(uuid::Uuid::from_u128(1)))
            .contains(&b_id)
    );
}

#[test]
fn browse_index_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let db = MenteDb::open(dir.path()).expect("open");
        let a = node(3, 3, MemoryType::Procedural, "persist me", &[]);
        id = a.id;
        db.store(a).unwrap();
        db.flush_full().expect("flush");
    }
    let db = MenteDb::open(dir.path()).expect("reopen");
    assert_eq!(db.memory_ids_of_type_str("procedural"), vec![id]);
    assert_eq!(
        db.memory_ids_for_user(UserId(uuid::Uuid::from_u128(3))),
        vec![id]
    );
}
