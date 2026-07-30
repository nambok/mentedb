//! The O(1) dashboard scope counts must mirror the store: correct totals by
//! type, owner, scope, and project; internal (hidden-tagged) working material
//! excluded; decremented on forget; and self-healing when an edit drifts the
//! maintained total away from the real count.

use mentedb::prelude::{AgentId, MemoryNode, MemoryType, UserId};
use mentedb::{CognitiveConfig, MenteDb};

fn config() -> CognitiveConfig {
    CognitiveConfig {
        // The host declares its internal-material spellings.
        hidden_count_tags: vec!["turn".to_string(), "action".to_string()],
        ..Default::default()
    }
}

fn open(dir: &std::path::Path) -> MenteDb {
    MenteDb::open_with_config(dir, config()).expect("open")
}

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
fn counts_by_type_owner_scope_excluding_hidden() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());

    db.store(node(1, 1, MemoryType::Semantic, "a", &[]))
        .unwrap();
    db.store(node(
        1,
        1,
        MemoryType::Procedural,
        "b",
        &["scope:project:apex"],
    ))
    .unwrap();
    db.store(node(1, 2, MemoryType::Semantic, "c", &[]))
        .unwrap();
    // A shared/global memory (nil owner) and a standing rule.
    db.store(node(0, 0, MemoryType::Semantic, "d", &["scope:always"]))
        .unwrap();
    // Internal working material: excluded from every count.
    db.store(node(1, 1, MemoryType::Episodic, "raw turn", &["turn"]))
        .unwrap();

    let c = db.scope_counts();
    assert_eq!(c.total, 4, "the raw turn is excluded");
    assert_eq!(c.global, 1, "the nil-owner memory");
    assert_eq!(c.always, 1);
    assert_eq!(c.by_type.get("semantic"), Some(&3));
    assert_eq!(c.by_type.get("procedural"), Some(&1));
    assert_eq!(
        c.by_type.get("episodic"),
        None,
        "the only episodic is hidden"
    );
    // User 1 owns a, b, and c (c is user 1 under agent 2), counted by user.
    assert_eq!(
        c.by_user.get(&UserId(uuid::Uuid::from_u128(1)).to_string()),
        Some(&3)
    );
    assert_eq!(
        c.by_agent
            .get(&AgentId(uuid::Uuid::from_u128(1)).to_string()),
        Some(&2)
    );
    assert_eq!(
        c.by_agent
            .get(&AgentId(uuid::Uuid::from_u128(2)).to_string()),
        Some(&1)
    );
    assert_eq!(c.by_project.get("apex"), Some(&1));
}

#[test]
fn forget_decrements_the_counts() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let a = node(1, 1, MemoryType::Semantic, "keep", &[]);
    let b = node(
        1,
        1,
        MemoryType::Procedural,
        "drop",
        &["scope:project:apex"],
    );
    let b_id = b.id;
    db.store(a).unwrap();
    db.store(b).unwrap();

    // Ground the counts (first read), then forget and confirm the decrement is
    // maintained without a full recount.
    assert_eq!(db.scope_counts().total, 2);
    db.forget(b_id).unwrap();
    let c = db.scope_counts();
    assert_eq!(c.total, 1);
    assert_eq!(c.by_type.get("procedural"), Some(&0));
    assert_eq!(c.by_project.get("apex"), Some(&0));
    assert_eq!(c.by_type.get("semantic"), Some(&1));
}

#[test]
fn first_read_grounds_writes_made_before_it() {
    // Stores happen before the counts are ever read (not yet ready), so they are
    // folded in by the grounding recount rather than lost.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.store(node(1, 1, MemoryType::Semantic, "a", &[]))
        .unwrap();
    db.store(node(1, 1, MemoryType::Semantic, "b", &[]))
        .unwrap();
    db.store(node(1, 1, MemoryType::Episodic, "turn", &["action"]))
        .unwrap();
    assert_eq!(
        db.scope_counts().total,
        2,
        "grounded from the store, hidden excluded"
    );
}

#[test]
fn counts_and_blobs_persist_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = MenteDb::open_with_config(dir.path(), config()).expect("open");
        db.store(node(1, 1, MemoryType::Semantic, "a", &[]))
            .unwrap();
        db.store(node(
            1,
            1,
            MemoryType::Procedural,
            "b",
            &["scope:project:apex"],
        ))
        .unwrap();
        db.store(node(1, 1, MemoryType::Episodic, "raw turn", &["turn"]))
            .unwrap();
        assert_eq!(db.scope_counts().total, 2, "grounds, hidden excluded");
        db.write_blob("stats", b"cached-stats-payload");
        db.flush_full().expect("flush");
    }
    // Reopen: counts come from the persisted snapshot (no recount), blob reads.
    let db = MenteDb::open_with_config(dir.path(), config()).expect("reopen");
    let c = db.scope_counts();
    assert_eq!(c.total, 2, "persisted counts survive a cold reopen");
    assert_eq!(c.by_type.get("procedural"), Some(&1));
    assert_eq!(c.by_project.get("apex"), Some(&1));
    assert_eq!(
        db.read_blob("stats").as_deref(),
        Some(&b"cached-stats-payload"[..])
    );
    assert_eq!(db.read_blob("never-written"), None);
}
