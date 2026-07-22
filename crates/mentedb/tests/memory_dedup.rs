//! Store-time paraphrase dedup for regular memories: reworded copies of the
//! same fact are skipped, while value updates (which share the frame but swap
//! a value token) stay storable so supersession can handle them.

use mentedb::MenteDb;
use mentedb::prelude::*;

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

fn node(agent: AgentId, content: &str, emb: Vec<f32>, at: u64) -> MemoryNode {
    let mut n = MemoryNode::new(agent, MemoryType::Semantic, content.into(), emb);
    n.created_at = at;
    n
}

#[test]
fn paraphrase_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();
    let t = now_us();

    // Same fact, reworded: near-identical embedding, high token overlap.
    let a = node(
        agent,
        "User is building a SaaS app called TaskPilot using Next.js and Supabase",
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        t,
    );
    let b = node(
        agent,
        "The user is building a SaaS app called TaskPilot using Next.js and Supabase",
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        t + 1,
    );
    let b_id = b.id;
    db.store(a).unwrap();
    db.store(b).unwrap();

    assert!(
        db.get_memory(b_id).is_err(),
        "a reworded copy of an existing fact must be skipped"
    );
    assert_eq!(db.memory_count(), 1, "only the original survives");
}

#[test]
fn value_update_is_not_swallowed() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let agent = AgentId::new();
    let t = now_us();

    // Same frame, different value: token overlap drops below the gate, so the
    // correction stores (and write inference supersedes the old fact).
    let a = node(
        agent,
        "The user's favorite coffee order is a cortado",
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        t,
    );
    let b = node(
        agent,
        "The user's favorite coffee order is a flat white",
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        t + 1,
    );
    let b_id = b.id;
    db.store(a).unwrap();
    db.store(b).unwrap();

    assert!(
        db.get_memory(b_id).is_ok(),
        "a value update must store, never be absorbed as a paraphrase"
    );
}

#[test]
fn cross_owner_is_never_deduped() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let t = now_us();

    let content = "The staging environment runs in eu-west-1";
    let emb = vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let a = node(AgentId::new(), content, emb.clone(), t);
    let b = node(AgentId::new(), content, emb, t + 1);
    let b_id = b.id;
    db.store(a).unwrap();
    db.store(b).unwrap();

    assert!(
        db.get_memory(b_id).is_ok(),
        "another owner's identical fact is their own memory, never deduped away"
    );
}

#[test]
fn disabled_config_stores_everything() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = mentedb::CognitiveConfig {
        memory_dedup: false,
        ..mentedb::CognitiveConfig::default()
    };
    let db = MenteDb::open_with_config(dir.path(), cfg).unwrap();
    let agent = AgentId::new();
    let t = now_us();

    let a = node(
        agent,
        "User is building a SaaS app called TaskPilot",
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        t,
    );
    let b = node(
        agent,
        "The user is building a SaaS app called TaskPilot",
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        t + 1,
    );
    let b_id = b.id;
    db.store(a).unwrap();
    db.store(b).unwrap();
    assert!(db.get_memory(b_id).is_ok(), "dedup off means both store");
}
