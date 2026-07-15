//! End-user isolation (multi-tenant safety).
//!
//! When each end-user of an application is given a distinct `agent_id`, one
//! user's private memories must never surface in another user's turn, through
//! ANY channel: retrieved context, proactive recalls, or pain warnings. Global
//! knowledge owned by nobody (nil agent) stays visible to everyone. This is the
//! "each user sees their own memories plus global, never another user's" contract.
//!
//! These tests run WITH an embedder (so the proactive-recall path actually
//! fires, the way it does in production) and are adversarial: the other user's
//! private memory is placed on the exact topic the querying user touches, so any
//! leak surfaces. A test that passed only because a code path stayed dormant
//! would be worthless for a security boundary.

use mentedb::CognitiveConfig;
use mentedb::MenteDb;
use mentedb::prelude::*;
use mentedb::process_turn::{ProcessTurnInput, ProcessTurnResult};
use mentedb_context::DeltaTracker;
use mentedb_embedding::{EmbeddingProvider, HashEmbeddingProvider};
use uuid::Uuid;

const DIM: usize = 384;

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

fn open_db() -> (MenteDb, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    // A real (deterministic) embedder, so process_turn embeds queries and the
    // proactive-recall / write-inference paths run exactly as in production.
    let db = MenteDb::open_with_embedder_and_config(
        dir.path(),
        Box::new(HashEmbeddingProvider::new(DIM)),
        CognitiveConfig::default(),
    )
    .unwrap();
    (db, dir)
}

/// Deterministic embedding for stored content, matching the db's embedder so
/// stored vectors and query vectors live in the same space.
fn emb(text: &str) -> Vec<f32> {
    HashEmbeddingProvider::new(DIM).embed(text).unwrap()
}

fn store_owned(db: &MenteDb, agent: AgentId, content: &str) -> MemoryId {
    let node = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        content.to_string(),
        emb(content),
    );
    let id = node.id;
    db.store(node).unwrap();
    id
}

/// Every content-bearing field of a turn result, flattened for leak checks.
/// If another user's text appears anywhere in here, that is a leak.
fn result_text(r: &ProcessTurnResult) -> String {
    let mut s = String::new();
    for c in &r.context {
        s.push_str(&c.memory.content);
        s.push('\n');
    }
    for p in &r.proactive_recalls {
        s.push_str(&p.content);
        s.push('\n');
    }
    for w in &r.pain_warnings {
        s.push_str(&w.description);
        s.push('\n');
    }
    for t in &r.predicted_topics {
        s.push_str(t);
        s.push('\n');
    }
    s
}

const ALICE_SECRET: &str = "ALICE_SECRET the deployment key is falcon9 do not share";
const GLOBAL_NOTE: &str = "GLOBAL the office deploy runbook lives in the wiki";

/// The core contract: nothing Alice stored privately may appear in Bob's turn,
/// through any output channel, including proactive recall.
#[test]
fn other_users_memory_never_leaks_into_a_turn() {
    let (db, _dir) = open_db();
    let mut delta = DeltaTracker::new();
    let alice = Uuid::new_v4();
    let bob = Uuid::new_v4();

    // Alice's private memory, on the very topic Bob will raise.
    store_owned(&db, AgentId(alice), ALICE_SECRET);
    // A global memory owned by nobody: both users may see it.
    store_owned(&db, AgentId::nil(), GLOBAL_NOTE);

    // Bob talks about deployment (action keywords fire proactive recall).
    let input = ProcessTurnInput {
        user_message: "help me deploy the new release to production".to_string(),
        assistant_response: Some("sure, starting the deploy now".to_string()),
        turn_id: 0,
        project_context: None,
        agent_id: Some(bob),
        session_id: None,
    };
    let bob_result = db.process_turn(&input, &mut delta).unwrap();
    let leaked = result_text(&bob_result);

    assert!(
        !leaked.contains("ALICE_SECRET") && !leaked.contains("falcon9"),
        "SECURITY: another user's private memory leaked into Bob's turn output:\n{leaked}"
    );
}

/// Isolation must not become a black hole: a scoped user still sees their own
/// memory and global (nil owned) knowledge, just never another user's.
#[test]
fn scoped_user_still_sees_own_and_global() {
    let (db, _dir) = open_db();
    let bob = AgentId::new();

    let bob_own = store_owned(&db, bob, "bob prefers dark mode in the deploy dashboard");
    let global = store_owned(&db, AgentId::nil(), GLOBAL_NOTE);
    let alice_secret = store_owned(&db, AgentId::new(), ALICE_SECRET);

    let hits = db
        .recall_hybrid_scoped_at_mode(
            &emb("deploy dashboard runbook"),
            Some("deploy dashboard runbook"),
            10,
            now_us(),
            None,
            false,
            None,
            Some(bob),
        )
        .unwrap();
    let ids: Vec<MemoryId> = hits.iter().map(|(id, _)| *id).collect();

    assert!(ids.contains(&bob_own), "user must see their own memory");
    assert!(
        ids.contains(&global),
        "user must see global (nil owned) memory"
    );
    assert!(
        !ids.contains(&alice_secret),
        "user must never see another user's private memory"
    );
}

/// Adversarial: Alice's private memory is the best match for Bob's query. It
/// must still be excluded from Bob's scoped recall.
#[test]
fn scoped_recall_excludes_other_users_even_as_top_match() {
    let (db, _dir) = open_db();
    let bob = AgentId::new();
    let secret = store_owned(&db, AgentId::new(), ALICE_SECRET);
    store_owned(&db, bob, "bob's unrelated note about lunch plans");

    let hits = db
        .recall_hybrid_scoped_at_mode(
            &emb(ALICE_SECRET),
            Some("deployment key falcon9"),
            10,
            now_us(),
            None,
            false,
            None,
            Some(bob),
        )
        .unwrap();
    let ids: Vec<MemoryId> = hits.iter().map(|(id, _)| *id).collect();
    assert!(
        !ids.contains(&secret),
        "SECURITY: another user's private memory must never appear, even as the top match"
    );
}

/// The enrichment helpers must be exact-owner scoped: the derived-knowledge
/// pipeline (profile, entities, communities) may only read the target owner's
/// memories, so one user's profile is never built from another user's facts.
#[test]
fn enrichment_helpers_are_owner_scoped() {
    let (db, _dir) = open_db();
    let alice = AgentId::new();
    let bob = AgentId::new();
    // Alice and Bob each store a semantic fact.
    let a = MemoryNode::new(
        alice,
        MemoryType::Semantic,
        "alice likes falcons".into(),
        emb("alice likes falcons"),
    );
    db.store(a).unwrap();
    let b = MemoryNode::new(
        bob,
        MemoryType::Semantic,
        "bob likes turtles".into(),
        emb("bob likes turtles"),
    );
    db.store(b).unwrap();

    let alice_facts = db.profile_facts(alice);
    assert!(
        alice_facts.iter().any(|f| f.contains("falcons")),
        "alice sees her own fact"
    );
    assert!(
        !alice_facts.iter().any(|f| f.contains("turtles")),
        "alice must NOT see bob's fact in her profile facts"
    );

    let bob_facts = db.profile_facts(bob);
    assert!(
        !bob_facts.iter().any(|f| f.contains("falcons")),
        "bob must NOT see alice's fact"
    );
    assert!(
        bob_facts.iter().any(|f| f.contains("turtles")),
        "bob sees his own fact"
    );
}
