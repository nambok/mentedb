//! Action recall (`recall_for_action`): the memories that bear on an action
//! the agent is about to take, surfaced by MEANING, not by a `trigger:` tag.
//!
//! The caller passes the embedding of the meaning of the action about to
//! happen; the engine returns memories that are both semantically near that
//! meaning and of a kind that GOVERNS behavior (a procedure, correction, or
//! anti-pattern outranks a topical fact or a past event). There is no tag, no
//! command classifier, and no string equality in the path.
//!
//! These tests pin the contract with controlled embeddings, so they are
//! deterministic and need no network embedder. The end-to-end proof that a
//! real embedder places "about to git commit" near "sign commits with
//! 1Password" is a separate live smoke against Bedrock; here we prove the
//! ranking machinery: semantic floor, type-governing weight, visibility, caps.

use mentedb::MenteDb;
use mentedb::prelude::*;
use mentedb_core::types::{AgentId, UserId};

const DIM: usize = 8;

/// A unit vector pointing along axis `i`, so two nodes on the same axis are
/// maximally similar and nodes on different axes are orthogonal (cosine 0).
fn axis(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[i] = 1.0;
    v
}

/// A vector mostly along `i` with a small tilt toward `j`, for a high-but-not
/// perfect similarity to `axis(i)`.
fn near(i: usize, j: usize, tilt: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[i] = 1.0;
    v[j] = tilt;
    v
}

fn node(agent: AgentId, ty: MemoryType, content: &str, emb: Vec<f32>) -> MemoryNode {
    MemoryNode::new(agent, ty, content.to_string(), emb)
}

#[test]
fn governing_type_outranks_topical_at_equal_similarity() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    // Both sit on the SAME axis as the action, so raw cosine is identical.
    // The procedure governs the action; the episode is merely topical. Type
    // weight, not similarity, must decide the order.
    let rule = node(
        AgentId::nil(),
        MemoryType::Procedural,
        "sign commits with 1Password",
        axis(0),
    );
    let rule_id = rule.id;
    db.store(rule).unwrap();

    let episode = node(
        AgentId::nil(),
        MemoryType::Episodic,
        "committed the auth fix yesterday",
        axis(0),
    );
    db.store(episode).unwrap();

    let out = db.recall_for_action(&axis(0), None, None, None, 8).unwrap();
    assert!(!out.is_empty(), "the governing rule must surface");
    assert_eq!(out[0].id, rule_id, "procedure outranks the topical episode");
}

#[test]
fn below_similarity_floor_is_excluded() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    // Orthogonal to the action meaning: cosine 0, well under the floor.
    let unrelated = node(
        AgentId::nil(),
        MemoryType::Procedural,
        "how to file an expense report",
        axis(3),
    );
    db.store(unrelated).unwrap();

    let out = db.recall_for_action(&axis(0), None, None, None, 8).unwrap();
    assert!(
        out.is_empty(),
        "an unrelated action pulls no governing rules, got {out:?}"
    );
}

#[test]
fn near_rule_surfaces_over_distant_rule() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    // Two procedures, equal type weight; only similarity separates them.
    let commit_rule = node(
        AgentId::nil(),
        MemoryType::Procedural,
        "sign commits with 1Password",
        near(0, 1, 0.1),
    );
    let commit_id = commit_rule.id;
    db.store(commit_rule).unwrap();

    let deploy_rule = node(
        AgentId::nil(),
        MemoryType::Procedural,
        "run terraform plan before apply",
        axis(4),
    );
    db.store(deploy_rule).unwrap();

    let out = db.recall_for_action(&axis(0), None, None, None, 8).unwrap();
    assert_eq!(out[0].id, commit_id, "the rule near the action wins");
    assert!(
        out.iter()
            .all(|n| n.content != "run terraform plan before apply"),
        "the orthogonal deploy rule is below the floor and excluded"
    );
}

#[test]
fn respects_agent_and_user_visibility() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    let a = AgentId::new();
    let b = AgentId::new();

    let mut a_rule = node(a, MemoryType::Procedural, "agent A commit rule", axis(0));
    a_rule.user_id = UserId::nil();
    db.store(a_rule).unwrap();

    let mut b_rule = node(b, MemoryType::Procedural, "agent B commit rule", axis(0));
    b_rule.user_id = UserId::nil();
    db.store(b_rule).unwrap();

    let shared = node(
        AgentId::nil(),
        MemoryType::Procedural,
        "shared commit rule",
        axis(0),
    );
    db.store(shared).unwrap();

    // Scoped to agent A: sees A's own rule plus the shared (nil) rule, never B's.
    let out = db
        .recall_for_action(&axis(0), None, Some(a), None, 8)
        .unwrap();
    let contents: Vec<&str> = out.iter().map(|n| n.content.as_str()).collect();
    assert!(contents.contains(&"agent A commit rule"));
    assert!(contents.contains(&"shared commit rule"));
    assert!(
        !contents.contains(&"agent B commit rule"),
        "must never surface another agent's rule"
    );
}

#[test]
fn honors_the_k_cap() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");
    for i in 0..6 {
        db.store(node(
            AgentId::nil(),
            MemoryType::Procedural,
            &format!("commit rule {i}"),
            near(0, (i % (DIM - 1)) + 1, 0.05),
        ))
        .unwrap();
    }
    let out = db.recall_for_action(&axis(0), None, None, None, 3).unwrap();
    assert_eq!(out.len(), 3, "returns at most k");
}

#[test]
fn keyword_admits_a_rule_when_the_vector_is_weak() {
    // The robustness case: the rule's embedding is ORTHOGONAL to the action
    // (cosine 0, well under the floor), as happens under a weak or
    // non-semantic embedder. The shared keyword must still surface it, and an
    // action with no shared keyword must still surface nothing.
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    let rule = node(
        AgentId::nil(),
        MemoryType::Procedural,
        "always sign the git commit with 1Password",
        axis(3),
    );
    let rule_id = rule.id;
    db.store(rule).unwrap();

    // Vector says nothing (query is orthogonal), but the words match.
    let out = db
        .recall_for_action(
            &axis(0),
            Some("about to run git commit -m fix"),
            None,
            None,
            8,
        )
        .unwrap();
    assert_eq!(
        out.first().map(|n| n.id),
        Some(rule_id),
        "a shared keyword must surface the rule even when the vector is weak"
    );

    // No shared keyword and no vector signal: nothing surfaces.
    let out = db
        .recall_for_action(&axis(0), Some("listing directory entries"), None, None, 8)
        .unwrap();
    assert!(
        out.is_empty(),
        "an unrelated action with no keyword and no vector match stays silent, got {out:?}"
    );
}

#[test]
fn empty_embedding_or_zero_k_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");
    db.store(node(
        AgentId::nil(),
        MemoryType::Procedural,
        "a rule",
        axis(0),
    ))
    .unwrap();
    assert!(
        db.recall_for_action(&[], None, None, None, 8)
            .unwrap()
            .is_empty()
    );
    assert!(
        db.recall_for_action(&axis(0), None, None, None, 0)
            .unwrap()
            .is_empty()
    );
}
