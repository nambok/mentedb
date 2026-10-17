//! Mode rules: standing directives ("longterm fixes only", "sound like a
//! scientist") activate when the turn's embedding lands near the rule's own
//! stored embedding, never through pattern matching. These tests use
//! constructed embeddings so the plumbing (enumeration, visibility,
//! validity, dedup, cap, reason) is pinned deterministically; the semantic
//! behavior on the real embedder is measured by the deep internal test.

use mentedb::injection::{InjectionConfig, InjectionQuery, SelectionReason};
use mentedb::prelude::*;
use mentedb::{CognitiveConfig, MenteDb};
use mentedb_core::types::AgentId;

const DIM: usize = 8;

fn axis(i: usize, weight: f32) -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[i] = weight;
    v
}

fn directive(content: &str, trigger: &str, embedding: Vec<f32>) -> MemoryNode {
    let mut node = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Procedural,
        content.to_string(),
        embedding,
    );
    node.tags = vec![format!("trigger:{trigger}")];
    node
}

fn open_with_gate(dir: &std::path::Path, gate: f32, max: usize) -> MenteDb {
    let config = CognitiveConfig {
        injection_config: InjectionConfig {
            mode_trigger_gate: gate,
            mode_rules_max: max,
            ..InjectionConfig::default()
        },
        ..CognitiveConfig::default()
    };
    MenteDb::open_with_config(dir, config).unwrap()
}

fn inject(db: &MenteDb, embedding: &[f32], k: usize) -> Vec<(SelectionReason, String)> {
    let query = InjectionQuery {
        embedding,
        query_text: None,
        session_id: None,
        exclude_ids: &[],
        max_items: k,
        max_episodic: 2,
        agent_id: None,
        user_id: None,
        current_project: None,
    };
    db.recall_for_injection(&query)
        .unwrap()
        .into_iter()
        .map(|c| (c.reason, c.node.content))
        .collect()
}

#[test]
fn directives_activate_by_activity_not_topic() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_gate(dir.path(), 0.4, 4);

    // Axis 0 is the working activity, axis 1 the replying activity, axis 3
    // the conversation's actual topic (payment retries, say).
    db.store(directive(
        "When working on code changes, longterm root cause fixes only, never quick patches",
        "code-change",
        axis(0, 1.0),
    ))
    .unwrap();
    db.store(directive(
        "Written replies stay precise and measured, like a careful scientist",
        "reply-style",
        axis(1, 1.0),
    ))
    .unwrap();
    // Topical memories about the task at hand dominate ordinary retrieval.
    for i in 0..12 {
        let node = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            format!("Topical fact {i} about payment retries"),
            axis(3, 1.0),
        );
        db.store(node).unwrap();
    }

    // The turn is mostly about its topic (0.89 to the topical cluster) and
    // only faintly near the work directive (0.45): the relevance tail gate
    // (0.55 of best cosine, 0.49 here) would DROP the directive, which is
    // the entire reason the mode channel exists. It must arrive as a
    // ModeRule anyway; the reply directive (cosine 0) stays out entirely.
    let mut turn = vec![0.0; DIM];
    turn[0] = 0.45;
    turn[3] = 0.89;
    let got = inject(&db, &turn, 6);
    assert!(
        got.iter()
            .any(|(r, c)| *r == SelectionReason::ModeRule && c.contains("root cause")),
        "the work directive must arrive through the mode channel: {got:?}"
    );
    assert!(
        !got.iter().any(|(_, c)| c.contains("scientist")),
        "the reply directive must stay inactive for a work turn: {got:?}"
    );

    // A turn orthogonal to both activities activates neither.
    let got = inject(&db, &axis(2, 1.0), 6);
    assert!(
        !got.iter().any(|(r, _)| *r == SelectionReason::ModeRule),
        "no directive activates for an unrelated turn: {got:?}"
    );
}

#[test]
fn superseded_and_foreign_directives_never_activate() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_gate(dir.path(), 0.5, 4);

    let old = directive(
        "Old rule: quick patches are fine",
        "code-change",
        axis(0, 1.0),
    );
    let old_id = old.id;
    db.store(old).unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    db.invalidate_memory(old_id, now).unwrap();

    let other_agent = AgentId::new();
    let mut foreign = directive(
        "Foreign agent directive about fixes",
        "code-change",
        axis(0, 1.0),
    );
    foreign.agent_id = other_agent;
    db.store(foreign).unwrap();

    let me = AgentId::new();
    let query = InjectionQuery {
        embedding: &axis(0, 1.0),
        query_text: None,
        session_id: None,
        exclude_ids: &[],
        max_items: 6,
        max_episodic: 2,
        agent_id: Some(me),
        user_id: None,
        current_project: None,
    };
    let got: Vec<_> = db
        .recall_for_injection(&query)
        .unwrap()
        .into_iter()
        .filter(|c| c.reason == SelectionReason::ModeRule)
        .map(|c| c.node.content)
        .collect();
    assert!(
        got.is_empty(),
        "superseded and foreign directives must not activate: {got:?}"
    );
}

#[test]
fn mode_rules_respect_cap_and_gate_zero_disables() {
    // Gate zero disables the channel entirely.
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_gate(dir.path(), 0.0, 4);
    db.store(directive("Any directive", "code-change", axis(0, 1.0)))
        .unwrap();
    let got = inject(&db, &axis(0, 1.0), 6);
    assert!(
        !got.iter().any(|(r, _)| *r == SelectionReason::ModeRule),
        "gate zero must disable mode rules: {got:?}"
    );

    // A tiny cap holds even when many rules activate, best similarities win.
    // Topical facts own the relevance slots so the directives can only
    // arrive through the mode channel, which is the channel under test.
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_gate(dir.path(), 0.5, 2);
    for i in 0..6 {
        db.store(directive(
            &format!("Directive number {i}"),
            &format!("mode-{i}"),
            axis(0, 1.0),
        ))
        .unwrap();
    }
    for i in 0..12 {
        let node = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            format!("Topical fact {i} about payment retries"),
            axis(3, 1.0),
        );
        db.store(node).unwrap();
    }
    // Topic dominant turn (0.8 to the topical cluster) with the activity
    // above the 0.5 gate (0.6 to the directives' axis).
    let mut turn = axis(3, 0.8);
    turn[0] = 0.6;
    let got = inject(&db, &turn, 12);
    let mode_count = got
        .iter()
        .filter(|(r, _)| *r == SelectionReason::ModeRule)
        .count();
    assert_eq!(mode_count, 2, "mode_rules_max must cap: {got:?}");
}

#[test]
fn exemplars_drive_activation_and_never_inject() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_gate(dir.path(), 0.4, 4);

    // The rule's own embedding is orthogonal to every turn (axis 4): with
    // rule based activation it could never fire. Its exemplar sits on the
    // turn axis, so exemplar activation carries it.
    db.store(directive(
        "When working on code, longterm root cause fixes only",
        "code-change",
        axis(4, 1.0),
    ))
    .unwrap();
    let mut exemplar = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        "fix this bug in the auth code".to_string(),
        axis(0, 1.0),
    );
    exemplar.tags = vec!["mode-exemplar:code-change".to_string()];
    db.store(exemplar).unwrap();
    // Topical distractors keep the ordinary relevance path busy so the rule
    // can only arrive through the mode channel.
    for i in 0..12 {
        let node = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            format!("Topical fact {i} about payment retries"),
            axis(3, 1.0),
        );
        db.store(node).unwrap();
    }

    let mut turn = vec![0.0; DIM];
    turn[0] = 0.45;
    turn[3] = 0.89;
    let got = inject(&db, &turn, 6);
    assert!(
        got.iter()
            .any(|(r, c)| *r == SelectionReason::ModeRule && c.contains("root cause")),
        "exemplar activation must inject the rule: {got:?}"
    );
    assert!(
        !got.iter()
            .any(|(_, c)| c.contains("fix this bug in the auth code")),
        "exemplars are activation metadata and never inject: {got:?}"
    );
}

/// Mode activation reads cached probes (embedding, validity, owner) instead of
/// loading each rule from storage. A write to a rule must reach the very next
/// recall: invalidating it deactivates it, and re-storing it with a new
/// embedding moves its activation.
#[test]
fn rule_writes_reach_the_next_recall() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_gate(dir.path(), 0.4, 4);
    let rule = directive(
        "When working on code changes, longterm root cause fixes only",
        "code-change",
        axis(0, 1.0),
    );
    let id = rule.id;
    db.store(rule.clone()).unwrap();
    // Topical memories dominate ordinary retrieval, so the rule can only
    // arrive through the mode channel.
    for i in 0..12 {
        db.store(MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            format!("Topical fact {i} about payment retries"),
            axis(3, 1.0),
        ))
        .unwrap();
    }
    // A turn mostly about the topic, faintly near activity axis `a`.
    let turn = |a: usize| {
        let mut t = vec![0.0; DIM];
        t[a] = 0.45;
        t[3] = 0.89;
        t
    };
    let is_active = |db: &MenteDb, emb: &[f32]| {
        inject(db, emb, 6)
            .iter()
            .any(|(r, c)| *r == SelectionReason::ModeRule && c.contains("longterm"))
    };
    // Twice, so the second recall runs from the cache.
    assert!(is_active(&db, &turn(0)));
    assert!(is_active(&db, &turn(0)));

    // Re-stored with an embedding on another axis: the old activity no
    // longer activates it, the new one does.
    let mut moved = rule.clone();
    moved.embedding = axis(2, 1.0);
    db.store(moved).unwrap();
    assert!(!is_active(&db, &turn(0)));
    assert!(is_active(&db, &turn(2)));

    // Invalidated: never activates again.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    db.invalidate_memory(id, now).unwrap();
    assert!(!is_active(&db, &turn(2)));
}

/// An active mode with hundreds of rules loads only the few it injects. Ranking
/// uses cached probes, so a warm recall loads at most `mode_rules_max` full
/// memories for the mode channel, not every rule in the mode.
#[test]
fn an_active_mode_loads_only_the_rules_it_keeps() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_gate(dir.path(), 0.4, 4);
    for i in 0..300 {
        db.store(directive(
            &format!("Working rule {i}: longterm root cause fixes only"),
            "code-change",
            axis(0, 1.0),
        ))
        .unwrap();
    }
    for i in 0..12 {
        db.store(MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            format!("Topical fact {i} about payment retries"),
            axis(3, 1.0),
        ))
        .unwrap();
    }
    let mut turn = vec![0.0; DIM];
    turn[0] = 0.45;
    turn[3] = 0.89;

    // The first recall fills the probe cache.
    let first = inject(&db, &turn, 6);
    let kept = first
        .iter()
        .filter(|(r, _)| *r == SelectionReason::ModeRule)
        .count();
    assert_eq!(kept, 4, "the mode injects its cap: {first:?}");

    let before = db.metrics().memory_loads;
    let second = inject(&db, &turn, 6);
    let loads = db.metrics().memory_loads - before;
    assert_eq!(first, second, "the warm recall picks the same memories");
    // Hybrid candidates plus the 4 kept rules; never the 300 in the mode.
    assert!(loads < 100, "warm recall loaded {loads} memories");
}
