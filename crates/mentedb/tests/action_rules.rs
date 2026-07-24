//! Action-cued recall (`recall_for_action`): standing rules tagged
//! `trigger:<action>` that surface at the moment an agent performs that
//! action, not by topic similarity. These tests pin the contract the hook
//! integrations depend on: strict trigger matching, agent and user
//! visibility identical to every other scoped recall, exclusion of
//! superseded rules, newest-first ordering, and the cap.

use mentedb::MenteDb;
use mentedb::prelude::*;
use mentedb_core::types::{AgentId, UserId};

fn rule(agent: AgentId, content: &str, tags: &[&str]) -> MemoryNode {
    let mut node = MemoryNode::new(agent, MemoryType::Procedural, content.to_string(), vec![]);
    node.tags = tags.iter().map(|t| t.to_string()).collect();
    node
}

#[test]
fn returns_only_matching_trigger() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    let commit = rule(
        AgentId::nil(),
        "never add Co-Authored-By trailers",
        &["trigger:git-commit"],
    );
    let commit_id = commit.id;
    db.store(commit).unwrap();

    let pr = rule(
        AgentId::nil(),
        "PR descriptions use Summary and Verification sections",
        &["trigger:pr-create"],
    );
    db.store(pr).unwrap();

    let untagged = rule(AgentId::nil(), "the user prefers dark mode", &[]);
    db.store(untagged).unwrap();

    let rules = db.recall_for_action("git-commit", None, None, 8).unwrap();
    assert_eq!(rules.len(), 1, "only the git-commit rule should match");
    assert_eq!(rules[0].id, commit_id);

    // A plain tag that happens to share the name must not match: the channel
    // is the `trigger:` namespace, not free-form tags.
    let plain = rule(AgentId::nil(), "notes about git-commit", &["git-commit"]);
    db.store(plain).unwrap();
    let rules = db.recall_for_action("git-commit", None, None, 8).unwrap();
    assert_eq!(
        rules.len(),
        1,
        "plain tags must not enter the trigger channel"
    );
}

#[test]
fn agent_isolation_matches_scoped_recall_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    let a = AgentId::new();
    let b = AgentId::new();

    let a_rule = rule(
        a,
        "agent A: single line commit subjects",
        &["trigger:git-commit"],
    );
    let a_id = a_rule.id;
    db.store(a_rule).unwrap();

    let b_rule = rule(b, "agent B: sign commits with gpg", &["trigger:git-commit"]);
    let b_id = b_rule.id;
    db.store(b_rule).unwrap();

    let global = rule(
        AgentId::nil(),
        "account rule: conventional commit prefixes",
        &["trigger:git-commit"],
    );
    let global_id = global.id;
    db.store(global).unwrap();

    // Agent A sees its own rule plus the account-global rule, never B's.
    let for_a = db
        .recall_for_action("git-commit", Some(a), None, 8)
        .unwrap();
    let ids: Vec<_> = for_a.iter().map(|n| n.id).collect();
    assert!(ids.contains(&a_id));
    assert!(ids.contains(&global_id));
    assert!(
        !ids.contains(&b_id),
        "agent B's rule leaked into agent A's action recall"
    );

    // Agent B: mirror image.
    let for_b = db
        .recall_for_action("git-commit", Some(b), None, 8)
        .unwrap();
    let ids: Vec<_> = for_b.iter().map(|n| n.id).collect();
    assert!(ids.contains(&b_id));
    assert!(ids.contains(&global_id));
    assert!(!ids.contains(&a_id));

    // Unscoped (admin) sees everything.
    let all = db.recall_for_action("git-commit", None, None, 8).unwrap();
    assert_eq!(all.len(), 3);
}

#[test]
fn user_isolation_within_shared_agent() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    let agent = AgentId::new();
    let u1 = UserId::new();
    let u2 = UserId::new();

    let mut u1_rule = rule(
        agent,
        "user one: no emojis in commits",
        &["trigger:git-commit"],
    );
    u1_rule.user_id = u1;
    let u1_id = u1_rule.id;
    db.store(u1_rule).unwrap();

    let mut u2_rule = rule(agent, "user two: emoji away", &["trigger:git-commit"]);
    u2_rule.user_id = u2;
    let u2_id = u2_rule.id;
    db.store(u2_rule).unwrap();

    let shared = rule(
        agent,
        "shared: imperative mood subjects",
        &["trigger:git-commit"],
    );
    let shared_id = shared.id;
    db.store(shared).unwrap();

    let for_u1 = db
        .recall_for_action("git-commit", Some(agent), Some(u1), 8)
        .unwrap();
    let ids: Vec<_> = for_u1.iter().map(|n| n.id).collect();
    assert!(ids.contains(&u1_id));
    assert!(
        ids.contains(&shared_id),
        "nil-user rules are shared knowledge"
    );
    assert!(
        !ids.contains(&u2_id),
        "user two's rule leaked into user one's action recall"
    );
}

#[test]
fn superseded_rule_stops_firing() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    let old = rule(
        AgentId::nil(),
        "commit messages may include emojis",
        &["trigger:git-commit"],
    );
    let old_id = old.id;
    db.store(old).unwrap();

    let new = rule(
        AgentId::nil(),
        "commit messages must not include emojis",
        &["trigger:git-commit"],
    );
    let new_id = new.id;
    db.store(new).unwrap();

    // Mirror what supersession does: a Supersedes edge plus temporal
    // invalidation of the replaced rule (write inference and the platform
    // both invalidate through this same path).
    db.relate(MemoryEdge {
        source: new_id,
        target: old_id,
        edge_type: EdgeType::Supersedes,
        weight: 1.0,
        created_at: 0,
        valid_from: None,
        valid_until: None,
        label: None,
    })
    .unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    db.invalidate_memory(old_id, now).unwrap();

    let rules = db.recall_for_action("git-commit", None, None, 8).unwrap();
    let ids: Vec<_> = rules.iter().map(|n| n.id).collect();
    assert!(ids.contains(&new_id), "the correction must fire");
    assert!(
        !ids.contains(&old_id),
        "a superseded rule fired: the exact bug this channel must never have"
    );
}

#[test]
fn newest_first_and_capped() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    // Explicit created_at stamps so the ordering assertion is deterministic.
    let base = 1_700_000_000_000_000u64;
    let mut ids = Vec::new();
    for i in 0..6u64 {
        let mut node = rule(
            AgentId::nil(),
            &format!("rule number {i}"),
            &["trigger:git-commit"],
        );
        node.created_at = base + i * 1_000_000;
        ids.push((node.id, node.created_at));
        db.store(node).unwrap();
    }

    let rules = db.recall_for_action("git-commit", None, None, 3).unwrap();
    assert_eq!(rules.len(), 3, "cap must hold");
    let got: Vec<_> = rules.iter().map(|n| n.created_at).collect();
    let mut expected: Vec<_> = ids.iter().map(|(_, t)| *t).collect();
    expected.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(got, expected[..3].to_vec(), "newest rules must lead");
}

#[test]
fn unknown_trigger_empty_trigger_and_zero_k() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    db.store(rule(
        AgentId::nil(),
        "never force push to main",
        &["trigger:git-push"],
    ))
    .unwrap();

    assert!(
        db.recall_for_action("deploy", None, None, 8)
            .unwrap()
            .is_empty()
    );
    assert!(db.recall_for_action("", None, None, 8).unwrap().is_empty());
    assert!(
        db.recall_for_action("   ", None, None, 8)
            .unwrap()
            .is_empty()
    );
    assert!(
        db.recall_for_action("git-push", None, None, 0)
            .unwrap()
            .is_empty()
    );
    // Case-normalized lookup: the documented tag form is lowercase.
    assert_eq!(
        db.recall_for_action("Git-Push", None, None, 8)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn correct_at_scale_without_scanning() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    // A realistic corpus: thousands of ordinary memories, a handful of rules.
    for i in 0..2000 {
        db.store(rule(
            AgentId::nil(),
            &format!("ordinary memory {i} about work"),
            &[],
        ))
        .unwrap();
    }
    let mut expected = Vec::new();
    for i in 0..3 {
        let node = rule(
            AgentId::nil(),
            &format!("commit rule {i}"),
            &["trigger:git-commit"],
        );
        expected.push(node.id);
        db.store(node).unwrap();
    }

    let rules = db.recall_for_action("git-commit", None, None, 8).unwrap();
    assert_eq!(rules.len(), 3);
    for id in expected {
        assert!(rules.iter().any(|n| n.id == id));
    }
}

#[test]
fn read_only_no_counter_or_decay_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    let node = rule(AgentId::nil(), "squash merge only", &["trigger:git-commit"]);
    let id = node.id;
    db.store(node).unwrap();

    let before = db.get_memory(id).unwrap();
    for _ in 0..5 {
        db.recall_for_action("git-commit", None, None, 8).unwrap();
    }
    let after = db.get_memory(id).unwrap();

    assert_eq!(before.access_count, after.access_count, "no access bumps");
    assert_eq!(
        before.accessed_at, after.accessed_at,
        "no decay clock touch"
    );
    assert_eq!(before.salience, after.salience, "no salience change");
}
