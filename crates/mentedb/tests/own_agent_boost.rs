//! Own agent boost: on an agent scoped query, the agent's own memories
//! (its ingested rules) must not be crowded out of the relevant slots by
//! the account's shared nil owned memories. Measured on a busy account,
//! shared memories otherwise flood the ranking.

use mentedb::injection::{InjectionConfig, InjectionQuery};
use mentedb::prelude::*;
use mentedb::{CognitiveConfig, MenteDb};
use mentedb_core::types::AgentId;

const DIM: usize = 8;

fn vec_with(i: usize, w: f32, j: usize, w2: f32) -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[i] = w;
    v[j] = w2;
    v
}

fn open_with(dir: &std::path::Path, boost: f32) -> MenteDb {
    let config = CognitiveConfig {
        injection_config: InjectionConfig {
            own_agent_boost: boost,
            ..InjectionConfig::default()
        },
        ..CognitiveConfig::default()
    };
    MenteDb::open_with_config(dir, config).unwrap()
}

fn seed(db: &MenteDb, me: AgentId) {
    // Shared account memories sit closer to the turn than the agent's own
    // rule AND are mutually diverse, so MMR keeps them and the own rule is
    // genuinely crowded out of the head without the boost.
    for i in 0..6 {
        let node = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            format!("Shared account note {i} about component styling work"),
            vec_with(0, 0.70, 1 + (i % 6), 0.70),
        );
        db.store(node).unwrap();
    }
    let mut own = MemoryNode::new(
        me,
        MemoryType::Procedural,
        "Own rule: style components with kialiStyle only".to_string(),
        vec_with(0, 0.62, 7, 0.78),
    );
    own.tags = vec!["source:agent-file".to_string()];
    db.store(own).unwrap();
}

fn got_scored(db: &MenteDb, me: AgentId, k: usize) -> Vec<(String, f32)> {
    let turn = vec_with(0, 1.0, 5, 0.0);
    let query = InjectionQuery {
        embedding: &turn,
        query_text: None,
        session_id: None,
        exclude_ids: &[],
        max_items: k,
        max_episodic: 2,
        agent_id: Some(me),
        user_id: None,
        current_project: None,
    };
    db.recall_for_injection(&query)
        .unwrap()
        .into_iter()
        .map(|c| (c.node.content, c.score))
        .collect()
}

fn got(db: &MenteDb, me: AgentId, k: usize) -> Vec<String> {
    let turn = vec_with(0, 1.0, 5, 0.0);
    let query = InjectionQuery {
        embedding: &turn,
        query_text: None,
        session_id: None,
        exclude_ids: &[],
        max_items: k,
        max_episodic: 2,
        agent_id: Some(me),
        user_id: None,
        current_project: None,
    };
    db.recall_for_injection(&query)
        .unwrap()
        .into_iter()
        .map(|c| c.node.content)
        .collect()
}

#[test]
fn own_rule_outranks_shared_noise_with_boost() {
    let dir = tempfile::tempdir().unwrap();
    let me = AgentId::new();
    let db = open_with(dir.path(), 1.25);
    seed(&db, me);
    let out = got(&db, me, 3);
    assert!(
        out.iter().any(|c| c.contains("kialiStyle")),
        "the agent's own rule must reach the head slots: {out:?}"
    );
}

#[test]
fn boost_multiplies_owned_scores_only() {
    let dir = tempfile::tempdir().unwrap();
    let me = AgentId::new();
    let db = open_with(dir.path(), 2.0);
    seed(&db, me);
    let boosted = got_scored(&db, me, 7);

    let dir2 = tempfile::tempdir().unwrap();
    let db2 = open_with(dir2.path(), 1.0);
    seed(&db2, me);
    let base = got_scored(&db2, me, 7);

    let find = |v: &Vec<(String, f32)>, needle: &str| {
        v.iter().find(|(c, _)| c.contains(needle)).map(|(_, s)| *s)
    };
    let own_b = find(&boosted, "kialiStyle").expect("own rule present when boosted");
    let own_0 = find(&base, "kialiStyle").expect("own rule present at baseline");
    assert!(
        (own_b / own_0 - 2.0).abs() < 0.05,
        "own rule score must scale by the boost: {own_0} -> {own_b}"
    );
    let sh_b = find(&boosted, "note 0").expect("shared note present");
    let sh_0 = find(&base, "note 0").expect("shared note present");
    assert!(
        (sh_b / sh_0 - 1.0).abs() < 0.05,
        "shared nil owned scores must not change: {sh_0} -> {sh_b}"
    );
}
