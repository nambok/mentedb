//! Cluster completion: when one section dominates the selected head, the
//! turn lives inside that cluster and the section's weak siblings must
//! arrive too, even past the relevance gates. When no section dominates,
//! selection is untouched.

use mentedb::injection::{InjectionConfig, InjectionQuery};
use mentedb::prelude::*;
use mentedb::{CognitiveConfig, MenteDb};
use mentedb_core::types::AgentId;

const DIM: usize = 8;

fn axis(i: usize, weight: f32) -> Vec<f32> {
    let mut v = vec![0.0; DIM];
    v[i] = weight;
    v
}

fn open_with(dir: &std::path::Path, fraction: f32) -> MenteDb {
    let config = CognitiveConfig {
        injection_config: InjectionConfig {
            cluster_dominant_fraction: fraction,
            ..InjectionConfig::default()
        },
        ..CognitiveConfig::default()
    };
    MenteDb::open_with_config(dir, config).unwrap()
}

fn node(content: &str, embedding: Vec<f32>, tags: &[&str]) -> MemoryNode {
    let mut n = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Procedural,
        content.to_string(),
        embedding,
    );
    n.tags = tags.iter().map(|t| t.to_string()).collect();
    n
}

fn seed(db: &MenteDb) {
    // Four strong cluster leaders and three medium off cluster distractors.
    for i in 0..4 {
        db.store(node(
            &format!("API sibling rule {i} about payload naming"),
            axis(0, 1.0),
            &["section:app-server-api"],
        ))
        .unwrap();
    }
    for i in 0..3 {
        let mut v = vec![0.0; DIM];
        v[0] = 0.62;
        v[5 + (i % 3).min(2)] = 0.78;
        db.store(node(&format!("Unrelated working note {i}"), v, &[]))
            .unwrap();
    }
    // The governing sibling: same section, but its own embedding sits at
    // cosine 0.36 to the turn, under the 0.55 relevance tail gate.
    let mut v = vec![0.0; DIM];
    v[0] = 0.35;
    v[4] = 0.90;
    db.store(node(
        "Wire fields are camelCase via serde rename_all",
        v,
        &["section:app-server-api"],
    ))
    .unwrap();
}

fn contents(db: &MenteDb, k: usize) -> Vec<String> {
    let turn = axis(0, 1.0);
    let query = InjectionQuery {
        embedding: &turn,
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
        .map(|c| c.node.content)
        .collect()
}

#[test]
fn dominant_cluster_pulls_its_gated_sibling_in() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with(dir.path(), 0.34);
    seed(&db);
    let got = contents(&db, 6);
    assert!(
        got.iter().any(|c| c.contains("camelCase")),
        "the gated sibling must arrive once its section dominates: {got:?}"
    );
}

#[test]
fn disabled_completion_leaves_gates_in_charge() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with(dir.path(), 0.0);
    seed(&db);
    let got = contents(&db, 6);
    assert!(
        !got.iter().any(|c| c.contains("camelCase")),
        "without completion the weak sibling stays gated: {got:?}"
    );
}

#[test]
fn no_dominant_section_means_no_swap() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with(dir.path(), 0.34);
    // Every atom in its own section: nothing dominates, output matches the
    // plain selection exactly.
    // Distinct directions (not magnitudes: cosine ignores magnitude) so the
    // six rules have strictly ordered similarities and both stores rank
    // them identically without leaning on tie-break randomness.
    for i in 0..6 {
        let mut v = axis(0, 1.0);
        v[7] = 0.1 * (i + 1) as f32;
        db.store(node(
            &format!("Standalone rule {i}"),
            v,
            &[&format!("section:sec-{i}")],
        ))
        .unwrap();
    }
    let mut v = vec![0.0; DIM];
    v[0] = 0.30;
    v[4] = 0.9;
    db.store(node("Far away rule", v, &["section:sec-far"]))
        .unwrap();
    let got = contents(&db, 5);

    let dir2 = tempfile::tempdir().unwrap();
    let db2 = open_with(dir2.path(), 0.0);
    for i in 0..6 {
        let mut v = axis(0, 1.0);
        v[7] = 0.1 * (i + 1) as f32;
        db2.store(node(
            &format!("Standalone rule {i}"),
            v,
            &[&format!("section:sec-{i}")],
        ))
        .unwrap();
    }
    let mut v = vec![0.0; DIM];
    v[0] = 0.30;
    v[4] = 0.9;
    db2.store(node("Far away rule", v, &["section:sec-far"]))
        .unwrap();
    let base = contents(&db2, 5);
    assert_eq!(got, base, "no dominance means output identical to disabled");
}
