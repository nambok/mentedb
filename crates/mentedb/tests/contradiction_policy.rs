//! Contradiction policy, end to end: the same contradicting pair either closes
//! the older memory (Supersede, the default) or keeps both valid and records a
//! Contradicts edge for the caller to resolve (Flag).

use mentedb::prelude::*;
use mentedb::{CognitiveConfig, MenteDb};
use mentedb_cognitive::ContradictionPolicy;

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

/// Store "cortado" then "flat white" on the same sentence frame, the shape the
/// value update rule treats as a contradiction. Returns (old_id, new_id).
fn store_pair(db: &MenteDb) -> (MemoryId, MemoryId) {
    let agent = AgentId::new();
    let mut old = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The user's favorite coffee order is a cortado".into(),
        vec![1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    old.created_at = now_us();
    let old_id = old.id;
    db.store(old).unwrap();

    let mut new = MemoryNode::new(
        agent,
        MemoryType::Semantic,
        "The user's favorite coffee order is a flat white".into(),
        vec![0.99, 0.08, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
    );
    new.created_at = now_us() + 1;
    let new_id = new.id;
    db.store(new).unwrap();
    (old_id, new_id)
}

fn recall_ids(db: &MenteDb) -> Vec<MemoryId> {
    db.recall_hybrid_at(
        &[1.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Some("favorite coffee order"),
        5,
        now_us() + 10,
        None,
        None,
        None,
    )
    .unwrap()
    .iter()
    .map(|(id, _)| *id)
    .collect()
}

fn edge_types_between(db: &MenteDb, a: MemoryId, b: MemoryId) -> Vec<EdgeType> {
    let graph = db.graph().graph();
    graph
        .outgoing(a)
        .into_iter()
        .filter(|(t, _)| *t == b)
        .chain(graph.outgoing(b).into_iter().filter(|(t, _)| *t == a))
        .map(|(_, e)| e.edge_type)
        .collect()
}

fn config(policy: ContradictionPolicy) -> CognitiveConfig {
    let mut cfg = CognitiveConfig::default();
    cfg.inference_config.contradiction_policy = policy;
    cfg
}

#[test]
fn default_policy_is_supersede() {
    assert_eq!(
        CognitiveConfig::default()
            .inference_config
            .contradiction_policy,
        ContradictionPolicy::Supersede
    );
}

#[test]
fn supersede_closes_the_older_memory() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_config(dir.path(), config(ContradictionPolicy::Supersede)).unwrap();
    let (old_id, new_id) = store_pair(&db);

    let old = db.get_memory(old_id).unwrap();
    assert!(
        old.valid_until.is_some(),
        "supersede must set valid_until on the older memory"
    );
    assert!(db.get_memory(new_id).unwrap().valid_until.is_none());

    let ids = recall_ids(&db);
    assert!(ids.contains(&new_id), "the newer memory must be recallable");
    assert!(
        !ids.contains(&old_id),
        "the superseded memory must be excluded from recall"
    );

    let edges = edge_types_between(&db, old_id, new_id);
    assert!(edges.contains(&EdgeType::Supersedes), "got {edges:?}");
    assert!(!edges.contains(&EdgeType::Contradicts), "got {edges:?}");
}

#[test]
fn flag_keeps_both_valid_and_records_a_contradicts_edge() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_config(dir.path(), config(ContradictionPolicy::Flag)).unwrap();
    let (old_id, new_id) = store_pair(&db);

    assert!(
        db.get_memory(old_id).unwrap().valid_until.is_none(),
        "flag must leave the older memory valid"
    );
    assert!(db.get_memory(new_id).unwrap().valid_until.is_none());

    let ids = recall_ids(&db);
    assert!(ids.contains(&new_id), "the newer memory must be recallable");
    assert!(
        ids.contains(&old_id),
        "the older memory must stay recallable"
    );

    let edges = edge_types_between(&db, old_id, new_id);
    assert!(edges.contains(&EdgeType::Contradicts), "got {edges:?}");
    assert!(!edges.contains(&EdgeType::Supersedes), "got {edges:?}");
}

#[test]
fn policy_deserializes_and_old_configs_still_load() {
    let cfg: mentedb_cognitive::WriteInferenceConfig =
        serde_json::from_str(r#"{"contradiction_threshold": 0.9}"#).unwrap();
    assert_eq!(cfg.contradiction_policy, ContradictionPolicy::Supersede);
    assert_eq!(cfg.contradiction_threshold, 0.9);

    let cfg: mentedb_cognitive::WriteInferenceConfig =
        serde_json::from_str(r#"{"contradiction_policy": "flag"}"#).unwrap();
    assert_eq!(cfg.contradiction_policy, ContradictionPolicy::Flag);
}
