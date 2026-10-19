//! Cold open serves the page map from the persisted memory map (plus the WAL)
//! instead of loading every page, and mode probes survive a reopen so the
//! first injection recall does not load every mode rule.

use std::collections::HashSet;

use mentedb::injection::{InjectionConfig, InjectionQuery, SelectionReason};
use mentedb::prelude::*;
use mentedb::{CognitiveConfig, MenteDb};
use mentedb_core::types::{AgentId, MemoryId};

fn config() -> CognitiveConfig {
    CognitiveConfig {
        write_inference: false,
        ..CognitiveConfig::default()
    }
}

fn open(dir: &std::path::Path) -> MenteDb {
    MenteDb::open_with_config(dir, config()).unwrap()
}

fn node(i: usize) -> MemoryNode {
    let mut emb = vec![0.0f32; 8];
    emb[i % 8] = 1.0;
    emb[(i / 8) % 8] += 0.25;
    MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        format!("fact {i}"),
        emb,
    )
}

fn store_many(db: &MenteDb, range: std::ops::Range<usize>) -> Vec<MemoryId> {
    let nodes: Vec<MemoryNode> = range.map(node).collect();
    db.store_batch(nodes).unwrap()
}

fn ids(db: &MenteDb) -> HashSet<MemoryId> {
    db.memory_ids().into_iter().collect()
}

#[test]
fn reopen_after_close_reads_no_pages() {
    let dir = tempfile::tempdir().unwrap();
    let stored: HashSet<MemoryId> = {
        let db = open(dir.path());
        let ids = store_many(&db, 0..600);
        db.close().unwrap();
        ids.into_iter().collect()
    };
    let db = open(dir.path());
    assert_eq!(ids(&db), stored);
    assert_eq!(
        db.storage_memory_loads(),
        0,
        "open must not load memories it already knows"
    );
    assert_eq!(db.storage_page_reads(), 0, "open must not read pages");
}

#[test]
fn crash_after_checkpoint_keeps_later_stores_and_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let mut expected: HashSet<MemoryId>;
    {
        let db = open(dir.path());
        let base = store_many(&db, 0..300);
        db.flush_full().unwrap();
        // After the checkpoint: new stores, deletes of old and new memories,
        // and a store that reuses a freed page.
        let mut later = Vec::new();
        for i in 300..305 {
            let n = node(i);
            later.push(n.id);
            db.store(n).unwrap();
        }
        db.forget(base[0]).unwrap();
        db.forget(base[1]).unwrap();
        db.forget(later[0]).unwrap();
        let reuse = node(999);
        let reuse_id = reuse.id;
        db.store(reuse).unwrap();
        expected = base[2..].iter().copied().collect();
        expected.extend(later[1..].iter().copied());
        expected.insert(reuse_id);
        assert_eq!(ids(&db), expected);
        db.simulate_crash();
    }
    let db = open(dir.path());
    assert_eq!(
        ids(&db),
        expected,
        "WAL tail replayed onto the persisted map"
    );
    let reads = db.storage_page_reads();
    assert!(
        reads < 40,
        "open read {reads} pages; the WAL touched only a handful of 300"
    );
    for id in &expected {
        db.get_memory(*id).unwrap();
    }
}

#[test]
fn corrupt_or_missing_memory_map_falls_back_to_scan() {
    let dir = tempfile::tempdir().unwrap();
    let stored: HashSet<MemoryId> = {
        let db = open(dir.path());
        let ids = store_many(&db, 0..120);
        db.close().unwrap();
        ids.into_iter().collect()
    };
    let snap = dir
        .path()
        .join(mentedb_storage::memory_map::MEMORY_MAP_FILE);
    let mut bytes = std::fs::read(&snap).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xa5;
    std::fs::write(&snap, &bytes).unwrap();
    {
        let db = open(dir.path());
        assert_eq!(ids(&db), stored, "corrupt map: same result via scan");
        assert!(db.storage_page_reads() >= 120);
        db.close().unwrap();
    }
    std::fs::remove_file(&snap).unwrap();
    {
        let db = open(dir.path());
        assert_eq!(ids(&db), stored, "missing map: same result via scan");
        db.close().unwrap();
    }
    let db = open(dir.path());
    assert_eq!(ids(&db), stored);
    assert_eq!(db.storage_page_reads(), 0, "the scan re-persisted the map");
}

#[test]
fn takeover_by_another_handle_is_seen_on_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let mut expected: HashSet<MemoryId> = HashSet::new();
    {
        let db = open(dir.path());
        expected.extend(store_many(&db, 0..50));
        db.close_quick().unwrap();
    }
    {
        // Another process takes over: writes, checkpoints, writes, dies.
        let db = open(dir.path());
        expected.extend(store_many(&db, 50..80));
        db.flush().unwrap();
        let gone = *expected.iter().next().unwrap();
        db.forget(gone).unwrap();
        expected.remove(&gone);
        expected.extend(store_many(&db, 80..85));
        db.simulate_crash();
    }
    let db = open(dir.path());
    assert_eq!(ids(&db), expected);
}

// ---- mode probes ----

const DIM: usize = 16;
const TRIGGERS: usize = 100;
const RULES_PER_TRIGGER: usize = 10;

/// Two-hot direction for trigger `t`; distinct for every t < 100.
fn trigger_dir(t: usize) -> Vec<f32> {
    let a = t % DIM;
    let b = (a + 1 + t / DIM) % DIM;
    let mut v = vec![0.0; DIM];
    v[a] = 1.0;
    v[b] = 1.0;
    v
}

fn mode_config() -> CognitiveConfig {
    CognitiveConfig {
        write_inference: false,
        injection_config: InjectionConfig {
            mode_trigger_gate: 0.9,
            mode_rules_max: 4,
            ..InjectionConfig::default()
        },
        ..CognitiveConfig::default()
    }
}

fn mode_rules(db: &MenteDb, embedding: &[f32]) -> Vec<String> {
    let query = InjectionQuery {
        embedding,
        query_text: None,
        session_id: None,
        exclude_ids: &[],
        max_items: 6,
        max_episodic: 2,
        agent_id: None,
        user_id: None,
        current_project: None,
    };
    db.recall_for_injection(&query)
        .unwrap()
        .into_iter()
        .filter(|c| c.reason == SelectionReason::ModeRule)
        .map(|c| c.node.content)
        .collect()
}

/// Loads `f` caused, with its result.
fn counted<T>(db: &MenteDb, f: impl FnOnce() -> T) -> (T, u64) {
    let before = db.storage_memory_loads();
    let out = f();
    (out, db.storage_memory_loads() - before)
}

fn rule_db(dir: &std::path::Path) -> (MenteDb, Vec<MemoryId>) {
    let db = MenteDb::open_with_config(dir, mode_config()).unwrap();
    let mut nodes = Vec::new();
    for t in 0..TRIGGERS {
        for r in 0..RULES_PER_TRIGGER {
            let mut n = MemoryNode::new(
                AgentId::nil(),
                MemoryType::Procedural,
                format!("rule {r} of mode {t}"),
                trigger_dir(t),
            );
            n.tags = vec![format!("trigger:mode{t}")];
            nodes.push(n);
        }
    }
    let ids = db.store_batch(nodes).unwrap();
    (db, ids)
}

#[test]
fn mode_probes_survive_reopen_without_loading_every_rule() {
    let dir = tempfile::tempdir().unwrap();
    let query = trigger_dir(37);
    let (warm, warm_loads) = {
        let (db, _) = rule_db(dir.path());
        // Fill the probe cache (the per request budget spreads a cold fill
        // over a few requests).
        for _ in 0..6 {
            mode_rules(&db, &query);
        }
        let (warm, loads) = counted(&db, || mode_rules(&db, &query));
        db.close().unwrap();
        (warm, loads)
    };
    assert_eq!(warm.len(), 4);
    assert!(warm.iter().all(|c| c.ends_with("of mode 37")), "{warm:?}");

    let db = MenteDb::open_with_config(dir.path(), mode_config()).unwrap();
    let (cold, cold_loads) = counted(&db, || mode_rules(&db, &query));
    assert_eq!(
        cold, warm,
        "the first recall after open activates the same modes"
    );
    assert!(
        cold_loads <= warm_loads,
        "first recall after open loaded {cold_loads} memories, a warm one {warm_loads}"
    );
}

#[test]
fn a_probe_written_after_its_snapshot_is_reloaded_not_trusted() {
    let dir = tempfile::tempdir().unwrap();
    let query = trigger_dir(5);
    let ids = {
        let (db, ids) = rule_db(dir.path());
        for _ in 0..6 {
            mode_rules(&db, &query);
        }
        db.close().unwrap();
        ids
    };
    // Reopen (probes restored), then retire every rule of the active mode
    // and crash before any snapshot: the probe file still holds them valid.
    let mode5: Vec<MemoryId> = ids[5 * RULES_PER_TRIGGER..6 * RULES_PER_TRIGGER].to_vec();
    {
        let db = MenteDb::open_with_config(dir.path(), mode_config()).unwrap();
        assert_eq!(mode_rules(&db, &query).len(), 4);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        for id in &mode5 {
            db.invalidate_memory(*id, now).unwrap();
        }
        assert!(mode_rules(&db, &query).is_empty());
        db.simulate_crash();
    }
    let db = MenteDb::open_with_config(dir.path(), mode_config()).unwrap();
    let got = mode_rules(&db, &query);
    assert!(
        got.is_empty(),
        "retired rules must not activate from stale persisted probes: {got:?}"
    );
}

#[test]
fn missing_probe_snapshot_is_bounded_by_the_load_budget() {
    let dir = tempfile::tempdir().unwrap();
    let query = trigger_dir(37);
    {
        let (db, _) = rule_db(dir.path());
        db.close().unwrap();
    }
    let _ = std::fs::remove_file(dir.path().join("mode_probes.bin"));
    let db = MenteDb::open_with_config(dir.path(), mode_config()).unwrap();
    let budget = InjectionConfig::default().mode_probe_load_budget as u64;
    let (_, loads) = counted(&db, || mode_rules(&db, &query));
    // Later requests finish the fill and then agree with a full computation.
    for _ in 0..6 {
        mode_rules(&db, &query);
    }
    let (got, warm) = counted(&db, || mode_rules(&db, &query));
    // A warm request's loads are the hybrid candidates plus the kept rules;
    // a cold one may add at most the budget of probe loads on top.
    assert!(
        loads <= warm + budget,
        "one cold request loaded {loads}, warm {warm}, budget {budget}"
    );
    assert!(loads < (TRIGGERS * RULES_PER_TRIGGER) as u64);
    assert_eq!(got.len(), 4);
    assert!(got.iter().all(|c| c.ends_with("of mode 37")), "{got:?}");
}
