//! Tests for the `ORDER BY salience DESC LIMIT k` top k walk: it must
//! return exactly what the full path (load every memory in page order,
//! filter, stable sort by decayed salience, truncate) returns, ties included,
//! while loading only the memories that can reach the top k.

use super::*;
use mentedb_core::memory::AttributeValue;

const BASE_TS: u64 = 1_000_000_000;
const HALF_LIFE: u64 = 1_000_000;
/// Every query runs at this instant.
const NOW: u64 = BASE_TS + 10 * HALF_LIFE;

const TYPES: [MemoryType; 4] = [
    MemoryType::Episodic,
    MemoryType::Semantic,
    MemoryType::Procedural,
    MemoryType::Correction,
];

fn config() -> CognitiveConfig {
    CognitiveConfig {
        write_inference: false,
        memory_dedup: false,
        decay_config: DecayConfig {
            half_life_us: HALF_LIFE,
            ..DecayConfig::default()
        },
        ..CognitiveConfig::default()
    }
}

/// Memory `i` of the mixed fixture. Kinds by `i % 8`:
/// 0 old high base (decays to a fraction), 1 fresh low base (no decay),
/// 2 exempt by attribute, 3 exempt by tag at the fresh tie value,
/// 4 old low base lifted by a large access count (effective above base),
/// 5 base below the decay floor (effective = the floor),
/// 6 tie group (identical base and access time), 7 spread values.
fn mixed_node(i: usize) -> MemoryNode {
    let mut node = MemoryNode::new(
        AgentId::nil(),
        TYPES[i % TYPES.len()],
        format!("mixed memory {i}"),
        vec![1.0, (i % 7) as f32, 0.0, 0.5],
    );
    node.created_at = BASE_TS + i as u64;
    node.tags = vec![format!("g{}", i % 5)];
    if i % 2 == 0 {
        node.tags.push("even".into());
    }
    let spread = (i as f32 + 1.0) / 1000.0;
    match i % 8 {
        0 => {
            node.salience = 0.95;
            node.accessed_at = BASE_TS + (i as u64 % 4) * HALF_LIFE;
        }
        1 => {
            node.salience = 0.2 + spread * 0.1;
            node.accessed_at = NOW;
        }
        2 => {
            node.salience = 0.4 + spread * 0.1;
            node.accessed_at = BASE_TS;
            node.attributes
                .insert("decay_exempt".into(), AttributeValue::Boolean(true));
        }
        3 => {
            node.salience = 0.25;
            node.accessed_at = BASE_TS;
            node.tags.push("scope:always".into());
        }
        4 => {
            node.salience = 0.05;
            node.accessed_at = BASE_TS;
            node.access_count = 50 + i as u32;
        }
        5 => {
            node.salience = 0.001;
            node.accessed_at = BASE_TS;
        }
        6 => {
            node.salience = 0.25;
            node.accessed_at = NOW;
        }
        _ => {
            node.salience = 0.1 + spread * 0.5;
            node.accessed_at = BASE_TS + (i as u64 % 10) * HALF_LIFE;
        }
    }
    node
}

fn mixed_db(dir: &Path, n: usize) -> MenteDb {
    let db = MenteDb::open_with_config(dir, config()).unwrap();
    db.store_batch((0..n).map(mixed_node).collect()).unwrap();
    db
}

/// The full path, written out: every memory in page order, the WHERE clause,
/// a stable sort by decayed salience, truncate.
fn reference(db: &MenteDb, mql: &str) -> Vec<MemoryId> {
    let plan = Mql::parse(mql).unwrap();
    let (filters, condition, limit, order_by, range) = match plan {
        QueryPlan::TagScan {
            tags,
            filters,
            limit,
            condition,
            order_by,
        } => {
            assert!(
                tags.is_empty(),
                "{mql}: tag-only queries take the hybrid path"
            );
            (filters, condition, limit, order_by, None)
        }
        QueryPlan::TemporalScan {
            start,
            end,
            filters,
            limit,
            order_by,
        } => (filters, None, limit, order_by, Some((start, end))),
        other => panic!("{mql}: unexpected plan {other:?}"),
    };
    let k = limit.unwrap_or(db.cognitive_config.mql_default_limit);
    let mut pages: Vec<PageId> = db.page_map.read().values().copied().collect();
    pages.sort_unstable_by_key(|p| p.0);
    let mut scored: Vec<ScoredMemory> = pages
        .into_iter()
        .filter_map(|p| db.storage.load_memory(p).ok())
        .filter(|m| range.is_none_or(|(s, e)| m.created_at >= s && m.created_at <= e))
        .filter(|m| match &condition {
            Some(c) => db.condition_matches(c, m, NOW),
            None => filters.iter().all(|f| db.filter_matches(f, m, NOW)),
        })
        .map(|memory| ScoredMemory { memory, score: 1.0 })
        .collect();
    let ob = order_by.unwrap_or_else(|| panic!("{mql}: expected ORDER BY"));
    db.apply_order_by(&mut scored, &ob, NOW);
    scored.truncate(k);
    scored.into_iter().map(|s| s.memory.id).collect()
}

fn run(db: &MenteDb, mql: &str) -> Vec<MemoryId> {
    db.query_at(mql, NOW)
        .unwrap_or_else(|e| panic!("{mql}: {e}"))
        .into_iter()
        .map(|s| s.memory.id)
        .collect()
}

const FILTERS: [&str; 9] = [
    "",
    "WHERE type = episodic",
    "WHERE type != correction AND tag = \"even\"",
    "WHERE type IN [semantic, correction] AND tag = \"g2\"",
    "WHERE salience > 0.2",
    "WHERE type = episodic OR tag = \"g3\"",
    "WHERE NOT tag = \"even\"",
    "WHERE content CONTAINS \"1\"",
    "WHERE type = procedural AND salience < 0.3",
];

const KS: [usize; 6] = [1, 3, 5, 17, 60, 1000];

fn assert_all_match(db: &MenteDb, label: &str) {
    for f in FILTERS {
        for k in KS {
            let mql = format!("RECALL memories {f} ORDER BY salience DESC LIMIT {k}");
            let want = reference(db, &mql);
            assert_eq!(run(db, &mql), want, "{label}: {mql}");
        }
    }
}

#[test]
fn fixture_has_the_cases_the_walk_must_handle() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed_db(dir.path(), 16);
    let nodes: Vec<MemoryNode> = (0..16).map(mixed_node).collect();
    let eff = |n: &MemoryNode| db.decayed_salience_at(n, NOW);
    // Access bonus and decay floor lift the effective value above the base.
    assert!(eff(&nodes[4]) > nodes[4].salience);
    assert!(eff(&nodes[5]) > nodes[5].salience);
    // Old high base decays below a fresh lower base.
    assert!(eff(&nodes[8]) < eff(&nodes[9]));
    // Exact ties across a decaying and an exempt memory.
    assert_eq!(eff(&nodes[6]), eff(&nodes[3]));
    assert_eq!(eff(&nodes[6]), eff(&nodes[14]));
}

#[test]
fn bounded_walk_equals_the_full_path() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed_db(dir.path(), 400);
    // Cold: every bound unknown. Warm: every bound cached.
    assert_all_match(&db, "cold");
    assert_all_match(&db, "warm");
}

#[test]
fn bounded_walk_equals_the_full_path_on_temporal_scans() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed_db(dir.path(), 400);
    for round in ["cold", "warm"] {
        for k in KS {
            let mql = format!(
                "RECALL memories WHERE created > {} AND created < {} ORDER BY salience DESC LIMIT {k}",
                BASE_TS + 50,
                BASE_TS + 300
            );
            assert_eq!(run(&db, &mql), reference(&db, &mql), "{round}: {mql}");
        }
    }
}

#[test]
fn writes_invalidate_cached_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed_db(dir.path(), 400);
    assert_all_match(&db, "before writes");

    // Raise a floor memory's base to the top, in place.
    let low = db
        .query_at(
            "RECALL memories WHERE content CONTAINS \"mixed memory 13\" LIMIT 1",
            NOW,
        )
        .unwrap()[0]
        .memory
        .clone();
    let mut raised = low.clone();
    raised.salience = 1.0;
    raised.accessed_at = NOW;
    let page = db.page_of(raised.id).unwrap();
    db.update_node(page, &raised).unwrap();
    let top = run(&db, "RECALL memories ORDER BY salience DESC LIMIT 1");
    assert_eq!(top, vec![raised.id]);

    // Access statistics: a decayed memory gains a large access bonus.
    let old = db
        .query_at(
            "RECALL memories WHERE content CONTAINS \"mixed memory 31\" LIMIT 1",
            NOW,
        )
        .unwrap()[0]
        .memory
        .id;
    for _ in 0..20 {
        db.note_access(std::iter::repeat_n(old, 1000));
    }
    db.flush().unwrap();
    assert_all_match(&db, "after writes");
}

/// 2000 memories with distinct bases, all accessed at the query instant, so
/// decayed salience equals the base and the bound is exact.
fn fresh_db(dir: &Path, n: usize) -> MenteDb {
    let db = MenteDb::open_with_config(dir, config()).unwrap();
    let nodes = (0..n)
        .map(|i| {
            let mut node = MemoryNode::new(
                AgentId::nil(),
                TYPES[i % TYPES.len()],
                format!("fresh memory {i}"),
                vec![1.0, 0.0, (i % 5) as f32, 0.0],
            );
            node.salience = 0.05 + 0.9 * (i as f32 + 1.0) / (n as f32 + 1.0);
            node.created_at = BASE_TS + i as u64;
            node.accessed_at = NOW;
            node
        })
        .collect();
    db.store_batch(nodes).unwrap();
    db
}

fn loads_for(db: &MenteDb, mql: &str) -> (Vec<MemoryId>, u64) {
    let before = db.storage_memory_loads();
    let got = run(db, mql);
    (got, db.storage_memory_loads() - before)
}

#[test]
fn order_by_salience_desc_limit_loads_only_the_top() {
    const N: usize = 2000;
    let dir = tempfile::tempdir().unwrap();
    let db = fresh_db(dir.path(), N);

    let mql = "RECALL memories ORDER BY salience DESC LIMIT 5";
    // Cold: no bound is known yet, so every memory loads once.
    let (cold, cold_loads) = loads_for(&db, mql);
    assert_eq!(cold, reference(&db, mql));
    assert_eq!(cold_loads, N as u64);

    // Warm: the top 5 load, then the 6th bound is below the 5th result.
    let (warm, warm_loads) = loads_for(&db, mql);
    assert_eq!(warm, cold);
    assert_eq!(warm_loads, 5, "loaded {warm_loads} of {N}");

    // With an indexed filter the walk visits only that type's top.
    let typed = "RECALL memories WHERE type = episodic ORDER BY salience DESC LIMIT 5";
    let (got, loads) = loads_for(&db, typed);
    assert_eq!(got, reference(&db, typed));
    assert_eq!(loads, 5, "loaded {loads} of {}", N / 4);
}

#[test]
fn bounds_survive_reopen() {
    const N: usize = 2000;
    let dir = tempfile::tempdir().unwrap();
    let mql = "RECALL memories ORDER BY salience DESC LIMIT 5";
    let want = {
        let db = fresh_db(dir.path(), N);
        let got = run(&db, mql);
        db.close().unwrap();
        got
    };
    let db = MenteDb::open_with_config(dir.path(), config()).unwrap();
    let (got, loads) = loads_for(&db, mql);
    assert_eq!(got, want);
    assert_eq!(loads, 5, "loaded {loads} of {N} after reopen");
}

#[test]
fn ascending_order_takes_the_full_path() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed_db(dir.path(), 200);
    for f in FILTERS {
        let mql = format!("RECALL memories {f} ORDER BY salience ASC LIMIT 7");
        assert_eq!(run(&db, &mql), reference(&db, &mql), "{mql}");
    }
}
