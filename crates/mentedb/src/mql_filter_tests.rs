//! Regression tests for MQL filters on the owner and score fields (agent,
//! space, salience, confidence) and for temporal scans, which must apply
//! every remaining filter and honor LIMIT and ORDER BY.
//!
//! Expected results are computed from the stored memories with predicates
//! written here, independently of `filter_matches`, so a filter that matches
//! every memory fails these tests. Queries run at the fixed instant `NOW`, and
//! salience predicates read the decayed salience at that instant, the value
//! MQL `salience` compares.

use std::collections::HashSet;

use super::*;
use mentedb_core::types::SpaceId;

const N: usize = 600;
const BASE_TS: u64 = 1_000_000;
const STEP: u64 = 1_000;
/// The instant every fixture query is evaluated at: just after the newest
/// access, so memory `i` was last accessed `(i + 1) * STEP` before it.
const NOW: u64 = BASE_TS + (N as u64 + 1) * STEP;
/// Half-life of the fixture's decay: long enough that decayed salience stays
/// spread out (the oldest access keeps about 84% of its base), short enough
/// that it differs from the stored base for every memory but the newest.
const HALF_LIFE: u64 = 4 * N as u64 * STEP;

const TYPES: [MemoryType; 4] = [
    MemoryType::Episodic,
    MemoryType::Semantic,
    MemoryType::Procedural,
    MemoryType::Correction,
];

struct Fixture {
    db: MenteDb,
    agents: [AgentId; 2],
    /// Owns only one memory in 60, spread through the store.
    rare_agent: AgentId,
    spaces: [SpaceId; 2],
}

fn uuid(i: u128) -> uuid::Uuid {
    uuid::Uuid::from_u128(0x1000 + i)
}

/// Memory `i`: type cycles, created and accessed run in opposite directions,
/// salience and confidence are distinct, two of each carry a round value for
/// equality tests, and owners and spaces vary.
fn fixture(dir: &Path) -> Fixture {
    let config = CognitiveConfig {
        write_inference: false,
        memory_dedup: false,
        decay_config: DecayConfig {
            half_life_us: HALF_LIFE,
            ..DecayConfig::default()
        },
        ..CognitiveConfig::default()
    };
    let db = MenteDb::open_with_config(dir, config).unwrap();
    let agents = [AgentId::from(uuid(1)), AgentId::from(uuid(2))];
    let rare_agent = AgentId::from(uuid(3));
    let spaces = [SpaceId::from(uuid(10)), SpaceId::from(uuid(11))];
    let dim = 16;
    let mut batch = Vec::new();
    for i in 0..N {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        emb[(i * 7 + 3) % dim] += 0.5;
        let agent = if i % 60 == 59 {
            rare_agent
        } else {
            agents[i % 2]
        };
        let mut node = MemoryNode::new(
            agent,
            TYPES[i % TYPES.len()],
            format!("fixture memory {i}"),
            emb,
        );
        node.space_id = if i % 5 == 0 {
            SpaceId::nil()
        } else {
            spaces[i % 2]
        };
        node.created_at = BASE_TS + i as u64 * STEP;
        node.accessed_at = BASE_TS + (N - i) as u64 * STEP;
        node.salience = (i as f32 + 1.0) / (N as f32 + 1.0);
        node.confidence = (((i * 37) % N) as f32 + 1.0) / (N as f32 + 1.0);
        if i == 7 || i == 300 {
            node.salience = 0.3;
        }
        if i == 11 || i == 400 {
            node.confidence = 0.8;
        }
        batch.push(node);
    }
    db.store_batch(batch).unwrap();
    Fixture {
        db,
        agents,
        rare_agent,
        spaces,
    }
}

/// Every stored memory in page order. Page ids are copied out first, so the
/// page map is not held across the loads.
fn all_nodes(db: &MenteDb) -> Vec<MemoryNode> {
    let mut pages: Vec<PageId> = db.page_map.read().values().copied().collect();
    pages.sort_unstable_by_key(|p| p.0);
    pages
        .into_iter()
        .filter_map(|p| db.storage.load_memory(p).ok())
        .collect()
}

fn expected(db: &MenteDb, pred: impl Fn(&MemoryNode) -> bool) -> HashSet<MemoryId> {
    all_nodes(db)
        .into_iter()
        .filter(|n| pred(n))
        .map(|n| n.id)
        .collect()
}

fn ids(db: &MenteDb, mql: &str) -> Vec<MemoryId> {
    db.query_at(mql, NOW)
        .unwrap_or_else(|e| panic!("{mql}: {e}"))
        .into_iter()
        .map(|s| s.memory.id)
        .collect()
}

fn id_set(db: &MenteDb, mql: &str) -> HashSet<MemoryId> {
    let got = ids(db, mql);
    let set: HashSet<MemoryId> = got.iter().copied().collect();
    assert_eq!(set.len(), got.len(), "{mql}: no duplicates");
    set
}

/// The query returns exactly the memories `pred` admits, and that set is a
/// real narrowing (neither empty nor every memory).
fn assert_query(db: &MenteDb, mql: &str, pred: impl Fn(&MemoryNode) -> bool) {
    let want = expected(db, pred);
    assert!(!want.is_empty(), "{mql}: fixture admits nothing");
    assert!(want.len() < N, "{mql}: fixture admits everything");
    assert_eq!(id_set(db, mql), want, "{mql}");
}

fn f32_of(x: f64) -> f32 {
    x as f32
}

/// The decayed salience of `n` at `NOW`: what MQL `salience` compares.
fn eff(db: &MenteDb, n: &MemoryNode) -> f32 {
    db.decayed_salience_at(n, NOW)
}

/// The decayed salience at `NOW` of the fixture memory whose content is
/// `fixture memory {i}`.
fn eff_of(db: &MenteDb, i: usize) -> f32 {
    let content = format!("fixture memory {i}");
    all_nodes(db)
        .iter()
        .find(|n| n.content == content)
        .map(|n| eff(db, n))
        .unwrap_or_else(|| panic!("no {content}"))
}

#[test]
fn agent_filters_compare_the_owner_agent() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let (a0, a1, r) = (fx.agents[0], fx.agents[1], fx.rare_agent);
    let db = &fx.db;
    assert_query(
        db,
        &format!("RECALL memories WHERE agent = {a0} LIMIT 100000"),
        |n| n.agent_id == a0,
    );
    // A quoted id parses to the same UUID.
    assert_query(
        db,
        &format!("RECALL memories WHERE agent = \"{a1}\" LIMIT 100000"),
        |n| n.agent_id == a1,
    );
    assert_query(
        db,
        &format!("RECALL memories WHERE agent != {a0} LIMIT 100000"),
        |n| n.agent_id != a0,
    );
    assert_query(
        db,
        &format!("RECALL memories WHERE agent IN [{a1}, {r}] LIMIT 100000"),
        |n| n.agent_id == a1 || n.agent_id == r,
    );
    assert_query(
        db,
        &format!("RECALL memories WHERE agent = {a0} AND type = episodic LIMIT 100000"),
        |n| n.agent_id == a0 && n.memory_type == MemoryType::Episodic,
    );
    assert_query(
        db,
        &format!("RECALL memories WHERE agent = {r} OR type = semantic LIMIT 100000"),
        |n| n.agent_id == r || n.memory_type == MemoryType::Semantic,
    );
    assert_query(
        db,
        &format!("RECALL memories WHERE NOT agent = {a0} LIMIT 100000"),
        |n| n.agent_id != a0,
    );
    // An agent that owns nothing matches nothing.
    let nobody = AgentId::from(uuid(99));
    assert!(ids(db, &format!("RECALL memories WHERE agent = {nobody}")).is_empty());
}

#[test]
fn space_filters_compare_the_space_id() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let (s0, s1) = (fx.spaces[0], fx.spaces[1]);
    let db = &fx.db;
    assert_query(
        db,
        &format!("RECALL memories WHERE space = {s0} LIMIT 100000"),
        |n| n.space_id == s0,
    );
    assert_query(
        db,
        &format!("RECALL memories WHERE space != {s0} LIMIT 100000"),
        |n| n.space_id != s0,
    );
    assert_query(
        db,
        &format!(
            "RECALL memories WHERE space IN [{s1}, \"{}\"] LIMIT 100000",
            SpaceId::nil()
        ),
        |n| n.space_id == s1 || n.space_id.is_nil(),
    );
    assert_query(
        db,
        &format!(
            "RECALL memories WHERE space = {s1} AND tag = \"none\" OR space = {s0} LIMIT 100000"
        ),
        |n| n.space_id == s0,
    );
}

#[test]
fn salience_filters_compare_the_decayed_salience() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let db = &fx.db;
    // The fixture decays: no memory but the newest keeps its stored base.
    assert!(
        all_nodes(db)
            .iter()
            .filter(|n| eff(db, n) != n.salience)
            .count()
            >= N - 1
    );
    let t = f32_of(0.5);
    assert_query(
        db,
        "RECALL memories WHERE salience > 0.5 LIMIT 100000",
        |n| eff(db, n) > t,
    );
    assert_query(
        db,
        "RECALL memories WHERE salience >= 0.5 LIMIT 100000",
        |n| eff(db, n) >= t,
    );
    assert_query(
        db,
        "RECALL memories WHERE salience < 0.5 LIMIT 100000",
        |n| eff(db, n) < t,
    );
    assert_query(
        db,
        "RECALL memories WHERE salience <= 0.5 LIMIT 100000",
        |n| eff(db, n) <= t,
    );
    // Equality on the decayed value of memory 300 (stored base 0.3).
    let eq = eff_of(db, 300);
    assert_ne!(eq, f32_of(0.3));
    assert_query(
        db,
        &format!("RECALL memories WHERE salience = {eq} LIMIT 100000"),
        |n| eff(db, n) == eq,
    );
    // The stored base is not what is compared.
    assert!(ids(db, "RECALL memories WHERE salience = 0.3 LIMIT 100000").is_empty());
    assert_query(
        db,
        &format!("RECALL memories WHERE salience != {eq} LIMIT 100000"),
        |n| eff(db, n) != eq,
    );
    let other = eff_of(db, 450);
    assert_query(
        db,
        &format!("RECALL memories WHERE salience IN [{eq}, {other}] LIMIT 100000"),
        |n| eff(db, n) == eq || eff(db, n) == other,
    );
    // Integer literals compare too: nothing is above 1.
    assert!(ids(db, "RECALL memories WHERE salience > 1 LIMIT 100000").is_empty());
    assert_query(
        db,
        "RECALL memories WHERE salience >= 0.2 AND salience < 0.25 AND type = semantic LIMIT 100000",
        |n| {
            eff(db, n) >= f32_of(0.2)
                && eff(db, n) < f32_of(0.25)
                && n.memory_type == MemoryType::Semantic
        },
    );
    assert_query(
        db,
        "RECALL memories WHERE NOT salience < 0.5 OR type = semantic LIMIT 100000",
        |n| eff(db, n) >= t || n.memory_type == MemoryType::Semantic,
    );
}

#[test]
fn confidence_filters_compare_the_stored_confidence() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let db = &fx.db;
    let t = f32_of(0.8);
    assert_query(
        db,
        "RECALL memories WHERE confidence >= 0.8 LIMIT 100000",
        |n| n.confidence >= t,
    );
    assert_query(
        db,
        "RECALL memories WHERE confidence > 0.8 LIMIT 100000",
        |n| n.confidence > t,
    );
    assert_query(
        db,
        "RECALL memories WHERE confidence < 0.8 LIMIT 100000",
        |n| n.confidence < t,
    );
    assert_query(
        db,
        "RECALL memories WHERE confidence <= 0.8 LIMIT 100000",
        |n| n.confidence <= t,
    );
    assert_query(
        db,
        "RECALL memories WHERE confidence = 0.8 LIMIT 100000",
        |n| n.confidence == t,
    );
    assert_eq!(
        ids(db, "RECALL memories WHERE confidence = 0.8 LIMIT 100000").len(),
        2
    );
    assert_query(
        db,
        "RECALL memories WHERE confidence != 0.8 LIMIT 100000",
        |n| n.confidence != t,
    );
    assert_query(
        db,
        "RECALL memories WHERE confidence IN [0.8, 0.1] LIMIT 100000",
        |n| n.confidence == t || n.confidence == f32_of(0.1),
    );
    assert_query(
        db,
        "RECALL memories WHERE NOT confidence < 0.5 LIMIT 100000",
        |n| n.confidence >= f32_of(0.5),
    );
}

#[test]
fn malformed_owner_and_score_filters_are_query_errors() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let db = &fx.db;
    let a0 = fx.agents[0];
    for q in [
        "RECALL memories WHERE agent = \"alice\"".to_string(),
        format!("RECALL memories WHERE agent > {a0}"),
        format!("RECALL memories WHERE agent CONTAINS \"{a0}\""),
        "RECALL memories WHERE space = 3".to_string(),
        "RECALL memories WHERE space IN [\"x\"]".to_string(),
        "RECALL memories WHERE salience = \"high\"".to_string(),
        "RECALL memories WHERE confidence IN [0.5, \"x\"]".to_string(),
        "RECALL memories WHERE confidence CONTAINS 0.5".to_string(),
        "RECALL memories WHERE type = episodic OR agent = \"alice\"".to_string(),
        "RECALL memories WHERE created > 5 AND salience = \"high\"".to_string(),
        "CONSOLIDATE WHERE agent = \"alice\"".to_string(),
    ] {
        let err = db.query(&q).err();
        assert!(
            matches!(err, Some(MenteError::Query(_))),
            "{q}: expected a query error, got {err:?}"
        );
    }
}

#[test]
fn agent_equality_is_answered_from_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let db = &fx.db;
    let r = fx.rare_agent;
    let want = expected(db, |n| n.agent_id == r);
    assert_eq!(want.len(), N / 60);

    // Indexed: only the rare agent's memories are loaded.
    let before = db.storage_memory_loads();
    let indexed = id_set(db, &format!("RECALL memories WHERE agent = {r} LIMIT 1000"));
    let loads = db.storage_memory_loads() - before;
    assert_eq!(indexed, want);
    assert_eq!(loads as usize, want.len(), "loaded {loads} memories");

    // The same filter through the boolean tree path, which never uses the
    // index, gives the same memories in the same order.
    let before = db.storage_memory_loads();
    let full = ids(
        db,
        &format!("RECALL memories WHERE agent = {r} OR agent = {r} LIMIT 1000"),
    );
    assert_eq!(db.storage_memory_loads() - before, N as u64);
    let indexed_list = ids(db, &format!("RECALL memories WHERE agent = {r} LIMIT 1000"));
    assert_eq!(indexed_list, full);

    // IN lists and intersections with type agree with the full scan too.
    let a1 = fx.agents[1];
    for (indexed_q, full_q) in [
        (
            format!("RECALL memories WHERE agent IN [{r}, {a1}] LIMIT 1000"),
            format!("RECALL memories WHERE agent = {r} OR agent = {a1} LIMIT 1000"),
        ),
        (
            format!("RECALL memories WHERE agent = {a1} AND type = semantic LIMIT 1000"),
            format!(
                "RECALL memories WHERE (agent = {a1} AND type = semantic) OR agent = {a1} AND type = semantic LIMIT 1000"
            ),
        ),
        (
            format!("RECALL memories WHERE agent = {a1} ORDER BY salience DESC LIMIT 7"),
            format!(
                "RECALL memories WHERE agent = {a1} OR agent = {a1} ORDER BY salience DESC LIMIT 7"
            ),
        ),
    ] {
        assert_eq!(ids(db, &indexed_q), ids(db, &full_q), "{indexed_q}");
    }

    // A LIMIT stops the indexed scan early.
    let before = db.storage_memory_loads();
    let got = ids(db, &format!("RECALL memories WHERE agent = {r} LIMIT 3"));
    assert_eq!(got.len(), 3);
    assert_eq!(db.storage_memory_loads() - before, 3);
}

#[test]
fn temporal_scan_applies_every_filter_and_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let db = &fx.db;
    let t = BASE_TS + 100 * STEP;
    let q = format!("RECALL memories WHERE created > {t} AND type = episodic LIMIT 10");
    assert!(matches!(
        Mql::parse(&q).unwrap(),
        QueryPlan::TemporalScan { .. }
    ));
    let got = db.query(&q).unwrap();
    assert_eq!(got.len(), 10, "{q}");
    for s in &got {
        assert_eq!(s.memory.memory_type, MemoryType::Episodic, "{q}");
        assert!(s.memory.created_at > t, "{q}");
    }

    // Without a cutting LIMIT the result is exactly the matching set.
    assert_query(
        db,
        &format!("RECALL memories WHERE created > {t} AND type = episodic LIMIT 100000"),
        |n| n.created_at > t && n.memory_type == MemoryType::Episodic,
    );
    // Strict and inclusive bounds on both sides.
    assert_query(
        db,
        &format!(
            "RECALL memories WHERE created >= {t} AND created < {} LIMIT 100000",
            t + 50 * STEP
        ),
        |n| n.created_at >= t && n.created_at < t + 50 * STEP,
    );
    assert_query(
        db,
        &format!(
            "RECALL memories WHERE created > {t} AND created <= {} LIMIT 100000",
            t + 50 * STEP
        ),
        |n| n.created_at > t && n.created_at <= t + 50 * STEP,
    );
    // The tighter of two lower bounds wins.
    assert_query(
        db,
        &format!(
            "RECALL memories WHERE created > {} AND created > {t} LIMIT 100000",
            BASE_TS
        ),
        |n| n.created_at > t,
    );
    // Owner and score filters inside a temporal scan.
    let a0 = fx.agents[0];
    assert_query(
        db,
        &format!(
            "RECALL memories WHERE created < {t} AND agent = {a0} AND salience > 0.05 AND confidence < 0.9 LIMIT 100000"
        ),
        |n| {
            n.created_at < t
                && n.agent_id == a0
                && eff(db, n) > f32_of(0.05)
                && n.confidence < f32_of(0.9)
        },
    );
    // More than the old fixed cap of 100 comes back when LIMIT asks for it.
    let wide = ids(
        db,
        &format!("RECALL memories WHERE created > {} LIMIT 500", BASE_TS),
    );
    assert_eq!(wide.len(), 500);
    // No LIMIT: the planner's default applies.
    let default = ids(db, &format!("RECALL memories WHERE created > {}", BASE_TS));
    assert_eq!(default.len(), 20);
}

#[test]
fn temporal_scan_honors_order_by() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let db = &fx.db;
    let t = BASE_TS + 200 * STEP;
    let in_range = |n: &MemoryNode| n.created_at >= t && n.memory_type != MemoryType::Correction;
    let mut nodes: Vec<MemoryNode> = all_nodes(db).into_iter().filter(in_range).collect();

    nodes.sort_by(|a, b| eff(db, b).total_cmp(&eff(db, a)));
    let want: Vec<MemoryId> = nodes.iter().take(6).map(|n| n.id).collect();
    let q = format!(
        "RECALL memories WHERE created >= {t} AND type != correction ORDER BY salience DESC LIMIT 6"
    );
    assert_eq!(ids(db, &q), want, "{q}");

    nodes.sort_by_key(|n| n.created_at);
    let want: Vec<MemoryId> = nodes.iter().take(4).map(|n| n.id).collect();
    let q = format!(
        "RECALL memories WHERE created >= {t} AND type != correction ORDER BY created ASC LIMIT 4"
    );
    assert_eq!(ids(db, &q), want, "{q}");

    nodes.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    let want: Vec<MemoryId> = nodes.iter().take(5).map(|n| n.id).collect();
    let q = format!(
        "RECALL memories WHERE created >= {t} AND type != correction ORDER BY created DESC LIMIT 5"
    );
    assert_eq!(ids(db, &q), want, "{q}");

    nodes.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));
    let want: Vec<MemoryId> = nodes.iter().take(5).map(|n| n.id).collect();
    let q = format!(
        "RECALL memories WHERE created >= {t} AND type != correction ORDER BY confidence ASC LIMIT 5"
    );
    assert_eq!(ids(db, &q), want, "{q}");
}

#[test]
fn accessed_ranges_filter_on_access_time() {
    let dir = tempfile::tempdir().unwrap();
    let fx = fixture(dir.path());
    let db = &fx.db;
    // accessed runs opposite to created, so a range read off the creation
    // index would return the wrong memories.
    let t = BASE_TS + 450 * STEP;
    assert_query(
        db,
        &format!("RECALL memories WHERE accessed > {t} LIMIT 100000"),
        |n| n.accessed_at > t,
    );
    let c = BASE_TS + 100 * STEP;
    assert_query(
        db,
        &format!("RECALL memories WHERE accessed > {t} AND created > {c} LIMIT 100000"),
        |n| n.accessed_at > t && n.created_at > c,
    );
}

/// A store under the default seven day half-life with memories whose stored
/// base salience and current decayed salience disagree, evaluated at `now`.
struct DecayFixture {
    db: MenteDb,
    now: u64,
}

const DAY_US: u64 = 24 * 3600 * 1_000_000;

fn decay_fixture(dir: &Path) -> DecayFixture {
    let config = CognitiveConfig {
        write_inference: false,
        memory_dedup: false,
        ..CognitiveConfig::default()
    };
    let db = MenteDb::open_with_config(dir, config).unwrap();
    let now = MenteDb::now_micros();
    let old = now - 60 * DAY_US;
    // (content, stored salience, last access, exempt tag, exempt attribute)
    let rows: [(&str, f32, u64, bool, bool); 5] = [
        ("stale", 0.95, old, false, false),
        ("fresh", 0.4, now, false, false),
        ("aging", 0.9, now - 7 * DAY_US, false, false),
        ("pinned", 0.85, old, true, false),
        ("exempt", 0.7, old, false, true),
    ];
    for (i, (content, salience, accessed, tagged, attr)) in rows.into_iter().enumerate() {
        let mut emb = vec![0.0f32; 8];
        emb[i] = 1.0;
        let mut n = MemoryNode::new(AgentId::nil(), MemoryType::Semantic, content.into(), emb);
        n.salience = salience;
        n.confidence = 0.9;
        n.created_at = accessed;
        n.accessed_at = accessed;
        if tagged {
            n.tags.push("scope:always".into());
        }
        if attr {
            n.attributes.insert(
                "decay_exempt".into(),
                mentedb_core::memory::AttributeValue::Boolean(true),
            );
        }
        db.store(n).unwrap();
    }
    DecayFixture { db, now }
}

fn contents_at(db: &MenteDb, mql: &str, now: u64) -> Vec<String> {
    db.query_at(mql, now)
        .unwrap_or_else(|e| panic!("{mql}: {e}"))
        .into_iter()
        .map(|s| s.memory.content)
        .collect()
}

fn content_set(db: &MenteDb, mql: &str, now: u64) -> HashSet<String> {
    contents_at(db, mql, now).into_iter().collect()
}

fn set_of(items: &[&str]) -> HashSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn salience_filters_use_the_decayed_value_not_the_stored_base() {
    let dir = tempfile::tempdir().unwrap();
    let DecayFixture { db, now } = decay_fixture(dir.path());
    let db = &db;
    // "stale" stores 0.95 but has not been touched in 60 days: its current
    // salience is near the floor, so a high threshold drops it and a low one
    // keeps it. The pinned memories keep their stored value.
    assert_eq!(
        content_set(db, "RECALL memories WHERE salience > 0.5 LIMIT 100", now),
        set_of(&["pinned", "exempt"])
    );
    assert_eq!(
        content_set(db, "RECALL memories WHERE salience < 0.2 LIMIT 100", now),
        set_of(&["stale"])
    );
    // One half-life: 0.9 decays to 0.45.
    assert_eq!(
        content_set(
            db,
            "RECALL memories WHERE salience > 0.44 AND salience < 0.46 LIMIT 100",
            now
        ),
        set_of(&["aging"])
    );
    // Equality and IN on current values; the stored 0.95 matches nothing.
    assert!(content_set(db, "RECALL memories WHERE salience = 0.95 LIMIT 100", now).is_empty());
    assert_eq!(
        content_set(
            db,
            "RECALL memories WHERE salience IN [0.4, 0.85, 0.95] LIMIT 100",
            now
        ),
        set_of(&["fresh", "pinned"])
    );
    // The same through a boolean tree.
    assert_eq!(
        content_set(
            db,
            "RECALL memories WHERE NOT salience >= 0.2 OR salience = 0.7 LIMIT 100",
            now
        ),
        set_of(&["stale", "exempt"])
    );
    // The value compared is the one compute_decayed_salience reports.
    let health: HashMap<String, f32> = all_nodes(db)
        .iter()
        .map(|n| (n.content.clone(), db.decayed_salience_at(n, now)))
        .collect();
    assert!(health["stale"] < 0.02, "{health:?}");
    assert_eq!(health["pinned"], 0.85);
    assert_eq!(health["exempt"], 0.7);
    // Confidence does not decay: every memory keeps its stored 0.9.
    assert_eq!(
        content_set(db, "RECALL memories WHERE confidence > 0.8 LIMIT 100", now).len(),
        5
    );
    // The public entry point reads the clock itself and agrees.
    let live: HashSet<String> = db
        .query("RECALL memories WHERE salience > 0.5 LIMIT 100")
        .unwrap()
        .into_iter()
        .map(|s| s.memory.content)
        .collect();
    assert_eq!(live, set_of(&["pinned", "exempt"]));
}

#[test]
fn order_by_salience_sorts_on_the_decayed_value() {
    let dir = tempfile::tempdir().unwrap();
    let DecayFixture { db, now } = decay_fixture(dir.path());
    let db = &db;
    // Stored order would be stale, aging, pinned, exempt, fresh.
    assert_eq!(
        contents_at(db, "RECALL memories ORDER BY salience DESC LIMIT 100", now),
        vec!["pinned", "exempt", "aging", "fresh", "stale"]
    );
    assert_eq!(
        contents_at(db, "RECALL memories ORDER BY salience ASC LIMIT 2", now),
        vec!["stale", "fresh"]
    );
    // Through the temporal scan path as well.
    assert_eq!(
        contents_at(
            db,
            "RECALL memories WHERE created > 0 ORDER BY salience DESC LIMIT 3",
            now
        ),
        vec!["pinned", "exempt", "aging"]
    );
}
