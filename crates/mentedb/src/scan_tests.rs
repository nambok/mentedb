//! Regression tests for the indexed MQL scan and for full passes that must
//! not hold the page map across page loads.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use super::*;

const TYPES: [MemoryType; 4] = [
    MemoryType::Episodic,
    MemoryType::Semantic,
    MemoryType::Procedural,
    MemoryType::Correction,
];

/// A database of `n` memories cycling through four types, tagged `even` on
/// every other memory and `g<i % 10>`, with unique salience so ORDER BY has
/// no ties.
fn mixed_db(dir: &Path, n: usize, dim: usize) -> MenteDb {
    let config = CognitiveConfig {
        write_inference: false,
        ..CognitiveConfig::default()
    };
    let db = MenteDb::open_with_config(dir, config).unwrap();
    let mut batch = Vec::new();
    for i in 0..n {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        emb[(i * 7 + 3) % dim] += 0.5;
        let mut node = MemoryNode::new(
            AgentId::nil(),
            TYPES[i % TYPES.len()],
            format!("memory number {i}"),
            emb,
        );
        node.salience = (i as f32 + 1.0) / (n as f32 + 1.0);
        node.tags = vec![format!("g{}", i % 10)];
        if i % 2 == 0 {
            node.tags.push("even".into());
        }
        batch.push(node);
        if batch.len() == 500 {
            db.store_batch(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        db.store_batch(batch).unwrap();
    }
    db
}

/// The tag-less TagScan exactly as it ran before the indexed scan: load every
/// memory through the page map, filter, order, truncate.
fn old_tag_scan(db: &MenteDb, mql: &str) -> Vec<MemoryId> {
    let plan = Mql::parse(mql).unwrap();
    let QueryPlan::TagScan {
        tags,
        filters,
        limit,
        condition,
        order_by,
    } = plan
    else {
        panic!("{mql}: expected a TagScan plan");
    };
    assert!(
        tags.is_empty(),
        "{mql}: tag-only queries take the hybrid path"
    );
    let k = limit.unwrap_or(20);
    let pm = db.page_map.read();
    let mut scored: Vec<ScoredMemory> = pm
        .values()
        .filter_map(|&pid| db.storage.load_memory(pid).ok())
        .map(|memory| ScoredMemory { memory, score: 1.0 })
        .collect();
    drop(pm);
    match &condition {
        Some(cond) => scored.retain(|sm| MenteDb::condition_matches(cond, &sm.memory)),
        None => scored.retain(|sm| {
            filters
                .iter()
                .all(|f| MenteDb::filter_matches(f, &sm.memory))
        }),
    }
    if let Some(ob) = &order_by {
        MenteDb::apply_order_by(&mut scored, ob);
    }
    scored.truncate(k);
    scored.into_iter().map(|s| s.memory.id).collect()
}

fn new_scan(db: &MenteDb, mql: &str) -> Vec<MemoryId> {
    db.query(mql)
        .unwrap()
        .into_iter()
        .map(|s| s.memory.id)
        .collect()
}

#[test]
fn indexed_scan_matches_the_full_scan() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed_db(dir.path(), 800, 16);

    // (query, ordered): ordered queries must match element for element,
    // unordered ones as sets, or as a subset of every match when LIMIT cuts.
    let queries: &[(&str, bool)] = &[
        ("RECALL memories WHERE type = episodic LIMIT 5000", false),
        (
            "RECALL memories WHERE type = episodic AND tag = \"even\" LIMIT 5000",
            false,
        ),
        (
            "RECALL memories WHERE type = semantic AND tag = \"even\" LIMIT 5000",
            false,
        ),
        (
            "RECALL memories WHERE type = semantic AND salience > 0.5 LIMIT 5000",
            false,
        ),
        (
            "RECALL memories WHERE type = procedural AND content CONTAINS \"1\" LIMIT 5000",
            false,
        ),
        (
            "RECALL memories WHERE type IN [episodic, correction] LIMIT 5000",
            false,
        ),
        (
            "RECALL memories WHERE type IN [episodic, correction] AND tag = \"g3\" LIMIT 5000",
            false,
        ),
        (
            "RECALL memories WHERE tag = \"even\" AND salience < 0.3 LIMIT 5000",
            false,
        ),
        ("RECALL memories WHERE type != episodic LIMIT 5000", false),
        (
            "RECALL memories WHERE type = episodic OR tag = \"g1\" LIMIT 5000",
            false,
        ),
        (
            "RECALL memories WHERE type = semantic ORDER BY salience DESC LIMIT 15",
            true,
        ),
        (
            "RECALL memories WHERE type = episodic AND tag = \"even\" ORDER BY salience ASC LIMIT 7",
            true,
        ),
        (
            "RECALL memories WHERE type = correction ORDER BY salience DESC",
            true,
        ),
        ("RECALL memories ORDER BY salience DESC LIMIT 9", true),
        // LIMIT cuts an unordered result: any LIMIT-sized subset is correct.
        ("RECALL memories WHERE type = episodic LIMIT 10", false),
        ("RECALL memories WHERE type = episodic", false),
        ("RECALL memories LIMIT 10", false),
        (
            "RECALL memories WHERE type = semantic AND tag = \"nope\" LIMIT 50",
            false,
        ),
    ];
    for &(q, ordered) in queries {
        let old = old_tag_scan(&db, q);
        let new = new_scan(&db, q);
        if ordered {
            assert_eq!(new, old, "{q}");
            continue;
        }
        // Every match, with the LIMIT lifted, to judge truncated results.
        let k = match Mql::parse(q).unwrap() {
            QueryPlan::TagScan { limit, .. } => limit.unwrap_or(20),
            _ => unreachable!(),
        };
        let unlimited = q
            .split(" LIMIT ")
            .next()
            .map(|base| format!("{base} LIMIT 100000"))
            .unwrap_or_default();
        let all: HashSet<MemoryId> = old_tag_scan(&db, &unlimited).into_iter().collect();
        let new_set: HashSet<MemoryId> = new.iter().copied().collect();
        assert_eq!(new_set.len(), new.len(), "{q}: no duplicates");
        assert_eq!(new.len(), all.len().min(k), "{q}: result count");
        assert!(new_set.is_subset(&all), "{q}: only matching memories");
        if all.len() <= k {
            let old_set: HashSet<MemoryId> = old.into_iter().collect();
            assert_eq!(new_set, old_set, "{q}");
        }
    }
}

#[test]
fn unordered_results_come_in_page_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed_db(dir.path(), 100, 8);
    let got = new_scan(&db, "RECALL memories WHERE type = episodic LIMIT 5");
    let pm = db.page_map.read();
    let pages: Vec<u64> = got.iter().map(|id| pm[id].0).collect();
    let mut sorted = pages.clone();
    sorted.sort_unstable();
    assert_eq!(pages, sorted, "results follow storage order");
    // And they are the first five episodic pages.
    let mut episodic: Vec<u64> = pm
        .iter()
        .filter(|(id, _)| {
            db.storage.load_memory(pm[*id]).unwrap().memory_type == MemoryType::Episodic
        })
        .map(|(_, p)| p.0)
        .collect();
    episodic.sort_unstable();
    assert_eq!(pages, episodic[..5].to_vec());
}

#[test]
fn type_filter_with_limit_loads_only_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let db = mixed_db(dir.path(), 2000, 8);

    let before = db.storage_memory_loads();
    let got = db
        .query("RECALL memories WHERE type = episodic LIMIT 10")
        .unwrap();
    let loads = db.storage_memory_loads() - before;
    assert_eq!(got.len(), 10);
    assert!(
        got.iter()
            .all(|s| s.memory.memory_type == MemoryType::Episodic)
    );
    assert_eq!(
        loads, 10,
        "loaded {loads} memories for a LIMIT 10 type query"
    );

    // Type plus tag intersects in the index: still only the limit.
    let before = db.storage_memory_loads();
    let got = db
        .query("RECALL memories WHERE type = semantic AND tag = \"g3\" LIMIT 10")
        .unwrap();
    let loads = db.storage_memory_loads() - before;
    assert_eq!(got.len(), 10);
    assert_eq!(loads, 10, "loaded {loads}");

    // Nothing indexable: the scan still stops at the limit.
    let before = db.storage_memory_loads();
    let got = db.query("RECALL memories LIMIT 10").unwrap();
    assert_eq!(got.len(), 10);
    assert_eq!(db.storage_memory_loads() - before, 10);

    // ORDER BY must see every candidate, but only the candidates.
    let before = db.storage_memory_loads();
    db.query("RECALL memories WHERE type = episodic ORDER BY salience DESC LIMIT 10")
        .unwrap();
    assert_eq!(db.storage_memory_loads() - before, 500);
}

/// Run `pass` in a thread over a database big enough that its page loads take
/// a while, and meanwhile repeatedly try to take the page map for writing (a
/// store needs it). Each attempt that overlaps the pass must succeed quickly.
fn assert_pass_leaves_page_map_free<F>(db: Arc<MenteDb>, name: &str, pass: F)
where
    F: FnOnce(&MenteDb) + Send + 'static,
{
    let running = Arc::new(AtomicBool::new(true));
    let start = db.storage_memory_loads();
    let handle = {
        let db = Arc::clone(&db);
        let running = Arc::clone(&running);
        std::thread::spawn(move || {
            pass(&db);
            running.store(false, Ordering::SeqCst);
        })
    };
    // Wait until the pass is loading pages.
    while running.load(Ordering::SeqCst) && db.storage_memory_loads() < start + 50 {
        std::thread::yield_now();
    }
    let mut overlapped = 0;
    let mut slowest = Duration::ZERO;
    while running.load(Ordering::SeqCst) {
        let t = std::time::Instant::now();
        let got = db
            .page_map
            .try_write_for(Duration::from_millis(100))
            .is_some();
        let waited = t.elapsed();
        if running.load(Ordering::SeqCst) {
            overlapped += 1;
            slowest = slowest.max(waited);
            assert!(
                got,
                "{name}: a writer waited over 100 ms for the page map while the pass loaded pages"
            );
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    handle.join().unwrap();
    assert!(
        overlapped > 0,
        "{name}: pass ended before any attempt overlapped it"
    );
    assert!(
        slowest < Duration::from_millis(100),
        "{name}: slowest {slowest:?}"
    );
}

#[test]
fn full_passes_do_not_hold_the_page_map_across_loads() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(mixed_db(dir.path(), 4000, 128));
    assert_pass_leaves_page_map_free(Arc::clone(&db), "archival", |db| {
        db.evaluate_archival_global().unwrap();
    });
    assert_pass_leaves_page_map_free(Arc::clone(&db), "consolidation", |db| {
        db.find_consolidation_candidates(3, 0.999).unwrap();
    });
    assert_pass_leaves_page_map_free(Arc::clone(&db), "mql scan", |db| {
        db.query("RECALL memories WHERE type = episodic ORDER BY salience DESC LIMIT 5")
            .unwrap();
        db.query(
            "RECALL memories WHERE content CONTAINS \"memory\" ORDER BY salience DESC LIMIT 5",
        )
        .unwrap();
    });
}
