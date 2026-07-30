//! The dashboard read path must answer "newest N", "ids with tag T", "count
//! excluding tags", and "conflict counts" from the in-memory indexes, loading
//! only the rows it returns rather than scanning every memory. These tests pin
//! that behaviour on a real engine: the results are correct, and (by
//! construction) they come from the temporal, bitmap, and graph indexes, not a
//! full scan.

use std::collections::HashSet;

use mentedb::MenteDb;
use mentedb::prelude::{AgentId, EdgeType, MemoryEdge, MemoryNode, MemoryType};

fn open() -> (tempfile::TempDir, MenteDb) {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");
    (dir, db)
}

/// Store a memory with a controlled creation time and tag set so the temporal
/// and bitmap indexes have deterministic contents.
fn store(
    db: &MenteDb,
    content: &str,
    created_at: u64,
    tags: &[&str],
) -> mentedb::prelude::MemoryId {
    let mut node = MemoryNode::new(
        AgentId(uuid::Uuid::nil()),
        MemoryType::Semantic,
        content.to_string(),
        Vec::new(),
    );
    node.created_at = created_at;
    node.tags = tags.iter().map(|t| t.to_string()).collect();
    let id = node.id;
    db.store(node).expect("store");
    id
}

const INTERNAL: &[&str] = &[
    "turn",
    "conversation-turn",
    "action",
    "ghost-memory",
    "entity",
];

#[test]
fn recent_memory_ids_are_newest_first_and_skip_excluded() {
    let (_dir, db) = open();
    let _oldest = store(&db, "oldest", 100, &[]);
    let a_turn = store(&db, "a raw turn", 150, &["turn"]);
    let middle = store(&db, "middle", 200, &[]);
    let an_action = store(&db, "an action capture", 250, &["action"]);
    let newest = store(&db, "newest", 300, &[]);

    // Newest first, internal working material excluded.
    let ids = db.recent_memory_ids(10, INTERNAL, None);
    assert_eq!(
        ids,
        vec![newest, middle, _oldest],
        "only the three user memories, newest first"
    );
    assert!(!ids.contains(&a_turn) && !ids.contains(&an_action));

    // A small limit returns only that many, still newest first.
    assert_eq!(db.recent_memory_ids(1, INTERNAL, None), vec![newest]);
}

#[test]
fn recent_memory_ids_paginate_after_a_cursor() {
    let (_dir, db) = open();
    let n3 = store(&db, "n3", 300, &[]);
    let n2 = store(&db, "n2", 200, &[]);
    let n1 = store(&db, "n1", 100, &[]);

    let page1 = db.recent_memory_ids(1, INTERNAL, None);
    assert_eq!(page1, vec![n3]);
    let page2 = db.recent_memory_ids(1, INTERNAL, Some(n3));
    assert_eq!(page2, vec![n2]);
    let page3 = db.recent_memory_ids(10, INTERNAL, Some(n2));
    assert_eq!(page3, vec![n1]);
}

#[test]
fn count_excluding_tags_ignores_internal_material() {
    let (_dir, db) = open();
    store(&db, "user 1", 100, &[]);
    store(&db, "user 2", 200, &["semantic"]);
    store(&db, "a turn", 300, &["turn"]);
    store(&db, "an action", 400, &["action"]);

    assert_eq!(db.memory_count(), 4, "all four are stored");
    assert_eq!(
        db.count_excluding_tags(INTERNAL),
        2,
        "only the two user memories count"
    );
}

#[test]
fn memory_ids_with_tag_finds_only_the_tagged() {
    let (_dir, db) = open();
    let pinned = store(&db, "a standing rule", 100, &["scope:always"]);
    store(&db, "not pinned", 200, &[]);
    let also = store(&db, "another rule", 300, &["scope:always"]);

    let ids: HashSet<_> = db.memory_ids_with_tag("scope:always").into_iter().collect();
    assert_eq!(ids, HashSet::from([pinned, also]));
    assert!(db.memory_ids_with_tag("no-such-tag").is_empty());
}

#[test]
fn conflict_edge_counts_tally_the_graph_without_loading_nodes() {
    let (_dir, db) = open();
    let a = store(&db, "the database uses Postgres", 100, &[]);
    let b = store(&db, "the database uses MySQL", 200, &[]);
    let c = store(&db, "retry limit is three", 300, &[]);
    let d = store(&db, "retry limit is five", 400, &[]);

    let edge = |source, target, kind| MemoryEdge {
        source,
        target,
        edge_type: kind,
        weight: 0.9,
        created_at: 1,
        valid_from: None,
        valid_until: None,
        label: None,
    };

    db.relate(edge(a, b, EdgeType::Contradicts)).unwrap();
    // The reverse direction of the same contradiction must not double count.
    db.relate(edge(b, a, EdgeType::Contradicts)).unwrap();
    db.relate(edge(d, c, EdgeType::Supersedes)).unwrap();

    let (contradicts, supersedes) = db.conflict_edge_counts();
    assert_eq!(contradicts, 1, "A<->B is one undirected contradiction");
    assert_eq!(supersedes, 1);
}
