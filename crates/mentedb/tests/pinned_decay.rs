//! Pinned memories (tagged `scope:always` by default) keep their salience and
//! are never archived, while ordinary memories of the same age fade.

use mentedb::MenteDb;
use mentedb_consolidation::archival::ArchivalDecision;
use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;

const YEAR_MICROS: u64 = 365 * 24 * 3600 * 1_000_000;

fn old_memory(content: &str, tags: &[&str]) -> MemoryNode {
    let mut n = MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        content.to_string(),
        vec![],
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    n.created_at = now - 3 * YEAR_MICROS;
    n.accessed_at = n.created_at;
    n.access_count = 0;
    n.salience = 0.3;
    n.tags = tags.iter().map(|t| t.to_string()).collect();
    n
}

#[test]
fn pinned_memories_never_fade_or_archive() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).unwrap();
    let pinned = old_memory("always answer in metric units", &["scope:always"]);
    let plain = old_memory("asked about the weather once", &[]);

    assert!(db.is_decay_exempt(&pinned));
    assert!(!db.is_decay_exempt(&plain));
    assert_eq!(
        db.compute_decayed_salience(&pinned),
        0.3,
        "pinned keeps its salience"
    );
    assert!(
        db.compute_decayed_salience(&plain) < 0.3,
        "an unpinned memory of the same age fades"
    );
    assert!(matches!(
        db.evaluate_archival(&pinned),
        ArchivalDecision::Keep
    ));
    assert!(
        !matches!(db.evaluate_archival(&plain), ArchivalDecision::Keep),
        "the unpinned three year old memory is archived or deleted"
    );

    let mut batch = vec![pinned.clone(), plain.clone()];
    db.apply_decay(&mut batch);
    assert_eq!(
        batch[0].salience, 0.3,
        "batch decay skips the pinned memory"
    );
    let decisions = db.evaluate_archival_batch(&[pinned.clone(), plain.clone()]);
    assert!(matches!(decisions[0].1, ArchivalDecision::Keep));
}

#[test]
fn exempt_tags_are_configurable() {
    let dir = tempfile::tempdir().unwrap();
    let config = mentedb::CognitiveConfig {
        decay_exempt_tags: vec!["keep-forever".to_string()],
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();
    assert!(db.is_decay_exempt(&old_memory("x", &["keep-forever"])));
    assert!(!db.is_decay_exempt(&old_memory("y", &["scope:always"])));
}
