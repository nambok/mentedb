//! The standing-rules (`scope:always`) cleanup policy lives in the engine so the
//! hosted platform, the local daemon, and every SDK share one behavior and it
//! cannot drift between them.

use mentedb::MenteDb;
use mentedb::prelude::*;
use mentedb_core::types::AgentId;

fn pinned(content: &str, extra_tags: &[&str]) -> MemoryNode {
    let mut node = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        content.to_string(),
        vec![],
    );
    node.tags = std::iter::once("scope:always".to_string())
        .chain(extra_tags.iter().map(|t| t.to_string()))
        .collect();
    node
}

#[test]
fn prune_unpins_auto_keeps_manual_and_profile_and_dedups() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open(dir.path()).expect("open");

    // Auto-pinned (enrichment): must be un-pinned, memory kept.
    let auto = pinned("a distilled fact about deploys", &["source:enrichment"]);
    let auto_id = auto.id;
    db.store(auto).expect("store");

    // Explicitly pinned by the user: must be kept as scope:always.
    let manual = pinned("never add Co-Authored-By", &["source:manual"]);
    let manual_id = manual.id;
    db.store(manual).expect("store");

    // The user profile: engine-pinned deliberately, must be kept.
    let profile = pinned(
        "Nam is a technical founder",
        &["user_profile", "source:enrichment"],
    );
    let profile_id = profile.id;
    db.store(profile).expect("store");

    // Two exact-content duplicates: one is removed (healthiest kept).
    let dup1 = pinned("always use rtk", &["source:enrichment"]);
    let dup1_id = dup1.id;
    db.store(dup1).expect("store");
    let dup2 = pinned("always use rtk", &["source:enrichment"]);
    let dup2_id = dup2.id;
    db.store(dup2).expect("store");

    let report = db.prune_standing_rules().expect("prune");

    assert_eq!(report.total_always, 5);

    // Exactly one duplicate collapsed and forgotten.
    assert_eq!(report.duplicate_groups, 1);
    assert_eq!(report.pruned.len(), 1);
    let removed = report.pruned[0];
    assert!(removed == dup1_id || removed == dup2_id);
    assert!(
        db.get_memory(removed).is_err(),
        "removed duplicate must be forgotten"
    );

    let is_always = |id: MemoryId| {
        db.get_memory(id)
            .map(|n| n.tags.iter().any(|t| t == "scope:always"))
            .unwrap_or(false)
    };

    // Auto-pinned rule un-pinned but still present (recalled by relevance).
    assert!(!is_always(auto_id), "auto-pinned rule must be un-pinned");
    assert!(db.get_memory(auto_id).is_ok(), "un-pinned memory is kept");

    // The surviving duplicate is enrichment-sourced, so also un-pinned.
    let surviving_dup = if removed == dup1_id { dup2_id } else { dup1_id };
    assert!(!is_always(surviving_dup));

    // Manual pin and the profile survive with scope:always intact.
    assert!(is_always(manual_id), "manual pin must be kept pinned");
    assert!(is_always(profile_id), "profile must stay pinned");

    // The report reflects exactly what was un-pinned.
    assert!(report.unpinned.contains(&auto_id));
    assert!(report.unpinned.contains(&surviving_dup));
    assert!(!report.unpinned.contains(&manual_id));
    assert!(!report.unpinned.contains(&profile_id));
}
