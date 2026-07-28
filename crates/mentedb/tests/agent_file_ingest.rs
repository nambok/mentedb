//! Agent file ingestion against a real engine: the repo's own CLAUDE.md is
//! the fixture (a real instruction file, not a synthetic one), plus a
//! generated large file for scale mechanics. Pins the properties the demo
//! and the coverage harness depend on: atomic granularity, honest counts,
//! zero always pins, action rules retrievable through recall_for_action,
//! and re ingest deduplicating instead of duplicating.

use mentedb::MenteDb;
use mentedb::agent_file::{AgentFileIngestOptions, plan_agent_file};
use mentedb_embedding::hash_provider::HashEmbeddingProvider;

const CLAUDE_MD: &str = include_str!("../../../CLAUDE.md");

fn open(dir: &std::path::Path) -> MenteDb {
    MenteDb::open_with_embedder(dir, Box::new(HashEmbeddingProvider::new(256))).expect("open")
}

#[test]
fn ingests_the_real_claude_md() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let opts = AgentFileIngestOptions::default();

    let report = db.ingest_agent_file(CLAUDE_MD, &opts).unwrap();

    assert!(
        report.candidates >= 25,
        "the real file must segment into atomic memories, got {}",
        report.candidates
    );
    assert_eq!(
        report.stored, report.candidates,
        "first ingest into an empty store keeps everything"
    );
    assert!(
        report.sections >= 4,
        "section paths must nest: {}",
        report.sections
    );
    // The deterministic parser assigns no action tags at all: which memories
    // bear on an action is decided at recall time by recall_for_action
    // (semantic), never stamped at ingest. So the report carries no triggers.
    assert_eq!(
        report.trigger_tagged, 0,
        "deterministic parser stamps no triggers"
    );
    assert!(report.procedural > 0 && report.semantic > 0);
    assert!(
        report.file_token_estimate > report.avg_memory_token_estimate * 4,
        "atoms must be much smaller than the file"
    );
}

#[test]
fn nothing_is_pinned_always() {
    let atoms = plan_agent_file(CLAUDE_MD, &AgentFileIngestOptions::default());
    assert!(!atoms.is_empty());
    assert!(
        atoms
            .iter()
            .all(|a| !a.tags.iter().any(|t| t == "scope:always")),
        "agent file ingest must never pin to every turn"
    );
}

#[test]
fn reingest_deduplicates_instead_of_duplicating() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let opts = AgentFileIngestOptions::default();

    let first = db.ingest_agent_file(CLAUDE_MD, &opts).unwrap();
    let second = db.ingest_agent_file(CLAUDE_MD, &opts).unwrap();

    assert!(
        second.stored < first.stored / 2,
        "re ingesting the identical file must dedup, first {} second {}",
        first.stored,
        second.stored
    );
    assert_eq!(
        second.deduplicated,
        second.candidates - second.stored,
        "report must account for every candidate"
    );
}

#[test]
fn reingest_syncs_memory_with_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let opts = AgentFileIngestOptions::default();

    let v1 = "# Rules\n\n\
        - Always run the complete test suite before committing anything.\n\
        - Indent with four spaces, tabs are forbidden in source files.\n\
        - Documentation comments stay in English regardless of code locale.\n";
    let r1 = db.ingest_agent_file(v1, &opts).unwrap();
    assert_eq!(r1.candidates, 3);
    assert_eq!((r1.stored, r1.removed), (3, 0));

    // The user edits their file: first rule unchanged, the indentation rule
    // rewritten, the documentation rule deleted, one rule added. Re ingest
    // must mirror that: dedup, replace, remove, add, no manual cleanup.
    let v2 = "# Rules\n\n\
        - Always run the complete test suite before committing anything.\n\
        - Prefer hard tabs over spaces when editing legacy makefiles.\n\
        - Wrap every user visible string in the translation helper function.\n";
    let r2 = db.ingest_agent_file(v2, &opts).unwrap();
    assert_eq!(r2.deduplicated, 1, "the unchanged rule deduplicates");
    assert_eq!(r2.stored, 2, "the rewritten and the new rule store");
    assert_eq!(
        r2.removed, 2,
        "the old edition and the deleted rule are forgotten"
    );

    // Sync converges: mirroring the same file again changes nothing.
    let r3 = db.ingest_agent_file(v2, &opts).unwrap();
    assert_eq!((r3.stored, r3.removed), (0, 0));
    assert_eq!(r3.deduplicated, 3);
}

#[test]
fn sync_never_touches_another_owners_memories() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let a = AgentFileIngestOptions {
        agent_id: mentedb_core::types::AgentId(uuid::Uuid::from_u128(1)),
        ..Default::default()
    };
    let b = AgentFileIngestOptions {
        agent_id: mentedb_core::types::AgentId(uuid::Uuid::from_u128(2)),
        ..Default::default()
    };
    let file_a = "# Rules\n\n- Agent alpha always answers in formal English sentences.\n";
    let file_b = "# Rules\n\n- Agent beta replies with short bullet lists only.\n";
    db.ingest_agent_file(file_a, &a).unwrap();
    let rb = db.ingest_agent_file(file_b, &b).unwrap();
    assert_eq!(rb.removed, 0, "one agent's sync must not prune another's");
    let ra = db.ingest_agent_file(file_a, &a).unwrap();
    assert_eq!(
        (ra.deduplicated, ra.removed),
        (1, 0),
        "the first agent's ingest is intact"
    );
}

#[test]
fn sync_off_keeps_reingest_additive() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let additive = AgentFileIngestOptions {
        sync: false,
        ..Default::default()
    };

    let v1 = "# Rules\n\n- Documentation comments stay in English regardless of code locale.\n";
    let v2 = "# Rules\n\n- Wrap every user visible string in the translation helper function.\n";
    db.ingest_agent_file(v1, &additive).unwrap();
    let r2 = db.ingest_agent_file(v2, &additive).unwrap();
    assert_eq!(r2.removed, 0, "sync off never removes");

    // Turning sync back on prunes the rule the file no longer contains.
    let r3 = db
        .ingest_agent_file(v2, &AgentFileIngestOptions::default())
        .unwrap();
    assert_eq!(r3.removed, 1, "the stale rule from v1 is forgotten");
}

#[test]
fn large_file_mechanics_hold() {
    // A generated large instruction file: mechanics only (segmentation,
    // atom caps, throughput), never a marketing number.
    let mut big = String::from("# Playbook\n\n");
    for section in 0..90 {
        big.push_str(&format!("## Area {section}\n\n"));
        big.push_str(
            "The team reviews the dashboard each morning before standup and notes anything unusual. \
             Long running migrations are announced in the channel a day ahead so nobody is surprised. \
             Rollbacks are practiced monthly and the runbook stays in the repository next to the code. \
             When an incident closes, the writeup lands within two days while details are fresh.\n\n",
        );
        for b in 0..6 {
            big.push_str(&format!(
                "- Area {section} guideline {b}: keep the change small and reviewed before it ships\n"
            ));
        }
        big.push('\n');
    }
    assert!(big.len() > 60_000, "fixture must be a genuinely large file");

    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let opts = AgentFileIngestOptions::default();
    let report = db.ingest_agent_file(&big, &opts).unwrap();

    assert!(
        report.candidates > 500,
        "a large file must atomize widely, got {}",
        report.candidates
    );
    assert!(report.sections >= 90);
    let atoms = plan_agent_file(&big, &opts);
    assert!(
        atoms
            .iter()
            .all(|a| a.content.len() <= opts.max_content_chars),
        "atom caps must hold"
    );
    // Near identical guidelines across sections stress write time dedup;
    // whatever it folds, the report must stay consistent.
    assert_eq!(report.stored + report.deduplicated, report.candidates);
}
