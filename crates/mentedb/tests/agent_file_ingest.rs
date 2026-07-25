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
    assert!(
        report.trigger_tagged >= 1,
        "the commit conventions rule must be action tagged"
    );
    assert!(report.procedural > 0 && report.semantic > 0);
    assert!(
        report.file_token_estimate > report.avg_memory_token_estimate * 4,
        "atoms must be much smaller than the file"
    );
}

#[test]
fn commit_rules_arrive_through_the_action_channel() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.ingest_agent_file(CLAUDE_MD, &AgentFileIngestOptions::default())
        .unwrap();

    let rules = db.recall_for_action("git-commit", None, None, 8).unwrap();
    assert!(
        !rules.is_empty(),
        "the commit style rule must fire at the commit moment"
    );
    assert!(
        rules
            .iter()
            .any(|r| r.content.to_lowercase().contains("commit")),
        "retrieved rule must be about commits: {:?}",
        rules.iter().map(|r| &r.content).collect::<Vec<_>>()
    );
    // And an action with no rules stays empty rather than guessing.
    let none = db.recall_for_action("deploy", None, None, 8).unwrap();
    assert!(none.is_empty());
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
