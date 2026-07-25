//! LLM parser path for agent file ingestion, tested with a mock provider:
//! open trigger vocabulary across domains, tolerant JSON handling, chunking
//! for large files, and the deterministic fallback when a completion is
//! unusable. Runs with `--features enrichment`.
#![cfg(feature = "enrichment")]

use std::sync::atomic::{AtomicUsize, Ordering};

use mentedb::MenteDb;
use mentedb::agent_file::AgentFileIngestOptions;
use mentedb_embedding::hash_provider::HashEmbeddingProvider;
use mentedb_extraction::error::ExtractionError;
use mentedb_extraction::provider::ExtractionProvider;

struct CannedProvider {
    response: String,
    calls: AtomicUsize,
}

impl ExtractionProvider for CannedProvider {
    async fn extract(
        &self,
        _conversation: &str,
        _system_prompt: &str,
    ) -> Result<String, ExtractionError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.response.clone())
    }
}

fn open(dir: &std::path::Path) -> MenteDb {
    MenteDb::open_with_embedder(dir, Box::new(HashEmbeddingProvider::new(256))).expect("open")
}

#[tokio::test]
async fn llm_parse_stores_open_vocabulary_triggers() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());

    // A support and trading agent file, nothing developer specific: the
    // model names the actions, the engine only validates the slugs.
    let provider = CannedProvider {
        response: r#"Here is the parse:
[
  {"content": "Refunds over 200 dollars require a manager approval before processing", "type": "procedural", "trigger": "Order Refund"},
  {"content": "Never promise a delivery date the carrier has not confirmed", "type": "anti_pattern"},
  {"content": "Entries are sized at one percent of the account per position", "type": "procedural", "trigger": "trade-entry"},
  {"content": "The support tone is warm and direct, two short paragraphs at most", "type": "semantic"},
  {"content": "x", "type": "semantic"},
  {"content": "Escalate chargebacks to the billing team the same day", "type": "procedural", "trigger": "always"}
]"#
        .to_string(),
        calls: AtomicUsize::new(0),
    };

    let opts = AgentFileIngestOptions::default();
    let report = db
        .ingest_agent_file_llm("any file text, format irrelevant here", &opts, &provider)
        .await
        .unwrap();

    assert_eq!(report.parsed_by, "llm");
    assert_eq!(report.llm_chunks, 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    // "x" is below min_atom_chars, everything else lands.
    assert_eq!(report.candidates, 5);
    assert_eq!(report.stored, 5);
    // "Order Refund" normalizes to order-refund; the "always" trigger is
    // rejected so that entry stays a plain memory.
    assert_eq!(report.trigger_tagged, 2);

    let refund = db.recall_for_action("order-refund", None, None, 8).unwrap();
    assert_eq!(refund.len(), 1);
    assert!(refund[0].content.contains("manager approval"));

    let trade = db.recall_for_action("trade-entry", None, None, 8).unwrap();
    assert_eq!(trade.len(), 1);
    assert!(trade[0].content.contains("one percent"));

    let none = db.recall_for_action("always", None, None, 8).unwrap();
    assert!(none.is_empty(), "always must never become a trigger");
}

#[tokio::test]
async fn large_files_are_chunked_at_sections() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());

    let mut big = String::new();
    for i in 0..40 {
        big.push_str(&format!("# Section {i}\n\n"));
        big.push_str(&"Guidance text for this section, long enough to matter. ".repeat(20));
        big.push('\n');
    }
    assert!(big.len() > 24_000);

    let provider = CannedProvider {
        response: r#"[{"content": "One canned rule that stands in for this chunk of the file", "type": "semantic"}]"#.to_string(),
        calls: AtomicUsize::new(0),
    };
    let opts = AgentFileIngestOptions::default();
    let report = db
        .ingest_agent_file_llm(&big, &opts, &provider)
        .await
        .unwrap();

    assert_eq!(report.parsed_by, "llm");
    assert!(
        report.llm_chunks >= 2,
        "large files must chunk: {}",
        report.llm_chunks
    );
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        report.llm_chunks,
        "one model call per chunk"
    );
}

#[tokio::test]
async fn unusable_completion_falls_back_to_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());

    let provider = CannedProvider {
        response: "I cannot help with that.".to_string(),
        calls: AtomicUsize::new(0),
    };
    let opts = AgentFileIngestOptions::default();
    let md =
        "# Rules\n\n- Refunds over 200 dollars need approval\n- Keep replies to two paragraphs\n";
    let report = db
        .ingest_agent_file_llm(md, &opts, &provider)
        .await
        .unwrap();

    assert_eq!(
        report.parsed_by, "deterministic",
        "a bad completion must never produce an empty ingest"
    );
    assert!(report.stored >= 2);
}

#[tokio::test]
async fn exemplars_store_as_activation_anchors_not_rules() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());

    let provider = CannedProvider {
        response: r#"[
  {"content": "When changing code, make longterm root cause fixes only, never quick patches", "type": "procedural", "trigger": "code-change", "exemplars": ["fix this bug in the auth flow", "refactor the retry logic", "add rate limiting to the API"]},
  {"content": "Commit messages are a single line with a conventional prefix", "type": "procedural", "trigger": "git-commit"}
]"#
        .to_string(),
        calls: AtomicUsize::new(0),
    };
    let opts = AgentFileIngestOptions::default();
    let report = db
        .ingest_agent_file_llm("# Ways of working\n", &opts, &provider)
        .await
        .unwrap();
    assert_eq!(report.parsed_by, "llm");
    // 2 rules + 3 exemplars all stored through the normal pipeline.
    assert_eq!(report.stored, 5, "{report:?}");

    // Exemplars carry the mode-exemplar tag for their trigger and never the
    // trigger tag itself, so the action channel returns only real rules.
    let rules = db.recall_for_action("code-change", None, None, 10).unwrap();
    assert_eq!(rules.len(), 1, "{rules:?}");
    assert!(rules[0].content.contains("root cause"));
    let anchors = db.recall_for_action("git-commit", None, None, 10).unwrap();
    assert_eq!(anchors.len(), 1, "{anchors:?}");
}
