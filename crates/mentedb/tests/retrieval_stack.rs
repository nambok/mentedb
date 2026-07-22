//! Internal end-to-end validation of the retrieval stack shipped in 0.22-0.23:
//! the contextual-retrieval hook, MMR diversity, the pluggable reranker, and
//! project scope weighting. Each scenario runs against a real (deterministic
//! hash) embedder and the full hybrid recall path, so it exercises the same code
//! production does, not a shortcut. These are the "do the updates actually work
//! together" checks.

use mentedb::prelude::*;
use mentedb::reranker::{RerankCandidate, Reranker};
use mentedb::{CognitiveConfig, MenteDb};
use mentedb_embedding::{EmbeddingProvider, HashEmbeddingProvider};

const DIM: usize = 384;

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

fn emb(text: &str) -> Vec<f32> {
    HashEmbeddingProvider::new(DIM).embed(text).unwrap()
}

fn open(cfg: CognitiveConfig) -> (MenteDb, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_embedder_and_config(
        dir.path(),
        Box::new(HashEmbeddingProvider::new(DIM)),
        cfg,
    )
    .unwrap();
    (db, dir)
}

/// Store a memory, embedding the contextualized text (the way a caller using the
/// contextual hook should), with optional context, project tag, and extra tags.
fn store(db: &MenteDb, content: &str, context: Option<&str>, project: Option<&str>) -> MemoryId {
    let mut node = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        content.to_string(),
        vec![],
    );
    if let Some(ctx) = context {
        node = node.with_context(ctx);
    }
    if let Some(p) = project {
        node.tags.push(format!("scope:project:{p}"));
    }
    node.embedding = emb(&node.indexed_text());
    let id = node.id;
    db.store(node).unwrap();
    id
}

fn recall_ids(db: &MenteDb, query_text: &str, k: usize, project: Option<&str>) -> Vec<MemoryId> {
    db.recall_hybrid_at(
        &emb(query_text),
        Some(query_text),
        k,
        now_us(),
        None,
        None,
        project,
    )
    .unwrap()
    .into_iter()
    .map(|(id, _)| id)
    .collect()
}

/// Contextual hook: a memory whose content never mentions the query terms is
/// still found because the caller-supplied context, indexed and embedded
/// alongside it, carries them.
#[test]
fn contextual_hook_surfaces_situated_memory() {
    let (db, _dir) = open(CognitiveConfig::default());
    let target = store(
        &db,
        "rolled back to the previous release tag",
        Some("resolving the payments service outage"),
        None,
    );
    store(&db, "updated the office wifi password", None, None);
    store(&db, "scheduled the quarterly planning offsite", None, None);

    let hits = recall_ids(&db, "payments outage", 3, None);
    assert!(
        hits.contains(&target),
        "a memory situated by its context must be found by context-only terms"
    );
    // The stored content is exactly what was written; context never leaks in.
    assert_eq!(
        db.get_memory(target).unwrap().content,
        "rolled back to the previous release tag"
    );
}

/// MMR: three near-duplicate memories plus one distinct. Without MMR the near
/// duplicates crowd the top; with MMR the distinct memory is pulled in.
#[test]
fn mmr_pulls_in_a_distinct_memory() {
    let dup_a = "the deployment pipeline uses blue green releases";
    let dup_b = "our deploy pipeline does blue green deployments";
    let dup_c = "blue green deployment is how the pipeline ships";
    let distinct = "the on call rotation hands off every monday";
    let query = "how do we deploy";

    // Default: MMR off. The near-duplicates dominate the top 2.
    let (db, _dir) = open(CognitiveConfig::default());
    store(&db, dup_a, None, None);
    store(&db, dup_b, None, None);
    store(&db, dup_c, None, None);
    let distinct_id = store(&db, distinct, None, None);
    let base = recall_ids(&db, query, 2, None);
    assert!(
        !base.contains(&distinct_id),
        "without MMR the near-duplicate deploy notes fill the top"
    );

    // MMR on: the distinct memory earns a slot.
    let (db2, _dir2) = open(CognitiveConfig {
        mmr_lambda: 0.5,
        ..CognitiveConfig::default()
    });
    store(&db2, dup_a, None, None);
    store(&db2, dup_b, None, None);
    store(&db2, dup_c, None, None);
    let distinct_id2 = store(&db2, distinct, None, None);
    let diversified = recall_ids(&db2, query, 3, None);
    assert!(
        diversified.contains(&distinct_id2),
        "MMR should surface the distinct memory over redundant duplicates"
    );
}

/// Pluggable reranker on the main hybrid path: an installed reranker's ordering
/// wins over the first pass.
#[test]
fn reranker_drives_final_order() {
    struct PromotePinned;
    impl Reranker for PromotePinned {
        fn rerank(&self, _q: &str, cands: &[RerankCandidate<'_>]) -> Vec<(MemoryId, f32)> {
            cands
                .iter()
                .map(|c| {
                    (
                        c.id,
                        if c.content.contains("PINNED") {
                            1.0
                        } else {
                            0.0
                        },
                    )
                })
                .collect()
        }
    }

    let (mut db, _dir) = open(CognitiveConfig::default());
    let pinned = store(&db, "PINNED note about the retrieval roadmap", None, None);
    store(
        &db,
        "a much closer note about the retrieval roadmap",
        None,
        None,
    );

    db.set_reranker(Box::new(PromotePinned));
    let hits = recall_ids(&db, "retrieval roadmap", 2, None);
    assert_eq!(
        hits.first().copied(),
        Some(pinned),
        "the reranker must decide the top slot"
    );
}

/// Project scope weighting: with a current project, same-topic memories from that
/// project outrank ones tagged for a different project.
#[test]
fn scope_weight_prefers_current_project() {
    let (db, _dir) = open(CognitiveConfig::default());
    // Same topic, but distinct wording per project so their embeddings differ
    // (identical vectors can orphan an HNSW node, which is a separate quirk from
    // the scope weighting under test).
    let in_project = store(
        &db,
        "the api gateway rate limit is a thousand requests per minute for alpha",
        None,
        Some("alpha"),
    );
    let other_project = store(
        &db,
        "the api gateway rate limit caps at a thousand requests each minute for beta",
        None,
        Some("beta"),
    );

    let hits = recall_ids(&db, "api gateway rate limit", 2, Some("alpha"));
    let pos = |id: MemoryId| hits.iter().position(|h| *h == id);
    assert!(
        pos(in_project) < pos(other_project),
        "the current project's memory must rank above the other project's: {hits:?}"
    );
    // Searching across all projects (no current project) does not penalize either.
    let all = recall_ids(&db, "api gateway rate limit", 2, None);
    assert!(all.contains(&in_project) && all.contains(&other_project));
}
