//! Verified erase: the cascade over derived memories, the on disk scrub
//! checked byte by byte across every file in the database directory, crash
//! safety, legacy copies, recall quality, and the receipt.

use std::collections::HashSet;
use std::path::Path;

use mentedb::cognitive::trajectory::{DecisionState, TrajectoryNode};
use mentedb::prelude::*;
use mentedb::{CognitiveConfig, EraseOptions, EraseReason, MemoryPatch};

/// Inference and dedup off, so the only edges are the ones a test makes.
fn config() -> CognitiveConfig {
    CognitiveConfig {
        write_inference: false,
        memory_dedup: false,
        ..CognitiveConfig::default()
    }
}

fn open(dir: &Path) -> MenteDb {
    MenteDb::open_with_config(dir, config()).expect("open")
}

/// Deterministic pseudo random unit-ish vector, distinct per seed.
fn embedding(seed: u64, dim: usize) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..dim)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn store(db: &MenteDb, content: &str, seed: u64) -> MemoryId {
    let node = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        content.to_string(),
        embedding(seed, 32),
    );
    let id = node.id;
    db.store(node).expect("store");
    id
}

/// `derived` was made from `source`, as the engine records it.
fn derive(db: &MenteDb, derived: MemoryId, source: MemoryId) {
    db.relate(MemoryEdge {
        source: derived,
        target: source,
        edge_type: EdgeType::Derived,
        weight: 1.0,
        created_at: 0,
        valid_from: None,
        valid_until: None,
        label: None,
    })
    .expect("relate");
}

fn ids(receipt: &mentedb::ErasureReceipt) -> HashSet<MemoryId> {
    receipt.erased.iter().map(|e| e.id).collect()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

/// Every file under `dir` holding `needle`, by path relative to `dir`.
/// WAL payloads are compressed, so WAL files are also read entry by entry.
fn files_with(dir: &Path, needle: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let rel = p.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
            let bytes = std::fs::read(&p).unwrap();
            let mut hit = contains(&bytes, needle);
            if rel.ends_with(".wal") || rel.starts_with("wal.log") {
                hit |= mentedb::storage::Wal::read_entries(&p)
                    .unwrap_or_default()
                    .iter()
                    .any(|e| contains(&e.data, needle));
            }
            if hit {
                out.push(rel);
            }
        }
    }
    out.sort();
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ---- cascade ----

/// turn <- fact <- consolidated, plus an entity node over two facts and a
/// second turn with its own fact.
struct Chain {
    turn: MemoryId,
    sibling_fact: MemoryId,
    fact: MemoryId,
    consolidated: MemoryId,
    entity: MemoryId,
    other_turn: MemoryId,
    other_fact: MemoryId,
}

fn chain(db: &MenteDb) -> Chain {
    let turn = store(
        db,
        "turn: I moved to Lisbon last spring and adopted a cat",
        1,
    );
    let fact = store(db, "fact: the user lives in Lisbon", 2);
    let sibling_fact = store(db, "fact: the user has a cat", 3);
    let other_turn = store(db, "turn: I work on compilers for a living", 4);
    let other_fact = store(db, "fact: the user works on compilers", 5);
    let consolidated = store(db, "summary: lives in Lisbon, works on compilers", 6);
    let entity = store(db, "entity: Lisbon, a city", 7);
    derive(db, fact, turn);
    derive(db, sibling_fact, turn);
    derive(db, other_fact, other_turn);
    derive(db, consolidated, fact);
    derive(db, consolidated, other_fact);
    derive(db, entity, fact);
    derive(db, entity, other_fact);
    Chain {
        turn,
        sibling_fact,
        fact,
        consolidated,
        entity,
        other_turn,
        other_fact,
    }
}

#[test]
fn erase_cascades_to_derived_memories_and_keeps_sources() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = chain(&db);

    let receipt = db.erase(c.fact, EraseOptions::default()).unwrap();
    assert_eq!(
        ids(&receipt),
        HashSet::from([c.fact, c.consolidated, c.entity])
    );
    assert_eq!(receipt.ancestors_kept, vec![c.turn]);
    for gone in [c.fact, c.consolidated, c.entity] {
        assert!(db.get_memory(gone).is_err());
    }
    // Sources and the other members of the shared nodes stay.
    for kept in [c.turn, c.sibling_fact, c.other_turn, c.other_fact] {
        db.get_memory(kept).unwrap();
    }
    let reason = |id| receipt.erased.iter().find(|e| e.id == id).unwrap().reason;
    assert_eq!(reason(c.fact), EraseReason::Requested);
    assert_eq!(reason(c.entity), EraseReason::Derived);
    assert!(!receipt.verification.unwrap().residual_found);
}

#[test]
fn include_sources_erases_ancestors_and_their_other_descendants() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = chain(&db);

    let receipt = db
        .erase(
            c.consolidated,
            EraseOptions {
                include_sources: true,
                ..EraseOptions::default()
            },
        )
        .unwrap();
    // Up: both facts and both turns. Down from them: the sibling fact (from
    // the same turn) and the entity node.
    assert_eq!(
        ids(&receipt),
        HashSet::from([
            c.consolidated,
            c.fact,
            c.other_fact,
            c.turn,
            c.other_turn,
            c.sibling_fact,
            c.entity,
        ])
    );
    assert!(receipt.ancestors_kept.is_empty());
    assert_eq!(db.memory_count(), 0);
}

#[test]
fn erasing_a_source_without_include_sources_keeps_unrelated_members() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = chain(&db);
    // Erasing the turn takes both facts from it, the consolidation and the
    // entity node built on one of them, never the other turn or its fact.
    let receipt = db.erase(c.turn, EraseOptions::default()).unwrap();
    assert_eq!(
        ids(&receipt),
        HashSet::from([c.turn, c.fact, c.sibling_fact, c.consolidated, c.entity])
    );
    assert!(receipt.ancestors_kept.is_empty());
    db.get_memory(c.other_turn).unwrap();
    db.get_memory(c.other_fact).unwrap();
}

#[test]
fn exact_duplicates_go_with_either_copy() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let text = "the user's passport number is X1234567";
    // Exact dedup links the keeper to the duplicate it absorbed.
    let keeper = store(&db, text, 10);
    let duplicate = store(&db, text, 11);
    let bystander = store(&db, "the user prefers window seats", 12);
    derive(&db, keeper, duplicate);

    // Erasing the keeper: the duplicate is its Derived target (an ancestor),
    // but it holds the same bytes, so it goes too.
    let receipt = db.erase(keeper, EraseOptions::default()).unwrap();
    assert_eq!(ids(&receipt), HashSet::from([keeper, duplicate]));
    let reason = receipt
        .erased
        .iter()
        .find(|e| e.id == duplicate)
        .unwrap()
        .reason;
    assert_eq!(reason, EraseReason::Identical);
    assert!(receipt.ancestors_kept.is_empty());
    db.get_memory(bystander).unwrap();

    // Erasing the duplicate: the keeper is derived from it.
    let keeper = store(&db, text, 13);
    let duplicate = store(&db, text, 14);
    derive(&db, keeper, duplicate);
    let receipt = db.erase(duplicate, EraseOptions::default()).unwrap();
    assert_eq!(ids(&receipt), HashSet::from([keeper, duplicate]));
}

#[test]
fn a_cascade_over_the_limit_is_refused_before_anything_changes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = chain(&db);
    let err = db
        .erase(
            c.turn,
            EraseOptions {
                max_cascade: 2,
                ..EraseOptions::default()
            },
        )
        .unwrap_err();
    assert!(err.to_string().contains("max_cascade"), "{err}");
    assert_eq!(db.memory_count(), 7);
    assert!(db.erase(MemoryId::new(), EraseOptions::default()).is_err());
}

// ---- bytes on disk ----

const GONE: &str = "ERASE-ME-7d41c2: my bank PIN is 4071";
const KEPT: [&str; 3] = [
    "KEEP-ME-a01: the user likes sailing",
    "KEEP-ME-b02: the user is allergic to peanuts",
    "KEEP-ME-c03: the user's sister lives in Oslo",
];
const GONE_TAG: &str = "topic:erase-me-7d41c2-only-tag";

/// A database where the memory to erase has bytes in every place they can
/// end up: the record file (a moved old copy in free space and the live
/// record), the WAL, the index and graph snapshots, a tag only it carries,
/// the speculative cache, the trajectory, and the cognitive snapshots.
fn seeded(dir: &Path) -> (MenteDb, MemoryId, Vec<f32>, MemoryId, Vec<MemoryId>) {
    let db = open(dir);
    let kept: Vec<MemoryId> = KEPT
        .iter()
        .enumerate()
        .map(|(i, t)| store(&db, t, 100 + i as u64))
        .collect();
    let gone = store(&db, GONE, 999);
    let gone_vec = embedding(999, 32);
    let fact = store(&db, "derived: ERASE-ME-7d41c2 PIN fact 4071", 998);
    derive(&db, fact, gone);
    db.flush_full().unwrap();
    // Grow the record so it moves: the old copy stays in the freed extent.
    db.update_memory_fields(
        gone,
        &MemoryPatch {
            tags: Some(vec![GONE_TAG.to_string(), "x".repeat(2_000)]),
            ..Default::default()
        },
    )
    .unwrap();
    db.record_trajectory_turn(TrajectoryNode {
        turn_id: 1,
        topic_embedding: vec![],
        topic_summary: GONE.to_string(),
        decision_state: DecisionState::Investigating,
        open_questions: vec![],
        timestamp: 1,
    });
    db.pre_assemble_speculative(vec!["bank pin".to_string()], |_| {
        Some((format!("context: {GONE}"), vec![gone], None))
    });
    db.flush_full().unwrap();
    // Written after the last checkpoint, so the WAL carries it too.
    db.update_memory_fields(
        gone,
        &MemoryPatch {
            tags: Some(vec![GONE_TAG.to_string()]),
            ..Default::default()
        },
    )
    .unwrap();
    (db, gone, gone_vec, fact, kept)
}

#[test]
fn erase_leaves_no_byte_of_the_memory_in_any_file() {
    let dir = tempfile::tempdir().unwrap();
    let (db, gone, gone_vec, fact, kept) = seeded(dir.path());
    // The bytes are everywhere before the erase.
    let before = files_with(dir.path(), GONE.as_bytes());
    for f in ["records.db", "records.wal", "cognitive/speculative.json"] {
        assert!(before.contains(&f.to_string()), "{f} in {before:?}");
    }
    assert!(!files_with(dir.path(), &f32_bytes(&gone_vec)).is_empty());
    assert!(!files_with(dir.path(), GONE_TAG.as_bytes()).is_empty());

    // Old versions of replaced snapshots: a reader holding the vector index
    // snapshot open sees what its unlinked blocks hold after the erase.
    let mut old_index = std::fs::File::open(dir.path().join("indexes/hnsw.bin")).unwrap();
    let receipt = db.erase(gone, EraseOptions::default()).unwrap();
    assert_eq!(ids(&receipt), HashSet::from([gone, fact]));
    let v = receipt.verification.as_ref().unwrap();
    assert!(!v.residual_found, "{:?}", v.hits);
    #[cfg(unix)]
    {
        use std::io::Read;
        let mut bytes = Vec::new();
        old_index.read_to_end(&mut bytes).unwrap();
        assert!(!bytes.is_empty() && bytes.iter().all(|b| *b == 0));
        assert!(receipt.steps.replaced_files_zeroed > 0);
    }
    #[cfg(not(unix))]
    let _ = &mut old_index;
    assert!(v.files_scanned >= 5);
    db.close().unwrap();

    assert_eq!(
        files_with(dir.path(), GONE.as_bytes()),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), b"ERASE-ME-7d41c2"),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), &f32_bytes(&gone_vec)),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), &f32_bytes(&embedding(998, 32))),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), GONE_TAG.as_bytes()),
        Vec::<String>::new()
    );
    // Its BM25 terms, lowercased by the tokenizer, went with its postings.
    assert_eq!(files_with(dir.path(), b"7d41c2"), Vec::<String>::new());
    // The neighbours are still on disk and readable.
    for t in KEPT {
        assert!(files_with(dir.path(), t.as_bytes()).contains(&"records.db".to_string()));
    }
    let db = open(dir.path());
    assert_eq!(db.memory_count(), 3);
    for (i, id) in kept.iter().enumerate() {
        assert_eq!(db.get_memory(*id).unwrap().content, KEPT[i]);
        let top = db
            .recall_similar(&embedding(100 + i as u64, 32), 1)
            .unwrap();
        assert_eq!(top[0].0, *id);
    }
    assert!(db.get_memory(gone).is_err());
    assert!(db.speculative_cache_entries().is_empty());
    assert!(db.get_trajectory().is_empty());
}

#[test]
fn erase_then_crash_never_brings_anything_back() {
    let dir = tempfile::tempdir().unwrap();
    let (db, gone, gone_vec, fact, kept) = seeded(dir.path());
    let receipt = db.erase(gone, EraseOptions::default()).unwrap();
    assert!(!receipt.verification.unwrap().residual_found);
    // No close: snapshots, caches and the lock are left as a crash leaves
    // them.
    db.simulate_crash();

    let db = open(dir.path());
    assert!(db.get_memory(gone).is_err());
    assert!(db.get_memory(fact).is_err());
    assert_eq!(db.memory_count(), 3);
    for (i, id) in kept.iter().enumerate() {
        assert_eq!(db.get_memory(*id).unwrap().content, KEPT[i]);
    }
    assert!(
        db.recall_similar(&gone_vec, 5)
            .unwrap()
            .iter()
            .all(|(m, _)| *m != gone)
    );
    db.close().unwrap();
    assert_eq!(
        files_with(dir.path(), GONE.as_bytes()),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), &f32_bytes(&gone_vec)),
        Vec::<String>::new()
    );
}

#[test]
fn a_crash_after_the_records_is_finished_by_the_next_open() {
    // The records are erased (and their extents zeroed) but the free space,
    // the WAL, the snapshots and the legacy copies are not yet scrubbed when
    // the process dies. The next open sees the pending marker and finishes.
    let dir = tempfile::tempdir().unwrap();
    let (db, gone, gone_vec, fact, kept) = seeded(dir.path());
    plant_legacy_copies(dir.path(), GONE);
    let err = db
        .erase(
            gone,
            EraseOptions {
                fault_after_records: true,
                ..EraseOptions::default()
            },
        )
        .unwrap_err();
    assert!(err.to_string().contains("fault injected"), "{err}");
    db.simulate_crash();
    assert!(dir.path().join("erase.pending").exists());
    // Not finished yet: copies remain (a moved old copy in free space, the
    // legacy files, the old index snapshot with the vector).
    assert!(!files_with(dir.path(), GONE.as_bytes()).is_empty());
    assert!(!files_with(dir.path(), &f32_bytes(&gone_vec)).is_empty());

    let db = open(dir.path());
    assert!(!dir.path().join("erase.pending").exists());
    assert!(db.get_memory(gone).is_err());
    assert!(db.get_memory(fact).is_err());
    for (i, id) in kept.iter().enumerate() {
        assert_eq!(db.get_memory(*id).unwrap().content, KEPT[i]);
    }
    assert_eq!(
        files_with(dir.path(), GONE.as_bytes()),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), b"ERASE-ME-7d41c2"),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), &f32_bytes(&gone_vec)),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), GONE_TAG.as_bytes()),
        Vec::<String>::new()
    );
    db.close().unwrap();
    assert_eq!(
        files_with(dir.path(), GONE.as_bytes()),
        Vec::<String>::new()
    );
    assert_eq!(
        files_with(dir.path(), &f32_bytes(&gone_vec)),
        Vec::<String>::new()
    );
    let db = open(dir.path());
    assert_eq!(db.memory_count(), 3);
}

#[test]
fn a_crash_part_way_through_a_cascade_leaves_the_request_to_retry() {
    // The cascade erases derived memories first and the requested one last,
    // so a crash in between leaves the requested memory, and erasing it
    // again finishes the job.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = chain(&db);
    db.flush_full().unwrap();
    // The state after the derived memories went and before the request did.
    db.forget(c.consolidated).unwrap();
    db.forget(c.entity).unwrap();
    db.simulate_crash();

    let db = open(dir.path());
    db.get_memory(c.fact).unwrap();
    let receipt = db.erase(c.fact, EraseOptions::default()).unwrap();
    assert_eq!(ids(&receipt), HashSet::from([c.fact]));
    assert!(db.get_memory(c.fact).is_err());
    db.get_memory(c.turn).unwrap();
}

// ---- legacy copies and the scan itself ----

/// Old format files a migration kept, holding a copy of `content`.
fn plant_legacy_copies(dir: &Path, content: &str) {
    for name in ["pages.db.v1", "wal.log.v1", "memory_map.bin.v1"] {
        let mut bytes = vec![7u8; 4096];
        bytes.extend_from_slice(content.as_bytes());
        std::fs::write(dir.join(name), bytes).unwrap();
    }
}

#[test]
fn erase_removes_legacy_copies() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let gone = store(&db, GONE, 999);
    plant_legacy_copies(dir.path(), GONE);
    let receipt = db.erase(gone, EraseOptions::default()).unwrap();
    let mut removed = receipt.steps.legacy_files_removed.clone();
    removed.sort();
    assert_eq!(
        removed,
        vec!["memory_map.bin.v1", "pages.db.v1", "wal.log.v1"]
    );
    for name in removed {
        assert!(!dir.path().join(name).exists());
    }
    assert!(!receipt.verification.unwrap().residual_found);
}

#[test]
fn verification_finds_what_a_skipped_step_leaves() {
    // Skipping a scrub step must show up in the receipt: the scan reads the
    // real files, it does not trust the steps.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let gone = store(&db, GONE, 999);
    plant_legacy_copies(dir.path(), GONE);
    let receipt = db
        .erase(
            gone,
            EraseOptions {
                remove_legacy_copies: false,
                ..EraseOptions::default()
            },
        )
        .unwrap();
    let v = receipt.verification.unwrap();
    assert!(v.residual_found);
    let files: HashSet<&str> = v.hits.iter().map(|h| h.file.as_str()).collect();
    assert_eq!(
        files,
        HashSet::from(["pages.db.v1", "wal.log.v1", "memory_map.bin.v1"])
    );
    assert!(
        v.hits
            .iter()
            .all(|h| h.memory == gone && h.kind == "content")
    );

    // An old copy left in record file free space by a move, with the free
    // space scrub skipped.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let gone = store(&db, GONE, 999);
    store(&db, KEPT[0], 100);
    db.update_memory_fields(
        gone,
        &MemoryPatch {
            tags: Some(vec!["y".repeat(2_000)]),
            ..Default::default()
        },
    )
    .unwrap();
    let receipt = db
        .erase(
            gone,
            EraseOptions {
                scrub_free_space: false,
                ..EraseOptions::default()
            },
        )
        .unwrap();
    assert_eq!(receipt.steps.free_space_bytes_zeroed, None);
    let v = receipt.verification.unwrap();
    assert!(v.residual_found);
    assert!(
        v.hits
            .iter()
            .any(|h| h.file == "records.db" && h.kind == "content"),
        "{:?}",
        v.hits
    );
    assert!(
        v.hits
            .iter()
            .any(|h| h.file == "records.db" && h.kind == "embedding"),
        "{:?}",
        v.hits
    );
}

// ---- recall and the receipt ----

#[test]
fn recall_of_unrelated_queries_is_unchanged_by_erase() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let all: Vec<MemoryId> = (0..400u64)
        .map(|i| store(&db, &format!("memory number {i} about topic {}", i % 17), i))
        .collect();
    let queries: Vec<Vec<f32>> = (0..40u64).map(|i| embedding(10_000 + i, 32)).collect();
    let k = 5;
    let before: Vec<Vec<MemoryId>> = queries
        .iter()
        .map(|q| {
            db.recall_similar(q, k)
                .unwrap()
                .iter()
                .map(|h| h.0)
                .collect()
        })
        .collect();
    let in_results: HashSet<MemoryId> = before.iter().flatten().copied().collect();
    // Erase a fifth of the memories, none of them in any result.
    let doomed: Vec<MemoryId> = all
        .iter()
        .copied()
        .filter(|id| !in_results.contains(id))
        .step_by(2)
        .take(80)
        .collect();
    assert_eq!(doomed.len(), 80);
    for id in &doomed {
        let r = db
            .erase(
                *id,
                EraseOptions {
                    verify: false,
                    ..EraseOptions::default()
                },
            )
            .unwrap();
        assert_eq!(r.steps.index_vectors_scrubbed, 1);
    }
    let after: Vec<Vec<MemoryId>> = queries
        .iter()
        .map(|q| {
            db.recall_similar(q, k)
                .unwrap()
                .iter()
                .map(|h| h.0)
                .collect()
        })
        .collect();
    assert_eq!(before, after);
    db.close().unwrap();
    let db = open(dir.path());
    let reopened: Vec<Vec<MemoryId>> = queries
        .iter()
        .map(|q| {
            db.recall_similar(q, k)
                .unwrap()
                .iter()
                .map(|h| h.0)
                .collect()
        })
        .collect();
    assert_eq!(before, reopened);
}

#[test]
fn receipt_carries_hashes_and_steps_never_content() {
    use sha2::{Digest, Sha256};
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    store(&db, KEPT[0], 100);
    let profile = db
        .store_user_profile("profile: likes sailing", UserId::nil(), AgentId::nil())
        .unwrap();
    let gone = store(&db, GONE, 999);
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    let receipt = db.erase(gone, EraseOptions::default()).unwrap();

    assert_eq!(receipt.requested, gone);
    assert_eq!(receipt.erased.len(), 1);
    let e = &receipt.erased[0];
    let hex = |b: &[u8]| -> String {
        Sha256::digest(b)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect()
    };
    assert_eq!(e.content_sha256, hex(GONE.as_bytes()));
    assert_eq!(
        e.embedding_sha256.as_deref(),
        Some(hex(&f32_bytes(&embedding(999, 32))).as_str())
    );
    assert!(receipt.erased_at_us >= started);
    assert_eq!(receipt.profiles_to_review, vec![profile]);
    let s = &receipt.steps;
    assert_eq!(s.records_zeroed, 1);
    assert_eq!(s.index_vectors_scrubbed, 1);
    assert!(s.free_space_bytes_zeroed.is_some());
    assert!(s.wal_scrubbed && s.snapshots_rewritten);
    let v = receipt.verification.as_ref().unwrap();
    assert!(!v.residual_found && v.unreadable.is_empty() && v.bytes_scanned > 0);

    let json = serde_json::to_string(&receipt).unwrap();
    assert!(!json.contains("ERASE-ME"), "{json}");
    for key in [
        "erased",
        "content_sha256",
        "ancestors_kept",
        "erased_at_us",
        "steps",
        "residual_found",
    ] {
        assert!(json.contains(key), "{key} in {json}");
    }
    let back: mentedb::ErasureReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(back.erased[0].content_sha256, e.content_sha256);

    // Short content is not searched for (it would match unrelated bytes);
    // the receipt says so.
    let short = store(&db, "PIN", 5);
    let r = db.erase(short, EraseOptions::default()).unwrap();
    assert_eq!(r.verification.unwrap().content_not_checked, vec![short]);
}
