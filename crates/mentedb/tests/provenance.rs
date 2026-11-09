//! Write time provenance: stamped on the first write, never changed after,
//! carried onto derived memories, indexed for session and source filters,
//! read back from databases written before it existed, and the basis of
//! erase by session.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use mentedb::prelude::*;
use mentedb::{
    CognitiveConfig, EraseOptions, EraseReason, HybridOptions, MemoryPatch, Provenance,
    ProvenanceFilter, SourceKind,
};

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

/// Deterministic pseudo random vector, distinct per seed.
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

fn node(content: &str, seed: u64) -> MemoryNode {
    MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        content.to_string(),
        embedding(seed, 32),
    )
}

fn chat(session: &str) -> Provenance {
    Provenance::new(SourceKind::Chat, AgentId::nil()).with_session(session)
}

fn store_in(db: &MenteDb, content: &str, seed: u64, p: Provenance) -> MemoryId {
    let n = node(content, seed).with_provenance(p);
    let id = n.id;
    db.store(n).expect("store");
    id
}

fn set(ids: impl IntoIterator<Item = MemoryId>) -> HashSet<MemoryId> {
    ids.into_iter().collect()
}

// ---- stamping and immutability ----

#[test]
fn first_write_without_provenance_is_stamped_unspecified() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let agent = AgentId::new();
    let mut n = node("a memory written with no provenance at all", 1);
    n.agent_id = agent;
    let id = n.id;
    let before = now_us();
    db.store(n).unwrap();
    let p = db.provenance_of(id).unwrap().expect("stamped");
    assert_eq!(p.source, SourceKind::Unspecified);
    assert_eq!(p.written_by, agent);
    assert!(p.written_at >= before);
    assert_eq!(p.session_id, None);
    assert_eq!(
        set(db.memory_ids_from_source(&SourceKind::Unspecified)),
        set([id])
    );
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}

#[test]
fn restore_of_an_existing_id_keeps_the_original_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let original = chat("s1").with_source_ref("msg-1");
    let id;
    {
        let db = open(dir.path());
        id = store_in(
            &db,
            "the user prefers tea in the morning",
            1,
            original.clone(),
        );

        // A re-store with new content and a different provenance.
        let mut again = db.get_memory(id).unwrap();
        again.content = "the user prefers coffee in the morning".into();
        again.provenance = Some(
            Provenance::new(SourceKind::Api, AgentId::new())
                .with_session("s2")
                .with_source_ref("forged"),
        );
        db.store(again).unwrap();

        let got = db.get_memory(id).unwrap();
        assert_eq!(got.content, "the user prefers coffee in the morning");
        assert_eq!(got.provenance.as_ref(), Some(&original));

        // A re-store that drops provenance keeps it too.
        let mut bare = db.get_memory(id).unwrap();
        bare.provenance = None;
        db.store(bare).unwrap();
        assert_eq!(db.provenance_of(id).unwrap().as_ref(), Some(&original));

        // The index follows the stored provenance, not the attempted one.
        assert_eq!(set(db.memory_ids_in_session("s1")), set([id]));
        assert!(db.memory_ids_in_session("s2").is_empty());
        assert!(db.memory_ids_from_source(&SourceKind::Api).is_empty());

        // The same through the batch path.
        let mut batched = db.get_memory(id).unwrap();
        batched.provenance = Some(chat("s3"));
        db.store_batch(vec![batched]).unwrap();
        assert_eq!(db.provenance_of(id).unwrap().as_ref(), Some(&original));
        assert!(db.memory_ids_in_session("s3").is_empty());
        db.close().unwrap();
    }
    let db = open(dir.path());
    assert_eq!(db.provenance_of(id).unwrap().as_ref(), Some(&original));
    assert_eq!(set(db.memory_ids_in_session("s1")), set([id]));
}

#[test]
fn updates_in_place_keep_provenance_and_its_index() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let original = chat("s1");
    let id = store_in(
        &db,
        "the deploy runs on every merge to main",
        1,
        original.clone(),
    );

    db.update_memory_fields(
        id,
        &MemoryPatch {
            tags: Some(vec!["topic:ci".into()]),
            ..MemoryPatch::default()
        },
    )
    .unwrap();
    db.invalidate_memory(id, now_us() + 1).unwrap();
    db.apply_decay_global().unwrap();

    assert_eq!(db.provenance_of(id).unwrap().as_ref(), Some(&original));
    // reindex_tags rebuilt the bitmap entries; the provenance keys survive.
    assert_eq!(set(db.memory_ids_in_session("s1")), set([id]));
    assert_eq!(set(db.memory_ids_from_source(&SourceKind::Chat)), set([id]));
}

// ---- derived memories ----

#[test]
fn consolidation_records_origins_and_a_shared_session() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let a = store_in(
        &db,
        "the user lives in Lisbon near the river",
        1,
        chat("s1"),
    );
    let b = store_in(&db, "the user lives in Lisbon with a cat", 2, chat("s1"));
    let c = store_in(&db, "the user lives in Lisbon and cycles", 3, chat("s2"));

    let same = db.consolidate_cluster(&[a, b]).unwrap();
    let p = db.provenance_of(same).unwrap().unwrap();
    assert_eq!(p.source, SourceKind::Derived);
    assert_eq!(set(p.derived_from.clone()), set([a, b]));
    assert_eq!(p.session_id.as_deref(), Some("s1"));
    assert!(db.memory_ids_in_session("s1").contains(&same));

    let mixed = db.consolidate_cluster(&[same, c]).unwrap();
    let p = db.provenance_of(mixed).unwrap().unwrap();
    assert_eq!(p.source, SourceKind::Derived);
    assert_eq!(set(p.derived_from.clone()), set([same, c]));
    assert_eq!(p.session_id, None, "origins from two sessions name none");
}

#[test]
fn extracted_and_community_memories_are_derived_from_their_sources() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let turn = store_in(
        &db,
        "turn: I moved to Lisbon and work on compilers",
        1,
        chat("s1"),
    );

    // Enrichment extraction: facts and entity nodes from the turn.
    let mut fact = node("fact: the user lives in Lisbon", 2);
    // A caller supplied provenance is replaced, keeping its reference.
    fact.provenance =
        Some(Provenance::new(SourceKind::Api, AgentId::nil()).with_source_ref("extractor-v2"));
    let fact_id = fact.id;
    let mut entity = node("entity: Lisbon, a city in Portugal", 3);
    entity.tags = vec!["entity:lisbon".into(), "category:place".into()];
    let entity_id = entity.id;
    let mut entity2 = node("entity: Porto, a city in Portugal", 4);
    entity2.tags = vec!["entity:porto".into(), "category:place".into()];
    let entity2_id = entity2.id;
    db.store_enrichment_memories(vec![fact, entity, entity2], &[turn])
        .unwrap();
    for id in [fact_id, entity_id, entity2_id] {
        let p = db.provenance_of(id).unwrap().unwrap();
        assert_eq!(p.source, SourceKind::Derived, "{id}");
        assert_eq!(p.derived_from, vec![turn]);
        assert_eq!(p.session_id.as_deref(), Some("s1"));
    }
    assert_eq!(
        db.provenance_of(fact_id)
            .unwrap()
            .unwrap()
            .source_ref
            .as_deref(),
        Some("extractor-v2")
    );

    // A community summary over the two entity nodes.
    let community = db
        .store_community_summary(
            "place",
            "places the user knows in Portugal",
            &["lisbon".into(), "porto".into()],
            UserId::nil(),
            AgentId::nil(),
        )
        .unwrap();
    let p = db.provenance_of(community).unwrap().unwrap();
    assert_eq!(p.source, SourceKind::Derived);
    assert_eq!(set(p.derived_from.clone()), set([entity_id, entity2_id]));
    assert_eq!(p.session_id.as_deref(), Some("s1"));

    // A profile names no origins.
    let profile = db
        .store_user_profile("likes Lisbon", UserId::nil(), AgentId::nil())
        .unwrap();
    let p = db.provenance_of(profile).unwrap().unwrap();
    assert_eq!(p.source, SourceKind::Derived);
    assert!(p.derived_from.is_empty());
    assert_eq!(p.session_id, None);
    // Rebuilding the profile in place keeps its provenance.
    db.store_user_profile("likes Lisbon and Porto", UserId::nil(), AgentId::nil())
        .unwrap();
    assert_eq!(db.provenance_of(profile).unwrap().unwrap(), p);
}

#[test]
fn process_turn_writes_chat_provenance_in_its_session() {
    let dir = tempfile::tempdir().unwrap();
    let db = MenteDb::open_with_embedder_and_config(
        dir.path(),
        Box::new(mentedb_embedding::HashEmbeddingProvider::new(32)),
        config(),
    )
    .unwrap();
    let mut delta = mentedb::context::DeltaTracker::new();
    let result = db
        .process_turn(
            &mentedb::process_turn::ProcessTurnInput {
                user_message: "I moved to Lisbon last spring".into(),
                assistant_response: Some("Nice, how do you like it?".into()),
                turn_id: 1,
                project_context: None,
                agent_id: None,
                user_id: None,
                session_id: Some("conv-7".into()),
            },
            &mut delta,
        )
        .unwrap();
    let episodic = result.episodic_id.expect("turn stored");
    let p = db.provenance_of(episodic).unwrap().unwrap();
    assert_eq!(p.source, SourceKind::Chat);
    assert_eq!(p.session_id.as_deref(), Some("conv-7"));
    assert!(db.memory_ids_in_session("conv-7").contains(&episodic));
}

// ---- filters ----

#[test]
fn list_recall_and_mql_filter_by_session_and_source_from_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let mut s1 = Vec::new();
    for i in 0..5 {
        s1.push(store_in(
            &db,
            &format!("session one memory number {i}"),
            i,
            chat("s1"),
        ));
    }
    let s1_file = store_in(
        &db,
        "session one file chunk about the handbook",
        10,
        Provenance::new(SourceKind::File, AgentId::nil())
            .with_session("s1")
            .with_source_ref("handbook.md"),
    );
    let mut others = Vec::new();
    for i in 0..20 {
        others.push(store_in(
            &db,
            &format!("session two memory number {i}"),
            100 + i,
            chat("s2"),
        ));
    }
    // Many session three memories packed around the query vector, so a
    // global top k (even over fetched) holds none of session one: the filter
    // must narrow before the search, not after.
    let query = embedding(100, 32);
    for i in 0..400u64 {
        let mut v = query.clone();
        v[(i % 32) as usize] += 0.001 * (1 + i / 32) as f32;
        let n = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            format!("session three crowd memory {i}"),
            v,
        )
        .with_provenance(chat("s3"));
        db.store(n).unwrap();
    }
    let legacy_like = node("a memory with no session", 500);
    let legacy_id = legacy_like.id;
    db.store(legacy_like).unwrap();

    let all_s1: HashSet<MemoryId> = s1.iter().copied().chain([s1_file]).collect();
    let now = now_us();

    // Listing.
    let page = db.list_memories_page_by(
        None,
        None,
        &ProvenanceFilter::session("s1"),
        true,
        now,
        0,
        100,
    );
    assert_eq!(set(page.memories.iter().map(|n| n.id)), all_s1);
    let page = db.list_memories_page_by(
        None,
        None,
        &ProvenanceFilter {
            session_id: Some("s1".into()),
            source: Some(SourceKind::File),
        },
        true,
        now,
        0,
        100,
    );
    assert_eq!(set(page.memories.iter().map(|n| n.id)), set([s1_file]));
    let page = db.list_memories_page_by(
        None,
        None,
        &ProvenanceFilter::source(SourceKind::Chat),
        true,
        now,
        0,
        100,
    );
    assert_eq!(page.memories.len(), 100, "the page limit");
    assert!(!page.memories.iter().any(|n| n.id == legacy_id));
    let none = db.list_memories_page_by(
        None,
        None,
        &ProvenanceFilter::session("nope"),
        true,
        now,
        0,
        100,
    );
    assert!(none.memories.is_empty());

    // Recall: query with an s2 memory's own vector, filtered to s1. The s2
    // memory is the nearest hit globally, yet none of s2 may come back, and
    // every s1 memory is still reachable.
    let filter = ProvenanceFilter::session("s1");
    let hits = db
        .recall_hybrid_scoped_with(
            &embedding(100, 32),
            None,
            10,
            now,
            None,
            false,
            None,
            None,
            None,
            None,
            &HybridOptions {
                provenance: Some(&filter),
                ..HybridOptions::default()
            },
        )
        .unwrap();
    assert_eq!(set(hits.iter().map(|(id, _)| *id)), all_s1);
    let unfiltered = db
        .recall_hybrid_scoped_with(
            &embedding(100, 32),
            None,
            1,
            now,
            None,
            false,
            None,
            None,
            None,
            None,
            &HybridOptions::default(),
        )
        .unwrap();
    assert!(!all_s1.contains(&unfiltered[0].0));

    // MQL.
    let got = db
        .query("RECALL memories WHERE session = \"s1\" LIMIT 100")
        .unwrap();
    assert_eq!(set(got.iter().map(|m| m.memory.id)), all_s1);
    let got = db
        .query("RECALL memories WHERE session = \"s1\" AND source = file LIMIT 100")
        .unwrap();
    assert_eq!(set(got.iter().map(|m| m.memory.id)), set([s1_file]));
    let got = db
        .query("RECALL memories WHERE source IN [\"file\", \"unspecified\"] LIMIT 100")
        .unwrap();
    assert_eq!(
        set(got.iter().map(|m| m.memory.id)),
        set([s1_file, legacy_id])
    );
    assert!(db.query("RECALL memories WHERE session > \"s1\"").is_err());

    // The sessions the index knows.
    assert_eq!(
        db.provenance_sessions(),
        vec!["s1".to_string(), "s2".to_string(), "s3".to_string()]
    );
}

// ---- erase by session ----

#[test]
fn erase_session_removes_the_session_and_what_was_derived_from_it() {
    let dir = tempfile::tempdir().unwrap();
    let a_text = "session one: the user is planning a trip to Kyoto in April";
    let b_text = "session one: the user is allergic to shellfish and peanuts";
    let keep_text = "session two: the user plays the cello on weekends";
    let (a, b, keep, consolidated, mixed);
    let a_emb = embedding(1, 32);
    {
        let db = open(dir.path());
        a = store_in(&db, a_text, 1, chat("s1"));
        b = store_in(&db, b_text, 2, chat("s1"));
        keep = store_in(&db, keep_text, 3, chat("s2"));
        // Derived inside the session: inherits it.
        consolidated = db.consolidate_cluster(&[a, b]).unwrap();
        // Derived across sessions: carries none, erased through the cascade.
        let mut fact = node("fact: the user travels and plays music", 9);
        fact.tags = vec!["kind:fact".into()];
        mixed = fact.id;
        db.store_enrichment_memories(vec![fact], &[b, keep])
            .unwrap();
        assert_eq!(db.provenance_of(mixed).unwrap().unwrap().session_id, None);

        let receipt = db.erase_session("s1", EraseOptions::default()).unwrap();
        assert_eq!(receipt.session_id, "s1");
        assert_eq!(set(receipt.requested.clone()), set([a, b, consolidated]));
        let erased: BTreeMap<MemoryId, EraseReason> =
            receipt.erased.iter().map(|e| (e.id, e.reason)).collect();
        assert_eq!(erased.len(), 4, "{erased:?}");
        assert_eq!(erased[&a], EraseReason::Requested);
        assert_eq!(erased[&consolidated], EraseReason::Requested);
        assert_eq!(erased[&mixed], EraseReason::Derived);
        assert!(!erased.contains_key(&keep));
        let v = receipt.verification.as_ref().expect("verified");
        assert!(!v.residual_found, "{:?}", v.hits);
        assert!(v.files_scanned > 0);

        assert!(db.memory_ids_in_session("s1").is_empty());
        assert!(!db.provenance_sessions().contains(&"s1".to_string()));
        assert!(db.get_memory(a).is_err());
        assert!(db.get_memory(mixed).is_err());
        assert_eq!(db.get_memory(keep).unwrap().content, keep_text);
        db.close().unwrap();
    }
    // Gone after a reopen, and not on disk.
    let db = open(dir.path());
    assert!(db.memory_ids_in_session("s1").is_empty());
    for id in [a, b, consolidated, mixed] {
        assert!(db.get_memory(id).is_err(), "{id} came back");
    }
    assert_eq!(set(db.memory_ids_in_session("s2")), set([keep]));
    drop(db);
    for needle in [a_text.as_bytes(), b_text.as_bytes()] {
        assert!(files_with(dir.path(), needle).is_empty());
    }
    let emb: Vec<u8> = a_emb.iter().flat_map(|f| f.to_le_bytes()).collect();
    assert!(files_with(dir.path(), &emb).is_empty());
}

#[test]
fn erase_of_an_unknown_session_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let keep = store_in(&db, "a memory in another session entirely", 1, chat("s2"));
    let receipt = db.erase_session("absent", EraseOptions::default()).unwrap();
    assert!(receipt.requested.is_empty());
    assert!(receipt.erased.is_empty());
    assert!(receipt.verification.is_none());
    assert!(db.get_memory(keep).is_ok());
}

#[test]
fn erase_session_respects_max_cascade_before_changing_anything() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let ids: Vec<MemoryId> = (0..3)
        .map(|i| store_in(&db, &format!("capped session memory {i}"), i, chat("big")))
        .collect();
    let err = db
        .erase_session(
            "big",
            EraseOptions {
                max_cascade: 2,
                ..EraseOptions::default()
            },
        )
        .unwrap_err();
    assert!(err.to_string().contains("session big"), "{err}");
    for id in ids {
        assert!(db.get_memory(id).is_ok());
    }
}

/// Every file under `dir` holding `needle`, WAL entries decoded.
fn files_with(dir: &Path, needle: &[u8]) -> Vec<String> {
    let contains = |hay: &[u8]| hay.windows(needle.len()).any(|w| w == needle);
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
            let mut hit = contains(&bytes);
            if rel.ends_with(".wal") {
                hit |= mentedb::storage::Wal::read_entries(&p)
                    .unwrap_or_default()
                    .iter()
                    .any(|e| contains(&e.data));
            }
            if hit {
                out.push(rel);
            }
        }
    }
    out
}

// ---- databases written before provenance ----

/// A database written by the 0.47.1 release (the last one before
/// provenance), copied so the fixture itself is never modified. Generated by
/// storing three memories and one edge with `mentedb = "=0.47.1"`, then
/// closing.
fn release_fixture() -> (tempfile::TempDir, PathBuf) {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/db_v0_47_1");
    let tmp = tempfile::tempdir().unwrap();
    let dst = tmp.path().join("db");
    copy_dir(&src, &dst);
    (tmp, dst)
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for e in std::fs::read_dir(src).unwrap().flatten() {
        let to = dst.join(e.file_name());
        if e.path().is_dir() {
            copy_dir(&e.path(), &to);
        } else {
            std::fs::copy(e.path(), to).unwrap();
        }
    }
}

fn file_bytes(dir: &Path, names: &[&str]) -> Vec<Vec<u8>> {
    names
        .iter()
        .map(|n| std::fs::read(dir.join(n)).unwrap())
        .collect()
}

fn fixture_id(n: u128) -> MemoryId {
    MemoryId(uuid::Uuid::from_u128(n))
}

#[test]
fn a_database_from_the_previous_release_opens_and_reads_unchanged() {
    let (_tmp, dir) = release_fixture();
    let record_files = ["records.db", "records_map.bin", "records.wal"];
    let before = file_bytes(&dir, &record_files);

    let db = open(&dir);
    // Nothing is rewritten on open.
    assert_eq!(
        file_bytes(&dir, &record_files),
        before,
        "open rewrote a record file"
    );

    assert_eq!(db.memory_count(), 3);
    let first = db.get_memory(fixture_id(0x100)).unwrap();
    assert_eq!(
        first.content,
        "The deploy pipeline runs on every merge to main"
    );
    assert_eq!(first.user_id, UserId(uuid::Uuid::from_u128(0xB)));
    assert_eq!(first.agent_id, AgentId(uuid::Uuid::from_u128(0xA)));
    assert_eq!(first.created_at, 1_700_000_000_000_000);
    assert_eq!(first.embedding, vec![1.0, 0.5, 0.0, 0.125]);
    assert!(first.tags.contains(&"session:legacy-1".to_string()));
    for i in 0..3u128 {
        assert_eq!(db.provenance_of(fixture_id(0x100 + i)).unwrap(), None);
    }
    // A pre provenance session tag is only a tag: no provenance session.
    assert!(db.memory_ids_in_session("legacy-1").is_empty());
    assert!(db.provenance_sessions().is_empty());
    // Search over the old index still works.
    let hits = db.recall_similar(&[1.0, 0.5, 0.0, 0.125], 1).unwrap();
    assert_eq!(hits[0].0, fixture_id(0x100));

    // A re-store of an old record does not invent an origin for it.
    let mut again = db.get_memory(fixture_id(0x101)).unwrap();
    again.content = "Alice prefers dark mode in every editor and terminal".into();
    again.provenance = Some(chat("new"));
    db.store(again).unwrap();
    assert_eq!(db.provenance_of(fixture_id(0x101)).unwrap(), None);
    assert!(db.memory_ids_in_session("new").is_empty());

    // New writes beside old records carry provenance.
    let mut fresh = MemoryNode::new(
        AgentId(uuid::Uuid::from_u128(0xA)),
        MemoryType::Semantic,
        "a memory written by this release".into(),
        vec![0.0, 1.0, 0.0, 0.0],
    )
    .with_provenance(chat("today"));
    fresh.user_id = UserId(uuid::Uuid::from_u128(0xB));
    let fresh_id = fresh.id;
    db.store(fresh).unwrap();
    db.close().unwrap();

    let db = open(&dir);
    assert_eq!(db.memory_count(), 4);
    assert_eq!(db.provenance_of(fixture_id(0x100)).unwrap(), None);
    assert_eq!(
        db.get_memory(fixture_id(0x101)).unwrap().content,
        "Alice prefers dark mode in every editor and terminal"
    );
    assert_eq!(set(db.memory_ids_in_session("today")), set([fresh_id]));
    let erased = db.erase_session("today", EraseOptions::default()).unwrap();
    assert_eq!(erased.requested, vec![fresh_id]);
    assert!(!erased.verification.unwrap().residual_found);
    assert_eq!(db.memory_count(), 3);
}
