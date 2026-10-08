//! A process killed mid-write used to leave an empty or truncated index
//! snapshot, and every later open of the database failed with a
//! serialization error. The indexes are derived data, so an unreadable
//! snapshot must cost a rebuild from storage, never the database.

use mentedb::MenteDb;
use mentedb::prelude::{AgentId, MemoryNode, MemoryType};
use mentedb_embedding::hash_provider::HashEmbeddingProvider;

fn open(dir: &std::path::Path) -> MenteDb {
    MenteDb::open_with_embedder(dir, Box::new(HashEmbeddingProvider::new(64))).expect("open")
}

fn seed(dir: &std::path::Path) -> Vec<String> {
    let db = open(dir);
    let texts: Vec<String> = (0..20)
        .map(|i| format!("the gateway deploys with a tag push, note {i}"))
        .collect();
    for t in &texts {
        let emb = db.embed_text(t).unwrap().unwrap();
        db.store(MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            t.clone(),
            emb,
        ))
        .unwrap();
    }
    db.flush_full().unwrap();
    texts
}

fn index_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(dir.join("indexes"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    files
}

#[test]
fn every_index_snapshot_truncated_still_opens_and_recalls() {
    let probe = tempfile::tempdir().unwrap();
    seed(probe.path());
    let names: Vec<_> = index_files(probe.path())
        .into_iter()
        .map(|p| p.file_name().unwrap().to_owned())
        .collect();
    assert!(!names.is_empty(), "flush_full writes index snapshots");

    for name in names {
        for corrupt in [&b""[..], &b"\x00\x01garbage"[..]] {
            let dir = tempfile::tempdir().unwrap();
            seed(dir.path());
            std::fs::write(dir.path().join("indexes").join(&name), corrupt).unwrap();

            let db = open(dir.path());
            let q = db
                .embed_text("how does the gateway deploy")
                .unwrap()
                .unwrap();
            let hits = db.recall_similar(&q, 5).unwrap();
            assert!(
                !hits.is_empty(),
                "{name:?} corrupted: database opened but recall found nothing"
            );
            assert_eq!(db.memory_count(), 20);
        }
    }
}

#[test]
fn snapshots_leave_no_temp_files_behind() {
    let dir = tempfile::tempdir().unwrap();
    seed(dir.path());
    let leftovers: Vec<_> = index_files(dir.path())
        .into_iter()
        .filter(|p| p.extension().is_some_and(|e| e == "tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}
