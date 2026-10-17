//! Concurrent writes and recalls on one database must never deadlock.
//!
//! `store_batch` used to take the page map write lock and then the graph write
//! lock, while hybrid recall took the graph read lock and then the page map
//! read lock. When the two overlapped, each waited on the other forever, every
//! later call for that database queued behind them, and in production the
//! stuck threads filled the blocking pool until the whole server stopped.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use mentedb::MenteDb;
use mentedb::prelude::{AgentId, MemoryNode, MemoryType};
use mentedb_embedding::hash_provider::HashEmbeddingProvider;

const ROUNDS: usize = 120;

fn node(db: &MenteDb, text: &str) -> MemoryNode {
    let emb = db.embed_text(text).unwrap().unwrap();
    MemoryNode::new(AgentId::nil(), MemoryType::Semantic, text.to_string(), emb)
}

#[test]
fn batch_stores_and_hybrid_recalls_never_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(
        MenteDb::open_with_embedder(dir.path(), Box::new(HashEmbeddingProvider::new(64)))
            .expect("open"),
    );
    for i in 0..50 {
        db.store(node(&db, &format!("seed memory about deploys {i}")))
            .unwrap();
    }
    let query = db.embed_text("deploys").unwrap().unwrap();

    let (done_tx, done_rx) = mpsc::channel();
    let mut workers = Vec::new();
    for w in 0..2 {
        let db = Arc::clone(&db);
        let done = done_tx.clone();
        workers.push(std::thread::spawn(move || {
            for r in 0..ROUNDS {
                let batch = (0..3)
                    .map(|j| node(&db, &format!("writer {w} round {r} item {j}")))
                    .collect();
                db.store_batch(batch).unwrap();
            }
            done.send(()).unwrap();
        }));
    }
    for _ in 0..4 {
        let db = Arc::clone(&db);
        let done = done_tx.clone();
        let query = query.clone();
        workers.push(std::thread::spawn(move || {
            for _ in 0..ROUNDS {
                db.recall_hybrid_at(&query, Some("deploys"), 10, u64::MAX, None, None, None)
                    .unwrap();
            }
            done.send(()).unwrap();
        }));
    }
    drop(done_tx);

    for _ in 0..workers.len() {
        done_rx
            .recv_timeout(Duration::from_secs(90))
            .expect("store_batch and recall deadlocked");
    }
    for w in workers {
        w.join().unwrap();
    }
}

/// Building a speculative entry embeds and recalls, which can take seconds or
/// hang on the network. The cache lock must not be held meanwhile, or every
/// recall's cache check and every flush waits behind it.
#[test]
fn speculative_build_runs_without_the_cache_lock() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(
        MenteDb::open_with_embedder(dir.path(), Box::new(HashEmbeddingProvider::new(64)))
            .expect("open"),
    );
    let (tx, rx) = mpsc::channel();
    let worker = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || {
            db.pre_assemble_speculative(vec!["deploys".to_string()], |topic| {
                // Reads the cache while building, as recall does concurrently.
                let _ = db.speculative_cache_stats();
                let _ = db.try_speculative_hit(topic, None);
                Some((format!("context for {topic}"), Vec::new(), None))
            });
            tx.send(db.speculative_cache_stats().cache_size).unwrap();
        })
    };
    let size = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("speculative build held the cache lock");
    assert_eq!(size, 1);
    worker.join().unwrap();
}
