//! Forget durability: a forgotten memory must never resurrect on reopen,
//! whether the database was closed cleanly or the process died before close.

use mentedb::MenteDb;
use mentedb::prelude::*;
use mentedb_core::types::AgentId;

fn open(dir: &std::path::Path) -> MenteDb {
    MenteDb::open(dir).expect("open")
}

fn store_one(db: &MenteDb, content: &str, embedding: Vec<f32>) -> MemoryId {
    let node = MemoryNode::new(
        AgentId::nil(),
        MemoryType::Semantic,
        content.to_string(),
        embedding,
    );
    let id = node.id;
    db.store(node).expect("store");
    id
}

#[test]
fn second_open_of_live_database_is_rejected() {
    // Two engine instances on one directory interleave stale cached state
    // (buffer pool, page map, index snapshots) and roll the database back
    // to one instance's open time snapshot. The process lock must refuse
    // the second open while the first is alive, and allow it again after
    // close and after a crash.
    let dir = tempfile::tempdir().unwrap();

    let db1 = open(dir.path());
    let second = MenteDb::open(dir.path());
    assert!(second.is_err(), "second concurrent open must fail");
    let msg = second.err().unwrap().to_string();
    assert!(msg.contains("locked"), "error names the lock: {msg}");

    db1.close().expect("close");
    let db2 = open(dir.path());
    db2.simulate_crash();

    // A crashed process releases the OS lock; reopen must succeed.
    let db3 = open(dir.path());
    db3.close().expect("close");
}

#[test]
fn forget_survives_clean_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let keep;
    let doomed;
    {
        let db = open(dir.path());
        keep = store_one(&db, "memory that stays", vec![0.9, 0.1, 0.0, 0.0]);
        doomed = store_one(
            &db,
            "memory that must not resurrect",
            vec![0.1, 0.9, 0.0, 0.0],
        );
        db.flush().expect("flush");
        db.close().expect("close");
    }

    {
        let db = open(dir.path());
        assert!(db.get_memory(doomed).is_ok(), "exists before forget");
        db.forget(doomed).expect("forget");
        assert!(db.get_memory(doomed).is_err(), "gone in same session");
        db.close().expect("close");
    }

    {
        let db = open(dir.path());
        assert!(db.get_memory(keep).is_ok(), "kept memory survives");
        assert!(
            db.get_memory(doomed).is_err(),
            "forgotten memory resurrected after clean close and reopen"
        );
        assert_eq!(db.memory_count(), 1);
    }
}

#[test]
fn forget_then_flush_survives_reopen() {
    // The exact production sequence: the gateway delete handler calls
    // forget() followed by flush(). A flush that checkpoints the WAL must
    // not erase the evidence of the deletion.
    let dir = tempfile::tempdir().unwrap();

    let keep;
    let doomed;
    {
        let db = open(dir.path());
        keep = store_one(&db, "flush test keeper", vec![0.9, 0.1, 0.0, 0.0]);
        doomed = store_one(&db, "flush test zombie", vec![0.1, 0.9, 0.0, 0.0]);
        db.flush().expect("flush");
        db.close().expect("close");
    }

    {
        let db = open(dir.path());
        db.forget(doomed).expect("forget");
        db.flush().expect("flush after forget");
        db.close().expect("close");
    }

    {
        let db = open(dir.path());
        assert!(db.get_memory(keep).is_ok(), "kept memory survives");
        assert!(
            db.get_memory(doomed).is_err(),
            "forgotten memory resurrected after forget+flush+reopen"
        );
    }
}

#[test]
fn forget_then_flush_then_crash_survives_reopen() {
    // The production lifecycle: forget() + flush() succeed (the flush may
    // checkpoint and truncate the WAL), then the process is killed hours
    // later without close() ever running. The deletion must still be
    // durable: either the checkpoint persists the free list, or the WAL
    // entry must survive the checkpoint.
    let dir = tempfile::tempdir().unwrap();

    let keep;
    let doomed;
    {
        let db = open(dir.path());
        keep = store_one(&db, "checkpoint keeper", vec![0.9, 0.1, 0.0, 0.0]);
        doomed = store_one(&db, "checkpoint zombie", vec![0.1, 0.9, 0.0, 0.0]);
        db.flush().expect("flush");
        db.close().expect("close");
    }

    {
        let db = open(dir.path());
        db.forget(doomed).expect("forget");
        db.flush().expect("flush after forget");
        // Crash without close: destructors skipped, nothing else written.
        db.simulate_crash();
    }

    {
        let db = open(dir.path());
        assert!(db.get_memory(keep).is_ok(), "kept memory survives");
        assert!(
            db.get_memory(doomed).is_err(),
            "forgotten memory resurrected after forget+flush+crash"
        );
    }
}

#[test]
fn forget_survives_crash_without_close() {
    let dir = tempfile::tempdir().unwrap();

    let keep;
    let doomed;
    {
        let db = open(dir.path());
        keep = store_one(&db, "crash test keeper", vec![0.9, 0.1, 0.0, 0.0]);
        doomed = store_one(&db, "crash test zombie candidate", vec![0.1, 0.9, 0.0, 0.0]);
        db.flush().expect("flush");
        db.close().expect("close");
    }

    {
        let db = open(dir.path());
        db.forget(doomed).expect("forget");
        // Simulate a crash: the destructor never runs, nothing else is
        // flushed. The WAL entry written inside forget must be enough.
        db.simulate_crash();
    }

    {
        let db = open(dir.path());
        assert!(db.get_memory(keep).is_ok(), "kept memory survives crash");
        assert!(
            db.get_memory(doomed).is_err(),
            "forgotten memory resurrected after crash-style stop"
        );
    }
}
