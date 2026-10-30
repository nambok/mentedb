//! Migration from the released page file layout (format 1) to the record
//! layout, including a crash at every step of the migration.
//!
//! Fixtures are built by the page file engine exactly as released in 0.44.0
//! (see `support/v1`), so the migration is tested against what production
//! databases actually contain.

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use mentedb_core::MemoryNode;
use mentedb_core::config::StorageConfig;
use mentedb_core::types::MemoryId;
use mentedb_storage::StorageEngine;
use mentedb_storage::legacy::{
    LEGACY_MAP_FILE, LEGACY_PAGE_FILE, LEGACY_PAGE_SIZE, LEGACY_WAL_FILE, MigrationFault,
};
use mentedb_storage::memory_map::MEMORY_MAP_FILE;
use mentedb_storage::record::{DATA_FILE, WAL_FILE};
use support::{typical, v1};

/// What the old engine reports for a database: page -> (id, stamp, node).
type Expected = BTreeMap<u64, (MemoryId, u64, MemoryNode)>;

/// Build an old format database the way production ones are: stores of
/// mixed sizes, deletes, updates (one that grows), a checkpoint with a
/// persisted memory map, then more writes that live only in the WAL, then a
/// crash.
fn legacy_fixture(dir: &Path) -> Expected {
    let e = v1::StorageEngine::open(dir).unwrap();
    let mut pages = Vec::new();
    for i in 0..40 {
        let mut n = typical(i);
        if i % 9 == 0 {
            // Large but within the old one page limit.
            n.content = n.content.repeat(150);
        }
        if i % 7 == 0 {
            n.embedding.clear();
        }
        pages.push((e.store_memory(&n).unwrap(), n));
    }
    let batch: Vec<MemoryNode> = (40..48).map(typical).collect();
    for (p, n) in e.store_memory_batch(&batch).unwrap().into_iter().zip(batch) {
        pages.push((p, n));
    }
    for i in [3, 11, 20] {
        e.delete_memory(pages[i].0).unwrap();
    }
    let mut grow = pages[5].1.clone();
    grow.content = grow.content.repeat(40);
    e.update_memory(pages[5].0, &grow).unwrap();
    e.memory_map().unwrap();
    e.checkpoint().unwrap();

    // After the checkpoint: only in the WAL.
    for i in 48..53 {
        e.store_memory(&typical(i)).unwrap();
    }
    e.delete_memory(pages[30].0).unwrap();
    let reuse = typical(60);
    e.store_memory(&reuse).unwrap();
    let mut changed = pages[8].1.clone();
    changed.content = "changed after the checkpoint".into();
    e.update_memory(pages[8].0, &changed).unwrap();

    let mut expected = Expected::new();
    for m in e.memory_map().unwrap() {
        let node = e.load_memory(m.page).unwrap();
        assert_eq!(node.id, m.id);
        expected.insert(m.page.0, (m.id, m.stamp, node));
    }
    assert_eq!(expected.len(), 48 - 4 + 5 + 1);
    // Crash: nothing beyond the WAL fsync is guaranteed.
    e.release_process_lock();
    expected
}

fn assert_same_node(a: &MemoryNode, b: &MemoryNode) {
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
}

/// Every memory on the same locator with the same content; stamps too when
/// `stamps` is set.
fn assert_migrated(engine: &StorageEngine, expected: &Expected, stamps: bool) {
    let map = engine.memory_map().unwrap();
    assert_eq!(map.len(), expected.len());
    for m in &map {
        let (id, stamp, node) = &expected[&m.page.0];
        assert_eq!(m.id, *id, "page {}", m.page.0);
        if stamps {
            assert_eq!(m.stamp, *stamp, "stamp of page {}", m.page.0);
        }
        assert_same_node(&engine.load_memory(m.page).unwrap(), node);
    }
}

fn legacy_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    [LEGACY_PAGE_FILE, LEGACY_WAL_FILE, LEGACY_MAP_FILE]
        .iter()
        .filter_map(|n| std::fs::read(dir.join(n)).ok().map(|b| (n.to_string(), b)))
        .collect()
}

fn copy_dir(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

#[test]
fn migration_keeps_every_memory_locator_and_stamp() {
    let dir = tempfile::tempdir().unwrap();
    let expected = legacy_fixture(dir.path());
    let before = std::fs::metadata(dir.path().join(LEGACY_PAGE_FILE))
        .unwrap()
        .len();

    let engine = StorageEngine::open(dir.path()).unwrap();
    assert_migrated(&engine, &expected, true);

    // The old files are gone, the new database is a fraction of the size.
    for n in [LEGACY_PAGE_FILE, LEGACY_WAL_FILE, LEGACY_MAP_FILE] {
        assert!(!dir.path().join(n).exists(), "{n} removed");
    }
    let after = engine.data_bytes();
    assert!(after * 5 < before, "{after} bytes vs {before} before");

    // New memories never reuse an old page number.
    let max_old = *expected.keys().max().unwrap();
    let fresh = typical(1000);
    let p = engine.store_memory(&fresh).unwrap();
    assert!(p.0 > max_old);
    engine.release_process_lock();
    drop(engine);

    // Reopening reads no records: the migrated map was persisted.
    let engine = StorageEngine::open(dir.path()).unwrap();
    assert_eq!(engine.page_reads(), 0);
    assert_same_node(&engine.load_memory(p).unwrap(), &fresh);
    let mut with_fresh = expected.clone();
    with_fresh.insert(p.0, (fresh.id, 0, fresh));
    assert_migrated(&engine, &with_fresh, false);
    assert_eq!(engine.scan_all_memories().len(), with_fresh.len());
}

#[test]
fn migration_without_a_snapshot_still_carries_every_memory() {
    let dir = tempfile::tempdir().unwrap();
    let expected = legacy_fixture(dir.path());
    std::fs::remove_file(dir.path().join(LEGACY_MAP_FILE)).unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();
    assert_migrated(&engine, &expected, false);
}

#[test]
fn migration_skips_corrupt_pages_and_keeps_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let expected = legacy_fixture(dir.path());
    // Flip a byte inside a checkpointed page that no later WAL entry rewrites.
    let victim = *expected.keys().next().unwrap();
    let path = dir.path().join(LEGACY_PAGE_FILE);
    let mut raw = std::fs::read(&path).unwrap();
    raw[victim as usize * LEGACY_PAGE_SIZE + 100] ^= 0xff;
    std::fs::write(&path, &raw).unwrap();

    let engine = StorageEngine::open(dir.path()).unwrap();
    let mut rest = expected.clone();
    rest.remove(&victim);
    assert_migrated(&engine, &rest, true);
}

#[test]
fn a_crash_at_any_migration_step_leaves_a_whole_database() {
    for fault in [
        MigrationFault::MidDataWrite,
        MigrationFault::AfterDataSync,
        MigrationFault::AfterWal,
        MigrationFault::BeforeCommit,
        MigrationFault::AfterCommit,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let expected = legacy_fixture(dir.path());
        let old = legacy_files(dir.path());

        let err =
            StorageEngine::open_with_migration_fault(dir.path(), &StorageConfig::default(), fault);
        assert!(err.is_err(), "{fault:?}: the injected crash stops the open");

        if fault == MigrationFault::AfterCommit {
            // Committed: the new database is complete, old files linger.
            assert!(dir.path().join(DATA_FILE).exists());
            assert_eq!(legacy_files(dir.path()), old);
        } else {
            // Not committed: the old database is untouched, byte for byte,
            // and there is no record file to mistake for a database.
            assert_eq!(legacy_files(dir.path()), old, "{fault:?}");
            assert!(!dir.path().join(DATA_FILE).exists(), "{fault:?}");

            // The released engine still opens it and sees everything.
            let copy = tempfile::tempdir().unwrap();
            copy_dir(dir.path(), copy.path());
            for stale in [
                DATA_FILE.to_string(),
                format!("{DATA_FILE}.tmp"),
                WAL_FILE.to_string(),
                MEMORY_MAP_FILE.to_string(),
            ] {
                let _ = std::fs::remove_file(copy.path().join(stale));
            }
            let old_engine = v1::StorageEngine::open(copy.path()).unwrap();
            let mut seen: Vec<(MemoryId, u64)> = old_engine
                .memory_map()
                .unwrap()
                .into_iter()
                .map(|m| (m.id, m.page.0))
                .collect();
            seen.sort_by_key(|(_, p)| *p);
            let want: Vec<(MemoryId, u64)> =
                expected.iter().map(|(p, (id, _, _))| (*id, *p)).collect();
            assert_eq!(seen, want, "{fault:?}: old engine sees the old data");
            old_engine.close().unwrap();
        }

        // The next open finishes the job, from scratch or from the commit.
        let engine = StorageEngine::open(dir.path()).unwrap();
        assert_migrated(&engine, &expected, true);
        for n in [LEGACY_PAGE_FILE, LEGACY_WAL_FILE, LEGACY_MAP_FILE] {
            assert!(!dir.path().join(n).exists(), "{fault:?}: {n} removed");
        }
        assert!(!dir.path().join(format!("{DATA_FILE}.tmp")).exists());
        engine.close().unwrap();
    }
}

#[test]
fn an_empty_legacy_database_migrates() {
    let dir = tempfile::tempdir().unwrap();
    {
        let e = v1::StorageEngine::open(dir.path()).unwrap();
        e.close().unwrap();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    assert!(engine.memory_map().unwrap().is_empty());
    assert!(!dir.path().join(LEGACY_PAGE_FILE).exists());
    let p = engine.store_memory(&typical(1)).unwrap();
    assert!(p.0 >= 1);
}
