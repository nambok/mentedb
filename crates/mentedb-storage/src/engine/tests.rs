use super::*;
use crate::memory_map::MappedMemory;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;

fn setup() -> (tempfile::TempDir, StorageEngine) {
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();
    (dir, engine)
}

fn node(content: &str) -> MemoryNode {
    MemoryNode::new(
        AgentId::new(),
        MemoryType::Semantic,
        content.to_string(),
        vec![1.0, 2.0],
    )
}

/// A node of roughly `bytes` encoded bytes: half embedding, half text.
fn sized(bytes: usize, tag: usize) -> MemoryNode {
    let dims = bytes / 8;
    MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        format!("{tag}:{}", "t".repeat(bytes / 2)),
        (0..dims).map(|i| (i + tag) as f32 * 0.001).collect(),
    )
}

fn same(a: &MemoryNode, b: &MemoryNode) {
    assert_eq!(a.id, b.id);
    assert_eq!(a.content, b.content);
    assert_eq!(a.embedding, b.embedding);
    assert_eq!(a.memory_type, b.memory_type);
}

/// (memory id, page) pairs, the part of the map a full scan can confirm.
fn ids(map: &[MappedMemory]) -> Vec<(MemoryId, PageId)> {
    map.iter().map(|m| (m.id, m.page)).collect()
}

/// Open a second, independent engine over the same files after `engine`
/// crashed, and return what a full read of the record file sees there.
fn scan_truth(dir: &Path) -> Vec<(MemoryId, PageId)> {
    let engine = StorageEngine::open(dir).unwrap();
    let scanned = engine.scan_all_memories();
    engine.release_process_lock();
    scanned
}

fn data_len(dir: &Path) -> u64 {
    std::fs::metadata(dir.join(DATA_FILE)).unwrap().len()
}

#[test]
fn test_store_and_load_memory() {
    let (_dir, engine) = setup();
    let n = MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        "The user prefers Rust over Go".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
    );
    let page_id = engine.store_memory(&n).unwrap();
    same(&engine.load_memory(page_id).unwrap(), &n);
}

#[test]
fn test_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let n = node("persist across close");
    let pid = {
        let engine = StorageEngine::open(dir.path()).unwrap();
        let pid = engine.store_memory(&n).unwrap();
        engine.close().unwrap();
        pid
    };
    let engine = StorageEngine::open(dir.path()).unwrap();
    same(&engine.load_memory(pid).unwrap(), &n);
}

#[test]
fn mixed_sizes_round_trip_including_records_larger_than_a_page() {
    let dir = tempfile::tempdir().unwrap();
    // From tiny to several times the old 64 KB page limit.
    let sizes = [
        10, 300, 511, 512, 4_000, 5_500, 65_000, 70_000, 300_000, 2_000,
    ];
    let nodes: Vec<MemoryNode> = sizes
        .iter()
        .enumerate()
        .map(|(i, &b)| sized(b, i))
        .collect();
    let pids = {
        let engine = StorageEngine::open(dir.path()).unwrap();
        let mut pids: Vec<PageId> = nodes
            .iter()
            .map(|n| engine.store_memory(n).unwrap())
            .collect();
        pids.extend(engine.store_memory_batch(&nodes[..3]).unwrap());
        for (pid, n) in pids.iter().zip(nodes.iter().chain(&nodes[..3])) {
            same(&engine.load_memory(*pid).unwrap(), n);
        }
        engine.close().unwrap();
        pids
    };
    // Clean reopen with a cold cache.
    let engine = StorageEngine::open(dir.path()).unwrap();
    for (pid, n) in pids.iter().zip(nodes.iter().chain(&nodes[..3])) {
        same(&engine.load_memory(*pid).unwrap(), n);
    }
    assert_eq!(engine.page_reads(), pids.len() as u64);
    // The file is about the size of the data, not a page per record.
    let payload: usize = nodes
        .iter()
        .chain(&nodes[..3])
        .map(|n| record::encode_node(n).unwrap().len())
        .sum();
    let file = data_len(dir.path()) as usize;
    assert!(
        file <= payload + pids.len() * BLOCK_SIZE + BLOCK_SIZE,
        "file {file} vs payload {payload}"
    );
}

#[test]
fn records_over_the_configured_limit_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = StorageConfig {
        max_record_bytes: 1000,
        ..Default::default()
    };
    let engine = StorageEngine::open_with_config(dir.path(), &cfg).unwrap();
    assert!(engine.store_memory(&sized(500, 0)).is_ok());
    let err = engine.store_memory(&sized(5000, 0)).unwrap_err();
    assert!(matches!(err, MenteError::CapacityExceeded(_)), "{err}");
}

#[test]
fn update_in_place_and_update_that_grows_keep_the_locator() {
    let dir = tempfile::tempdir().unwrap();
    let (pid, other, grown, shrunk);
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        let mut n = sized(1000, 1);
        pid = engine.store_memory(&n).unwrap();
        other = engine.store_memory(&sized(1000, 2)).unwrap();
        let before = engine.data_bytes();

        // Same size: rewritten in place, the file does not grow.
        n.content = n.content.replace('t', "u");
        engine.update_memory(pid, &n).unwrap();
        assert_eq!(engine.data_bytes(), before);
        same(&engine.load_memory(pid).unwrap(), &n);

        // Grows past its extent: moves, keeps its locator, frees the old one.
        n.content = "g".repeat(20_000);
        engine.update_memory(pid, &n).unwrap();
        grown = n.clone();
        same(&engine.load_memory(pid).unwrap(), &grown);
        assert!(engine.free_bytes() >= 1000, "old extent freed");
        assert_eq!(engine.memory_map().unwrap().len(), 2, "no orphan copy");

        // Shrinks: stays in its (larger) extent.
        let at = engine.data_bytes();
        n.content = "s".into();
        engine.update_memory(pid, &n).unwrap();
        shrunk = n.clone();
        assert_eq!(engine.data_bytes(), at);
        assert_eq!(engine.scan_all_memories().len(), 2);
        engine.close().unwrap();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    same(&engine.load_memory(pid).unwrap(), &shrunk);
    assert_eq!(
        engine.load_memory(other).unwrap().content,
        sized(1000, 2).content
    );
    assert_eq!(engine.scan_all_memories().len(), 2);
    assert_ne!(grown.content, shrunk.content);
}

#[test]
fn update_of_a_missing_record_fails() {
    let (_dir, engine) = setup();
    assert!(engine.update_memory(PageId(42), &node("x")).is_err());
    assert!(engine.delete_memory(PageId(42)).is_err());
}

#[test]
fn deleted_space_is_reused_and_the_file_stops_growing() {
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();
    let mut pids: Vec<PageId> = (0..50)
        .map(|i| engine.store_memory(&sized(4_000, i)).unwrap())
        .collect();
    let full = engine.data_bytes();
    // Churn: delete and store the same volume, many times over.
    for round in 0..5 {
        for pid in pids.drain(..25) {
            engine.delete_memory(pid).unwrap();
            assert!(engine.load_memory(pid).is_err());
        }
        for i in 0..25 {
            pids.push(engine.store_memory(&sized(4_000, 100 * round + i)).unwrap());
        }
    }
    assert_eq!(engine.data_bytes(), full, "freed extents were reused");
    // Small records fill holes left by large ones.
    let big = engine.store_memory(&sized(20_000, 7)).unwrap();
    let grown = engine.data_bytes();
    engine.delete_memory(big).unwrap();
    for i in 0..4 {
        engine.store_memory(&sized(4_000, 900 + i)).unwrap();
    }
    assert_eq!(engine.data_bytes(), grown);
    engine.close().unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();
    assert_eq!(engine.memory_map().unwrap().len(), 54);
    assert_eq!(engine.scan_all_memories().len(), 54);
    assert_eq!(engine.data_bytes(), grown);
}

#[test]
fn test_crash_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let mut stored = Vec::new();
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        for i in 0..3 {
            let n = node(&format!("checkpointed-{i}"));
            stored.push((engine.store_memory(&n).unwrap(), n));
        }
        engine.checkpoint().unwrap();
        for i in 3..6 {
            let n = node(&format!("uncheckpointed-{i}"));
            stored.push((engine.store_memory(&n).unwrap(), n));
        }
        engine.release_process_lock();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    for (pid, n) in &stored {
        same(&engine.load_memory(*pid).unwrap(), n);
    }
}

#[test]
fn test_delete_survives_crash() {
    let dir = tempfile::tempdir().unwrap();
    let pid;
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        pid = engine.store_memory(&node("crash delete")).unwrap();
        engine.delete_memory(pid).unwrap();
        engine.release_process_lock();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    assert!(engine.load_memory(pid).is_err(), "no resurrection");
    assert!(engine.scan_all_memories().is_empty());
    assert!(engine.memory_map().unwrap().is_empty());
}

#[test]
fn test_delete_reuse_crash_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let (a_pid, b_pid, b);
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        a_pid = engine.store_memory(&node("a")).unwrap();
        engine.delete_memory(a_pid).unwrap();
        b = node("b");
        b_pid = engine.store_memory(&b).unwrap();
        // The WAL now holds the free and a write into the same blocks.
        engine.release_process_lock();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    same(&engine.load_memory(b_pid).unwrap(), &b);
    assert!(engine.load_memory(a_pid).is_err());
    // The reused blocks are not free: a new store must not clobber b.
    let c = engine.store_memory(&node("c")).unwrap();
    assert_ne!(c, b_pid);
    same(&engine.load_memory(b_pid).unwrap(), &b);
    assert_eq!(engine.scan_all_memories().len(), 2);
}

#[test]
fn test_concurrent_open_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let engine1 = StorageEngine::open(dir.path()).unwrap();
    let second = StorageEngine::open(dir.path());
    assert!(second.is_err(), "second concurrent open must fail");
    let msg = second.err().unwrap().to_string();
    assert!(msg.contains("locked"), "error names the lock: {msg}");
    engine1.close().unwrap();
    StorageEngine::open(dir.path()).unwrap().close().unwrap();
}

#[test]
fn test_concurrent_writes_and_reads_from_threads() {
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let cfg = StorageConfig {
        cache_bytes: 0,
        ..Default::default()
    };
    let engine = Arc::new(StorageEngine::open_with_config(dir.path(), &cfg).unwrap());
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let eng = Arc::clone(&engine);
            std::thread::spawn(move || {
                let mut mine = Vec::new();
                for i in 0..40 {
                    let mut n = sized(600 + 50 * i, t * 100 + i);
                    let pid = eng.store_memory(&n).unwrap();
                    if i % 3 == 0 {
                        n.content.push_str(&"z".repeat(3000));
                        eng.update_memory(pid, &n).unwrap();
                    }
                    if i % 5 == 0 {
                        eng.delete_memory(pid).unwrap();
                    } else {
                        mine.push((pid, n));
                    }
                    for (p, m) in mine.iter().rev().take(3) {
                        assert_eq!(&eng.load_memory(*p).unwrap().content, &m.content);
                    }
                }
                mine
            })
        })
        .collect();
    let all: Vec<(PageId, MemoryNode)> = handles
        .into_iter()
        .flat_map(|h| h.join().unwrap())
        .collect();
    for (pid, n) in &all {
        same(&engine.load_memory(*pid).unwrap(), n);
    }
    assert_eq!(engine.scan_all_memories().len(), all.len());
}

#[test]
fn cache_serves_repeat_reads_and_is_resizable() {
    let dir = tempfile::tempdir().unwrap();
    let pids: Vec<PageId> = {
        let engine = StorageEngine::open(dir.path()).unwrap();
        let p = (0..100)
            .map(|i| engine.store_memory(&sized(5_000, i)).unwrap())
            .collect();
        engine.close().unwrap();
        p
    };
    // A cache sized to the data: the second pass reads nothing.
    let cfg = StorageConfig {
        cache_bytes: 1 << 20,
        ..Default::default()
    };
    let engine = StorageEngine::open_with_config(dir.path(), &cfg).unwrap();
    for p in &pids {
        engine.load_memory(*p).unwrap();
    }
    let first = engine.page_reads();
    assert_eq!(first, 100);
    for p in &pids {
        engine.load_memory(*p).unwrap();
    }
    assert_eq!(engine.page_reads(), first, "second scan served from memory");
    let s = engine.buffer_stats();
    assert_eq!(s.hits, 100);
    assert!(s.resident_bytes <= 1 << 20);

    // A small cache keeps at most its budget resident.
    engine.set_cache_capacity(50_000);
    assert!(engine.buffer_stats().resident_bytes <= 50_000);
    assert_eq!(engine.cache_capacity(), 50_000);
    for p in &pids {
        engine.load_memory(*p).unwrap();
    }
    assert!(engine.page_reads() > first);
}

// ---- persisted record map ----

#[test]
fn open_after_checkpoint_reads_no_records() {
    let dir = tempfile::tempdir().unwrap();
    let mut stored = Vec::new();
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        for i in 0..300 {
            let m = node(&format!("m{i}"));
            stored.push((m.id, engine.store_memory(&m).unwrap()));
        }
        assert_eq!(ids(&engine.memory_map().unwrap()), stored);
        engine.checkpoint().unwrap();
        engine.close().unwrap();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    assert_eq!(ids(&engine.memory_map().unwrap()), stored);
    assert_eq!(engine.page_reads(), 0, "open must not read records");
    assert_eq!(engine.bytes_read(), 0);
}

#[test]
fn open_after_crash_replays_the_wal_without_reading_records() {
    let dir = tempfile::tempdir().unwrap();
    let mut stored = Vec::new();
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        for i in 0..200 {
            let m = node(&format!("base{i}"));
            stored.push((m.id, engine.store_memory(&m).unwrap()));
        }
        engine.checkpoint().unwrap();
        for i in 0..5 {
            let m = node(&format!("after{i}"));
            stored.push((m.id, engine.store_memory(&m).unwrap()));
        }
        engine.release_process_lock();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    assert_eq!(ids(&engine.memory_map().unwrap()), stored);
    assert_eq!(
        engine.page_reads(),
        0,
        "replay writes from the WAL, reads nothing"
    );
    engine.release_process_lock();
    assert_eq!(scan_truth(dir.path()), stored);
}

#[test]
fn deletes_and_updates_after_checkpoint_are_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let (keep, gone, updated, updated_pid);
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        let k = node("keep");
        let g = node("gone");
        let mut u = node("old");
        let kp = engine.store_memory(&k).unwrap();
        let gp = engine.store_memory(&g).unwrap();
        updated_pid = engine.store_memory(&u).unwrap();
        engine.checkpoint().unwrap();
        engine.delete_memory(gp).unwrap();
        u.content = "new".repeat(2000);
        engine.update_memory(updated_pid, &u).unwrap();
        keep = (k.id, kp);
        gone = g.id;
        updated = u.id;
        engine.release_process_lock();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    let map = engine.memory_map().unwrap();
    assert!(
        map.iter().all(|m| m.id != gone),
        "deleted memory stays deleted"
    );
    assert!(ids(&map).contains(&keep));
    assert!(ids(&map).contains(&(updated, updated_pid)));
    assert_eq!(
        engine.load_memory(updated_pid).unwrap().content,
        "new".repeat(2000)
    );
    assert_eq!(map.len(), 2);
    engine.release_process_lock();
    assert_eq!(scan_truth(dir.path()), ids(&map));
}

#[test]
fn missing_corrupt_or_stale_snapshot_falls_back_to_scan() {
    let dir = tempfile::tempdir().unwrap();
    let mut stored = Vec::new();
    let snap = dir.path().join(memory_map::MEMORY_MAP_FILE);
    let stale;
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        for i in 0..20 {
            let m = node(&format!("m{i}"));
            stored.push((m.id, engine.store_memory(&m).unwrap()));
        }
        engine.checkpoint().unwrap();
        // A snapshot from an earlier checkpoint whose token later leaves the
        // WAL, even though the file itself stays intact.
        stale = std::fs::read(&snap).unwrap();
        engine.delete_memory(stored[0].1).unwrap();
        stored.remove(0);
        for i in 20..40 {
            let m = node(&format!("m{i}"));
            stored.push((m.id, engine.store_memory(&m).unwrap()));
        }
        engine.checkpoint().unwrap();
        engine.close().unwrap();
    }

    let check = |label: &str| {
        let engine = StorageEngine::open(dir.path()).unwrap();
        assert_eq!(
            ids(&engine.memory_map().unwrap()),
            stored,
            "{label}: same result as a scan"
        );
        assert!(
            engine.page_reads() >= 39,
            "{label}: must have scanned, read {}",
            engine.page_reads()
        );
        engine.close().unwrap();
        // The scan re-persisted a valid map.
        let engine = StorageEngine::open(dir.path()).unwrap();
        assert_eq!(ids(&engine.memory_map().unwrap()), stored);
        assert_eq!(engine.page_reads(), 0, "{label}: next open is fast again");
        engine.close().unwrap();
    };

    std::fs::remove_file(&snap).unwrap();
    check("missing");

    let mut bad = std::fs::read(&snap).unwrap();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x55;
    std::fs::write(&snap, &bad).unwrap();
    check("corrupt");

    std::fs::write(&snap, &stale).unwrap();
    check("stale");
}

#[test]
fn handoff_between_handles_sees_writes_and_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let mut stored = Vec::new();
    {
        let e1 = StorageEngine::open(dir.path()).unwrap();
        for i in 0..20 {
            let m = node(&format!("h1-{i}"));
            stored.push((m.id, e1.store_memory(&m).unwrap()));
        }
        e1.checkpoint().unwrap();
        e1.close().unwrap();
    }
    {
        let e2 = StorageEngine::open(dir.path()).unwrap();
        for i in 0..10 {
            let m = node(&format!("h2-{i}"));
            stored.push((m.id, e2.store_memory(&m).unwrap()));
        }
        e2.checkpoint().unwrap();
        e2.delete_memory(stored[0].1).unwrap();
        stored.remove(0);
        let m = node("h2-last");
        stored.push((m.id, e2.store_memory(&m).unwrap()));
        e2.release_process_lock();
    }
    let e3 = StorageEngine::open(dir.path()).unwrap();
    let mut got = ids(&e3.memory_map().unwrap());
    got.sort_by_key(|(_, p)| p.0);
    stored.sort_by_key(|(_, p)| p.0);
    assert_eq!(got, stored);
    assert_eq!(e3.page_reads(), 0);
    e3.release_process_lock();
    assert_eq!(scan_truth(dir.path()), stored);
}

#[test]
fn live_stamps_survive_checkpoint_and_crash_replay() {
    let dir = tempfile::tempdir().unwrap();
    let (b_page, b_live);
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine.store_memory(&node("a")).unwrap();
        engine.checkpoint().unwrap();
        let c = engine.store_memory(&node("c")).unwrap();
        let c_live = engine.memory_stamp(c).unwrap();
        engine.checkpoint().unwrap();
        assert_eq!(
            engine.memory_stamp(c).unwrap(),
            c_live,
            "checkpoint keeps it"
        );
        b_page = engine.store_memory(&node("b")).unwrap();
        b_live = engine.memory_stamp(b_page).unwrap();
        engine.release_process_lock();
    }
    let engine = StorageEngine::open(dir.path()).unwrap();
    let map = engine.memory_map().unwrap();
    let b = map.iter().find(|m| m.page == b_page).unwrap();
    assert_eq!(
        (b.id, b.stamp),
        b_live,
        "WAL replay reproduces the live stamp"
    );
}

#[test]
fn unchanged_checkpoints_keep_the_snapshot_and_stamps() {
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();
    let pid = engine.store_memory(&node("a")).unwrap();
    engine.checkpoint().unwrap();
    let snap = dir.path().join(memory_map::MEMORY_MAP_FILE);
    let before = std::fs::metadata(&snap).unwrap().modified().unwrap();
    let stamp = engine.memory_stamp(pid).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    engine.checkpoint().unwrap();
    engine.checkpoint().unwrap();
    assert_eq!(
        std::fs::metadata(&snap).unwrap().modified().unwrap(),
        before
    );
    assert_eq!(engine.memory_stamp(pid).unwrap(), stamp);
    let mut n = engine.load_memory(pid).unwrap();
    n.content = "b".into();
    engine.update_memory(pid, &n).unwrap();
    assert_ne!(engine.memory_stamp(pid).unwrap().1, stamp.1);
    engine.checkpoint().unwrap();
    engine.close().unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();
    let map = engine.memory_map().unwrap();
    assert_eq!(map[0].stamp, engine.memory_stamp(pid).unwrap().1);
}

// ---- crash at every write step ----

/// Byte ranges of the entries in a WAL file.
fn wal_entry_bounds(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 4 <= bytes.len() {
        let len = record::le_u32(&bytes[at..at + 4]) as usize;
        let end = at + 4 + len + 4;
        if end > bytes.len() {
            break;
        }
        out.push((at, end));
        at = end;
    }
    out
}

/// The expected memories after each prefix of an operation script.
#[derive(Clone)]
enum Step {
    Store(MemoryNode),
    Update(usize, MemoryNode),
    Delete(usize),
}

#[test]
fn wal_replay_recovers_a_crash_at_every_write_step() {
    let base = tempfile::tempdir().unwrap();
    // Base state: checkpointed records, clean close.
    let mut locs: Vec<PageId> = Vec::new();
    let mut base_nodes = Vec::new();
    {
        let engine = StorageEngine::open(base.path()).unwrap();
        for i in 0..6 {
            let n = sized(1_500 + 700 * i, i);
            locs.push(engine.store_memory(&n).unwrap());
            base_nodes.push(n);
        }
        engine.checkpoint().unwrap();
        engine.close().unwrap();
    }
    let base_data = std::fs::read(base.path().join(DATA_FILE)).unwrap();
    let base_map = std::fs::read(base.path().join(memory_map::MEMORY_MAP_FILE)).unwrap();
    let base_wal = std::fs::read(base.path().join(WAL_FILE)).unwrap();

    // Every kind of write: store into a hole, store at the end, in place
    // update, growing update (moves), delete, and a store that reuses the
    // blocks freed by the move.
    let script = vec![
        Step::Delete(1),
        Step::Store(sized(1_000, 50)),
        Step::Store(sized(9_000, 51)),
        Step::Update(0, {
            let mut n = base_nodes[0].clone();
            n.content = n.content.replace('t', "u");
            n
        }),
        Step::Update(2, {
            let mut n = base_nodes[2].clone();
            n.content = "w".repeat(30_000);
            n
        }),
        Step::Store(sized(2_000, 52)),
        Step::Delete(5),
    ];

    // Run the script once to capture the full WAL.
    let run = tempfile::tempdir().unwrap();
    for (name, bytes) in [
        (DATA_FILE, &base_data),
        (memory_map::MEMORY_MAP_FILE, &base_map),
        (WAL_FILE, &base_wal),
    ] {
        std::fs::write(run.path().join(name), bytes).unwrap();
    }
    let mut expected: Vec<Vec<(PageId, MemoryNode)>> = Vec::new();
    {
        let engine = StorageEngine::open(run.path()).unwrap();
        let mut live: Vec<(PageId, MemoryNode)> = locs
            .iter()
            .copied()
            .zip(base_nodes.iter().cloned())
            .collect();
        expected.push(live.clone());
        for step in &script {
            match step.clone() {
                Step::Store(n) => {
                    let p = engine.store_memory(&n).unwrap();
                    live.push((p, n));
                }
                Step::Update(i, n) => {
                    engine.update_memory(locs[i], &n).unwrap();
                    let at = live.iter().position(|(p, _)| *p == locs[i]).unwrap();
                    live[at].1 = n;
                }
                Step::Delete(i) => {
                    engine.delete_memory(locs[i]).unwrap();
                    live.retain(|(p, _)| *p != locs[i]);
                }
            }
            expected.push(live.clone());
        }
        engine.release_process_lock();
    }
    let full_wal = std::fs::read(run.path().join(WAL_FILE)).unwrap();
    let bounds = wal_entry_bounds(&full_wal);
    // The checkpoint entry, then one entry per step.
    assert_eq!(bounds.len(), script.len() + 1);
    let ops: Vec<RecordOp> = Wal::read_entries(&run.path().join(WAL_FILE))
        .unwrap()
        .into_iter()
        .skip(1)
        .map(|e| {
            RecordOp::decode(
                e.entry_type == WalEntryType::RecordWrite,
                e.page_id,
                &e.data,
            )
            .unwrap()
        })
        .collect();

    // Crash with k entries durable, in three data file states: none of their
    // writes reached the file, all but the last did and the last is torn,
    // and all did. Then also with a torn, never synced entry k + 1.
    for k in 0..=script.len() {
        for data_state in 0..3 {
            for torn_wal_tail in [false, true] {
                if torn_wal_tail && k == script.len() {
                    continue;
                }
                let dir = tempfile::tempdir().unwrap();
                let data_path = dir.path().join(DATA_FILE);
                std::fs::write(&data_path, &base_data).unwrap();
                std::fs::write(dir.path().join(memory_map::MEMORY_MAP_FILE), &base_map).unwrap();
                let mut wal = full_wal[..bounds[k].1].to_vec();
                if torn_wal_tail {
                    let (s, e) = bounds[k + 1];
                    wal.extend_from_slice(&full_wal[s..s + (e - s) / 2]);
                }
                std::fs::write(dir.path().join(WAL_FILE), &wal).unwrap();
                {
                    let f = std::fs::OpenOptions::new()
                        .write(true)
                        .open(&data_path)
                        .unwrap();
                    let applied = match data_state {
                        0 => 0,
                        _ => k,
                    };
                    for (i, op) in ops.iter().take(applied).enumerate() {
                        if data_state == 1 && i + 1 == applied {
                            // Torn: only the first half of the last write.
                            if let RecordOp::Write {
                                locator,
                                id,
                                extent,
                                payload,
                                ..
                            } = op
                            {
                                let img = record::live_image(*locator, *id, payload, extent.blocks);
                                record::write_at(&f, &img[..img.len() / 2], extent.offset())
                                    .unwrap();
                            }
                        } else {
                            op.redo(&f).unwrap();
                        }
                    }
                }
                let label = format!("k={k} data_state={data_state} torn_wal={torn_wal_tail}");
                let engine = StorageEngine::open(dir.path()).unwrap();
                let want = &expected[k];
                let map = engine.memory_map().unwrap();
                assert_eq!(map.len(), want.len(), "{label}");
                for (p, n) in want {
                    let got = engine
                        .load_memory(*p)
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    same(&got, n);
                }
                // Writes after recovery are durable too (a torn WAL tail must
                // not swallow them), and the file chain is intact.
                let extra = sized(700, 999);
                let xp = engine.store_memory(&extra).unwrap();
                engine.release_process_lock();
                drop(engine);
                let engine = StorageEngine::open(dir.path()).unwrap();
                same(&engine.load_memory(xp).unwrap(), &extra);
                assert_eq!(engine.scan_all_memories().len(), want.len() + 1, "{label}");
                engine.close().unwrap();
            }
        }
    }
}
