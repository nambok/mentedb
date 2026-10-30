//! On-disk size must be proportional to the data.

mod support;

use mentedb_core::MemoryNode;
use mentedb_storage::StorageEngine;
use mentedb_storage::record::DATA_FILE;
use support::{json_len, typical, v1};

/// The bound under test: the store holds `count` memories whose serialized
/// (JSON) size totals `serialized` bytes in at most twice that on disk.
fn within_bound(file_bytes: u64, serialized: u64) -> bool {
    file_bytes <= 2 * serialized
}

#[test]
fn ten_thousand_typical_memories_take_about_their_serialized_size() {
    let n = 10_000;
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();
    let mut serialized = 0u64;
    for chunk in (0..n).collect::<Vec<_>>().chunks(500) {
        let nodes: Vec<MemoryNode> = chunk.iter().map(|&i| typical(i)).collect();
        serialized += nodes.iter().map(json_len).sum::<u64>();
        engine.store_memory_batch(&nodes).unwrap();
    }
    engine.checkpoint().unwrap();
    engine.close().unwrap();
    drop(engine);

    let file = std::fs::metadata(dir.path().join(DATA_FILE)).unwrap().len();
    let per_memory = file / n as u64;
    eprintln!(
        "{n} memories: file {file} bytes ({per_memory} per memory), serialized {serialized} bytes"
    );
    assert!(
        within_bound(file, serialized),
        "file {file} bytes exceeds twice the serialized {serialized} bytes"
    );
    // Tighter: a 1024 dimension memory needs about 5 KB, not a 64 KB page.
    assert!(per_memory <= 6 * 1024, "{per_memory} bytes per memory");

    // And it all reads back.
    let engine = StorageEngine::open(dir.path()).unwrap();
    let map = engine.memory_map().unwrap();
    assert_eq!(map.len(), n);
    for m in map.iter().step_by(97) {
        engine.load_memory(m.page).unwrap();
    }
}

/// Mutation check for the test above: the same bound, applied to the page
/// file the released engine writes for the same memories, fails. One page per
/// memory is about 64 KB each, over five times the serialized size.
#[test]
fn the_released_page_layout_fails_the_same_bound() {
    let n = 300;
    let dir = tempfile::tempdir().unwrap();
    let engine = v1::StorageEngine::open(dir.path()).unwrap();
    let nodes: Vec<MemoryNode> = (0..n).map(typical).collect();
    let serialized: u64 = nodes.iter().map(json_len).sum();
    engine.store_memory_batch(&nodes).unwrap();
    engine.checkpoint().unwrap();
    engine.close().unwrap();
    let file = std::fs::metadata(dir.path().join("pages.db"))
        .unwrap()
        .len();
    assert_eq!(file, (n as u64 + 1) * v1::PAGE_SIZE as u64);
    assert!(
        !within_bound(file, serialized),
        "page layout: {file} bytes for {serialized} serialized"
    );
    assert!(file > 5 * serialized);
}
