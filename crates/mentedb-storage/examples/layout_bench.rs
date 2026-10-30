//! Page file layout (released 0.44.0) versus record layout, same memories.
//!
//! ```text
//! cargo run --release -p mentedb-storage --example layout_bench -- <dir> [count]
//! ```
//!
//! Builds `count` (default 100000) typical memories (1024 dimension
//! embedding, about 300 characters of text) in each layout under `<dir>` and
//! reports file size, open time with and without a persisted map, full scan
//! time and bytes read, point read and store latency, and the time to migrate
//! the page file database. The bulk of each database is loaded in batches;
//! store latency is sampled over the last stores, one fsync each, against
//! the full database. Bytes read are what the engine requested from the file
//! system: on a network file system they are the billed reads. Wall times on
//! a local disk include the OS page cache.

#[path = "../tests/support/mod.rs"]
mod support;

use std::path::Path;
use std::time::{Duration, Instant};

use mentedb_core::MemoryNode;
use mentedb_core::config::StorageConfig;
use mentedb_storage::StorageEngine;
use support::{typical, v1};

const SAMPLED_STORES: usize = 2_000;
const POINT_READS: usize = 10_000;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn pct(mut v: Vec<Duration>, p: f64) -> f64 {
    v.sort();
    let i = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[i].as_secs_f64() * 1e6
}

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok()?.metadata().ok())
        .map(|m| m.len())
        .sum()
}

fn rng(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

struct Row {
    name: &'static str,
    file_bytes: u64,
    open_map_ms: f64,
    open_map_reads: u64,
    open_scan_ms: f64,
    open_scan_bytes: u64,
    scan_ms: f64,
    scan_bytes: u64,
    rescan_ms: f64,
    rescan_bytes: u64,
    point_p50_us: f64,
    point_p99_us: f64,
    store_p50_us: f64,
    store_p99_us: f64,
}

fn bench_v1(dir: &Path, n: usize) -> Row {
    let page = v1::PAGE_SIZE as u64;
    let bulk = n - SAMPLED_STORES;
    let e = v1::StorageEngine::open(dir).unwrap();
    for start in (0..bulk).step_by(1000) {
        let nodes: Vec<MemoryNode> = (start..(start + 1000).min(bulk)).map(typical).collect();
        e.store_memory_batch(&nodes).unwrap();
    }
    let mut lat = Vec::with_capacity(SAMPLED_STORES);
    for i in bulk..n {
        let node = typical(i);
        let t = Instant::now();
        e.store_memory(&node).unwrap();
        lat.push(t.elapsed());
    }
    e.memory_map().unwrap();
    e.checkpoint().unwrap();
    e.close().unwrap();
    drop(e);
    let file_bytes = dir_bytes(dir);

    let t = Instant::now();
    let e = v1::StorageEngine::open(dir).unwrap();
    let map = e.memory_map().unwrap();
    let open_map_ms = ms(t.elapsed());
    let open_map_reads = e.page_reads();
    assert_eq!(map.len(), n);
    e.close().unwrap();
    drop(e);

    std::fs::remove_file(dir.join("memory_map.bin")).unwrap();
    let t = Instant::now();
    let e = v1::StorageEngine::open(dir).unwrap();
    e.memory_map().unwrap();
    let open_scan_ms = ms(t.elapsed());
    let open_scan_bytes = e.page_reads() * page;
    e.close().unwrap();
    drop(e);

    let e = v1::StorageEngine::open(dir).unwrap();
    let map = e.memory_map().unwrap();
    let t = Instant::now();
    for m in &map {
        e.load_memory(m.page).unwrap();
    }
    let scan_ms = ms(t.elapsed());
    let scan_bytes = e.page_reads() * page;
    let before = e.page_reads();
    let t = Instant::now();
    for m in &map {
        e.load_memory(m.page).unwrap();
    }
    let rescan_ms = ms(t.elapsed());
    let rescan_bytes = (e.page_reads() - before) * page;
    e.close().unwrap();
    drop(e);

    let e = v1::StorageEngine::open(dir).unwrap();
    let map = e.memory_map().unwrap();
    let mut seed = 0x5eed_u64;
    let mut reads = Vec::with_capacity(POINT_READS);
    for _ in 0..POINT_READS {
        let m = map[(rng(&mut seed) % map.len() as u64) as usize];
        let t = Instant::now();
        e.load_memory(m.page).unwrap();
        reads.push(t.elapsed());
    }
    e.close().unwrap();

    Row {
        name: "page file (0.44.0)",
        file_bytes,
        open_map_ms,
        open_map_reads,
        open_scan_ms,
        open_scan_bytes,
        scan_ms,
        scan_bytes,
        rescan_ms,
        rescan_bytes,
        point_p50_us: pct(reads.clone(), 0.5),
        point_p99_us: pct(reads, 0.99),
        store_p50_us: pct(lat.clone(), 0.5),
        store_p99_us: pct(lat, 0.99),
    }
}

fn bench_records(dir: &Path, n: usize, cache_bytes: u64, name: &'static str) -> Row {
    let cfg = StorageConfig {
        cache_bytes,
        ..Default::default()
    };
    let bulk = n - SAMPLED_STORES;
    let e = StorageEngine::open_with_config(dir, &cfg).unwrap();
    for start in (0..bulk).step_by(1000) {
        let nodes: Vec<MemoryNode> = (start..(start + 1000).min(bulk)).map(typical).collect();
        e.store_memory_batch(&nodes).unwrap();
    }
    let mut lat = Vec::with_capacity(SAMPLED_STORES);
    for i in bulk..n {
        let node = typical(i);
        let t = Instant::now();
        e.store_memory(&node).unwrap();
        lat.push(t.elapsed());
    }
    e.checkpoint().unwrap();
    e.close().unwrap();
    drop(e);
    let file_bytes = dir_bytes(dir);

    let t = Instant::now();
    let e = StorageEngine::open_with_config(dir, &cfg).unwrap();
    let map = e.memory_map().unwrap();
    let open_map_ms = ms(t.elapsed());
    let open_map_reads = e.page_reads();
    assert_eq!(map.len(), n);
    e.close().unwrap();
    drop(e);

    std::fs::remove_file(dir.join(mentedb_storage::memory_map::MEMORY_MAP_FILE)).unwrap();
    let t = Instant::now();
    let e = StorageEngine::open_with_config(dir, &cfg).unwrap();
    e.memory_map().unwrap();
    let open_scan_ms = ms(t.elapsed());
    let open_scan_bytes = e.bytes_read();
    e.close().unwrap();
    drop(e);

    let e = StorageEngine::open_with_config(dir, &cfg).unwrap();
    let map = e.memory_map().unwrap();
    let t = Instant::now();
    for m in &map {
        e.load_memory(m.page).unwrap();
    }
    let scan_ms = ms(t.elapsed());
    let scan_bytes = e.bytes_read();
    let t = Instant::now();
    for m in &map {
        e.load_memory(m.page).unwrap();
    }
    let rescan_ms = ms(t.elapsed());
    let rescan_bytes = e.bytes_read() - scan_bytes;
    e.close().unwrap();
    drop(e);

    let e = StorageEngine::open_with_config(dir, &cfg).unwrap();
    let map = e.memory_map().unwrap();
    let mut seed = 0x5eed_u64;
    let mut reads = Vec::with_capacity(POINT_READS);
    for _ in 0..POINT_READS {
        let m = map[(rng(&mut seed) % map.len() as u64) as usize];
        let t = Instant::now();
        e.load_memory(m.page).unwrap();
        reads.push(t.elapsed());
    }
    e.close().unwrap();

    Row {
        name,
        file_bytes,
        open_map_ms,
        open_map_reads,
        open_scan_ms,
        open_scan_bytes,
        scan_ms,
        scan_bytes,
        rescan_ms,
        rescan_bytes,
        point_p50_us: pct(reads.clone(), 0.5),
        point_p99_us: pct(reads, 0.99),
        store_p50_us: pct(lat.clone(), 0.5),
        store_p99_us: pct(lat, 0.99),
    }
}

fn mb(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = Path::new(args.get(1).expect("usage: layout_bench <dir> [count]"));
    let n: usize = args.get(2).map_or(100_000, |s| s.parse().unwrap());
    assert!(n > SAMPLED_STORES);
    std::fs::create_dir_all(root).unwrap();
    let serialized: u64 = (0..n).map(|i| support::json_len(&typical(i))).sum();
    println!("{n} memories, {:.1} MB serialized as JSON", mb(serialized));

    let v1_dir = root.join("page_file");
    let rec_dir = root.join("records");
    let full_dir = root.join("records_full_cache");
    for d in [&v1_dir, &rec_dir, &full_dir] {
        let _ = std::fs::remove_dir_all(d);
    }
    let old = bench_v1(&v1_dir, n);
    let new = bench_records(
        &rec_dir,
        n,
        StorageConfig::default().cache_bytes,
        "records (64 MB cache)",
    );
    let full = bench_records(&full_dir, n, 2 << 30, "records (2 GB cache)");

    println!(
        "| layout | file MB | open with map ms | records read | open by scan ms | scan open MB read | full scan ms | full scan MB read | second scan ms | second scan MB read | point read p50 us | p99 us | store p50 us | p99 us |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in [&old, &new, &full] {
        println!(
            "| {} | {:.1} | {:.1} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |",
            r.name,
            mb(r.file_bytes),
            r.open_map_ms,
            r.open_map_reads,
            r.open_scan_ms,
            mb(r.open_scan_bytes),
            r.scan_ms,
            mb(r.scan_bytes),
            r.rescan_ms,
            mb(r.rescan_bytes),
            r.point_p50_us,
            r.point_p99_us,
            r.store_p50_us,
            r.store_p99_us
        );
    }

    // Migrate the page file database built above.
    let before = dir_bytes(&v1_dir);
    let t = Instant::now();
    let e = StorageEngine::open(&v1_dir).unwrap();
    let migrate_ms = ms(t.elapsed());
    assert_eq!(e.memory_map().unwrap().len(), n);
    e.close().unwrap();
    println!(
        "migration: {:.1} MB to {:.1} MB in {:.0} ms",
        mb(before),
        mb(dir_bytes(&v1_dir)),
        migrate_ms
    );
}
