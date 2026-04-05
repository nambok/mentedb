use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mentedb::MenteDb;
use mentedb::core::memory::{MemoryNode, MemoryType};
use mentedb_core::types::AgentId;
use tempfile::TempDir;

fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed.wrapping_mul(31).wrapping_add(i as u64)) % 1000) as f32 / 1000.0)
        .collect()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");

    for count in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("memories", count), &count, |b, &count| {
            b.iter_with_setup(
                || {
                    let tmp = TempDir::new().unwrap();
                    let db = MenteDb::open(tmp.path()).unwrap();
                    (db, tmp)
                },
                |(mut db, _tmp)| {
                    for i in 0..count {
                        let emb = generate_embedding(128, i as u64);
                        let node = MemoryNode::new(
                            AgentId::new(),
                            MemoryType::Semantic,
                            format!("Memory content number {i} with some meaningful text"),
                            emb,
                        );
                        db.store(node).unwrap();
                    }
                },
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_insert);
criterion_main!(benches);
