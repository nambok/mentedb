use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use mentedb::MenteDb;
use mentedb::core::memory::{MemoryNode, MemoryType};
use mentedb_context::{AssemblyConfig, ContextAssembler, ScoredMemory};
use mentedb_core::types::AgentId;
use tempfile::TempDir;

fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed.wrapping_mul(31).wrapping_add(i as u64)) % 1000) as f32 / 1000.0)
        .collect()
}

/// Pre-populate a database with `count` memories.
/// Returns the db, tmpdir, and a query embedding for searching.
fn setup_db(count: usize) -> (MenteDb, TempDir) {
    let tmp = TempDir::new().unwrap();
    let mut db = MenteDb::open(tmp.path()).unwrap();
    for i in 0..count {
        let emb = generate_embedding(128, i as u64);
        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            format!("Memory content number {i} with meaningful context for assembly"),
            emb,
        );
        db.store(node).unwrap();
    }
    (db, tmp)
}

fn bench_context_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_assembly");

    let config = AssemblyConfig {
        token_budget: 4096,
        ..AssemblyConfig::default()
    };

    for count in [100, 1_000, 10_000] {
        let (mut db, _tmp) = setup_db(count);
        let query = generate_embedding(128, 999_999);

        group.bench_with_input(
            BenchmarkId::new("recall_and_assemble", count),
            &count,
            |b, _| {
                b.iter(|| {
                    // Recall top-50 similar memories
                    let results = db.recall_similar(&query, 50).unwrap();

                    // Convert to ScoredMemory by loading each node
                    let scored: Vec<ScoredMemory> = results
                        .iter()
                        .filter_map(|&(id, score)| {
                            db.get_memory(id).ok().map(|memory| ScoredMemory { memory, score })
                        })
                        .collect();

                    // Assemble context within token budget
                    black_box(ContextAssembler::assemble(scored, vec![], &config));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_context_assembly);
criterion_main!(benches);
