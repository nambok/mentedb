use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mentedb::context::{AssemblyConfig, ContextAssembler, ScoredMemory};
use mentedb::core::memory::{MemoryNode, MemoryType};
use mentedb::index::hnsw::{HnswConfig, HnswIndex};
use mentedb::query::mql::Mql;
use mentedb::storage::StorageEngine;
use mentedb_core::types::{AgentId};

fn random_embedding(dim: usize) -> Vec<f32> {
    // Simple pseudo-random via system time mixed with counter.
    // Sufficient for benchmarking; avoids adding rand as a dependency.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut state = hasher.finish();

    (0..dim)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64);
            // Map to [-1, 1]
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn make_memory(i: usize) -> MemoryNode {
    MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        format!("benchmark memory content for item number {i}"),
        random_embedding(128),
    )
}

fn bench_storage_write(c: &mut Criterion) {
    c.bench_function("storage_write_1000", |b| {
        b.iter_with_setup(
            || {
                let dir = tempfile::tempdir().unwrap();
                let engine = StorageEngine::open(dir.path()).unwrap();
                let memories: Vec<MemoryNode> = (0..1000).map(make_memory).collect();
                (dir, engine, memories)
            },
            |(_dir, mut engine, memories)| {
                for mem in &memories {
                    black_box(engine.store_memory(mem).unwrap());
                }
                engine.close().unwrap();
            },
        );
    });
}

fn bench_hnsw_search(c: &mut Criterion) {
    // Build the index once outside the benchmark loop.
    let index = HnswIndex::new(HnswConfig::default());
    for _ in 0..10_000 {
        index
            .insert(AgentId::new(), &random_embedding(128))
            .unwrap();
    }

    c.bench_function("hnsw_search_top10", |b| {
        b.iter(|| {
            let query = random_embedding(128);
            black_box(index.search(&query, 10));
        });
    });
}

fn bench_context_assembly(c: &mut Criterion) {
    let memories: Vec<ScoredMemory> = (0..100)
        .map(|i| {
            let score = (100 - i) as f32 / 100.0;
            ScoredMemory {
                memory: make_memory(i),
                score,
            }
        })
        .collect();
    let config = AssemblyConfig::default();

    c.bench_function("context_assembly_100", |b| {
        b.iter(|| {
            let mems = memories.clone();
            black_box(ContextAssembler::assemble(mems, vec![], &config));
        });
    });
}

fn bench_mql_parse(c: &mut Criterion) {
    let queries: Vec<String> = (0..100)
        .map(|i| match i % 5 {
            0 => format!(
                "RECALL memories NEAR [{}] LIMIT 10",
                (0..10)
                    .map(|j| format!("{:.2}", (i * 10 + j) as f32 / 100.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            1 => format!("RECALL memories WHERE type = episodic AND tag = \"bench{i}\" LIMIT 5"),
            2 => format!(r#"RECALL memories WHERE content ~> "topic number {i}" LIMIT 10"#),
            3 => format!(
                "TRAVERSE {} DEPTH 3 WHERE edge_type = caused",
                AgentId::new()
            ),
            _ => format!("RECALL memories WHERE salience > 0.{i:02} LIMIT 20"),
        })
        .collect();

    c.bench_function("mql_parse_100", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(Mql::parse(q).unwrap());
            }
        });
    });
}

criterion_group!(
    benches,
    bench_storage_write,
    bench_hnsw_search,
    bench_context_assembly,
    bench_mql_parse,
);
criterion_main!(benches);
