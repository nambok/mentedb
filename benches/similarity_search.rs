use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use mentedb::MenteDb;
use mentedb::core::memory::{MemoryNode, MemoryType};
use mentedb_core::types::AgentId;
use tempfile::TempDir;

fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed.wrapping_mul(31).wrapping_add(i as u64)) % 1000) as f32 / 1000.0)
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

/// Pre-populate a database with `count` memories and return (db, tmpdir, embeddings).
fn setup_db(count: usize) -> (MenteDb, TempDir, Vec<Vec<f32>>) {
    let tmp = TempDir::new().unwrap();
    let mut db = MenteDb::open(tmp.path()).unwrap();
    let mut embeddings = Vec::with_capacity(count);
    for i in 0..count {
        let emb = generate_embedding(128, i as u64);
        embeddings.push(emb.clone());
        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            format!("Memory content number {i}"),
            emb,
        );
        db.store(node).unwrap();
    }
    (db, tmp, embeddings)
}

fn bench_similarity_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_search");

    for count in [100, 1_000, 10_000] {
        let (mut db, _tmp, embeddings) = setup_db(count);
        let query = generate_embedding(128, 999_999);

        group.bench_with_input(
            BenchmarkId::new("recall_similar_k10", count),
            &count,
            |b, _| {
                b.iter(|| {
                    black_box(db.recall_similar(&query, 10).unwrap());
                });
            },
        );

        // Verify recall@10 accuracy: compare HNSW results against brute-force.
        let hnsw_results = db.recall_similar(&query, 10).unwrap();
        let mut brute_force: Vec<(usize, f32)> = embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine_similarity(&query, emb)))
            .collect();
        brute_force.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let brute_top10: Vec<f32> = brute_force.iter().take(10).map(|(_, s)| *s).collect();
        let hnsw_scores: Vec<f32> = hnsw_results.iter().map(|(_, s)| *s).collect();

        // Log recall accuracy for this size.
        let overlap = hnsw_scores
            .iter()
            .filter(|s| brute_top10.iter().any(|b| (b - *s).abs() < 1e-4))
            .count();
        eprintln!(
            "[recall@10 accuracy] db_size={count}, overlap={overlap}/10, \
             hnsw_best={:.4}, brute_best={:.4}",
            hnsw_scores.first().copied().unwrap_or(0.0),
            brute_top10.first().copied().unwrap_or(0.0),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_similarity_search);
criterion_main!(benches);
