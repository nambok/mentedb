use criterion::{Criterion, black_box, criterion_group, criterion_main};
use mentedb_cognitive::speculative::SpeculativeCache;
use mentedb_cognitive::trajectory::{
    DecisionState, TrajectoryNode, TrajectoryTracker, TransitionMap,
};
use mentedb_core::types::MemoryId;

fn generate_topics(n: usize) -> Vec<String> {
    let domains = [
        "auth",
        "database",
        "deployment",
        "testing",
        "monitoring",
        "caching",
        "logging",
        "security",
        "networking",
        "storage",
        "api",
        "frontend",
        "backend",
        "infra",
        "ci-cd",
        "migrations",
        "indexing",
        "search",
        "payments",
        "notifications",
    ];
    let actions = [
        "setup",
        "config",
        "debug",
        "optimize",
        "refactor",
        "review",
        "fix",
        "design",
        "implement",
        "test",
        "deploy",
        "monitor",
        "scale",
        "migrate",
        "document",
        "benchmark",
        "profile",
        "audit",
        "upgrade",
        "rollback",
        "troubleshoot",
        "investigate",
        "plan",
        "estimate",
        "validate",
    ];
    (0..n)
        .map(|i| {
            let d = &domains[i % domains.len()];
            let a = &actions[i % actions.len()];
            format!("{} {} task {}", d, a, i)
        })
        .collect()
}

fn generate_embedding(seed: usize) -> Vec<f32> {
    let mut emb = vec![0.0f32; 32];
    for (i, v) in emb.iter_mut().enumerate() {
        *v = ((seed * 7 + i * 13) as f32 % 100.0) / 100.0;
    }
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut emb {
            *v /= norm;
        }
    }
    emb
}

fn bench_transition_map_500(c: &mut Criterion) {
    let topics = generate_topics(500);

    let mut group = c.benchmark_group("transition_map_500");

    // Build a map with 500 sequential transitions
    group.bench_function("record_500_transitions", |b| {
        b.iter(|| {
            let mut map = TransitionMap::default();
            for pair in topics.windows(2) {
                map.record(&pair[0], &pair[1]);
            }
            black_box(&map);
        });
    });

    // Predict from a populated map
    let mut populated = TransitionMap::default();
    for pair in topics.windows(2) {
        populated.record(&pair[0], &pair[1]);
    }
    // Add repeat patterns so some transitions have count > 1
    for i in 0..100 {
        populated.record(&topics[i], &topics[(i + 1) % 500]);
    }

    group.bench_function("predict_from_populated", |b| {
        b.iter(|| {
            for i in (0..500).step_by(10) {
                black_box(populated.predict_from(&topics[i], 3));
            }
        });
    });

    // Reinforce + decay cycle
    group.bench_function("reinforce_decay_cycle", |b| {
        let mut map = populated.clone();
        b.iter(|| {
            for i in 0..100 {
                map.reinforce(&topics[i], &topics[(i + 1) % 500]);
            }
            for i in 100..200 {
                map.decay(&topics[i], &topics[(i + 1) % 500]);
            }
            black_box(&map);
        });
    });

    // Save and load roundtrip
    group.bench_function("save_load_roundtrip", |b| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench_transitions.json");
        b.iter(|| {
            populated.save(&path, 1).unwrap();
            let mut loaded = TransitionMap::default();
            loaded.load(&path).unwrap();
            black_box(&loaded);
        });
    });

    group.finish();
}

fn make_full_cache(topics: &[String], with_embeddings: bool) -> SpeculativeCache {
    let mut cache = SpeculativeCache::new(500, 0.5, 0.4);
    cache.pre_assemble(topics.to_vec(), |topic| {
        let emb = if with_embeddings {
            Some(generate_embedding(topic.len()))
        } else {
            None
        };
        Some((format!("Context for {}", topic), vec![MemoryId::new()], emb))
    });
    cache
}

fn bench_speculative_cache_500(c: &mut Criterion) {
    let topics = generate_topics(500);

    let mut group = c.benchmark_group("speculative_cache_500");

    // Pre-assemble 500 entries (cache holds max 500)
    group.bench_function("pre_assemble_500", |b| {
        b.iter(|| {
            let cache = make_full_cache(&topics, true);
            black_box(&cache);
        });
    });

    // try_hit with embeddings on 500-entry cache
    group.bench_function("try_hit_embedding_500", |b| {
        let mut cache = make_full_cache(&topics, true);
        b.iter(|| {
            for i in (0..500).step_by(5) {
                let emb = generate_embedding(topics[i].len());
                black_box(cache.try_hit(&topics[i], Some(&emb)));
            }
        });
    });

    // try_hit keyword fallback on 500-entry cache
    group.bench_function("try_hit_keyword_500", |b| {
        let mut cache = make_full_cache(&topics, false);
        b.iter(|| {
            for i in (0..500).step_by(5) {
                black_box(cache.try_hit(&topics[i], None));
            }
        });
    });

    // Save/load roundtrip with 500 entries
    group.bench_function("save_load_500", |b| {
        let mut cache = make_full_cache(&topics, true);
        // Hit everything so min_hits=1 keeps all
        for t in &topics {
            let emb = generate_embedding(t.len());
            cache.try_hit(t, Some(&emb));
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bench_cache.json");
        b.iter(|| {
            cache.save(&path, 1).unwrap();
            let mut loaded = SpeculativeCache::new(500, 0.5, 0.4);
            loaded.load(&path).unwrap();
            black_box(&loaded);
        });
    });

    group.finish();
}

fn bench_trajectory_500(c: &mut Criterion) {
    let topics = generate_topics(500);

    let mut group = c.benchmark_group("trajectory_500");

    // Record 500 turns and predict
    group.bench_function("record_500_predict", |b| {
        b.iter(|| {
            let mut tracker = TrajectoryTracker::new(500);
            for (i, topic) in topics.iter().enumerate() {
                tracker.record_turn(TrajectoryNode {
                    turn_id: i as u64,
                    topic_embedding: generate_embedding(i),
                    topic_summary: topic.clone(),
                    decision_state: DecisionState::Investigating,
                    open_questions: vec![],
                    timestamp: i as u64 * 1000,
                });
            }
            black_box(tracker.predict_next_topics());
        });
    });

    // Reinforce/decay from full trajectory
    let mut tracker = TrajectoryTracker::new(500);
    for (i, topic) in topics.iter().enumerate() {
        tracker.record_turn(TrajectoryNode {
            turn_id: i as u64,
            topic_embedding: generate_embedding(i),
            topic_summary: topic.clone(),
            decision_state: DecisionState::Investigating,
            open_questions: vec![],
            timestamp: i as u64 * 1000,
        });
    }

    group.bench_function("reinforce_decay_500", |b| {
        let mut t = tracker.transitions.clone();
        b.iter(|| {
            for i in 0..100 {
                t.reinforce(&topics[i], &topics[(i + 1) % 500]);
            }
            for i in 100..200 {
                t.decay(&topics[i], &topics[(i + 1) % 500]);
            }
            black_box(&t);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_transition_map_500,
    bench_speculative_cache_500,
    bench_trajectory_500,
);
criterion_main!(benches);
