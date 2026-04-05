use mentedb_core::memory::{MemoryNode, MemoryType};
use mentedb_core::types::AgentId;
use mentedb_embedding::{
    EmbeddingCache, EmbeddingManager, EmbeddingProvider, HashEmbeddingProvider,
};

#[test]
fn test_hash_provider_consistency() {
    let provider = HashEmbeddingProvider::default_384();
    let e1 = provider.embed("the quick brown fox").unwrap();
    let e2 = provider.embed("the quick brown fox").unwrap();
    assert_eq!(e1, e2, "Same text must produce identical embeddings");
    assert_eq!(e1.len(), 384);
}

#[test]
fn test_hash_provider_different_texts() {
    let provider = HashEmbeddingProvider::default_384();
    let e1 = provider.embed("hello world").unwrap();
    let e2 = provider.embed("goodbye world").unwrap();

    // Embeddings should differ
    assert_ne!(
        e1, e2,
        "Different texts should produce different embeddings"
    );

    // Both should be normalized
    let norm1: f32 = e1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = e2.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm1 - 1.0).abs() < 1e-5);
    assert!((norm2 - 1.0).abs() < 1e-5);
}

#[test]
fn test_cache_lru_eviction() {
    let mut cache = EmbeddingCache::new(3);

    cache.put("text_a", "model", vec![1.0, 2.0]);
    cache.put("text_b", "model", vec![3.0, 4.0]);
    cache.put("text_c", "model", vec![5.0, 6.0]);

    // Access text_a to make it recently used
    assert!(cache.get("text_a", "model").is_some());

    // Insert text_d — should evict text_b (LRU)
    cache.put("text_d", "model", vec![7.0, 8.0]);

    assert!(
        cache.get("text_a", "model").is_some(),
        "text_a was recently accessed"
    );
    assert!(
        cache.get("text_b", "model").is_none(),
        "text_b should have been evicted"
    );
    assert!(cache.get("text_c", "model").is_some());
    assert!(cache.get("text_d", "model").is_some());

    let stats = cache.stats();
    assert!(stats.evictions >= 1);
}

#[test]
fn test_manager_embed_and_cache() {
    let provider = HashEmbeddingProvider::default_384();
    let mut manager = EmbeddingManager::new(Box::new(provider), 100);

    let e1 = manager.embed("test text").unwrap();
    assert_eq!(e1.len(), 384);

    // Second call should hit cache
    let e2 = manager.embed("test text").unwrap();
    assert_eq!(e1, e2);

    let stats = manager.stats();
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.total_embeddings, 1);
}

#[test]
fn test_manager_embed_batch() {
    let provider = HashEmbeddingProvider::default_384();
    let mut manager = EmbeddingManager::new(Box::new(provider), 100);

    // Pre-cache one text
    let _ = manager.embed("alpha").unwrap();

    let batch = manager.embed_batch(&["alpha", "beta", "gamma"]).unwrap();

    assert_eq!(batch.len(), 3);
    assert_eq!(batch[0].len(), 384);
    assert_eq!(batch[1].len(), 384);
    assert_eq!(batch[2].len(), 384);

    // "alpha" was cached, "beta" and "gamma" were computed
    let stats = manager.stats();
    assert_eq!(stats.total_embeddings, 3); // 1 from pre-cache + 2 from batch
}

#[test]
fn test_embed_memory_sets_embedding() {
    let provider = HashEmbeddingProvider::default_384();
    let mut manager = EmbeddingManager::new(Box::new(provider), 100);

    let mut node = MemoryNode::new(
        AgentId::new(),
        MemoryType::Semantic,
        "This is a test memory about Rust programming.".to_string(),
        vec![], // empty embedding
    );

    assert!(node.embedding.is_empty());

    manager.embed_memory(&mut node).unwrap();

    assert_eq!(node.embedding.len(), 384);
    assert!(!node.embedding.is_empty());

    // Verify it's normalized
    let norm: f32 = node.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}
