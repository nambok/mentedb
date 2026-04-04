//! Embedding manager that wraps a provider with caching and statistics.

use mentedb_core::MemoryNode;
use mentedb_core::error::MenteResult;

use crate::cache::EmbeddingCache;
use crate::provider::EmbeddingProvider;

/// Statistics about embedding operations.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_embeddings: u64,
}

/// Manages embedding generation with caching.
pub struct EmbeddingManager {
    provider: Box<dyn EmbeddingProvider>,
    cache: EmbeddingCache,
    total_embeddings: u64,
}

impl EmbeddingManager {
    /// Create a new embedding manager with the given provider and cache size.
    pub fn new(provider: Box<dyn EmbeddingProvider>, cache_size: usize) -> Self {
        Self {
            provider,
            cache: EmbeddingCache::new(cache_size),
            total_embeddings: 0,
        }
    }

    /// Generate an embedding for the given text, using the cache when possible.
    pub fn embed(&mut self, text: &str) -> MenteResult<Vec<f32>> {
        let model = self.provider.model_name().to_string();

        if let Some(cached) = self.cache.get(text, &model) {
            return Ok(cached.to_vec());
        }

        let embedding = self.provider.embed(text)?;
        self.cache.put(text, &model, embedding.clone());
        self.total_embeddings += 1;
        Ok(embedding)
    }

    /// Generate embeddings for a batch of texts, using the cache for already-computed ones.
    pub fn embed_batch(&mut self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        let model = self.provider.model_name().to_string();

        let mut results: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut missing_indices: Vec<usize> = Vec::new();
        let mut missing_texts: Vec<&str> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(cached) = self.cache.get(text, &model) {
                results.push(Some(cached.to_vec()));
            } else {
                results.push(None);
                missing_indices.push(i);
                missing_texts.push(text);
            }
        }

        if !missing_texts.is_empty() {
            let missing_refs: Vec<&str> = missing_texts.iter().copied().collect();
            let computed = self.provider.embed_batch(&missing_refs)?;

            for (idx, embedding) in missing_indices.into_iter().zip(computed) {
                self.cache.put(texts[idx], &model, embedding.clone());
                self.total_embeddings += 1;
                results[idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Embed a memory node's content and set its embedding field.
    pub fn embed_memory(&mut self, node: &mut MemoryNode) -> MenteResult<()> {
        let embedding = self.embed(&node.content)?;
        node.embedding = embedding;
        Ok(())
    }

    /// Get statistics about embedding operations.
    pub fn stats(&self) -> EmbeddingStats {
        let cache_stats = self.cache.stats();
        EmbeddingStats {
            cache_hits: cache_stats.hits,
            cache_misses: cache_stats.misses,
            total_embeddings: self.total_embeddings,
        }
    }

    /// Get the dimensionality of the underlying provider.
    pub fn dimensions(&self) -> usize {
        self.provider.dimensions()
    }

    /// Get the model name of the underlying provider.
    pub fn model_name(&self) -> &str {
        self.provider.model_name()
    }

    /// Clear the embedding cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}
