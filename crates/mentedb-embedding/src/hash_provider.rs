//! A deterministic, zero-dependency embedding provider for testing and development.
//!
//! Generates embeddings by hashing text content. Same text always produces the same
//! embedding vector. NOT suitable for production similarity search.

use mentedb_core::error::MenteResult;

use crate::provider::{AsyncEmbeddingProvider, EmbeddingProvider};

/// Deterministic hash-based embedding provider.
///
/// Useful for testing the full embedding pipeline without requiring an ML model.
/// The same input text always produces the same embedding vector.
pub struct HashEmbeddingProvider {
    dimensions: usize,
    model_name: String,
}

impl HashEmbeddingProvider {
    /// Create a new hash embedding provider with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            model_name: format!("hash-embedding-{dimensions}d"),
        }
    }

    /// Create a new hash embedding provider with the default 384 dimensions.
    pub fn default_384() -> Self {
        Self::new(384)
    }

    /// Hash text combined with a dimension index to produce a deterministic f32 value.
    fn hash_dimension(text: &str, dim: usize) -> f32 {
        // Simple FNV-1a-inspired hash combining text bytes with dimension index
        let mut hash: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;

        // Mix in the dimension index
        for byte in dim.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        // Mix in the text bytes
        for byte in text.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        // Convert to f32 in [-1.0, 1.0]
        (((hash as f64) / (u64::MAX as f64)) * 2.0 - 1.0) as f32
    }

    /// Generate a normalized embedding for the given text.
    fn compute_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding: Vec<f32> = (0..self.dimensions)
            .map(|dim| Self::hash_dimension(text, dim))
            .collect();

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

impl EmbeddingProvider for HashEmbeddingProvider {
    fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
        Ok(self.compute_embedding(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.compute_embedding(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl AsyncEmbeddingProvider for HashEmbeddingProvider {
    async fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
        Ok(self.compute_embedding(text))
    }

    async fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.compute_embedding(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let provider = HashEmbeddingProvider::default_384();
        let e1 = EmbeddingProvider::embed(&provider, "hello world").unwrap();
        let e2 = EmbeddingProvider::embed(&provider, "hello world").unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_correct_dimensions() {
        let provider = HashEmbeddingProvider::new(128);
        let emb = EmbeddingProvider::embed(&provider, "test").unwrap();
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_normalized() {
        let provider = HashEmbeddingProvider::default_384();
        let emb = EmbeddingProvider::embed(&provider, "test normalization").unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
