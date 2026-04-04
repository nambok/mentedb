//! Embedding provider traits for synchronous and asynchronous embedding generation.

use mentedb_core::error::MenteResult;

/// Synchronous embedding provider.
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding vector for a single text.
    fn embed(&self, text: &str) -> MenteResult<Vec<f32>>;

    /// Generate embedding vectors for a batch of texts.
    fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>>;

    /// The dimensionality of the embedding vectors produced by this provider.
    fn dimensions(&self) -> usize;

    /// The name of the model used by this provider.
    fn model_name(&self) -> &str;
}

/// Asynchronous embedding provider.
pub trait AsyncEmbeddingProvider: Send + Sync {
    /// Generate an embedding vector for a single text.
    fn embed(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = MenteResult<Vec<f32>>> + Send;

    /// Generate embedding vectors for a batch of texts.
    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> impl std::future::Future<Output = MenteResult<Vec<Vec<f32>>>> + Send;

    /// The dimensionality of the embedding vectors produced by this provider.
    fn dimensions(&self) -> usize;

    /// The name of the model used by this provider.
    fn model_name(&self) -> &str;
}
