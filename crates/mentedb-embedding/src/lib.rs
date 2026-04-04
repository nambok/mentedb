//! Embedding model integration for MenteDB.
//!
//! Provides a trait-based architecture for embedding providers with built-in
//! caching and a hash-based provider for testing.

pub mod cache;
pub mod hash_provider;
pub mod http_provider;
pub mod manager;
pub mod provider;

pub use cache::{CacheStats, CachedEmbedding, EmbeddingCache};
pub use hash_provider::HashEmbeddingProvider;
pub use http_provider::{HttpEmbeddingConfig, HttpEmbeddingProvider};
pub use manager::{EmbeddingManager, EmbeddingStats};
pub use provider::{AsyncEmbeddingProvider, EmbeddingProvider};
