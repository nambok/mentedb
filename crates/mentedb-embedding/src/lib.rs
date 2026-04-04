//! Embedding model integration for MenteDB.
//!
//! Provides a trait-based architecture for embedding providers with built-in
//! caching and a hash-based provider for testing.

/// LRU embedding cache with hit/miss tracking.
pub mod cache;
/// Deterministic hash based embedding provider for testing.
pub mod hash_provider;
/// HTTP based embedding provider for remote model APIs.
pub mod http_provider;
/// Embedding manager that wraps providers with caching.
pub mod manager;
/// Trait definitions for sync and async embedding providers.
pub mod provider;

pub use cache::{CacheStats, CachedEmbedding, EmbeddingCache};
pub use hash_provider::HashEmbeddingProvider;
pub use http_provider::{HttpEmbeddingConfig, HttpEmbeddingProvider};
pub use manager::{EmbeddingManager, EmbeddingStats};
pub use provider::{AsyncEmbeddingProvider, EmbeddingProvider};
