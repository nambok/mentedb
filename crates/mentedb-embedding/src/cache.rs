//! LRU cache for computed embeddings to avoid recomputation.

use std::collections::{HashMap, VecDeque};

/// Statistics about cache usage.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub max_size: usize,
    pub evictions: u64,
}

/// A cached embedding entry.
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    pub embedding: Vec<f32>,
    pub created_at: u64,
    pub hit_count: u32,
}

/// LRU cache for embedding vectors, keyed by hash of (model_name + text).
pub struct EmbeddingCache {
    max_size: usize,
    cache: HashMap<u64, CachedEmbedding>,
    order: VecDeque<u64>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl EmbeddingCache {
    /// Create a new cache with the given maximum number of entries.
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            cache: HashMap::with_capacity(max_size.min(1024)),
            order: VecDeque::with_capacity(max_size.min(1024)),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Create a cache with the default size of 10,000 entries.
    pub fn default_size() -> Self {
        Self::new(10_000)
    }

    /// Compute a cache key from model name and text.
    fn cache_key(model: &str, text: &str) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;

        for byte in model.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(prime);
        }
        // Separator
        hash ^= 0xff;
        hash = hash.wrapping_mul(prime);

        for byte in text.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        hash
    }

    /// Look up a cached embedding. Returns `None` on cache miss.
    pub fn get(&mut self, text: &str, model: &str) -> Option<&[f32]> {
        let key = Self::cache_key(model, text);

        if self.cache.contains_key(&key) {
            self.hits += 1;

            // Move to back (most recently used)
            self.order.retain(|k| *k != key);
            self.order.push_back(key);

            let entry = self.cache.get_mut(&key).unwrap();
            entry.hit_count += 1;
            Some(&entry.embedding)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert an embedding into the cache. Evicts the LRU entry if full.
    pub fn put(&mut self, text: &str, model: &str, embedding: Vec<f32>) {
        let key = Self::cache_key(model, text);

        // If key already exists, update it
        if self.cache.contains_key(&key) {
            self.order.retain(|k| *k != key);
            self.order.push_back(key);
            self.cache.insert(
                key,
                CachedEmbedding {
                    embedding,
                    created_at: Self::now_micros(),
                    hit_count: 0,
                },
            );
            return;
        }

        // Evict LRU if at capacity
        while self.cache.len() >= self.max_size {
            if let Some(evict_key) = self.order.pop_front() {
                self.cache.remove(&evict_key);
                self.evictions += 1;
            } else {
                break;
            }
        }

        self.cache.insert(
            key,
            CachedEmbedding {
                embedding,
                created_at: Self::now_micros(),
                hit_count: 0,
            },
        );
        self.order.push_back(key);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            size: self.cache.len(),
            max_size: self.max_size,
            evictions: self.evictions,
        }
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }

    fn now_micros() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit_miss() {
        let mut cache = EmbeddingCache::new(10);
        assert!(cache.get("hello", "model").is_none());
        assert_eq!(cache.stats().misses, 1);

        cache.put("hello", "model", vec![1.0, 2.0, 3.0]);
        let result = cache.get("hello", "model");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = EmbeddingCache::new(3);

        cache.put("a", "m", vec![1.0]);
        cache.put("b", "m", vec![2.0]);
        cache.put("c", "m", vec![3.0]);

        // Cache is full, inserting "d" should evict "a" (LRU)
        cache.put("d", "m", vec![4.0]);

        assert!(cache.get("a", "m").is_none());
        assert!(cache.get("b", "m").is_some());
        assert!(cache.get("c", "m").is_some());
        assert!(cache.get("d", "m").is_some());
        assert_eq!(cache.stats().evictions, 1);
    }
}
