use mentedb_core::types::{MemoryId, Timestamp};

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub topic: String,
    pub topic_embedding: Option<Vec<f32>>,
    pub context_text: String,
    pub memory_ids: Vec<MemoryId>,
    pub created_at: Timestamp,
    pub hit_count: u32,
    last_accessed: Timestamp,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub cache_size: usize,
}

pub struct SpeculativeCache {
    entries: Vec<CacheEntry>,
    stats: CacheStats,
    max_size: usize,
    keyword_threshold: f32,
    embedding_threshold: f32,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn keyword_overlap_score(query: &str, topic: &str) -> f32 {
    let query_words: ahash::AHashSet<String> = query
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(String::from)
        .collect();
    let topic_words: ahash::AHashSet<String> = topic
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(String::from)
        .collect();

    if query_words.is_empty() || topic_words.is_empty() {
        return 0.0;
    }

    let intersection = query_words.intersection(&topic_words).count();
    let union = query_words.union(&topic_words).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

impl SpeculativeCache {
    pub fn new(max_size: usize, keyword_threshold: f32, embedding_threshold: f32) -> Self {
        Self {
            entries: Vec::new(),
            stats: CacheStats::default(),
            max_size,
            keyword_threshold,
            embedding_threshold,
        }
    }

    pub fn pre_assemble(
        &mut self,
        predictions: Vec<String>,
        builder: impl Fn(&str) -> Option<(String, Vec<MemoryId>, Option<Vec<f32>>)>,
    ) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for topic in predictions {
            if self.entries.iter().any(|e| e.topic == topic) {
                continue;
            }

            if let Some((context_text, memory_ids, embedding)) = builder(&topic) {
                if self.entries.len() >= self.max_size {
                    self.evict_lru();
                }

                self.entries.push(CacheEntry {
                    topic,
                    topic_embedding: embedding,
                    context_text,
                    memory_ids,
                    created_at: now,
                    hit_count: 0,
                    last_accessed: now,
                });
            }
        }
    }

    /// Try to find a cached context for this query. Uses cosine similarity on
    /// embeddings when available, falls back to keyword overlap only when
    /// no embeddings exist.
    pub fn try_hit(&mut self, query: &str, query_embedding: Option<&[f32]>) -> Option<CacheEntry> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let mut best_idx = None;
        let mut best_score = 0.0f32;
        let mut used_embeddings = false;

        for (i, entry) in self.entries.iter().enumerate() {
            let score = match (query_embedding, &entry.topic_embedding) {
                (Some(qe), Some(te)) => {
                    used_embeddings = true;
                    cosine_similarity(qe, te)
                }
                _ => keyword_overlap_score(query, &entry.topic),
            };

            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        let threshold = if used_embeddings {
            self.embedding_threshold
        } else {
            self.keyword_threshold
        };

        if best_score > threshold
            && let Some(idx) = best_idx
        {
            self.entries[idx].hit_count += 1;
            self.entries[idx].last_accessed = now;
            self.stats.hits += 1;
            return Some(self.entries[idx].clone());
        }

        self.stats.misses += 1;
        None
    }

    pub fn evict_stale(&mut self, max_age_us: u64, now: Timestamp) {
        let before = self.entries.len();
        self.entries.retain(|e| now - e.created_at <= max_age_us);
        let evicted = before - self.entries.len();
        self.stats.evictions += evicted as u64;
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            cache_size: self.entries.len(),
            ..self.stats.clone()
        }
    }

    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let lru_idx = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(i, _)| i)
            .unwrap();
        self.entries.remove(lru_idx);
        self.stats.evictions += 1;
    }
}

impl Default for SpeculativeCache {
    fn default() -> Self {
        Self::new(10, 0.5, 0.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_assemble_and_hit() {
        let mut cache = SpeculativeCache::default();
        cache.pre_assemble(
            vec![
                "database schema design".to_string(),
                "API authentication".to_string(),
            ],
            |topic| {
                Some((
                    format!("Context for {}", topic),
                    vec![MemoryId::new()],
                    None,
                ))
            },
        );

        assert_eq!(cache.stats().cache_size, 2);

        let hit = cache.try_hit("database schema", None);
        assert!(hit.is_some());
        assert!(hit.unwrap().context_text.contains("database schema design"));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = SpeculativeCache::default();
        cache.pre_assemble(vec!["database schema".to_string()], |topic| {
            Some((format!("Context for {}", topic), vec![], None))
        });

        let hit = cache.try_hit("cooking recipes", None);
        assert!(hit.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_embedding_hit() {
        let mut cache = SpeculativeCache::default();
        let topic_emb = vec![1.0, 0.0, 0.0, 0.0];
        cache.pre_assemble(vec!["rust ownership".to_string()], |_topic| {
            Some((
                "Context about ownership".to_string(),
                vec![],
                Some(vec![1.0, 0.0, 0.0, 0.0]),
            ))
        });

        // Query with similar embedding but different words
        let query_emb = vec![0.95, 0.1, 0.0, 0.0];
        let hit = cache.try_hit("memory safety borrow checker", Some(&query_emb));
        assert!(hit.is_some());
    }

    #[test]
    fn test_embedding_miss() {
        let mut cache = SpeculativeCache::default();
        cache.pre_assemble(vec!["rust ownership".to_string()], |_topic| {
            Some((
                "Context about ownership".to_string(),
                vec![],
                Some(vec![1.0, 0.0, 0.0, 0.0]),
            ))
        });

        // Orthogonal embedding should miss
        let query_emb = vec![0.0, 0.0, 0.0, 1.0];
        let hit = cache.try_hit("cooking recipes", Some(&query_emb));
        assert!(hit.is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = SpeculativeCache::new(10, 0.5, 0.4);
        for i in 0..12 {
            cache.pre_assemble(vec![format!("topic {}", i)], |topic| {
                Some((format!("Context for {}", topic), vec![], None))
            });
        }
        assert!(cache.stats().cache_size <= 10);
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_evict_stale() {
        let mut cache = SpeculativeCache::default();
        cache.pre_assemble(vec!["old topic".to_string()], |topic| {
            Some((format!("Context for {}", topic), vec![], None))
        });
        cache.evict_stale(0, u64::MAX);
        assert_eq!(cache.stats().cache_size, 0);
    }
}
