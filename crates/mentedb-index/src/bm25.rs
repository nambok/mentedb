//! BM25 full-text index for keyword-based memory retrieval.
//!
//! Provides term-frequency / inverse-document-frequency scoring to complement
//! vector similarity search. Particularly effective for exact entity names,
//! dates, numbers, and other terms that embedding models may conflate.

use std::collections::HashMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;

/// BM25 tuning parameters.
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Minimum token length to index (skip single chars, "a", "I", etc.)
const MIN_TOKEN_LEN: usize = 2;

/// Common English stop words to skip during tokenization.
const STOP_WORDS: &[&str] = &[
    "the", "be", "to", "of", "and", "in", "that", "have", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by",
    "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one",
    "all", "would", "there", "their", "what", "so", "up", "out", "if", "about",
    "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
    "no", "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back", "after",
    "use", "two", "how", "our", "work", "first", "well", "way", "even", "new",
    "want", "because", "any", "these", "give", "day", "most", "us", "is", "was",
    "are", "were", "been", "has", "had", "did", "am",
];

/// Inverted index entry: which documents contain a term and how many times.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostingList {
    /// (memory_id, term_frequency) pairs
    entries: Vec<(MemoryId, u32)>,
}

/// BM25 full-text search index.
pub struct Bm25Index {
    inner: RwLock<Bm25Inner>,
}

struct Bm25Inner {
    /// term → posting list (which docs contain it and how often)
    inverted: HashMap<String, PostingList>,
    /// doc_id → document length in tokens
    doc_lengths: HashMap<MemoryId, u32>,
    /// total number of indexed documents
    doc_count: u32,
    /// sum of all document lengths (for computing avgdl)
    total_length: u64,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Bm25Inner {
                inverted: HashMap::new(),
                doc_lengths: HashMap::new(),
                doc_count: 0,
                total_length: 0,
            }),
        }
    }

    /// Tokenize text into lowercase terms, filtering stop words and short tokens.
    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '\'')
            .map(|w| w.trim_matches('\'').to_lowercase())
            .filter(|w| w.len() >= MIN_TOKEN_LEN && !STOP_WORDS.contains(&w.as_str()))
            .collect()
    }

    /// Index a document (memory content) for BM25 search.
    pub fn insert(&self, id: MemoryId, content: &str) {
        let tokens = Self::tokenize(content);
        if tokens.is_empty() {
            return;
        }

        // Count term frequencies
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0) += 1;
        }

        let doc_len = tokens.len() as u32;

        let mut inner = self.inner.write();

        // Update doc metadata
        inner.doc_lengths.insert(id, doc_len);
        inner.doc_count += 1;
        inner.total_length += doc_len as u64;

        // Update inverted index
        for (term, freq) in tf {
            inner
                .inverted
                .entry(term.to_string())
                .or_insert_with(|| PostingList {
                    entries: Vec::new(),
                })
                .entries
                .push((id, freq));
        }
    }

    /// Remove a document from the index.
    pub fn remove(&self, id: MemoryId) {
        let mut inner = self.inner.write();

        if let Some(doc_len) = inner.doc_lengths.remove(&id) {
            inner.doc_count = inner.doc_count.saturating_sub(1);
            inner.total_length = inner.total_length.saturating_sub(doc_len as u64);

            // Remove from all posting lists
            for posting in inner.inverted.values_mut() {
                posting.entries.retain(|(did, _)| *did != id);
            }
            // Clean up empty posting lists
            inner.inverted.retain(|_, pl| !pl.entries.is_empty());
        }
    }

    /// Search for documents matching the query, returning top-k by BM25 score.
    pub fn search(&self, query: &str, k: usize) -> Vec<(MemoryId, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let query_tokens = Self::tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let inner = self.inner.read();

        if inner.doc_count == 0 {
            return Vec::new();
        }

        let avgdl = inner.total_length as f32 / inner.doc_count as f32;
        let n = inner.doc_count as f32;

        // Accumulate BM25 scores per document
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        for token in &query_tokens {
            if let Some(posting) = inner.inverted.get(token.as_str()) {
                // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let df = posting.entries.len() as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(doc_id, tf) in &posting.entries {
                    let dl = inner.doc_lengths.get(&doc_id).copied().unwrap_or(1) as f32;
                    let tf_f = tf as f32;

                    // BM25 score for this term in this document
                    let numerator = tf_f * (BM25_K1 + 1.0);
                    let denominator = tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl);
                    let term_score = idf * numerator / denominator;

                    *scores.entry(doc_id).or_insert(0.0) += term_score;
                }
            }
        }

        // Sort by score descending, take top k
        let mut results: Vec<(MemoryId, f32)> = scores.into_iter().collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Returns the number of indexed documents.
    pub fn len(&self) -> usize {
        self.inner.read().doc_count as usize
    }

    /// Returns true if no documents are indexed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// --- Persistence ---

#[derive(Serialize, Deserialize)]
struct Bm25Snapshot {
    inverted: Vec<(String, Vec<(MemoryId, u32)>)>,
    doc_lengths: Vec<(MemoryId, u32)>,
    doc_count: u32,
    total_length: u64,
}

impl Bm25Index {
    pub fn save(&self, path: &std::path::Path) -> MenteResult<()> {
        let inner = self.inner.read();
        let snapshot = Bm25Snapshot {
            inverted: inner
                .inverted
                .iter()
                .map(|(k, v)| (k.clone(), v.entries.clone()))
                .collect(),
            doc_lengths: inner.doc_lengths.iter().map(|(&k, &v)| (k, v)).collect(),
            doc_count: inner.doc_count,
            total_length: inner.total_length,
        };
        let data =
            serde_json::to_vec(&snapshot).map_err(|e| MenteError::Serialization(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> MenteResult<Self> {
        let data = std::fs::read(path)?;
        let snapshot: Bm25Snapshot =
            serde_json::from_slice(&data).map_err(|e| MenteError::Serialization(e.to_string()))?;

        let inverted = snapshot
            .inverted
            .into_iter()
            .map(|(k, entries)| (k, PostingList { entries }))
            .collect();

        let doc_lengths = snapshot.doc_lengths.into_iter().collect();

        Ok(Self {
            inner: RwLock::new(Bm25Inner {
                inverted,
                doc_lengths,
                doc_count: snapshot.doc_count,
                total_length: snapshot.total_length,
            }),
        })
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let idx = Bm25Index::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();

        idx.insert(a, "Sarah recommended an Italian restaurant in downtown Portland");
        idx.insert(b, "The project uses PostgreSQL for the database backend");
        idx.insert(c, "We had dinner at the new Thai restaurant last Friday");

        let results = idx.search("restaurant Sarah", 10);
        assert!(!results.is_empty());
        // 'a' should be top result — matches both "restaurant" and "Sarah"
        assert_eq!(results[0].0, a);
    }

    #[test]
    fn test_exact_name_match() {
        let idx = Bm25Index::new();
        let a = MemoryId::new();
        let b = MemoryId::new();

        idx.insert(a, "Meeting with Dr. Chen about the quarterly review");
        idx.insert(b, "Scheduled a meeting about project planning for next quarter");

        let results = idx.search("Dr. Chen", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, a);
    }

    #[test]
    fn test_remove() {
        let idx = Bm25Index::new();
        let a = MemoryId::new();
        idx.insert(a, "Important fact about PostgreSQL");

        assert_eq!(idx.len(), 1);
        idx.remove(a);
        assert_eq!(idx.len(), 0);

        let results = idx.search("PostgreSQL", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_query() {
        let idx = Bm25Index::new();
        let a = MemoryId::new();
        idx.insert(a, "Some content here");

        assert!(idx.search("", 10).is_empty());
        assert!(idx.search("the", 10).is_empty()); // stop word only
    }

    #[test]
    fn test_persistence() {
        let idx = Bm25Index::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        idx.insert(a, "User prefers dark mode for all applications");
        idx.insert(b, "The deployment uses Kubernetes on AWS");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bm25.json");
        idx.save(&path).unwrap();

        let loaded = Bm25Index::load(&path).unwrap();
        let results = loaded.search("dark mode", 10);
        assert_eq!(results[0].0, a);
        assert_eq!(loaded.len(), 2);
    }
}
