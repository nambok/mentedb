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
    "the", "be", "to", "of", "and", "in", "that", "have", "it", "for", "not", "on", "with", "he",
    "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out",
    "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "was", "are",
    "were", "been", "has", "had", "did", "am",
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
    /// doc_id → the distinct terms it contributes. Lets removal touch only this
    /// document's posting lists instead of scanning every term, and lets insert
    /// be a safe upsert (drop the prior version first) rather than double count.
    doc_terms: HashMap<MemoryId, Vec<String>>,
    /// total number of indexed documents
    doc_count: u32,
    /// sum of all document lengths (for computing avgdl)
    total_length: u64,
}

impl Bm25Inner {
    /// Remove a document's entire contribution from the index. Cheap: visits only
    /// the terms this document actually contains, via the forward index. A no-op
    /// if the document is not indexed.
    fn remove_doc(&mut self, id: MemoryId) {
        let Some(doc_len) = self.doc_lengths.remove(&id) else {
            return;
        };
        self.doc_count = self.doc_count.saturating_sub(1);
        self.total_length = self.total_length.saturating_sub(doc_len as u64);
        if let Some(terms) = self.doc_terms.remove(&id) {
            for term in terms {
                if let Some(posting) = self.inverted.get_mut(&term) {
                    posting.entries.retain(|(did, _)| *did != id);
                    if posting.entries.is_empty() {
                        self.inverted.remove(&term);
                    }
                }
            }
        }
    }
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Bm25Inner {
                inverted: HashMap::new(),
                doc_lengths: HashMap::new(),
                doc_terms: HashMap::new(),
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

    /// Index a document (memory content) for BM25 search. Idempotent per id: a
    /// second call for the same id replaces the prior version rather than adding
    /// to it, so re-indexing after an edit (which the engine does routinely, e.g.
    /// on un-pinning) cannot double count the document and corrupt IDF/avgdl.
    pub fn insert(&self, id: MemoryId, content: &str) {
        let tokens = Self::tokenize(content);

        // Count term frequencies
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.as_str()).or_insert(0) += 1;
        }

        let doc_len = tokens.len() as u32;

        let mut inner = self.inner.write();

        // Upsert: drop any prior version of this document first. Without this,
        // a re-index pushes a second posting entry per term and increments
        // doc_count/total_length again, inflating N and avgdl for every query.
        inner.remove_doc(id);

        // Indexing empty (or all stop-word) content just removes the document.
        if tokens.is_empty() {
            return;
        }

        // Update doc metadata
        inner.doc_lengths.insert(id, doc_len);
        inner.doc_count += 1;
        inner.total_length += doc_len as u64;

        // Update the inverted index and record this document's terms so it can be
        // removed cheaply later.
        let mut terms: Vec<String> = Vec::with_capacity(tf.len());
        for (term, freq) in tf {
            let term = term.to_string();
            inner
                .inverted
                .entry(term.clone())
                .or_insert_with(|| PostingList {
                    entries: Vec::new(),
                })
                .entries
                .push((id, freq));
            terms.push(term);
        }
        inner.doc_terms.insert(id, terms);
    }

    /// Remove a document from the index.
    pub fn remove(&self, id: MemoryId) {
        self.inner.write().remove_doc(id);
    }

    /// Search for documents matching the query, returning top-k by BM25 score.
    pub fn search(&self, query: &str, k: usize) -> Vec<(MemoryId, f32)> {
        self.search_impl(query, k, None)
    }

    /// BM25 search restricted to a pre-filtered candidate set.
    pub fn search_filtered(
        &self,
        query: &str,
        k: usize,
        candidates: &std::collections::HashSet<MemoryId>,
    ) -> Vec<(MemoryId, f32)> {
        self.search_impl(query, k, Some(candidates))
    }

    fn search_impl(
        &self,
        query: &str,
        k: usize,
        candidates: Option<&std::collections::HashSet<MemoryId>>,
    ) -> Vec<(MemoryId, f32)> {
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
                    // Skip if not in candidate set
                    if let Some(cands) = candidates
                        && !cands.contains(&doc_id)
                    {
                        continue;
                    }
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
            bincode::serialize(&snapshot).map_err(|e| MenteError::Serialization(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> MenteResult<Self> {
        let data = std::fs::read(path)?;
        let snapshot: Bm25Snapshot = bincode::deserialize(&data)
            .or_else(|_| serde_json::from_slice(&data))
            .map_err(|e| MenteError::Serialization(e.to_string()))?;

        // Rebuild the forward index (doc_id -> terms) from the posting lists, and
        // self-heal any duplicate posting entries left by indexes written before
        // upsert-safe insert existed. Duplicates are collapsed to the highest term
        // frequency, and the document counts are recomputed from the authoritative
        // doc_lengths so a previously inflated doc_count/avgdl is corrected on load.
        let mut inverted: HashMap<String, PostingList> = HashMap::new();
        let mut doc_terms: HashMap<MemoryId, Vec<String>> = HashMap::new();
        for (term, entries) in snapshot.inverted {
            let mut best: HashMap<MemoryId, u32> = HashMap::new();
            for (id, tf) in entries {
                let slot = best.entry(id).or_insert(0);
                *slot = (*slot).max(tf);
            }
            if best.is_empty() {
                continue;
            }
            for &id in best.keys() {
                doc_terms.entry(id).or_default().push(term.clone());
            }
            inverted.insert(
                term,
                PostingList {
                    entries: best.into_iter().collect(),
                },
            );
        }

        let doc_lengths: HashMap<MemoryId, u32> = snapshot.doc_lengths.into_iter().collect();
        let doc_count = doc_lengths.len() as u32;
        let total_length: u64 = doc_lengths.values().map(|&v| v as u64).sum();

        Ok(Self {
            inner: RwLock::new(Bm25Inner {
                inverted,
                doc_lengths,
                doc_terms,
                doc_count,
                total_length,
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

        idx.insert(
            a,
            "Sarah recommended an Italian restaurant in downtown Portland",
        );
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
        idx.insert(
            b,
            "Scheduled a meeting about project planning for next quarter",
        );

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
    fn reindex_does_not_double_count() {
        // Re-indexing the same id (as happens after an edit) must replace, not
        // accumulate: the doc is counted once and its old terms disappear.
        let idx = Bm25Index::new();
        let a = MemoryId::new();
        idx.insert(a, "espresso machine grinder beans");
        idx.insert(a, "gardening tools soil trowel");
        idx.insert(a, "gardening tools soil trowel");

        assert_eq!(
            idx.len(),
            1,
            "same id must count once no matter how often re-indexed"
        );
        // Stale terms from the first version are gone.
        assert!(idx.search("espresso", 10).is_empty());
        // Current terms resolve, and to exactly one hit.
        let hits = idx.search("gardening trowel", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, a);
    }

    #[test]
    fn reindex_keeps_scoring_uncorrupted() {
        // A neighbor doc's IDF/avgdl must not drift because another doc was
        // re-indexed several times: the winner for a shared term stays stable.
        let idx = Bm25Index::new();
        let a = MemoryId::new();
        let b = MemoryId::new();
        idx.insert(a, "the postgres database migration plan");
        idx.insert(b, "database backup schedule");
        for _ in 0..5 {
            idx.insert(a, "the postgres database migration plan");
        }
        assert_eq!(idx.len(), 2);
        let hits = idx.search("postgres migration", 10);
        assert_eq!(hits[0].0, a, "re-indexing a must not corrupt ranking");
    }

    #[test]
    fn load_self_heals_duplicate_postings() {
        // Simulate a pre-fix snapshot: duplicate posting entries for one id plus
        // an inflated doc_count/total_length. Load must collapse and recompute.
        let a = MemoryId::new();
        let legacy = Bm25Snapshot {
            inverted: vec![
                ("postgres".to_string(), vec![(a, 1), (a, 1)]),
                ("migration".to_string(), vec![(a, 1), (a, 1)]),
            ],
            doc_lengths: vec![(a, 2)],
            doc_count: 2,
            total_length: 4,
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.bin");
        std::fs::write(&path, bincode::serialize(&legacy).unwrap()).unwrap();

        let idx = Bm25Index::load(&path).unwrap();
        assert_eq!(idx.len(), 1, "doc_count recomputed from doc_lengths");
        let hits = idx.search("postgres", 10);
        assert_eq!(hits.len(), 1, "duplicate posting entries collapsed to one");
        assert_eq!(hits[0].0, a);
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
