//! Pluggable reranking: an optional second pass that reorders recall candidates
//! by a signal beyond the first-pass vector/BM25 score.
//!
//! Off by default. The engine ships a dependency-free [`LexicalReranker`], and the
//! [`Reranker`] trait lets an embedder plug a stronger model (a cross-encoder, or
//! an LLM judge) without the engine taking on that dependency. Apply it with
//! [`crate::MenteDb::recall_reranked`].

use std::collections::HashSet;

use mentedb_core::types::MemoryId;

/// One candidate handed to a reranker: its id, text, and first-pass score.
pub struct RerankCandidate<'a> {
    pub id: MemoryId,
    pub content: &'a str,
    pub score: f32,
}

/// Reorders recall candidates for a query by returning a new score per id (higher
/// is more relevant). The caller sorts descending by the returned score.
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, candidates: &[RerankCandidate<'_>]) -> Vec<(MemoryId, f32)>;
}

/// A dependency-free lexical reranker: blends the first-pass score with the
/// fraction of query terms that appear in the candidate. Cheap and deterministic,
/// a reasonable default when no model is available, and it lifts exact-term matches
/// that pure vector similarity can bury.
pub struct LexicalReranker {
    /// Weight on lexical overlap versus the first-pass score, clamped to `[0, 1]`.
    overlap_weight: f32,
}

impl Default for LexicalReranker {
    fn default() -> Self {
        Self {
            overlap_weight: 0.5,
        }
    }
}

impl LexicalReranker {
    pub fn new(overlap_weight: f32) -> Self {
        Self {
            overlap_weight: overlap_weight.clamp(0.0, 1.0),
        }
    }
}

fn tokens(s: &str) -> HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

impl Reranker for LexicalReranker {
    fn rerank(&self, query: &str, candidates: &[RerankCandidate<'_>]) -> Vec<(MemoryId, f32)> {
        let q = tokens(query);
        candidates
            .iter()
            .map(|c| {
                let ct = tokens(c.content);
                let overlap = if q.is_empty() {
                    0.0
                } else {
                    // Fraction of query terms present in the candidate.
                    q.intersection(&ct).count() as f32 / q.len() as f32
                };
                let blended = (1.0 - self.overlap_weight) * c.score + self.overlap_weight * overlap;
                (c.id, blended)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u128) -> MemoryId {
        uuid::Uuid::from_u128(n).into()
    }

    #[test]
    fn lexical_reranker_lifts_exact_term_matches() {
        // Candidate B has the lower first-pass score but contains both query terms;
        // with enough overlap weight it should overtake A.
        let a = id(1);
        let b = id(2);
        let cands = vec![
            RerankCandidate {
                id: a,
                content: "a note about gardening tools",
                score: 0.9,
            },
            RerankCandidate {
                id: b,
                content: "the espresso machine broke",
                score: 0.6,
            },
        ];
        let rr = LexicalReranker::new(0.8);
        let mut scored = rr.rerank("espresso machine", &cands);
        scored.sort_by(|x, y| y.1.total_cmp(&x.1));
        assert_eq!(scored[0].0, b, "exact-term candidate should rank first");
    }

    #[test]
    fn empty_query_leaves_first_pass_order() {
        let a = id(1);
        let b = id(2);
        let cands = vec![
            RerankCandidate {
                id: a,
                content: "alpha",
                score: 0.8,
            },
            RerankCandidate {
                id: b,
                content: "beta",
                score: 0.5,
            },
        ];
        let rr = LexicalReranker::default();
        let scored = rr.rerank("", &cands);
        // With no query terms, overlap is 0 for all, so scores scale the first pass
        // uniformly and preserve order.
        assert!(scored[0].1 > scored[1].1);
    }
}
