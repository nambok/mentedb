//! Maximal Marginal Relevance (MMR): a diversity-aware selection over recall
//! candidates.
//!
//! Greedily picks items that are relevant yet dissimilar to what is already
//! chosen, so near-duplicate memories do not all consume the fixed, U-curve
//! context budget. Applied as an optional built-in stage after the first pass
//! and any reranker; disabled by default (`mmr_lambda = 1.0`, pure relevance).

/// Cosine similarity, guarding against zero-norm vectors and length mismatch.
pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Greedy MMR ordering. `rel[i]` is candidate `i`'s relevance; `emb[i]` its
/// embedding (`None` means no diversity penalty can be computed for it, so it is
/// ranked on relevance alone). `lambda` in `[0, 1]`: `1.0` is pure relevance
/// (order by `rel`), `0.0` is pure diversity. Returns candidate indices in
/// selection order, at most `k` of them.
///
/// Relevance is min-max normalized to `[0, 1]` so it trades off against cosine
/// (also `[0, 1]` for the non-negative embeddings recall produces) on a common
/// scale, which keeps `lambda` meaningful regardless of the score magnitudes the
/// first pass emits.
pub(crate) fn mmr_select(rel: &[f32], emb: &[Option<&[f32]>], lambda: f32, k: usize) -> Vec<usize> {
    let n = rel.len();
    debug_assert_eq!(n, emb.len());
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let lambda = lambda.clamp(0.0, 1.0);

    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &r in rel {
        lo = lo.min(r);
        hi = hi.max(r);
    }
    let span = hi - lo;
    let reln = |i: usize| {
        if span > 0.0 {
            (rel[i] - lo) / span
        } else {
            1.0
        }
    };

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut chosen = vec![false; n];
    while selected.len() < k {
        let mut best: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..n {
            if chosen[i] {
                continue;
            }
            // Diversity penalty: the largest cosine to anything already picked.
            let max_sim = match emb[i] {
                Some(ei) => selected
                    .iter()
                    .filter_map(|&s| emb[s].map(|es| cosine(ei, es)))
                    .fold(0.0f32, f32::max),
                None => 0.0,
            };
            let mmr = lambda * reln(i) - (1.0 - lambda) * max_sim;
            if mmr > best_score {
                best_score = mmr;
                best = Some(i);
            }
        }
        match best {
            Some(i) => {
                chosen[i] = true;
                selected.push(i);
            }
            None => break,
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_one_is_pure_relevance_order() {
        // With lambda=1 the diversity term vanishes, so the order is exactly by
        // descending relevance regardless of embeddings.
        let rel = [0.2, 0.9, 0.5];
        let a = [1.0f32, 0.0];
        let emb = [Some(&a[..]), Some(&a[..]), Some(&a[..])];
        let order = mmr_select(&rel, &emb, 1.0, 3);
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn diversity_demotes_a_near_duplicate() {
        // Candidate 0 is the most relevant. Candidate 1 is a near-duplicate of 0
        // (identical vector) but slightly less relevant; candidate 2 is a bit
        // less relevant still but points a different direction. With diversity
        // weighted in, the second pick should be the diverse 2, not the
        // duplicate 1.
        let rel = [1.0, 0.9, 0.8];
        let v0 = [1.0f32, 0.0, 0.0];
        let v1 = [1.0f32, 0.0, 0.0]; // duplicate of v0
        let v2 = [0.0f32, 1.0, 0.0]; // orthogonal
        let emb = [Some(&v0[..]), Some(&v1[..]), Some(&v2[..])];
        let order = mmr_select(&rel, &emb, 0.5, 3);
        assert_eq!(order[0], 0, "most relevant is picked first");
        assert_eq!(order[1], 2, "diverse candidate beats the near-duplicate");
        assert_eq!(order[2], 1);
    }

    #[test]
    fn missing_embeddings_fall_back_to_relevance() {
        let rel = [0.3, 0.7];
        let emb: [Option<&[f32]>; 2] = [None, None];
        let order = mmr_select(&rel, &emb, 0.5, 2);
        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn k_is_clamped_and_zero_is_empty() {
        let rel = [0.5, 0.4];
        let emb: [Option<&[f32]>; 2] = [None, None];
        assert_eq!(mmr_select(&rel, &emb, 0.5, 0), Vec::<usize>::new());
        assert_eq!(mmr_select(&rel, &emb, 0.5, 9).len(), 2);
    }
}
