"""Optional cross-encoder reranking for MenteDB search results.

Uses sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2) to re-score
search results by query-document relevance. Blends with original retrieval
scores for improved ranking.

Install: pip install sentence-transformers
"""

from __future__ import annotations

import os
from typing import Optional

_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
        model_name = os.environ.get(
            "MENTEDB_CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def rerank_results(
    query: str,
    results: list[dict],
    content_key: str = "content",
    score_key: str = "score",
    blend_original: float = 0.7,
    blend_ce: float = 0.3,
    top_k: Optional[int] = None,
) -> list[dict]:
    """Re-rank search results using cross-encoder scores.

    Args:
        query: The search query.
        results: List of dicts with at least content_key and score_key.
        content_key: Key for document content in each result dict.
        score_key: Key for original score in each result dict.
        blend_original: Weight for original score (default 0.7).
        blend_ce: Weight for cross-encoder score (default 0.3).
        top_k: Return only top K results. None = return all.

    Returns:
        Re-ranked list of result dicts with updated scores.
    """
    if not results:
        return results

    ce = _get_cross_encoder()

    # Build query-document pairs
    pairs = [(query, r[content_key]) for r in results]

    # Score all pairs in one batch
    ce_scores = ce.predict(pairs)

    # Sigmoid normalize to [0, 1]
    import math
    ce_normalized = [1.0 / (1.0 + math.exp(-float(s))) for s in ce_scores]

    # Normalize original scores to [0, 1]
    orig_scores = [r.get(score_key, 0.0) for r in results]
    max_orig = max(orig_scores) if orig_scores else 1.0
    if max_orig == 0:
        max_orig = 1.0
    orig_normalized = [s / max_orig for s in orig_scores]

    # Blend scores
    reranked = []
    for i, r in enumerate(results):
        blended = blend_original * orig_normalized[i] + blend_ce * ce_normalized[i]
        entry = dict(r)
        entry[score_key] = blended
        entry["_ce_score"] = ce_normalized[i]
        entry["_orig_score"] = orig_normalized[i]
        reranked.append(entry)

    # Sort by blended score descending
    reranked.sort(key=lambda x: x[score_key], reverse=True)

    if top_k is not None:
        reranked = reranked[:top_k]

    return reranked
