from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model=None, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = model
        self._model_name = model_name

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        """
        Rerank candidates using the cross-encoder.
        candidates: list of (candidate_id, text) tuples.
        Returns list of (candidate_id, score) sorted descending by score.
        """
        if self._model is None:
            return self._rerank_fallback(query, candidates)

        try:
            pairs = [(query, text) for _, text in candidates]
            scores = self._model.predict(pairs, show_progress_bar=False)
            scores = 1.0 / (1.0 + np.exp(-np.array(scores, dtype=np.float32)))
            results = []
            for (cid, _), s in zip(candidates, scores):
                results.append((cid, float(s)))
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            log.warning("Cross-encoder inference failed (%s), using fallback", e)
            return self._rerank_fallback(query, candidates)

    def _rerank_fallback(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, float]]:
        """Fallback: score based on token overlap (simple lexical)."""
        query_terms = set(query.lower().split())
        results: list[tuple[str, float]] = []
        for cid, text in candidates:
            terms = set(text.lower().split())
            overlap = len(query_terms & terms) / max(len(query_terms | terms), 1)
            results.append((cid, overlap))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
