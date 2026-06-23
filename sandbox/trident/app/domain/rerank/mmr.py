"""
MMR (Maximal Marginal Relevance) diversity reranking.

MMR(c) = λ * relevance(c) - (1 - λ) * max_{c' ∈ selected} similarity(c, c')

At each step, selects the candidate that maximizes this trade-off between
relevance and diversity. Uses candidate embedding cosine similarity as the
diversity measure.

Default λ = 0.7 — favors relevance but still penalizes redundancy.
"""

from __future__ import annotations

import numpy as np


class MMRDiversityReranker:
    def __init__(self, lambda_: float = 0.7):
        self._lambda = lambda_

    def rerank(
        self,
        candidates: list[tuple[str, float]],
        embeddings: dict[str, np.ndarray],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        k = top_k or len(candidates)
        selected_indices: list[int] = []
        remaining_indices = set(range(len(candidates)))

        # Extract relevance scores
        relevance = np.array([s for _, s in candidates], dtype=np.float32)
        if len(relevance) == 0:
            return []

        # Build embedding matrix
        emb_matrix = self._build_embedding_matrix(candidates, embeddings)
        sim_matrix = self._compute_similarity_matrix(emb_matrix)

        while len(selected_indices) < min(k, len(candidates)):
            best_idx = -1
            best_mmr = -float("inf")

            for i in remaining_indices:
                mmr_i = self._lambda * relevance[i]
                if selected_indices:
                    max_sim = max(sim_matrix[i][j] for j in selected_indices)
                    mmr_i -= (1 - self._lambda) * max_sim
                if mmr_i > best_mmr:
                    best_mmr = mmr_i
                    best_idx = i

            if best_idx == -1:
                break
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def _build_embedding_matrix(
        self,
        candidates: list[tuple[str, float]],
        embeddings: dict[str, np.ndarray],
    ) -> np.ndarray:
        dim = 0
        vecs = []
        for cid, _ in candidates:
            v = embeddings.get(cid)
            if v is not None:
                if dim == 0:
                    dim = len(v)
                vecs.append(np.array(v, dtype=np.float32))
            else:
                vecs.append(np.zeros(dim, dtype=np.float32) if dim > 0 else np.array([0.0]))
        if not vecs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(vecs)

    def _compute_similarity_matrix(self, emb_matrix: np.ndarray) -> np.ndarray:
        n = emb_matrix.shape[0]
        if n == 0:
            return np.zeros((0, 0))
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normed = emb_matrix / norms
        sim = emb_normed @ emb_normed.T
        sim = np.clip(sim, 0.0, 1.0)
        return sim
