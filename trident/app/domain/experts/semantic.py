from __future__ import annotations

import numpy as np

from app.domain.experts.base import Expert
from app.domain.models import JobPosting, TextView, ExpertScore


class SemanticExpert:
    name = "semantic"

    def __init__(self, temperature: float = 0.05):
        self._temperature = temperature

    async def score(
        self,
        job: JobPosting,
        candidates: list[TextView],
        job_embedding: np.ndarray | None = None,
        **kwargs,
    ) -> list[ExpertScore]:
        results: list[ExpertScore] = []
        for c in candidates:
            if c.embedding is None or job_embedding is None:
                results.append(ExpertScore(
                    candidate_id=c.candidate_id,
                    score=0.5,
                    confidence=0.0,
                    low_confidence=True,
                    metadata={"error": "missing_embedding"},
                ))
                continue

            job_vec = np.array(job_embedding, dtype=np.float32).flatten()
            cand_vec = np.array(c.embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(job_vec) * np.linalg.norm(cand_vec)
            if norm == 0:
                cos = 0.0
            else:
                cos = float(np.dot(job_vec, cand_vec) / norm)

            score = 1.0 / (1.0 + np.exp(-cos / self._temperature))
            score = float(np.clip(score, 0.0, 1.0))
            results.append(ExpertScore(
                candidate_id=c.candidate_id,
                score=score,
                confidence=1.0,
                metadata={"cosine_similarity": cos},
            ))
        return results
