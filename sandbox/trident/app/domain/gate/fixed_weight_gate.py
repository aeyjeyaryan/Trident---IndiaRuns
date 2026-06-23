"""
Fixed-Weight Gate — Fallback fusion strategy for cold-start scenarios.

When labeled data isn't available to train the learned gate, or during
testing/demo, this gate applies a fixed set of weights to the three expert
scores. The default weights [0.4, 0.35, 0.25] prioritize semantic text fit
slightly over career trajectory, with behavioral engagement as a supporting
signal — a reasonable starting heuristic.

This class implements the same Gate interface as LearnedGate, so swapping
between them is transparent to the pipeline.
"""

from __future__ import annotations

import numpy as np

from app.domain.models import JobPosting


class FixedWeightGate:
    def __init__(self, weights: tuple[float, float, float] | None = None):
        self._weights = list(weights) if weights else [0.4, 0.35, 0.25]

    def fuse(
        self,
        job: JobPosting,
        semantic_scores: list[float],
        career_scores: list[float],
        behavioral_scores: list[float],
        candidate_ids: list[str],
    ) -> list[tuple[str, float, list[float]]]:
        w = self._weights
        results: list[tuple[str, float, list[float]]] = []
        for i, cid in enumerate(candidate_ids):
            s_sem = semantic_scores[i] if i < len(semantic_scores) else 0.0
            s_car = career_scores[i] if i < len(career_scores) else 0.0
            s_beh = behavioral_scores[i] if i < len(behavioral_scores) else 0.0
            fused = w[0] * s_sem + w[1] * s_car + w[2] * s_beh
            results.append((cid, fused, w))
        return results
