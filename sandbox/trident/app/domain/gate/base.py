from __future__ import annotations
from typing import Protocol
import numpy as np

from app.domain.models import JobPosting


class Gate(Protocol):
    def fuse(
        self,
        job: JobPosting,
        semantic_scores: list[float],
        career_scores: list[float],
        behavioral_scores: list[float],
        candidate_ids: list[str],
    ) -> list[tuple[str, float, list[float]]]:
        ...
