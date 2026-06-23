from __future__ import annotations
from typing import Protocol
from app.domain.models import JobPosting, ExpertScore


class Expert(Protocol):
    name: str

    async def score(
        self,
        job: JobPosting,
        candidates: list,
        **kwargs,
    ) -> list[ExpertScore]:
        ...
