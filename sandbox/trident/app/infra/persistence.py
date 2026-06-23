"""
Repository interfaces and implementations for candidate/job storage.

Provides a clean abstraction over data storage so the pipeline never touches
files or databases directly. The POC uses in-memory storage; the interface is
designed so a SQLite/Postgres implementation can be swapped in by replacing
the dependency injection binding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from app.domain.models import CandidateProfile, JobPosting, TextView, CareerView, BehavioralView

log = logging.getLogger(__name__)


class CandidateRepository(Protocol):
    async def get_all(self) -> list[CandidateProfile]:
        ...

    async def get_by_id(self, candidate_id: str) -> CandidateProfile | None:
        ...

    async def add(self, candidate: CandidateProfile) -> None:
        ...

    async def add_many(self, candidates: list[CandidateProfile]) -> None:
        ...

    @property
    def size(self) -> int:
        ...


class InMemoryCandidateRepository:
    def __init__(self):
        self._store: dict[str, CandidateProfile] = {}

    async def get_all(self) -> list[CandidateProfile]:
        return list(self._store.values())

    async def get_by_id(self, candidate_id: str) -> CandidateProfile | None:
        return self._store.get(candidate_id)

    async def add(self, candidate: CandidateProfile) -> None:
        self._store[candidate.candidate_id] = candidate

    async def add_many(self, candidates: list[CandidateProfile]) -> None:
        for c in candidates:
            self._store[c.candidate_id] = c

    async def load_from_jsonl(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            log.warning("Data file not found at %s", path)
            return 0
        count = 0
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    candidate = self._dict_to_candidate(raw)
                    self._store[candidate.candidate_id] = candidate
                    count += 1
                except json.JSONDecodeError:
                    log.warning("Skipping invalid JSON line")
        log.info("Loaded %d candidates from %s", count, path)
        return count

    @property
    def size(self) -> int:
        return len(self._store)

    @staticmethod
    def _dict_to_candidate(raw: dict) -> CandidateProfile:
        profile_data = raw.get("profile", {})
        redrob = raw.get("redrob_signals", {}) or {}
        career_raw = raw.get("career_history", []) or []
        education_raw = raw.get("education", []) or []
        skills_raw = raw.get("skills", []) or []

        from app.domain.models import (
            CareerEntry, Education, Skill, RedrobSignals,
        )

        return CandidateProfile(
            candidate_id=raw.get("candidate_id", ""),
            anonymized_name=profile_data.get("anonymized_name", ""),
            headline=profile_data.get("headline", ""),
            summary=profile_data.get("summary", ""),
            location=profile_data.get("location", ""),
            country=profile_data.get("country", ""),
            years_of_experience=profile_data.get("years_of_experience", 0.0),
            current_title=profile_data.get("current_title", ""),
            current_company=profile_data.get("current_company", ""),
            current_company_size=profile_data.get("current_company_size", ""),
            current_industry=profile_data.get("current_industry", ""),
            career_history=[CareerEntry(**r) for r in career_raw],
            education=[Education(**e) for e in education_raw],
            skills=[Skill(**s) for s in skills_raw],
            redrob_signals=RedrobSignals(**redrob),
        )
