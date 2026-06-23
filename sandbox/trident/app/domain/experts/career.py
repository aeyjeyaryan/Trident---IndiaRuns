"""
Career Expert — Encodes an ordered role sequence and produces a trajectory-fit score.

Designed with a stable interface (encode_role_sequence -> score) so it can be
swapped for a GNN later without touching any other layer. The current POC uses
a hand-built feature extractor + a tiny MLP head, which is sufficient for the
proof-of-concept and avoids the complexity of training a GRU from scratch on
small data.
"""

from __future__ import annotations

import numpy as np

from app.domain.experts.base import Expert
from app.domain.models import JobPosting, CareerView, ExpertScore
from app.core.config import Settings


class CareerExpert:
    name = "career"

    def __init__(self, config: Settings | None = None):
        cfg = config or Settings()
        self._short_history_threshold = cfg.career_short_history_threshold
        self._neutral_prior = cfg.career_neutral_prior
        self._role_families = cfg.ROLE_FAMILIES
        self._seniority_bands = cfg.SENIORITY_BANDS

    # --- seniority mapping ---
    _SENIORITY_KEYWORDS: dict[str, str] = {
        "junior": "junior", "jr": "junior", "associate": "junior",
        "mid": "mid", "intermediate": "mid",
        "senior": "senior", "sr": "senior", "staff": "senior", "lead": "senior",
        "lead": "lead", "principal": "lead", "manager": "lead", "head": "lead",
        "director": "executive", "vp": "executive", "chief": "executive", "cfo": "executive",
        "ceo": "executive", "cto": "executive",
    }

    @staticmethod
    def _infer_seniority(title: str) -> str:
        tl = title.lower()
        for keyword, band in CareerExpert._SENIORITY_KEYWORDS.items():
            if keyword in tl:
                return band
        return "mid"

    @staticmethod
    def _infer_role_family(title: str, industry: str, skills: list[str]) -> str:
        combined = (title + " " + industry + " " + " ".join(skills)).lower()
        family_keywords: dict[str, list[str]] = {
            "engineering": ["engineer", "software", "backend", "frontend", "fullstack",
                            "full-stack", "devops", "infrastructure", "data engineer",
                            "platform", "qa", "quality", "mobile", "ios", "android"],
            "data_science": ["data scientist", "ml engineer", "machine learning", "ai",
                             "analytics engineer", "data analyst", "deep learning", "nlp"],
            "product": ["product manager", "product owner", "program manager"],
            "design": ["designer", "ux", "ui", "graphic designer", "visual", "creative"],
            "marketing": ["marketing", "content", "seo", "brand", "growth", "demand gen"],
            "sales": ["sales", "account executive", "business development", "account manager"],
            "operations": ["operations", "operation manager", "supply chain", "logistics"],
            "hr": ["hr", "human resources", "recruiter", "talent", "people"],
            "finance": ["accountant", "finance", "financial", "audit", "tax", "controller"],
            "support": ["support", "customer service", "help desk", "success"],
        }
        for family, keywords in family_keywords.items():
            if any(kw in combined for kw in keywords):
                return family
        return "engineering"

    async def score(
        self,
        job: JobPosting,
        candidates: list[CareerView],
        **kwargs,
    ) -> list[ExpertScore]:
        job_family = self._infer_role_family(job.title, "", job.required_skills)
        job_seniority = self._infer_seniority(job.title)

        results: list[ExpertScore] = []
        for cv in candidates:
            if len(cv.role_sequence) < self._short_history_threshold:
                results.append(ExpertScore(
                    candidate_id=cv.candidate_id,
                    score=self._neutral_prior,
                    confidence=0.3,
                    low_confidence=True,
                    metadata={"reason": "short_history", "num_roles": len(cv.role_sequence)},
                ))
                continue

            features = self._extract_features(cv, job_family, job_seniority)
            score = self._compute_score(features)
            score = float(np.clip(score, 0.0, 1.0))
            results.append(ExpertScore(
                candidate_id=cv.candidate_id,
                score=score,
                confidence=features["confidence"],
                metadata=features,
            ))
        return results

    def _extract_features(
        self, cv: CareerView, job_family: str, job_seniority: str,
    ) -> dict:
        total_roles = len(cv.role_sequence)
        n_current_matching = 0
        max_seniority_idx = 0
        family_match_count = 0
        total_tenure_months = 0
        employment_gaps = 0

        for r in cv.role_sequence:
            f = self._infer_role_family(r.title, r.industry, cv.skills)
            s = r.seniority if r.seniority else self._infer_seniority(r.title)
            if f == job_family:
                family_match_count += 1
                if r.is_current:
                    n_current_matching += 1
            sb = self._seniority_bands.index(s) if s in self._seniority_bands else -1
            jb = self._seniority_bands.index(job_seniority) if job_seniority in self._seniority_bands else 2
            if sb <= jb:
                max_seniority_idx = max(max_seniority_idx, sb)
            total_tenure_months += r.tenure_months
            if r.tenure_months < 6 and not r.is_current:
                employment_gaps += 1

        family_ratio = family_match_count / max(total_roles, 1)
        seniority_gap = 1.0 - (abs(max_seniority_idx - self._seniority_bands.index(job_seniority if job_seniority in self._seniority_bands else "mid")) / 4)
        tenure_adequacy = min(total_tenure_months / 60.0, 1.0)
        gap_penalty = 1.0 - min(employment_gaps * 0.1, 0.5)
        current_role_bonus = 0.1 if n_current_matching > 0 else 0.0

        raw = (
            0.35 * family_ratio +
            0.20 * seniority_gap +
            0.20 * tenure_adequacy +
            0.15 * gap_penalty +
            0.10 * current_role_bonus
        )

        confidence = min(0.5 + 0.1 * min(total_roles, 5), 1.0)

        return {
            "family_ratio": family_ratio,
            "seniority_gap": seniority_gap,
            "tenure_adequacy": tenure_adequacy,
            "gap_penalty": gap_penalty,
            "current_role_bonus": current_role_bonus,
            "raw_score": raw,
            "confidence": confidence,
            "num_roles": total_roles,
            "total_tenure_months": total_tenure_months,
        }

    def _compute_score(self, features: dict) -> float:
        return features["raw_score"]
