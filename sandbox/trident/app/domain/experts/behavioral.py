"""
Behavioral Expert — Closed-form, decay-weighted scoring over timestamped events.

Scoring is organized into three families, each with its own exponential decay
half-life:

  * Acute Intent (λ_acute; default HL ~10 days):
    Recent job applications, profile views, open-to-work flag — signals the
    candidate is actively looking now.
  * Responsiveness (λ_responsiveness; default HL ~45 days):
    Recruiter response rate, avg response time, interview completion rate —
    signals how engaged the candidate is with recruiter outreach.
  * Platform Presence (λ_presence; default HL ~75 days):
    Connection count, endorsements, search appearances, profile completeness,
    GitHub activity — signals general career platform engagement.

Each event is weighted by a configurable per-event importance weight and decayed
exponentially from its timestamp. The three family scores are combined via a
learned/configurable linear combination followed by a sigmoid to produce the
final [0,1] score.

If a candidate has no behavioral events, the expert returns the neutral prior
(0.5) with a low_confidence: true flag — never a 0.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, date

from app.domain.experts.base import Expert
from app.domain.models import JobPosting, BehavioralView, ExpertScore
from app.core.config import Settings


class BehavioralExpert:
    name = "behavioral"

    def __init__(self, config: Settings | None = None):
        cfg = config or Settings()
        self._hl_acute = cfg.behavioral_hl_acute_days
        self._hl_responsiveness = cfg.behavioral_hl_responsiveness_days
        self._hl_presence = cfg.behavioral_hl_presence_days
        self._w_acute = cfg.behavioral_w_acute
        self._w_responsiveness = cfg.behavioral_w_responsiveness
        self._w_presence = cfg.behavioral_w_presence
        self._neutral_prior = cfg.behavioral_neutral_prior

    @staticmethod
    def _lambda_from_hl(hl_days: float) -> float:
        """λ = ln(2) / half_life — converts half-life to decay constant."""
        if hl_days <= 0:
            return 1.0
        return np.log(2) / hl_days

    async def score(
        self,
        job: JobPosting,
        candidates: list[BehavioralView],
        **kwargs,
    ) -> list[ExpertScore]:
        results: list[ExpertScore] = []
        for bv in candidates:
            score, confidence, meta = self._score_single(bv)
            score = float(np.clip(score, 0.0, 1.0))
            results.append(ExpertScore(
                candidate_id=bv.candidate_id,
                score=score,
                confidence=confidence,
                low_confidence=confidence < 0.5,
                metadata=meta,
            ))
        return results

    def _score_single(self, bv: BehavioralView) -> tuple[float, float, dict]:
        now = date(2026, 6, 22)
        signup = self._parse_date(bv.signup_date) if bv.signup_date else None
        last_active = self._parse_date(bv.last_active_date) if bv.last_active_date else None

        # Check if we have any meaningful data
        has_any_data = (
            bv.profile_views > 0
            or bv.applications_submitted > 0
            or bv.connection_count > 0
            or bv.search_appearance_30d > 0
            or bv.saved_by_recruiters_30d > 0
            or bv.recruiter_response_rate > 0
            or bv.avg_response_time_hours > 0
        )

        if not has_any_data:
            return self._neutral_prior, 0.2, {"reason": "no_behavioral_data"}

        # --- Acute Intent family ---
        acute_score = self._score_acute(bv, now, signup)

        # --- Responsiveness family ---
        responsiveness_score = self._score_responsiveness(bv)

        # --- Platform Presence family ---
        presence_score = self._score_presence(bv, now, signup, last_active)

        # Linear combination
        combined = (
            self._w_acute * acute_score
            + self._w_responsiveness * responsiveness_score
            + self._w_presence * presence_score
        )
        # Sigmoid squashing
        final = 1.0 / (1.0 + np.exp(-combined))

        confidence = self._estimate_confidence(bv)

        meta = {
            "acute_score": acute_score,
            "responsiveness_score": responsiveness_score,
            "presence_score": presence_score,
            "combined_raw": combined,
            "confidence": confidence,
            "w_acute": self._w_acute,
            "w_responsiveness": self._w_responsiveness,
            "w_presence": self._w_presence,
        }
        return final, confidence, meta

    def _score_acute(self, bv: BehavioralView, now: date, signup: date | None) -> float:
        """Score for recent active-job-seeking signals, decay-weighted."""
        lam = self._lambda_from_hl(self._hl_acute)
        t_elapsed_days = 1.0
        if signup:
            t_elapsed_days = max((now - signup).days, 1)
        decay = np.exp(-lam * min(t_elapsed_days, 90))

        views_score = min(bv.profile_views / 50.0, 1.0) * 0.25
        apps_score = min(bv.applications_submitted / 10.0, 1.0) * 0.35
        open_to_work_score = (0.40 if bv.open_to_work else 0.0)

        raw = (views_score + apps_score + open_to_work_score) * decay
        return min(raw, 1.0)

    def _score_responsiveness(self, bv: BehavioralView) -> float:
        """Score for recruiter-engagement signals."""
        lam = self._lambda_from_hl(self._hl_responsiveness)
        decay = np.exp(-lam * 30)

        resp_rate = bv.recruiter_response_rate if bv.recruiter_response_rate >= 0 else 0.0
        resp_time_norm = 1.0 - min(bv.avg_response_time_hours / 168.0, 1.0) if bv.avg_response_time_hours > 0 else 0.0
        interview_norm = bv.interview_completion_rate if bv.interview_completion_rate >= 0 else 0.0

        raw = (0.4 * resp_rate + 0.3 * resp_time_norm + 0.3 * interview_norm) * decay
        return min(raw, 1.0)

    def _score_presence(self, bv: BehavioralView, now: date, signup: date | None, last_active: date | None) -> float:
        """Score for general platform presence and engagement."""
        lam = self._lambda_from_hl(self._hl_presence)
        t_since_last = 30
        if last_active:
            t_since_last = max((now - last_active).days, 1)
        decay = np.exp(-lam * min(t_since_last, 180))

        completeness = bv.profile_completeness / 100.0
        connections = min(bv.connection_count / 500.0, 1.0)
        endorsements = min(bv.endorsements_received / 50.0, 1.0)
        search_app = min(bv.search_appearance_30d / 200.0, 1.0)
        saved = min(bv.saved_by_recruiters_30d / 20.0, 1.0)
        github = max(bv.github_activity_score / 100.0, 0.0) if bv.github_activity_score >= 0 else 0.0

        score = (0.20 * completeness + 0.20 * connections + 0.15 * endorsements
                 + 0.15 * search_app + 0.15 * saved + 0.15 * github)
        return min(score * decay, 1.0)

    def _estimate_confidence(self, bv: BehavioralView) -> float:
        n_signals = 0
        if bv.profile_views > 0:
            n_signals += 1
        if bv.applications_submitted > 0:
            n_signals += 1
        if bv.recruiter_response_rate >= 0:
            n_signals += 1
        if bv.avg_response_time_hours > 0:
            n_signals += 1
        if bv.connection_count > 0:
            n_signals += 1
        if bv.search_appearance_30d > 0:
            n_signals += 1
        if bv.saved_by_recruiters_30d > 0:
            n_signals += 1
        if bv.github_activity_score >= 0:
            n_signals += 1
        return min(0.3 + 0.1 * n_signals, 1.0)

    @staticmethod
    def _parse_date(d: str) -> date | None:
        if not d:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(d, fmt).date()
            except ValueError:
                continue
        return None
