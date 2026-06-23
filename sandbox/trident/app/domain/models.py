from __future__ import annotations
from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import Any


# ─── Raw input models ───────────────────────────────────────────────────────

class Skill(BaseModel):
    name: str
    proficiency: str = ""
    endorsements: int = 0
    duration_months: int = 0


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int | None = None
    end_year: int | None = None
    grade: str | None = None
    tier: str = "unknown"


class CareerEntry(BaseModel):
    company: str
    title: str
    start_date: str | None = None
    end_date: str | None = None
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""


class RedrobSignals(BaseModel):
    profile_completeness_score: float = 0.0
    signup_date: str = ""
    last_active_date: str = ""
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: dict[str, float] = {}
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    expected_salary_range_inr_lpa: dict[str, float] = {}
    preferred_work_mode: str = ""
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


class CandidateProfile(BaseModel):
    candidate_id: str
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""
    career_history: list[CareerEntry] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)


class JobPosting(BaseModel):
    job_id: str = "default"
    title: str = ""
    description: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    seniority_band: str = "mid"
    role_family: str = "engineering"
    location: str = ""
    country: str = ""
    min_years_experience: int = 0
    max_years_experience: int = 50


# ─── Three views (stage 0 output) ───────────────────────────────────────────

class TextView(BaseModel):
    candidate_id: str
    text: str
    embedding: list[float] | None = None


class RoleTuple(BaseModel):
    title: str
    seniority: str
    employer: str
    industry: str
    tenure_months: int
    is_current: bool
    start_date: str | None = None
    end_date: str | None = None


class CareerView(BaseModel):
    candidate_id: str
    role_sequence: list[RoleTuple] = Field(default_factory=list)
    total_experience_years: float = 0.0
    current_title: str = ""
    current_industry: str = ""
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)


class BehavioralView(BaseModel):
    candidate_id: str
    profile_views: int = 0
    applications_submitted: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    connection_count: int = 0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    github_activity_score: float = -1.0
    profile_completeness: float = 0.0
    open_to_work: bool = False
    signup_date: str = ""
    last_active_date: str = ""
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)
    endorsements_received: int = 0
    verified_email: bool = False
    verified_phone: bool = False


# ─── Stage outputs ──────────────────────────────────────────────────────────

class ExpertScore(BaseModel):
    candidate_id: str
    score: float
    confidence: float = 1.0
    low_confidence: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusedScore(BaseModel):
    candidate_id: str
    fused_score: float
    semantic_score: float
    career_score: float
    behavioral_score: float
    gate_weights: list[float] = Field(default_factory=lambda: [0.4, 0.35, 0.25])
    low_confidence: bool = False


class RankedCandidate(BaseModel):
    candidate_id: str
    rank: int
    score: float
    fused_score: float
    semantic_score: float = 0.0
    career_score: float = 0.0
    behavioral_score: float = 0.0
    gate_weights: list[float] = Field(default_factory=lambda: [0.4, 0.35, 0.25])
    reasoning: str = ""
    low_confidence: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
