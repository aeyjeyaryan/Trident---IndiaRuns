from pydantic import BaseModel, Field
from typing import Any


class ScoreBreakdown(BaseModel):
    semantic: float = 0.0
    career: float = 0.0
    behavioral: float = 0.0


class GateWeights(BaseModel):
    semantic: float = 0.4
    career: float = 0.35
    behavioral: float = 0.25


class RankRequest(BaseModel):
    job_title: str = Field(default="", description="Job title")
    job_description: str = Field(..., description="Full job description text")
    required_skills: list[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: list[str] = Field(default_factory=list, description="Preferred skills")
    seniority_band: str = Field(default="mid", description="seniority_band")
    role_family: str = Field(default="engineering", description="role_family")
    top_k: int = Field(default=50, description="Number of ranked candidates to return")
    candidate_ids: list[str] | None = Field(default=None, description="Filter to specific candidates")
    min_years_experience: int = Field(default=0)
    max_years_experience: int = Field(default=50)


class RankedCandidateResponse(BaseModel):
    candidate_id: str
    rank: int
    score: float
    semantic_score: float = 0.0
    career_score: float = 0.0
    behavioral_score: float = 0.0
    gate_weights: GateWeights = Field(default_factory=GateWeights)
    reasoning: str = ""
    low_confidence: bool = False


class RankResponse(BaseModel):
    job_id: str
    candidates: list[RankedCandidateResponse]
    total_candidates_considered: int = 0
    pipeline_latency_ms: float = 0.0


class IngestRequest(BaseModel):
    candidates: list[dict] = Field(default_factory=list, description="List of candidate dicts")


class IngestResponse(BaseModel):
    ingested: int = 0
    total: int = 0


class IndexResponse(BaseModel):
    indexed: int = 0
    dim: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    candidates_loaded: int = 0
    index_size: int = 0
    gate_type: str = ""


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
