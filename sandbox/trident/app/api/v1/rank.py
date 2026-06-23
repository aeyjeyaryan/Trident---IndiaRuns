import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.schemas import (
    RankRequest,
    RankResponse,
    RankedCandidateResponse,
    GateWeights,
    IngestRequest,
    IngestResponse,
    IndexResponse,
    HealthResponse,
)
from app.core.config import Settings
from app.domain.pipeline import RankingPipeline
from app.domain.models import JobPosting, CandidateProfile

log = logging.getLogger(__name__)

router = APIRouter()


def get_settings() -> Settings:
    from fastapi import Request
    request: Request = Request  # placeholder — real resolver via dependency override
    return Settings()


@router.post("/rank", response_model=RankResponse)
async def rank_candidates(
    req: RankRequest,
    pipeline: RankingPipeline = Depends(lambda: _get_pipeline()),
):
    start = time.monotonic()

    job = JobPosting(
        job_id="default",
        title=req.job_title,
        description=req.job_description,
        required_skills=req.required_skills,
        preferred_skills=req.preferred_skills,
        seniority_band=req.seniority_band,
        role_family=req.role_family,
        min_years_experience=req.min_years_experience,
        max_years_experience=req.max_years_experience,
    )

    try:
        ranked = await pipeline.rank(job, candidate_ids=req.candidate_ids)
    except Exception as e:
        log.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))

    top_k = ranked[: req.top_k]

    candidates_resp = []
    for rc in top_k:
        gw = rc.gate_weights
        candidates_resp.append(RankedCandidateResponse(
            candidate_id=rc.candidate_id,
            rank=rc.rank,
            score=rc.score,
            semantic_score=rc.semantic_score,
            career_score=rc.career_score,
            behavioral_score=rc.behavioral_score,
            gate_weights=GateWeights(
                semantic=gw[0] if isinstance(gw, list) and len(gw) >= 1 else 0.4,
                career=gw[1] if isinstance(gw, list) and len(gw) >= 2 else 0.35,
                behavioral=gw[2] if isinstance(gw, list) and len(gw) >= 3 else 0.25,
            ),
            reasoning=rc.reasoning,
            low_confidence=rc.low_confidence,
        ))

    return RankResponse(
        job_id="default",
        candidates=candidates_resp,
        total_candidates_considered=len(ranked),
        pipeline_latency_ms=round((time.monotonic() - start) * 1000, 1),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_candidates(
    req: IngestRequest,
    pipeline: RankingPipeline = Depends(lambda: _get_pipeline()),
):
    candidates = []
    for raw in req.candidates:
        candidates.append(_dict_to_candidate(raw))
    result = await pipeline.ingest_candidates(candidates)
    return IngestResponse(ingested=result["ingested"], total=result["total"])


@router.post("/index", response_model=IndexResponse)
async def build_index(
    pipeline: RankingPipeline = Depends(lambda: _get_pipeline()),
):
    result = await pipeline.build_index()
    return IndexResponse(
        indexed=result["indexed"],
        dim=result["dim"],
        message=f"Indexed {result['indexed']} candidates with dim={result['dim']}",
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    pipeline: RankingPipeline = Depends(lambda: _get_pipeline()),
    settings: Settings = Depends(get_settings),
):
    from app.domain.gate.learned_gate import LearnedGate
    gate_type = "learned" if isinstance(pipeline._gate, LearnedGate) and pipeline._gate.is_trained else "fixed_weight"
    return HealthResponse(
        status="ok",
        candidates_loaded=pipeline._candidate_repo.size,
        index_size=pipeline._vector_index.size,
        gate_type=gate_type,
    )


# ─── Module-level state (set by main.py at startup) ─────────────────────

_pipeline_instance: RankingPipeline | None = None


def _get_pipeline() -> RankingPipeline:
    if _pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline_instance


def set_pipeline(pipeline: RankingPipeline) -> None:
    global _pipeline_instance
    _pipeline_instance = pipeline


def _dict_to_candidate(raw: dict) -> CandidateProfile:
    from app.domain.models import CareerEntry, Education, Skill, RedrobSignals
    profile_data = raw.get("profile", {}) or {}
    redrob = raw.get("redrob_signals", {}) or {}
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
        career_history=[CareerEntry(**r) for r in (raw.get("career_history", []) or [])],
        education=[Education(**e) for e in (raw.get("education", []) or [])],
        skills=[Skill(**s) for s in (raw.get("skills", []) or [])],
        redrob_signals=RedrobSignals(**redrob),
    )
