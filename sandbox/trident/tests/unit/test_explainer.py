import pytest

from app.domain.explanation.explainer import Explainer
from app.domain.models import RankedCandidate, JobPosting, CandidateProfile, RedrobSignals, Skill


@pytest.fixture
def job():
    return JobPosting(
        title="Senior AI Engineer",
        description="AI/ML role with embeddings, retrieval, ranking focus",
        required_skills=["Python", "Embeddings", "NLP"],
        preferred_skills=["FAISS", "NDCG"],
        seniority_band="senior",
        role_family="data_science",
    )


@pytest.fixture
def ranked_candidate():
    return RankedCandidate(
        candidate_id="c1",
        rank=1,
        score=0.85,
        fused_score=0.85,
        semantic_score=0.9,
        career_score=0.8,
        behavioral_score=0.7,
        gate_weights=[0.4, 0.35, 0.25],
        low_confidence=False,
    )


@pytest.fixture
def profile():
    return CandidateProfile(
        candidate_id="c1",
        current_title="ML Engineer",
        current_company="Acme",
        years_of_experience=6.5,
        location="Pune, Maharashtra",
        country="India",
        current_industry="Technology",
        skills=[
            Skill(name="Python", proficiency="expert", duration_months=60),
            Skill(name="Embeddings", proficiency="advanced", duration_months=24),
            Skill(name="FAISS", proficiency="intermediate", duration_months=12),
            Skill(name="NLP", proficiency="advanced", duration_months=36),
        ],
        redrob_signals=RedrobSignals(
            last_active_date="2026-05-26",
            recruiter_response_rate=0.85,
            open_to_work_flag=True,
            notice_period_days=45,
            github_activity_score=50.0,
            saved_by_recruiters_30d=12,
            profile_views_received_30d=45,
        ),
    )


def test_template_explain_with_profile(ranked_candidate, job, profile):
    explainer = Explainer()
    explanation = explainer.explain(ranked_candidate, job, profile=profile)
    assert "ML Engineer" in explanation
    assert "6.5" in explanation
    assert "Acme" in explanation
    assert "Python" in explanation
    assert "Embeddings" in explanation
    assert "active" in explanation
    assert "Pune" in explanation


def test_no_profile_fallback(ranked_candidate):
    explainer = Explainer()
    explanation = explainer.explain(ranked_candidate, "Test")
    assert "Score" in explanation


def test_template_explain_low_confidence(job, profile):
    profile2 = profile.model_copy()
    profile2.redrob_signals = RedrobSignals(
        last_active_date="2025-01-01",
        recruiter_response_rate=0.1,
        open_to_work_flag=False,
        github_activity_score=-1,
    )
    rc = RankedCandidate(
        candidate_id="c2",
        rank=5,
        score=0.5,
        fused_score=0.5,
        semantic_score=0.5,
        career_score=0.5,
        behavioral_score=0.5,
        gate_weights=[0.4, 0.35, 0.25],
        low_confidence=True,
    )
    explainer = Explainer()
    explanation = explainer.explain(rc, job, profile=profile2)
    assert explanation
    assert "Concerns" in explanation
