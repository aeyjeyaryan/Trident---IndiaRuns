import pytest

from app.domain.experts.behavioral import BehavioralExpert
from app.domain.models import JobPosting, BehavioralView, ExpertScore


@pytest.fixture
def expert():
    return BehavioralExpert()


@pytest.fixture
def job():
    return JobPosting(job_id="test", title="Test Role", description="Test description")


@pytest.mark.asyncio
async def test_active_candidate_high_score(expert, job):
    bv = BehavioralView(
        candidate_id="active_cand",
        profile_views=50,
        applications_submitted=8,
        recruiter_response_rate=0.9,
        avg_response_time_hours=2.0,
        connection_count=500,
        search_appearance_30d=200,
        saved_by_recruiters_30d=20,
        interview_completion_rate=0.95,
        offer_acceptance_rate=0.7,
        github_activity_score=80,
        profile_completeness=95,
        open_to_work=True,
        signup_date="2024-01-01",
        last_active_date="2026-06-20",
        skill_assessment_scores={"Python": 90},
        endorsements_received=100,
        verified_email=True,
        verified_phone=True,
    )

    scores = await expert.score(job, [bv])
    assert len(scores) == 1
    assert scores[0].score > 0.6  # Active candidate should score high
    assert scores[0].low_confidence is False


@pytest.mark.asyncio
async def test_inactive_candidate_low_score(expert, job):
    bv = BehavioralView(
        candidate_id="inactive_cand",
        profile_views=0,
        applications_submitted=0,
        recruiter_response_rate=0.0,
        avg_response_time_hours=0.0,
        connection_count=10,
        search_appearance_30d=0,
        saved_by_recruiters_30d=0,
        interview_completion_rate=0.0,
        offer_acceptance_rate=-1.0,
        github_activity_score=-1.0,
        profile_completeness=20,
        open_to_work=False,
        signup_date="2024-06-01",
        last_active_date="2025-01-01",
        endorsements_received=0,
    )

    scores = await expert.score(job, [bv])
    assert len(scores) == 1
    # Inactive candidate should score near or below neutral prior
    assert scores[0].score <= 0.55


@pytest.mark.asyncio
async def test_empty_behavioral_data_returns_prior(expert, job):
    """Graceful degradation: no behavioral data → neutral prior 0.5, not 0."""
    bv = BehavioralView(
        candidate_id="empty_cand",
    )

    scores = await expert.score(job, [bv])
    assert len(scores) == 1
    assert scores[0].score == 0.5
    assert scores[0].low_confidence is True
    assert scores[0].metadata.get("reason") == "no_behavioral_data"


@pytest.mark.asyncio
async def test_all_scores_in_range(expert, job):
    candidates = [
        BehavioralView(candidate_id=f"c{i}",
                       profile_views=i * 10,
                       applications_submitted=i,
                       recruiter_response_rate=i * 0.1,
                       avg_response_time_hours=100 - i * 10,
                       connection_count=i * 100,
                       search_appearance_30d=i * 20,
                       saved_by_recruiters_30d=i,
                       interview_completion_rate=i * 0.1,
                       github_activity_score=i * 10 - 1,
                       profile_completeness=i * 10,
                       open_to_work=bool(i % 2),
                       signup_date="2024-01-01",
                       last_active_date="2026-06-01",
                       endorsements_received=i * 10,
                       verified_email=True,
                       verified_phone=True,
                       )
        for i in range(5)
    ]
    scores = await expert.score(job, candidates)
    for s in scores:
        assert 0.0 <= s.score <= 1.0
