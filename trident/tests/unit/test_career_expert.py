import pytest

from app.domain.experts.career import CareerExpert
from app.domain.models import JobPosting, CareerView, RoleTuple, Education, Skill


@pytest.fixture
def job():
    return JobPosting(
        job_id="test",
        title="Senior ML Engineer",
        description="ML engineering role",
        required_skills=["Python", "ML"],
        seniority_band="senior",
        role_family="data_science",
    )


@pytest.fixture
def expert():
    return CareerExpert()


@pytest.mark.asyncio
async def test_strong_candidate_scoring(expert, job):
    cv = CareerView(
        candidate_id="c1",
        role_sequence=[
            RoleTuple(
                title="Senior ML Engineer", seniority="senior", employer="TechCorp",
                industry="Technology", tenure_months=36, is_current=True,
            ),
            RoleTuple(
                title="Data Scientist", seniority="mid", employer="StartupAI",
                industry="Technology", tenure_months=24, is_current=False,
            ),
        ],
        total_experience_years=6.0,
        current_title="Senior ML Engineer",
        current_industry="Technology",
        skills=["Python", "PyTorch"],
        education=[
            Education(institution="MIT", degree="M.S.", field_of_study="CS"),
        ],
    )

    scores = await expert.score(job, [cv])
    assert len(scores) == 1
    assert scores[0].candidate_id == "c1"
    assert 0.0 <= scores[0].score <= 1.0
    assert scores[0].score > 0.5  # Should be a decent match


@pytest.mark.asyncio
async def test_weak_candidate_scoring(expert, job):
    cv = CareerView(
        candidate_id="c2",
        role_sequence=[
            RoleTuple(
                title="Junior Accountant", seniority="junior", employer="FinanceCorp",
                industry="Finance", tenure_months=12, is_current=True,
            ),
        ],
        total_experience_years=2.0,
        current_title="Junior Accountant",
        current_industry="Finance",
        skills=["Excel"],
    )

    scores = await expert.score(job, [cv])
    assert len(scores) == 1
    # Short history triggers neutral prior (0.5), not a strong match
    assert scores[0].score <= 0.5


@pytest.mark.asyncio
async def test_short_history_fallback(expert, job):
    cv = CareerView(
        candidate_id="c3",
        role_sequence=[
            RoleTuple(
                title="Intern", seniority="junior", employer="SomeCo",
                industry="Tech", tenure_months=3, is_current=True,
            ),
        ],
        total_experience_years=0.3,
        current_title="Intern",
        current_industry="Tech",
        skills=[],
    )

    scores = await expert.score(job, [cv])
    assert scores[0].low_confidence is True
    assert scores[0].metadata.get("reason") == "short_history"


@pytest.mark.asyncio
async def test_empty_history_fallback(expert, job):
    cv = CareerView(
        candidate_id="c4",
        role_sequence=[],
        total_experience_years=0,
        current_title="",
        current_industry="",
        skills=[],
    )

    scores = await expert.score(job, [cv])
    assert scores[0].low_confidence is True
    assert scores[0].metadata.get("num_roles", 0) == 0
