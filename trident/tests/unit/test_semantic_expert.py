import numpy as np
import pytest

from app.domain.experts.semantic import SemanticExpert
from app.domain.models import JobPosting, TextView, ExpertScore


@pytest.fixture
def job():
    return JobPosting(
        job_id="test",
        title="ML Engineer",
        description="Looking for an ML engineer with NLP experience.",
        required_skills=["Python", "ML"],
    )


@pytest.mark.asyncio
async def test_semantic_scoring(job):
    expert = SemanticExpert(temperature=0.05)
    views = [
        TextView(
            candidate_id="c1",
            text="ML engineer with NLP experience",
            embedding=[0.1] * 384,
        ),
        TextView(
            candidate_id="c2",
            text="Accountant with finance background",
            embedding=[-0.1] * 384,
        ),
    ]
    job_emb = np.array([0.2] * 384, dtype=np.float32)

    scores = await expert.score(job, views, job_embedding=job_emb)
    assert len(scores) == 2
    assert scores[0].candidate_id == "c1"
    assert scores[1].candidate_id == "c2"
    assert 0.0 <= scores[0].score <= 1.0
    assert 0.0 <= scores[1].score <= 1.0
    # c1 should score higher because both embeddings point in similar direction
    assert scores[0].score > scores[1].score


@pytest.mark.asyncio
async def test_missing_embedding_fallback(job):
    expert = SemanticExpert(temperature=0.05)
    views = [
        TextView(candidate_id="c1", text="some text", embedding=None),
    ]

    scores = await expert.score(job, views, job_embedding=None)
    assert len(scores) == 1
    assert scores[0].score == 0.5
    assert scores[0].low_confidence is True


@pytest.mark.asyncio
async def test_scores_in_range(job):
    expert = SemanticExpert(temperature=0.1)
    views = [
        TextView(
            candidate_id="c1",
            text="perfect match candidate",
            embedding=[0.5] * 384,
        ),
    ]
    job_emb = np.array([0.5] * 384, dtype=np.float32)

    scores = await expert.score(job, views, job_embedding=job_emb)
    assert 0.0 <= scores[0].score <= 1.0
