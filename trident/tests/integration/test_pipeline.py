"""
Integration test that runs the full pipeline end-to-end on a tiny fixture dataset.
"""

import pytest

from app.core.config import Settings
from app.domain.pipeline import RankingPipeline
from app.domain.gate.fixed_weight_gate import FixedWeightGate
from app.domain.retrieval.vector_index import FAISSIndex
from app.domain.rerank.cross_encoder import CrossEncoderReranker
from app.domain.explanation.explainer import Explainer
from app.infra.embeddings import EmbeddingService
from app.infra.persistence import InMemoryCandidateRepository
from tests.fixtures.sample_data import (
    make_job,
    make_candidate_strong,
    make_candidate_weak,
    make_candidate_empty_behavioral,
)


@pytest.fixture
def config():
    c = Settings()
    c.retrieval_top_k = 10
    c.rerank_k = 5
    c.mmr_lambda = 0.7
    return c


@pytest.fixture
def repo():
    return InMemoryCandidateRepository()


@pytest.fixture
def pipeline(config, repo):
    emb = EmbeddingService(model=None)
    idx = FAISSIndex(dim=384)
    ce = CrossEncoderReranker(model=None)
    expl = Explainer()
    gate = FixedWeightGate(weights=config.gate_fallback_weights)
    return RankingPipeline(
        config=config,
        candidate_repo=repo,
        embedding_service=emb,
        vector_index=idx,
        cross_encoder=ce,
        explainer=expl,
        gate=gate,
    )


@pytest.mark.asyncio
async def test_pipeline_end_to_end(pipeline, repo):
    """Run full pipeline with fixture data; expect ranked candidates back."""
    strong = make_candidate_strong()
    weak = make_candidate_weak()
    empty = make_candidate_empty_behavioral()
    await repo.add_many([strong, weak, empty])

    job = make_job()
    ranked = await pipeline.rank(job)

    assert len(ranked) > 0
    # The strong candidate should rank higher
    ranked_ids = [r.candidate_id for r in ranked]
    assert strong.candidate_id in ranked_ids
    assert weak.candidate_id in ranked_ids

    # Check rank ordering
    for i in range(len(ranked) - 1):
        assert ranked[i].score >= ranked[i + 1].score

    # Check all scores in [0, 1]
    for r in ranked:
        assert 0.0 <= r.score <= 1.0
        assert 0.0 <= r.semantic_score <= 1.0
        assert 0.0 <= r.career_score <= 1.0
        assert 0.0 <= r.behavioral_score <= 1.0

    # Check reasoning is not empty
    for r in ranked:
        assert len(r.reasoning) > 0


@pytest.mark.asyncio
async def test_pipeline_graceful_degradation(pipeline, repo):
    """Pipeline should handle candidates with missing behavioral data."""
    empty = make_candidate_empty_behavioral()
    await repo.add_many([empty])

    job = make_job()
    ranked = await pipeline.rank(job, candidate_ids=["CAND_TEST_003"])

    assert len(ranked) == 1
    # Should still produce a score, not error
    assert ranked[0].score > 0
    assert ranked[0].behavioral_score > 0


@pytest.mark.asyncio
async def test_pipeline_ranking_matches_expected(pipeline, repo):
    """Strong candidate should rank higher than weak candidate."""
    strong = make_candidate_strong()
    weak = make_candidate_weak()
    await repo.add_many([strong, weak])

    job = make_job()
    ranked = await pipeline.rank(job)

    strong_rank = next((r.rank for r in ranked if r.candidate_id == strong.candidate_id), None)
    weak_rank = next((r.rank for r in ranked if r.candidate_id == weak.candidate_id), None)

    assert strong_rank is not None
    assert weak_rank is not None
    # Strong candidate should rank higher (lower rank number)
    assert strong_rank < weak_rank
