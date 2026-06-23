import numpy as np
import pytest

from app.domain.rerank.mmr import MMRDiversityReranker


def test_mmr_basic():
    mmr = MMRDiversityReranker(lambda_=0.7)
    candidates = [
        ("c1", 0.9),
        ("c2", 0.85),
        ("c3", 0.8),
        ("c4", 0.75),
    ]
    embeddings = {
        "c1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "c2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "c3": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "c4": np.array([0.5, 0.5, 0.0], dtype=np.float32),
    }

    result = mmr.rerank(candidates, embeddings, top_k=2)
    assert len(result) == 2
    # c1 should be first (highest relevance)
    assert result[0][0] == "c1"


def test_mmr_empty():
    mmr = MMRDiversityReranker()
    result = mmr.rerank([], {})
    assert result == []


def test_mmr_preserves_order_when_diverse():
    mmr = MMRDiversityReranker(lambda_=0.7)
    candidates = [
        ("c1", 0.9),
        ("c2", 0.1),
    ]
    embeddings = {
        "c1": np.array([1.0, 0.0], dtype=np.float32),
        "c2": np.array([0.0, 1.0], dtype=np.float32),
    }
    result = mmr.rerank(candidates, embeddings, top_k=2)
    assert len(result) == 2
    assert result[0][0] == "c1"
