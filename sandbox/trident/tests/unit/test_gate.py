import pytest

from app.domain.gate.fixed_weight_gate import FixedWeightGate
from app.domain.models import JobPosting


@pytest.fixture
def job():
    return JobPosting(
        job_id="test", title="ML Engineer",
        description="ML role", role_family="data_science",
        seniority_band="senior",
    )


def test_fixed_gate_fusion(job):
    gate = FixedWeightGate(weights=(0.5, 0.3, 0.2))
    cids = ["c1", "c2"]
    sem = [0.9, 0.3]
    car = [0.8, 0.4]
    beh = [0.7, 0.5]

    results = gate.fuse(job, sem, car, beh, cids)

    assert len(results) == 2
    # c1: 0.5*0.9 + 0.3*0.8 + 0.2*0.7 = 0.45 + 0.24 + 0.14 = 0.83
    assert results[0][1] == pytest.approx(0.83)
    assert results[0][2] == [0.5, 0.3, 0.2]


def test_fixed_gate_default_weights(job):
    gate = FixedWeightGate()
    results = gate.fuse(job, [0.8], [0.7], [0.6], ["c1"])
    # default: 0.4, 0.35, 0.25 => 0.4*0.8 + 0.35*0.7 + 0.25*0.6 = 0.32 + 0.245 + 0.15 = 0.715
    assert results[0][1] == pytest.approx(0.715)


def test_fixed_gate_mismatched_lengths(job):
    gate = FixedWeightGate()
    results = gate.fuse(job, [0.8], [0.7, 0.6], [0.6], ["c1", "c2"])
    assert len(results) == 2
    # Should handle gracefully with out-of-range defaults
    assert results[0][1] > 0
