import pytest
from openvox.enroll.config import EnrollConfig, evals_for_quality


def test_defaults():
    c = EnrollConfig()
    assert c.device == "cuda" and c.quality == "balanced"
    assert c.max_evals is None and c.enhance_clips is True
    assert c.min_clips == 1 and c.exaggeration == 0.5


def test_evals_for_quality():
    assert evals_for_quality("fast") == 15
    assert evals_for_quality("balanced") == 40
    assert evals_for_quality("thorough") == 100
    with pytest.raises(ValueError):
        evals_for_quality("nope")
