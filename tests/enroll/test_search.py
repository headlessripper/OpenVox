import numpy as np
from openvox.enroll.search import maximize


def test_maximize_improves_toward_optimum():
    target = np.array([0.3, -0.7, 0.5])
    def score_fn(x):
        return -float(np.sum((x - target) ** 2))   # optimum at x == target
    x0 = np.zeros(3)
    best_x, best_score = maximize(score_fn, x0, max_evals=200, seed=0)
    assert best_score > score_fn(x0)               # strictly better than start
    assert np.linalg.norm(best_x - target) < np.linalg.norm(x0 - target)


def test_maximize_respects_eval_budget():
    calls = {"n": 0}
    def score_fn(x):
        calls["n"] += 1
        return -float(np.sum(x ** 2))
    maximize(score_fn, np.zeros(4), max_evals=10, seed=0)
    assert calls["n"] <= 10
