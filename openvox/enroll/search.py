import numpy as np


def maximize(score_fn, x0, max_evals, seed=0, init_step=0.5, min_step=0.03):
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).copy()
    best = score_fn(x)
    evals = 1
    step = init_step
    dim = x.size
    while evals < max_evals and step >= min_step:
        improved = False
        for i in rng.permutation(dim):
            if evals >= max_evals:
                break
            for sign in (1.0, -1.0):
                if evals >= max_evals:
                    break
                cand = x.copy()
                cand[i] += sign * step
                s = score_fn(cand)
                evals += 1
                if s > best:
                    best, x, improved = s, cand, True
                    break
        if not improved:
            step *= 0.5
    return x, best
