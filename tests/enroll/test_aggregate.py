import numpy as np
from openvox.enroll.aggregate import l2_normalize, cosine, robust_centroid


def test_l2_normalize_and_cosine():
    v = np.array([3.0, 4.0])
    n = l2_normalize(v)
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.isclose(cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])), 1.0)


def test_robust_centroid_drops_outlier():
    # three tight vectors + one far outlier
    base = np.array([1.0, 0.0, 0.0])
    embs = np.stack([
        base + 0.01, base - 0.01, base,
        np.array([0.0, 1.0, 0.0]),   # outlier
    ])
    centroid, kept, rep = robust_centroid(embs, outlier_threshold=0.6)
    assert kept[3] == False            # outlier dropped
    assert kept[:3].all()
    assert np.isclose(np.linalg.norm(centroid), 1.0)
    assert rep in (0, 1, 2)


def test_robust_centroid_never_drops_all():
    embs = np.stack([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])
    centroid, kept, rep = robust_centroid(embs, outlier_threshold=0.99)
    assert kept.all()                  # would drop everything -> keep all
