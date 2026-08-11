import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize vectors to unit length along specified axis."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, 1e-9, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)


def robust_centroid(
    embs: np.ndarray, outlier_threshold: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute a robust centroid from speaker embeddings.

    Drops embeddings whose cosine similarity to the initial centroid
    falls below the outlier threshold. Never drops all embeddings.

    Args:
        embs: (N, D) array of embeddings
        outlier_threshold: cosine similarity threshold for outlier detection

    Returns:
        centroid: normalized (D,) centroid vector
        kept: (N,) boolean mask of kept embeddings
        representative_idx: index of the kept embedding closest to final centroid
    """
    unit = l2_normalize(np.asarray(embs, dtype=np.float32), axis=1)  # (N, D)
    initial = l2_normalize(unit.mean(axis=0))
    sims = unit @ initial
    kept = sims >= outlier_threshold
    if not kept.any():
        kept = np.ones(len(unit), dtype=bool)  # never drop everything
    centroid = l2_normalize(unit[kept].mean(axis=0))
    # representative = surviving clip closest to the final centroid
    surviving_idx = np.where(kept)[0]
    rep = int(surviving_idx[np.argmax(unit[kept] @ centroid)])
    return centroid.astype(np.float32), kept, rep
