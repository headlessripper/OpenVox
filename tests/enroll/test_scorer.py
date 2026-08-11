import numpy as np
from openvox.enroll.scorer import score_embedding

class FakeBackend:
    """generate() echoes the conditioning; ve_embed() returns it -> score == cos(emb, target)."""
    def make_conditionals(self, emb, reference, exaggeration):
        return np.asarray(emb, dtype=np.float32)
    def generate(self, conds, text, seed):
        return conds, 16000
    def ve_embed(self, wav, sr):
        return np.asarray(wav, dtype=np.float32)

def test_closer_embedding_scores_higher():
    target = np.array([1.0, 0.0, 0.0])
    clips = np.stack([target, target])           # on-manifold
    be = FakeBackend()
    kw = dict(reference=None, target_centroid=target, probes=["hi"],
              realizability_lambda=0.0, clip_embs=clips, exaggeration=0.5, seed=0)
    near = score_embedding(be, np.array([0.9, 0.1, 0.0]), **kw)
    far = score_embedding(be, np.array([0.0, 1.0, 0.0]), **kw)
    assert near > far

def test_realizability_penalty_reduces_offmanifold_score():
    target = np.array([1.0, 0.0, 0.0])
    clips = np.stack([target])
    be = FakeBackend()
    emb = np.array([1.0, 0.0, 0.0])
    no_pen = score_embedding(be, emb, reference=None, target_centroid=target,
                             probes=["hi"], realizability_lambda=0.0,
                             clip_embs=np.stack([np.array([0.0, 1.0, 0.0])]),
                             exaggeration=0.5, seed=0)
    with_pen = score_embedding(be, emb, reference=None, target_centroid=target,
                               probes=["hi"], realizability_lambda=1.0,
                               clip_embs=np.stack([np.array([0.0, 1.0, 0.0])]),
                               exaggeration=0.5, seed=0)
    assert with_pen < no_pen
