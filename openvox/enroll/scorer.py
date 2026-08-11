import numpy as np

from openvox.enroll.aggregate import l2_normalize, cosine

def score_embedding(backend, emb, reference, target_centroid, probes,
                    realizability_lambda, clip_embs, exaggeration, seed) -> float:
    conds = backend.make_conditionals(emb, reference, exaggeration)
    sims = []
    for text in probes:
        wav, sr = backend.generate(conds, text, seed)
        out = backend.ve_embed(wav, sr)
        sims.append(cosine(out, target_centroid))
    similarity = float(np.mean(sims))
    unit_emb = l2_normalize(np.asarray(emb, dtype=np.float32))
    unit_clips = l2_normalize(np.asarray(clip_embs, dtype=np.float32), axis=1)
    penalty = float(np.min(np.linalg.norm(unit_clips - unit_emb, axis=1)))
    return similarity - realizability_lambda * penalty
