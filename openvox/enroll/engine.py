import dataclasses
import hashlib
import logging

import numpy as np

from openvox.enroll.aggregate import l2_normalize, robust_centroid
from openvox.enroll.backend import EnrollBackend
from openvox.enroll.chatterbox_backend import ChatterboxEnrollBackend
from openvox.enroll.config import EnrollConfig, evals_for_quality
from openvox.enroll.prep import prepare
from openvox.enroll.profile import VoiceProfile
from openvox.enroll.scorer import score_embedding
from openvox.enroll.search import maximize

log = logging.getLogger(__name__)

DEFAULT_PROBES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the shore on a bright morning.",
    "How much wood would a woodchuck chuck if it could?",
    "We drove through six zigzagging mountain villages.",
    "Please call Stella and ask her to bring these things.",
    "Authentic voices carry warmth, weight, and quiet detail.",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


class VoiceEnrollEngine:
    def __init__(self, device: str | None = None, quality: str | None = None,
                 config: EnrollConfig | None = None) -> None:
        c = dataclasses.replace(config) if config is not None else EnrollConfig()
        if device is not None:
            c.device = device
        if quality is not None:
            c.quality = quality
        self._config = c
        self._backend: EnrollBackend = ChatterboxEnrollBackend(device=c.device)

    def enroll(self, clips: list[str]) -> VoiceProfile:
        if not clips:
            raise ValueError("provide at least one audio clip to enroll")

        c = self._config
        probes = c.probes if c.probes else DEFAULT_PROBES
        clips_16k, clips_24k = prepare(
            clips, enhance=c.enhance_clips, device=c.device,
            min_rms=c.min_rms, min_dur_s=c.min_dur_s, max_clip_s=c.max_clip_s,
        )
        if len(clips_16k) < c.min_clips:
            raise ValueError(
                f"only {len(clips_16k)} usable clip(s) after prep; need >= {c.min_clips}"
            )

        embs = self._backend.embed_clips(clips_16k)                 # (N, 256)
        centroid, kept, rep = robust_centroid(embs, c.outlier_threshold)
        kept_embs = l2_normalize(embs[kept], axis=1)
        reference = self._backend.reference_from_clip(clips_16k[rep], clips_24k[rep])

        def score_of(emb):
            return score_embedding(
                self._backend, emb, reference, centroid, probes,
                c.realizability_lambda, kept_embs, c.exaggeration, c.seed,
            )

        score_a = score_of(centroid)
        final_emb, final_score, stage = centroid, score_a, "A"

        if self._backend.cuda_available:
            max_evals = c.max_evals or evals_for_quality(c.quality)
            def weight_score(w):
                emb = l2_normalize(kept_embs.T @ _softmax(w))
                return score_of(emb)
            best_w, score_b = maximize(
                weight_score, np.zeros(len(kept_embs)), max_evals=max_evals, seed=c.seed,
            )
            if score_b >= score_a + c.accept_margin:
                final_emb = l2_normalize(kept_embs.T @ _softmax(best_w))
                final_score, stage = score_b, "B"
        else:
            log.info("No CUDA; enrolling with Stage A only (the search needs a GPU).")

        conds = self._backend.make_conditionals(final_emb, reference, c.exaggeration)
        metadata = {
            "stage": stage,
            "algorithm": "embedding-optimization",
            "schema_note": "openvox.enroll 2B.2b",
            "sample_rate": self._backend.model_sr,
            "n_clips_used": int(kept.sum()),
            "clip_sha1": [self._sha1(p) for p in clips],
        }
        return VoiceProfile(conditionals=conds, score=float(final_score), metadata=metadata)

    @staticmethod
    def _sha1(path: str) -> str | None:
        h = hashlib.sha1()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except OSError:
            log.warning("Could not hash clip %r for metadata; leaving clip_sha1 unset.", path)
            return None
        return h.hexdigest()
