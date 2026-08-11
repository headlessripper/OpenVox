import numpy as np
import pytest

from openvox.enroll import VoiceEnrollEngine, VoiceProfile
from openvox.enroll.config import EnrollConfig

class FakeBackend:
    def __init__(self, cuda):
        self._cuda = cuda
    @property
    def cuda_available(self):
        return self._cuda
    @property
    def model_sr(self):
        return 24000
    def embed_clips(self, wavs_16k):
        # two identical clips pointing at a fixed identity
        return np.stack([np.array([1.0, 0.0, 0.0], dtype=np.float32)] * len(wavs_16k))
    def reference_from_clip(self, clip_16k, clip_24k):
        return {"gen": "ref", "cond_prompt": None}
    def make_conditionals(self, speaker_emb, reference, exaggeration):
        return np.asarray(speaker_emb, dtype=np.float32)
    def generate(self, conds, text, seed):
        return conds, 16000
    def ve_embed(self, wav, sr):
        return np.asarray(wav, dtype=np.float32)

@pytest.fixture
def patched_prep(monkeypatch):
    def fake_prepare(paths, **kw):
        one = np.ones(16000, dtype=np.float32)
        return [one, one], [one, one]      # clips_16k, clips_24k
    monkeypatch.setattr("openvox.enroll.engine.prepare", fake_prepare)

def _engine(cuda, patched_prep):
    eng = VoiceEnrollEngine(device="cpu", config=EnrollConfig(enhance_clips=False, seed=0))
    eng._backend = FakeBackend(cuda=cuda)
    return eng

def test_enroll_stage_a_only_without_gpu(patched_prep):
    eng = _engine(cuda=False, patched_prep=patched_prep)
    prof = eng.enroll(["a.wav"])
    assert isinstance(prof, VoiceProfile)
    assert prof.metadata["stage"] == "A"
    assert prof.score > 0.9                    # matches the fixed identity

def test_enroll_runs_stage_b_with_gpu(patched_prep):
    eng = _engine(cuda=True, patched_prep=patched_prep)
    prof = eng.enroll(["a.wav"])
    assert prof.metadata["stage"] in ("A", "B")   # B only ships if it wins
    assert prof.score >= 0.9

def test_enroll_requires_survivors(patched_prep, monkeypatch):
    monkeypatch.setattr("openvox.enroll.engine.prepare", lambda paths, **kw: ([], []))
    eng = _engine(cuda=False, patched_prep=patched_prep)
    with pytest.raises(ValueError, match="clip"):
        eng.enroll(["a.wav"])
