import numpy as np
import pytest

from openvox.clone import VoiceCloneEngine
from openvox.enroll import VoiceProfile

class _Conds:
    def to(self, device):
        return self

def test_profile_and_reference_are_mutually_exclusive():
    eng = VoiceCloneEngine(device="cpu")
    with pytest.raises(ValueError, match="either"):
        eng.clone("hi", reference_audio="ref.wav", profile=VoiceProfile(_Conds(), 0.9, {}))

def test_clone_from_profile_injects_conditionals(monkeypatch):
    eng = VoiceCloneEngine(device="cpu")
    captured = {}
    from openvox.tts.backend import TTSResult
    def fake_cfp(text, conditionals, exaggeration, cfg):
        captured["conds"] = conditionals
        captured["text"] = text
        return TTSResult(np.zeros(10, dtype=np.float32), 24000)
    monkeypatch.setattr(eng._backend, "clone_from_profile", fake_cfp)
    conds = _Conds()
    out = eng.clone("speak this", profile=VoiceProfile(conds, 0.9, {}))
    assert out.sample_rate == 24000
    assert captured["conds"] is conds and captured["text"] == "speak this"

def test_clone_requires_a_source():
    eng = VoiceCloneEngine(device="cpu")
    with pytest.raises(ValueError, match="reference_audio or profile"):
        eng.clone("hi")
