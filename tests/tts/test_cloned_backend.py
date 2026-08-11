import numpy as np
from openvox.tts.cloned_backend import ClonedVoiceBackend, CLONED_VOICE_ID
from openvox.tts.backend import TTSResult

class _FakeConds:
    pass

class _FakeProfile:
    conditionals = _FakeConds()

def test_synthesize_clones_each_segment(monkeypatch):
    calls = {}
    class _FakeChatBackend:
        def __init__(self, device):
            calls["device"] = device
        def clone_from_profile(self, text, conditionals, exaggeration, cfg):
            calls.setdefault("texts", []).append(text)
            calls["exaggeration"] = exaggeration
            return TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)
    # patch the lazy import target
    import openvox.clone.chatterbox_backend as cb
    monkeypatch.setattr(cb, "ChatterboxBackend", _FakeChatBackend)

    be = ClonedVoiceBackend(_FakeProfile(), device="cpu", exaggeration=0.7)
    out = list(be.stream_segments(["one", "two"], voice=CLONED_VOICE_ID, speed=1.0))
    assert len(out) == 2 and out[0].sample_rate == 24000
    assert calls["texts"] == ["one", "two"]
    assert calls["device"] == "cpu" and calls["exaggeration"] == 0.7
