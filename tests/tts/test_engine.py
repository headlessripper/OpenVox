import numpy as np
import pytest
from openvox.tts.engine import TTSEngine
from openvox.tts.backend import TTSResult

# Unit tests: patch the backend so no model is loaded.
class _FakeBackend:
    def __init__(self, *a, **k): pass
    def synthesize(self, text, voice, speed):
        return TTSResult(audio=np.zeros(24000, dtype=np.float32), sample_rate=24000)

@pytest.fixture
def patched(monkeypatch):
    import openvox.tts.engine as eng
    monkeypatch.setattr(eng, "KokoroBackend", _FakeBackend)

def test_voices_listed(patched):
    assert "af_heart" in TTSEngine().voices()

def test_empty_text_raises(patched):
    with pytest.raises(ValueError):
        TTSEngine().synthesize("   ")

def test_unknown_voice_raises(patched):
    with pytest.raises(ValueError):
        TTSEngine().synthesize("hello", voice="bogus_voice")

def test_synthesize_returns_result(patched):
    r = TTSEngine(voice="am_michael").synthesize("hello")
    assert r.sample_rate == 24000
    assert len(r.audio) == 24000

def test_config_not_mutated(patched):
    from openvox.tts.config import TTSConfig
    cfg = TTSConfig()
    TTSEngine(voice="am_michael", config=cfg)
    assert cfg.voice == "af_heart"     # caller's config untouched

def test_nonpositive_speed_raises(patched):
    with pytest.raises(ValueError, match="speed must be > 0"):
        TTSEngine().synthesize("hello", speed=0)
    with pytest.raises(ValueError, match="speed must be > 0"):
        TTSEngine().synthesize("hello", speed=-1)
