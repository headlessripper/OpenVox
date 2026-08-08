import numpy as np
import pytest
from openvox.clone.engine import VoiceCloneEngine
from openvox.tts.backend import TTSResult

class _FakeBackend:
    def __init__(self, *a, **k): pass
    def clone(self, text, reference_path, exaggeration, cfg):
        return TTSResult(audio=np.zeros(24000, dtype=np.float32), sample_rate=24000)

@pytest.fixture
def patched(monkeypatch):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _FakeBackend)

def test_empty_text_raises(patched, tmp_path):
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    with pytest.raises(ValueError):
        VoiceCloneEngine().clone("   ", str(ref))

def test_missing_reference_raises(patched):
    with pytest.raises(FileNotFoundError):
        VoiceCloneEngine().clone("hello", "does_not_exist.wav")

def test_empty_reference_raises(patched, tmp_path):
    ref = tmp_path / "empty.wav"; ref.write_bytes(b"")
    with pytest.raises(ValueError):
        VoiceCloneEngine().clone("hello", str(ref))

def test_clone_returns_result(patched, tmp_path):
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    r = VoiceCloneEngine().clone("hello", str(ref))
    assert r.sample_rate == 24000 and len(r.audio) == 24000

def test_config_not_mutated(patched, tmp_path):
    from openvox.clone.config import CloneConfig
    cfg = CloneConfig()
    VoiceCloneEngine(device="cpu", config=cfg)
    assert cfg.device == "cuda"   # caller's config untouched
