import numpy as np
import pytest
from openvox.enhance.engine import EnhanceEngine
from openvox.tts.backend import TTSResult

class _FakeBackend:
    def __init__(self, *a, **k): pass
    def enhance(self, audio, sample_rate):
        return TTSResult(audio=np.zeros(44100, dtype=np.float32), sample_rate=44100)

@pytest.fixture
def patched(monkeypatch):
    import openvox.enhance.engine as eng
    monkeypatch.setattr(eng, "ResembleEnhanceBackend", _FakeBackend)

def test_empty_array_raises(patched):
    with pytest.raises(ValueError):
        EnhanceEngine().enhance(np.zeros(0, dtype=np.float32), 16000)

def test_missing_file_raises(patched):
    with pytest.raises(FileNotFoundError):
        EnhanceEngine().enhance_file("does_not_exist.wav")

def test_empty_file_raises(patched, tmp_path):
    p = tmp_path / "empty.wav"; p.write_bytes(b"")
    with pytest.raises(ValueError):
        EnhanceEngine().enhance_file(str(p))

def test_enhance_returns_result(patched):
    r = EnhanceEngine().enhance(np.ones(1000, dtype=np.float32), 16000)
    assert r.sample_rate == 44100 and len(r.audio) == 44100

def test_config_not_mutated(patched):
    from openvox.enhance.config import EnhanceConfig
    cfg = EnhanceConfig()
    EnhanceEngine(device="cpu", denoise_only=True, config=cfg)
    assert cfg.device == "cuda" and cfg.denoise_only is False
