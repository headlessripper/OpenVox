import numpy as np
import pytest
from openvox.clone.engine import VoiceCloneEngine
from openvox.tts.backend import TTSResult

class _RecordingBackend:
    def __init__(self, *a, **k): self.ref = None
    def clone(self, text, reference_path, exaggeration, cfg):
        self.ref = reference_path
        return TTSResult(audio=np.zeros(10, dtype=np.float32), sample_rate=24000)

def _wav(path):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.zeros(16000, dtype=np.int16)).tobytes())

def test_enhance_false_uses_raw_reference(monkeypatch, tmp_path):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _RecordingBackend)
    ref = tmp_path / "ref.wav"; _wav(ref)
    e = VoiceCloneEngine()
    e.clone("hi", str(ref), enhance=False)
    assert e._backend.ref == str(ref)              # raw reference passed

def test_enhance_true_uses_cleaned_reference(monkeypatch, tmp_path):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _RecordingBackend)

    class _FakeEnhance:
        def __init__(self, *a, **k): pass
        def enhance_file(self, path):
            return TTSResult(audio=np.zeros(44100, dtype=np.float32), sample_rate=44100)
    import openvox.enhance
    monkeypatch.setattr(openvox.enhance, "EnhanceEngine", _FakeEnhance)

    ref = tmp_path / "ref.wav"; _wav(ref)
    e = VoiceCloneEngine()
    e.clone("hi", str(ref), enhance=True)
    assert e._backend.ref != str(ref)              # a cleaned (cached) path
    assert e._backend.ref.endswith(".wav")

def test_enhance_unavailable_degrades_to_raw(monkeypatch, tmp_path):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _RecordingBackend)

    class _Boom:
        def __init__(self, *a, **k): raise RuntimeError("enhance deps missing")
    import openvox.enhance
    monkeypatch.setattr(openvox.enhance, "EnhanceEngine", _Boom)

    ref = tmp_path / "ref.wav"; _wav(ref)
    e = VoiceCloneEngine()
    e.clone("hi", str(ref), enhance=True)          # must not raise
    assert e._backend.ref == str(ref)              # degraded to raw reference

def test_stat_failure_degrades_to_raw(monkeypatch, tmp_path):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _RecordingBackend)
    ref = tmp_path / "ref.wav"; _wav(ref)
    e = VoiceCloneEngine()
    # Patch cache_dir to raise, forcing error inside _enhanced_reference's try block
    import openvox._paths
    monkeypatch.setattr(openvox._paths, "cache_dir", lambda *a, **k: (_ for _ in ()).throw(OSError("cache_dir boom")))
    e.clone("hi", str(ref), enhance=True)          # must not raise
    assert e._backend.ref == str(ref)              # degraded to raw
