import numpy as np
import pytest
from openvox.tts import TTSEngine, TTSResult
from openvox.tts.engine import TTSEngine as _Eng

class _FakeKokoro:
    def synthesize(self, text, voice, speed):
        return TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)
    def stream_segments(self, segments, voice, speed):
        for _ in segments:
            yield self.synthesize("", voice, speed)

def _engine():
    eng = _Eng.__new__(_Eng)          # bypass real Kokoro asset load
    from openvox.tts.config import TTSConfig
    eng._config = TTSConfig(device="cpu")
    eng._backend = _FakeKokoro()
    return eng

def test_stream_yields_one_chunk_per_segment():
    eng = _engine()
    chunks = list(eng.stream("Hello there. How are you?", voice="af_heart"))
    assert len(chunks) == 2 and all(c.sample_rate == 24000 for c in chunks)

def test_stream_rejects_empty_text():
    with pytest.raises(ValueError):
        list(_engine().stream("   "))

def test_say_stream_returns_handle_that_completes(monkeypatch):
    # inject a fake player so no audio device is needed
    import openvox.tts.engine as eng_mod
    from openvox.tts.stream import SpeechHandle
    class FakePlayer:
        def start(self): pass
        def put(self, a, sr): pass
        def finish(self): pass
        def abort(self): pass
        def wait_drain(self, timeout=None): pass
        def close(self): pass
    monkeypatch.setattr(eng_mod, "_StreamPlayer", lambda sr, qs: FakePlayer())
    h = _engine().say_stream("Hello there.", voice="af_heart")
    h.wait()
    assert h.done is True
