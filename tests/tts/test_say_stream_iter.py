import numpy as np
import pytest
from openvox.tts import TTSResult
from openvox.tts.engine import TTSEngine
from openvox.tts.config import TTSConfig

class _FakeKokoro:
    def __init__(self): self.segs = []
    def synthesize(self, text, voice, speed):
        self.segs.append(text)
        return TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)
    def stream_segments(self, segments, voice, speed):
        for seg in segments:
            yield self.synthesize(seg, voice, speed)

def _engine():
    eng = TTSEngine.__new__(TTSEngine)
    eng._config = TTSConfig(device="cpu")
    eng._backend = _FakeKokoro()
    return eng

def test_say_stream_accepts_chunk_iterator(monkeypatch):
    import openvox.tts.engine as em
    captured = {}
    class FakePlayer:
        def start(self): pass
        def put(self, a, sr): pass
        def finish(self): pass
        def abort(self): pass
        def wait_drain(self, timeout=None): pass
        def close(self): pass
    monkeypatch.setattr(em, "_StreamPlayer", lambda sr, qs: FakePlayer())
    eng = _engine()
    chunks = iter(["Hello there. ", "How are you? ", "Bye"])
    h = eng.say_stream(chunks)      # an iterator, not a string
    h.wait()
    # the fake backend saw sentence-segmented text, in order
    assert eng._backend.segs == ["Hello there.", "How are you?", "Bye"]

def test_say_stream_string_unchanged(monkeypatch):
    import openvox.tts.engine as em
    class FakePlayer:
        def start(self): pass
        def put(self, a, sr): pass
        def finish(self): pass
        def abort(self): pass
        def wait_drain(self, timeout=None): pass
        def close(self): pass
    monkeypatch.setattr(em, "_StreamPlayer", lambda sr, qs: FakePlayer())
    eng = _engine()
    eng.say_stream("One sentence. Two sentences.").wait()
    assert eng._backend.segs == ["One sentence.", "Two sentences."]
