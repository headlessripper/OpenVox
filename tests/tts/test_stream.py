import threading
import numpy as np
import pytest
from openvox.tts.stream import SpeechHandle
from openvox.tts.backend import TTSResult

class FakePlayer:
    def __init__(self):
        self.puts, self.finished, self.aborted, self.drained = [], False, False, False
    def start(self): pass
    def put(self, audio, sample_rate): self.puts.append((len(audio), sample_rate))
    def finish(self): self.finished = True
    def abort(self): self.aborted = True
    def wait_drain(self, timeout=None): self.drained = True
    def close(self): pass

class ListBackend:
    def __init__(self, n): self._n = n
    def stream_segments(self, segments, voice, speed):
        for _ in segments:
            yield TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)

def test_full_playthrough_puts_all_chunks_and_finishes():
    p = FakePlayer()
    h = SpeechHandle(p)
    h.start(ListBackend(3), "af_heart", 1.0, ["a", "b", "c"])
    h.wait()
    assert len(p.puts) == 3 and p.finished is True and h.done is True

def test_stop_is_idempotent_and_sets_done():
    p = FakePlayer()
    h = SpeechHandle(p)
    h.stop(); h.stop()
    assert p.aborted is True and h.done is True

def test_stop_mid_stream_discards_inflight_and_returns_promptly():
    release = threading.Event()
    started = threading.Event()
    class BlockingBackend:
        def stream_segments(self, segments, voice, speed):
            yield TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)  # 1st
            started.set()
            release.wait(5)                                                          # 2nd blocks
            yield TTSResult(audio=np.ones(4, dtype=np.float32), sample_rate=24000)
    p = FakePlayer()
    h = SpeechHandle(p)
    h.start(BlockingBackend(), "af_heart", 1.0, ["a", "b"])
    assert started.wait(5)
    h.stop()                       # returns promptly while 2nd segment is blocked
    assert h.done is True and p.aborted is True
    release.set()
    h.wait()
    assert len(p.puts) == 1        # in-flight/next chunk never enqueued

def test_producer_exception_surfaces_from_wait():
    class BoomBackend:
        def stream_segments(self, segments, voice, speed):
            raise RuntimeError("boom")
            yield  # pragma: no cover
    h = SpeechHandle(FakePlayer())
    h.start(BoomBackend(), "af_heart", 1.0, ["a"])
    with pytest.raises(RuntimeError, match="boom"):
        h.wait()
