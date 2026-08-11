import numpy as np
from openvox.tts.backend import TTSBackend, TTSResult

class FakeBackend(TTSBackend):
    def __init__(self):
        self.calls = []
    def synthesize(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        return TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)

def test_stream_segments_yields_one_result_per_segment_in_order():
    be = FakeBackend()
    out = list(be.stream_segments(["a", "b", "c"], voice="af_heart", speed=1.0))
    assert [r.sample_rate for r in out] == [24000, 24000, 24000]
    assert be.calls == [("a", "af_heart", 1.0), ("b", "af_heart", 1.0), ("c", "af_heart", 1.0)]
