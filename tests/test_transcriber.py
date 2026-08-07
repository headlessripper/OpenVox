import numpy as np
from nectarstt.engine.transcriber import StreamingTranscriber
from nectarstt.engine.backend import StreamingBackend, BackendResult
from nectarstt.events import WordTiming, PartialResult, FinalResult
from nectarstt.config import Config

class ListSource:
    def __init__(self, frames): self._frames = frames
    def frames(self): return iter(self._frames)
    def close(self): pass

class ScriptedBackend(StreamingBackend):
    """Returns a growing transcript based on how many speech frames are in the buffer."""
    def transcribe(self, audio, sample_rate, language, word_timestamps):
        n = int(round(len(audio) / 512))  # 1 token per ~frame of speech
        tokens = ["the", "quick", "brown", "fox"][:max(1, min(n, 4))]
        words = [WordTiming(t, i * 0.1, i * 0.1 + 0.1, 1.0) for i, t in enumerate(tokens)]
        return BackendResult(text=" ".join(tokens), words=words if word_timestamps else [])

class ScriptVAD:
    """Speech for the first `speech_frames` frames, then silence."""
    def __init__(self, speech_frames): self._left = speech_frames
    def is_speech(self, frame):
        if self._left > 0:
            self._left -= 1
            return True
        return False
    def reset(self): pass

def test_emits_partials_then_final():
    cfg = Config(window_interval_ms=64, min_silence_ms=64, min_speech_ms=32,
                 sample_rate=16000, device="cpu", compute_type="int8")
    frame = np.zeros(512, dtype=np.float32)          # 32ms frames
    frames = [frame] * 4 + [frame] * 4               # 4 speech, 4 silence
    t = StreamingTranscriber(ScriptedBackend(), ScriptVAD(speech_frames=4), cfg)
    events = list(t.run(ListSource(frames)))
    partials = [e for e in events if isinstance(e, PartialResult)]
    finals = [e for e in events if isinstance(e, FinalResult)]
    assert len(partials) >= 1
    assert len(finals) == 1
    assert finals[0].text == "the quick brown fox"
    assert len(finals[0].words) == 4                 # word timestamps on final only
