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

class PatternVAD:
    """Returns is_speech per a fixed list of booleans (one per call), then silence forever."""
    def __init__(self, pattern):
        self._pattern = list(pattern)
        self._i = 0
    def is_speech(self, frame):
        if self._i < len(self._pattern):
            v = self._pattern[self._i]
            self._i += 1
            return v
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

def test_two_utterances_reset_cleanly():
    """Two separate speech->silence runs must yield two independent finals,
    proving buffer, VAD, and LocalAgreement are reset between utterances
    (a stale LocalAgreement would corrupt the second final's committed prefix)."""
    cfg = Config(window_interval_ms=64, min_silence_ms=64, min_speech_ms=32,
                 sample_rate=16000, device="cpu", compute_type="int8")
    frame = np.zeros(512, dtype=np.float32)          # 32ms frames
    pattern = [True] * 4 + [False] * 4 + [True] * 4 + [False] * 4
    frames = [frame] * len(pattern)
    t = StreamingTranscriber(ScriptedBackend(), PatternVAD(pattern), cfg)
    events = list(t.run(ListSource(frames)))

    final_indices = [i for i, e in enumerate(events) if isinstance(e, FinalResult)]
    assert len(final_indices) == 2
    for i in final_indices:
        f = events[i]
        assert f.text == "the quick brown fox"
        assert len(f.words) == 4

    # Every partial for the first utterance precedes its final, and every
    # partial for the second utterance precedes the second final.
    idx1, idx2 = final_indices
    partials_before_first = [e for e in events[:idx1] if isinstance(e, PartialResult)]
    partials_between = [e for e in events[idx1 + 1:idx2] if isinstance(e, PartialResult)]
    assert len(partials_before_first) >= 1
    assert len(partials_between) >= 1

def test_finalizes_trailing_buffer_at_eof():
    """A stream that ends mid-speech (no trailing silence) must still emit
    exactly one FinalResult for the buffered speech at end-of-stream."""
    cfg = Config(window_interval_ms=64, min_silence_ms=64, min_speech_ms=32,
                 sample_rate=16000, device="cpu", compute_type="int8")
    frame = np.zeros(512, dtype=np.float32)          # 32ms frames
    frames = [frame] * 4                              # all speech, stream ends here
    t = StreamingTranscriber(ScriptedBackend(), ScriptVAD(speech_frames=4), cfg)
    events = list(t.run(ListSource(frames)))
    finals = [e for e in events if isinstance(e, FinalResult)]
    assert len(finals) == 1
    assert finals[0].text == "the quick brown fox"
    assert len(finals[0].words) == 4

def test_subthreshold_blip_is_discarded_not_merged():
    """A speech blip shorter than min_speech_ms must be discarded silently
    (no FinalResult) so a later real utterance is not concatenated onto it."""
    cfg = Config(window_interval_ms=64, min_silence_ms=64, min_speech_ms=50,
                 sample_rate=16000, device="cpu", compute_type="int8")
    frame = np.zeros(512, dtype=np.float32)          # 32ms frames
    # 1 speech frame (32ms < 50ms min_speech_ms) -> sub-threshold blip
    # 2 silence frames (64ms) -> silence threshold reached, blip discarded
    # 2 speech frames (64ms >= 50ms) -> real utterance
    # 2 silence frames (64ms) -> real utterance finalizes
    pattern = [True] + [False] * 2 + [True] * 2 + [False] * 2
    frames = [frame] * len(pattern)
    t = StreamingTranscriber(ScriptedBackend(), PatternVAD(pattern), cfg)
    events = list(t.run(ListSource(frames)))
    finals = [e for e in events if isinstance(e, FinalResult)]
    assert len(finals) == 1                           # blip did not merge into the real utterance
    assert finals[0].text == "the quick"
    assert len(finals[0].words) == 2
