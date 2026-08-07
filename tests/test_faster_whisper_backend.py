import wave
import numpy as np
import pytest
from nectarstt.engine.faster_whisper_backend import FasterWhisperBackend

pytestmark = pytest.mark.integration

def _load(path):
    with wave.open(path) as w:
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr

def test_transcribes_hello_world():
    audio, sr = _load("tests/fixtures/hello_world.wav")
    backend = FasterWhisperBackend(model="tiny", device="cpu", compute_type="int8")
    result = backend.transcribe(audio, sr, language="en", word_timestamps=True)
    assert "hello" in result.text.lower()
    assert "world" in result.text.lower()
    assert len(result.words) >= 2
    assert result.words[0].end >= result.words[0].start
