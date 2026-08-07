import wave

import numpy as np
import pytest

import nectarstt
from nectarstt import STTEngine, FileSource, FinalResult, PartialResult
from nectarstt.engine.backend import StreamingBackend, BackendResult


@pytest.mark.integration  # uses the real tiny backend + VAD
def test_stream_over_file_yields_final():
    engine = STTEngine(model="tiny", device="cpu", language="en")
    events = list(engine.stream(source=FileSource("tests/fixtures/hello_world.wav")))
    finals = [e for e in events if isinstance(e, FinalResult)]
    assert len(finals) >= 1
    joined = " ".join(f.text.lower() for f in finals)
    assert "hello" in joined and "world" in joined

@pytest.mark.integration  # uses the real tiny backend + VAD
def test_transcribe_file_batch():
    engine = STTEngine(model="tiny", device="cpu", language="en")
    result = engine.transcribe_file("tests/fixtures/hello_world.wav")
    assert isinstance(result, FinalResult)
    assert "hello" in result.text.lower()


class _UnreachableBackend(StreamingBackend):
    """Stub backend used so STTEngine can be constructed without loading a
    real model. transcribe() must never be called for a malformed WAV, since
    FileSource validation should raise before the backend is touched."""
    def transcribe(self, audio, sample_rate, language, word_timestamps):
        raise AssertionError(
            "backend.transcribe() should not be reached for a malformed WAV")


def test_transcribe_file_rejects_malformed_wav(tmp_path, monkeypatch):
    """transcribe_file must route through FileSource for validation, so a
    stereo (or otherwise malformed) WAV raises ValueError instead of being
    silently read as garbage via a raw np.frombuffer reinterpretation.

    The real FasterWhisperBackend is stubbed out via monkeypatch so this test
    stays fast and deterministic (no model download/load) and can run as a
    normal, non-integration test.
    """
    monkeypatch.setattr(
        nectarstt, "FasterWhisperBackend", lambda **kwargs: _UnreachableBackend())

    p = tmp_path / "stereo.wav"
    n = int(0.1 * 16000)
    data = (np.sin(np.linspace(0, 100, n)) * 10000).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)  # stereo -> invalid
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(data.tobytes())

    engine = STTEngine(model="tiny", device="cpu", language="en")
    with pytest.raises(ValueError, match="mono"):
        engine.transcribe_file(str(p))
