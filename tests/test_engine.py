import pytest
from nectarstt import STTEngine, FileSource, FinalResult, PartialResult

pytestmark = pytest.mark.integration  # uses the real tiny backend + VAD

def test_stream_over_file_yields_final():
    engine = STTEngine(model="tiny", device="cpu", language="en")
    engine._config.compute_type = "int8"  # ensure CPU-friendly
    events = list(engine.stream(source=FileSource("tests/fixtures/hello_world.wav")))
    finals = [e for e in events if isinstance(e, FinalResult)]
    assert len(finals) >= 1
    joined = " ".join(f.text.lower() for f in finals)
    assert "hello" in joined and "world" in joined

def test_transcribe_file_batch():
    engine = STTEngine(model="tiny", device="cpu", language="en")
    result = engine.transcribe_file("tests/fixtures/hello_world.wav")
    assert isinstance(result, FinalResult)
    assert "hello" in result.text.lower()
