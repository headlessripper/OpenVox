# tests/tts/test_integration_stream.py
import numpy as np
import pytest

pytestmark = pytest.mark.integration

def test_kokoro_stream_yields_multiple_nonsilent_chunks():
    from openvox.tts import TTSEngine
    eng = TTSEngine(device="cpu")   # CPU is fine; Kokoro is fast
    chunks = list(eng.stream("Hello there. How are you today?", voice="af_heart"))
    assert len(chunks) >= 2
    for c in chunks:
        assert c.sample_rate == 24000
        assert c.audio.size > 0
    assert any(float(np.sqrt(np.mean(c.audio ** 2))) > 1e-4 for c in chunks)
