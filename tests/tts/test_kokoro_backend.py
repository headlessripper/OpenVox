import numpy as np
import pytest
from openvox.tts.kokoro_backend import KokoroBackend
from openvox.tts.models import KOKORO_VOICES

pytestmark = pytest.mark.integration

def test_synthesizes_audio():
    backend = KokoroBackend(device="cpu")
    result = backend.synthesize("Hello world, this is OpenVox.", "af_heart", 1.0)
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 24000
    assert len(result.audio) > 24000                 # > ~1s of audio
    rms = float(np.sqrt(np.mean(result.audio ** 2)))
    assert rms > 0.01                                # not silence

def test_get_voices_superset_of_registry():
    backend = KokoroBackend(device="cpu")
    got = set(backend.get_voices())
    assert KOKORO_VOICES.issubset(got)               # drift guard
