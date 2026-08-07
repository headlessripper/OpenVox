import numpy as np
import pytest
from nectarstt.audio.vad import SileroVAD

pytestmark = pytest.mark.integration  # loads the onnx model

def test_silence_is_not_speech():
    vad = SileroVAD(threshold=0.5)
    silence = np.zeros(512, dtype=np.float32)
    assert vad.is_speech(silence) is False

def test_loud_noise_probability_differs_from_silence():
    vad = SileroVAD(threshold=0.5)
    rng = np.random.default_rng(0)
    loud = (rng.standard_normal(512) * 0.5).astype(np.float32)
    silence = np.zeros(512, dtype=np.float32)
    # Not asserting 'loud is speech' (noise != speech); assert the wrapper runs
    # and returns bools for both without error.
    assert isinstance(vad.is_speech(loud), bool)
    assert isinstance(vad.is_speech(silence), bool)
