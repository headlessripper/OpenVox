import numpy as np
import pytest
from openvox.enhance.backend import EnhanceBackend
from openvox.tts.backend import TTSResult

def test_backend_abc_not_instantiable():
    with pytest.raises(TypeError):
        EnhanceBackend()

def test_concrete_backend_ok():
    class Fake(EnhanceBackend):
        def enhance(self, audio, sample_rate):
            return TTSResult(audio=np.zeros(10, dtype=np.float32), sample_rate=44100)
    r = Fake().enhance(np.zeros(4, dtype=np.float32), 16000)
    assert r.sample_rate == 44100
