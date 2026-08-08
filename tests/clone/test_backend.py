import numpy as np
import pytest
from openvox.clone.backend import CloneBackend
from openvox.tts.backend import TTSResult

def test_backend_abc_not_instantiable():
    with pytest.raises(TypeError):
        CloneBackend()

def test_concrete_backend_ok():
    class Fake(CloneBackend):
        def clone(self, text, reference_path, exaggeration, cfg):
            return TTSResult(audio=np.zeros(10, dtype=np.float32), sample_rate=24000)
    r = Fake().clone("hi", "ref.wav", 0.5, 0.5)
    assert r.sample_rate == 24000
