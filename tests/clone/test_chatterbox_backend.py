import numpy as np
import pytest
from openvox.clone.chatterbox_backend import _resolve_device, ChatterboxBackend

def test_resolve_cuda_available():
    assert _resolve_device("cuda", True) == "cuda"

def test_resolve_cuda_unavailable():
    assert _resolve_device("cuda", False) == "cpu"

def test_resolve_cpu():
    assert _resolve_device("cpu", True) == "cpu"

@pytest.mark.integration
def test_clones_audio():
    backend = ChatterboxBackend(device="cpu")
    result = backend.clone(
        "Hello world, this is a cloned voice.",
        "tests/stt/fixtures/hello_world.wav", 0.5, 0.5)
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 24000
    assert len(result.audio) > 12000                 # > ~0.5s
    rms = float(np.sqrt(np.mean(result.audio ** 2)))
    assert rms > 0.01                                # not silence
