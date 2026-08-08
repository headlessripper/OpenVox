import numpy as np
import pytest
from openvox.enhance.resemble_backend import _resolve_device, ResembleEnhanceBackend
from openvox.enhance.config import EnhanceConfig

def test_resolve_cuda_available():
    assert _resolve_device("cuda", True) == "cuda"

def test_resolve_cuda_unavailable():
    assert _resolve_device("cuda", False) == "cpu"

def test_resolve_cpu():
    assert _resolve_device("cpu", True) == "cpu"

@pytest.mark.integration
def test_enhances_and_upsamples():
    # 1 second of 16 kHz noise-ish speech proxy from the STT fixture
    import wave
    with wave.open("tests/stt/fixtures/hello_world.wav") as w:
        sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    backend = ResembleEnhanceBackend(EnhanceConfig(device="cpu", nfe=32))
    result = backend.enhance(audio, sr)
    assert result.audio.dtype == np.float32
    assert result.sample_rate == 44100                  # bandwidth-extended
    assert len(result.audio) > len(audio)               # upsampled 16k -> 44.1k
    rms = float(np.sqrt(np.mean(result.audio ** 2)))
    assert rms > 0.005
