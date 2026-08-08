import wave
import numpy as np
import pytest
from openvox.tts.backend import TTSResult, TTSBackend

def test_ttsresult_duration():
    r = TTSResult(audio=np.zeros(24000, dtype=np.float32), sample_rate=24000)
    assert abs(r.duration - 1.0) < 1e-6

def test_ttsresult_save_wav(tmp_path):
    audio = np.concatenate([np.ones(12000, dtype=np.float32),   # +1.0 must not wrap
                            -np.ones(12000, dtype=np.float32)])
    p = tmp_path / "out.wav"
    TTSResult(audio=audio, sample_rate=24000).save_wav(str(p))
    with wave.open(str(p)) as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 24000
        pcm = np.frombuffer(w.readframes(24000), dtype=np.int16)
    assert pcm.max() == 32767 and pcm.min() == -32768   # no overflow wrap at +1.0

def test_backend_abc_not_instantiable():
    with pytest.raises(TypeError):
        TTSBackend()

def test_concrete_backend_ok():
    class Fake(TTSBackend):
        def synthesize(self, text, voice, speed):
            return TTSResult(audio=np.zeros(10, dtype=np.float32), sample_rate=24000)
    assert Fake().synthesize("hi", "af_heart", 1.0).sample_rate == 24000
