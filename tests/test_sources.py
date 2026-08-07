import wave
import numpy as np
from nectarstt.audio.sources import FileSource

def _write_wav(path, seconds, sr=16000):
    n = int(seconds * sr)
    data = (np.sin(np.linspace(0, 100, n)) * 10000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data.tobytes())

def test_filesource_frame_shape_and_count(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, seconds=1.0)                      # 16000 samples
    src = FileSource(str(p), frame_ms=32, sample_rate=16000)
    frames = list(src.frames())
    assert all(f.dtype == np.float32 for f in frames)
    assert all(f.shape[0] == 512 for f in frames)   # 32ms * 16000 = 512
    assert len(frames) == 32                         # 16000/512 = 31.25 -> 32 (padded)

def test_filesource_values_normalized(tmp_path):
    p = tmp_path / "b.wav"
    _write_wav(p, seconds=0.1)
    src = FileSource(str(p))
    f = next(src.frames())
    assert f.max() <= 1.0 and f.min() >= -1.0
