import wave
import numpy as np
import pytest
from nectarstt.audio.sources import FileSource, MicSource

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

def test_filesource_rejects_stereo(tmp_path):
    """FileSource should reject stereo WAV files."""
    p = tmp_path / "stereo.wav"
    n = int(0.1 * 16000)
    data = (np.sin(np.linspace(0, 100, n)) * 10000).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)  # stereo
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(data.tobytes())

    src = FileSource(str(p))
    with pytest.raises(ValueError, match="mono"):
        next(src.frames())

def test_filesource_rejects_wrong_sample_rate(tmp_path):
    """FileSource should reject WAV files with wrong sample rate."""
    p = tmp_path / "wrong_sr.wav"
    n = int(0.1 * 44100)
    data = (np.sin(np.linspace(0, 100, n)) * 10000).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)  # wrong sample rate
        w.writeframes(data.tobytes())

    src = FileSource(str(p), sample_rate=16000)
    with pytest.raises(ValueError, match="16000 Hz"):
        next(src.frames())

def test_filesource_rejects_wrong_bit_depth(tmp_path):
    """FileSource should reject WAV files with wrong bit depth."""
    p = tmp_path / "wrong_depth.wav"
    n = int(0.1 * 16000)
    data = (np.sin(np.linspace(0, 100, n)) * 127).astype(np.int8)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)  # 8-bit instead of 16-bit
        w.setframerate(16000)
        w.writeframes(data.tobytes())

    src = FileSource(str(p))
    with pytest.raises(ValueError, match="16-bit"):
        next(src.frames())

def test_microsource_close_sets_stop_flag():
    """MicSource.close() should set the stop flag."""
    src = MicSource()
    assert not src._stop.is_set()
    src.close()
    assert src._stop.is_set()
