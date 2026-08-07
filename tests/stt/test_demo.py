import wave

import numpy as np
import pytest

from openvox.stt import demo
from openvox.stt.events import PartialResult, FinalResult

def test_build_parser_defaults():
    args = demo.build_parser().parse_args([])
    assert args.model == "distil-large-v3"
    assert args.device == "cuda"
    assert args.language == "en"
    assert args.file is None
    assert args.full is False

def test_parser_accepts_file():
    args = demo.build_parser().parse_args(["--file", "x.wav", "--device", "cpu"])
    assert args.file == "x.wav"
    assert args.device == "cpu"

def test_parser_full_flag():
    assert demo.build_parser().parse_args(["--full"]).full is True

# --- render_event: partials hidden by default, shown only with --full ---

def test_render_hides_partial_by_default():
    p = PartialResult(text="hello wor", committed_prefix="hello", volatile_tail="wor")
    assert demo.render_event(p, full=False) is None

def test_render_shows_partial_in_full_mode():
    p = PartialResult(text="hello wor", committed_prefix="hello", volatile_tail="wor")
    assert demo.render_event(p, full=True) == "~ hello wor"

def test_render_always_shows_final():
    f = FinalResult(text="hello world", words=[], start=0.0, end=1.0)
    assert demo.render_event(f, full=False) == "✓ hello world"
    assert demo.render_event(f, full=True) == "✓ hello world"

# --- audio loader ---

def _write_wav(path, seconds, sr, channels=1, sampwidth=2):
    n = int(seconds * sr)
    data = (np.sin(np.linspace(0, 200, n * channels)) * 8000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels); w.setsampwidth(sampwidth); w.setframerate(sr)
        w.writeframes(data.tobytes())

def test_load_conforming_wav_fast_path(tmp_path):
    # Already 16k mono 16-bit: loads directly, no PyAV required.
    p = tmp_path / "ok.wav"
    _write_wav(p, seconds=0.5, sr=16000)
    audio = demo.load_audio_16k_mono(str(p))
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 8000) <= 1           # 0.5s * 16000
    assert audio.max() <= 1.0 and audio.min() >= -1.0

@pytest.mark.integration
def test_load_resamples_non_16k_wav(tmp_path):
    # 48kHz stereo -> resampled to 16k mono float32 via PyAV.
    p = tmp_path / "hi.wav"
    _write_wav(p, seconds=1.0, sr=48000, channels=2)
    audio = demo.load_audio_16k_mono(str(p))
    assert audio.dtype == np.float32
    assert audio.ndim == 1                        # downmixed to mono
    assert abs(len(audio) - 16000) < 400          # ~1.0s at 16k (resampler edge slack)
    assert audio.max() <= 1.0 and audio.min() >= -1.0
