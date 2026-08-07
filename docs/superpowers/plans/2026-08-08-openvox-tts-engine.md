# OpenVox TTS Engine Implementation Plan (Sub-project 2A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline text-to-speech engine at `openvox/tts/` that synthesizes genuinely human-sounding speech via Kokoro (torch-free ONNX), with built-in voices, WAV save, and playback.

**Architecture:** A `TTSEngine` façade parallel to `STTEngine`. Text → `KokoroBackend` (kokoro-onnx: phonemize → ONNX inference with a voice embedding → 24 kHz float32 waveform) → `TTSResult`, which can `save_wav()` or be played via `sounddevice`. The backend sits behind a `TTSBackend` ABC. Model + voices assets download once to the shared `openvox` cache.

**Tech Stack:** Python 3.11+, kokoro-onnx 0.5.x (onnxruntime + phonemizer-fork + espeakng-loader, all pip wheels — no system install), sounddevice, numpy, pytest.

## Global Constraints

- Python `>=3.11`. No PyTorch (kokoro-onnx is ONNX-based).
- No hardcoded filesystem paths — assets resolve via `openvox._paths.cache_dir("tts/models")`.
- Only `openvox/tts/kokoro_backend.py` may import `kokoro_onnx`; consumers go through the `TTSBackend` ABC.
- Audio contract: 24000 Hz, mono, float32 in [-1, 1] (Kokoro's native output).
- English-first: synthesis uses `lang="en-us"`.
- `import openvox` stays lean; TTS is reached via `openvox.tts` and needs no `openvox.stt` import.
- Default voice `af_heart`; default device `cuda` with CPU fallback. Tests run on CPU.
- TDD: failing test first, watch it fail, implement, watch it pass, commit. Playback (`play`/`say`) is validated manually, never in CI (needs speakers).
- **Deviation from spec §11 (intentional):** the Kokoro v1.0 assets are GitHub-release files, so they download via stdlib `urllib` (no `huggingface_hub`); the `[tts]` extra is `kokoro-onnx` + `sounddevice` only.

## File Structure

- `openvox/tts/__init__.py` — exports `TTSEngine`, `TTSResult` (filled in Task 5).
- `openvox/tts/config.py` — `TTSConfig` dataclass.
- `openvox/tts/backend.py` — `TTSResult`, `TTSBackend` ABC, `AudioDeviceError`.
- `openvox/tts/models.py` — asset URLs + `ensure_assets()`, `KOKORO_VOICES`, `voices()`, `validate_voice()`.
- `openvox/tts/kokoro_backend.py` — `KokoroBackend`.
- `openvox/tts/engine.py` — `TTSEngine`.
- `openvox/tts/demo.py` — `python -m openvox.tts.demo`.
- `tests/tts/` — mirrors the package.

---

## Task 1: Package scaffold + TTSConfig + TTSResult + backend ABC

**Files:**
- Create: `openvox/tts/__init__.py`, `openvox/tts/config.py`, `openvox/tts/backend.py`, `tests/tts/__init__.py`
- Test: `tests/tts/test_config.py`, `tests/tts/test_backend.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `TTSConfig(device="cuda", voice="af_heart", speed=1.0, sample_rate=24000)` dataclass.
  - `TTSResult(audio: np.ndarray, sample_rate: int)` with `duration: float` property and `save_wav(path: str) -> None` (24 kHz mono 16-bit WAV).
  - `TTSBackend` ABC with abstract `synthesize(text: str, voice: str, speed: float) -> TTSResult`.
  - `AudioDeviceError(RuntimeError)`.

- [ ] **Step 1: Add the `[tts]` extra + entry point to `pyproject.toml`**

In `[project.optional-dependencies]` add `tts` and extend `all`; add the demo script. (`kokoro-onnx` pulls CPU `onnxruntime`; for CUDA the user additionally installs `kokoro-onnx[gpu]` / `onnxruntime-gpu` — not required for tests, which run on CPU.)
```toml
tts = ["kokoro-onnx>=0.5.0", "sounddevice>=0.4.6"]
all = ["openvox[stt,stt-demo,tts]"]
```
```toml
[project.scripts]
openvox-stt-demo = "openvox.stt.demo:main"
openvox-tts-demo = "openvox.tts.demo:main"
```

- [ ] **Step 2: Create package init files**

`openvox/tts/__init__.py`:
```python
"""OpenVox TTS — offline text-to-speech."""
```
Create empty `tests/tts/__init__.py`.

- [ ] **Step 3: Write the failing tests**

`tests/tts/test_config.py`:
```python
from openvox.tts.config import TTSConfig

def test_defaults():
    c = TTSConfig()
    assert c.device == "cuda"
    assert c.voice == "af_heart"
    assert c.speed == 1.0
    assert c.sample_rate == 24000
```

`tests/tts/test_backend.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/tts/ -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5: Implement `openvox/tts/config.py`**

```python
from dataclasses import dataclass

@dataclass
class TTSConfig:
    device: str = "cuda"
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000
```

- [ ] **Step 6: Implement `openvox/tts/backend.py`**

```python
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

class AudioDeviceError(RuntimeError):
    pass

@dataclass
class TTSResult:
    audio: np.ndarray          # mono float32 in [-1, 1]
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    def save_wav(self, path: str) -> None:
        # Scale-then-clip so a +1.0 sample maps to 32767, not a wrapped -32768.
        pcm = np.clip(np.asarray(self.audio, dtype=np.float32) * 32768.0,
                      -32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())

class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float) -> TTSResult:
        """Synthesize speech for text with the given voice and speed."""
        raise NotImplementedError
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pip install -e ".[dev]"` then `pytest tests/tts/ -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml openvox/tts/__init__.py openvox/tts/config.py openvox/tts/backend.py tests/tts/
git commit -m "feat(tts): scaffold openvox.tts with TTSConfig, TTSResult, backend ABC"
```

---

## Task 2: Model & voice management

**Files:**
- Create: `openvox/tts/models.py`
- Test: `tests/tts/test_models.py`

**Interfaces:**
- Consumes: `openvox._paths.cache_dir` (already exists).
- Produces:
  - `KOKORO_VOICES: frozenset[str]` (the built-in English voices).
  - `voices() -> list[str]` (sorted).
  - `validate_voice(name: str) -> None` (raises `ValueError` for unknown).
  - `ensure_assets() -> tuple[str, str]` returning `(model_path, voices_path)`, downloading each to `cache_dir("tts/models")` on first use.

- [ ] **Step 1: Write the failing tests**

`tests/tts/test_models.py` (all unit — no network; the real download is exercised by Task 3's backend integration test):
```python
import os
import pytest
from openvox.tts import models

def test_voices_nonempty_and_known():
    vs = models.voices()
    assert isinstance(vs, list) and len(vs) >= 20
    assert "af_heart" in vs
    assert "am_michael" in vs
    assert vs == sorted(vs)

def test_validate_voice_ok():
    models.validate_voice("af_heart")   # no raise

def test_validate_voice_unknown_raises():
    with pytest.raises(ValueError):
        models.validate_voice("nonexistent_voice")

def test_ensure_assets_builds_paths_and_calls_download(tmp_path, monkeypatch):
    import openvox._paths as paths
    monkeypatch.setattr(paths, "user_cache_dir", lambda app: str(tmp_path))
    calls = []
    monkeypatch.setattr(models, "_download_if_missing",
                        lambda url, dest: calls.append((url, dest)))
    model_path, voices_path = models.ensure_assets()
    assert model_path.endswith("kokoro-v1.0.onnx")
    assert voices_path.endswith("voices-v1.0.bin")
    assert os.path.join("tts", "models") in os.path.normpath(model_path)
    assert len(calls) == 2                          # both assets requested

def test_download_if_missing_skips_existing(tmp_path):
    dest = tmp_path / "asset.bin"
    dest.write_bytes(b"already here")
    models._download_if_missing("http://invalid.invalid/x", str(dest))  # no network
    assert dest.read_bytes() == b"already here"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tts/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `openvox/tts/models.py`**

```python
import os
import urllib.request

from openvox._paths import cache_dir

_MODEL_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
              "model-files-v1.0/kokoro-v1.0.onnx")
_VOICES_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
               "model-files-v1.0/voices-v1.0.bin")

# Built-in English voices (American af_/am_, British bf_/bm_) shipped with
# Kokoro v1.0. Integration tests cross-check this against the model's own list.
KOKORO_VOICES: frozenset[str] = frozenset({
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
})

def voices() -> list[str]:
    return sorted(KOKORO_VOICES)

def validate_voice(name: str) -> None:
    if name not in KOKORO_VOICES:
        raise ValueError(
            f"Unknown voice '{name}'. Available voices: {voices()}"
        )

def ensure_assets() -> tuple[str, str]:
    root = cache_dir("tts/models")
    model_path = os.path.join(root, "kokoro-v1.0.onnx")
    voices_path = os.path.join(root, "voices-v1.0.bin")
    _download_if_missing(_MODEL_URL, model_path)
    _download_if_missing(_VOICES_URL, voices_path)
    return model_path, voices_path

def _download_if_missing(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Failed to download TTS asset from {url} to {dest}: {exc}. "
            "Check your connection, or pre-place the file at that path."
        ) from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/tts/test_models.py -v`
Expected: PASS (5 unit tests, no network). The real asset download is exercised by Task 3's backend integration test.

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/models.py tests/tts/test_models.py
git commit -m "feat(tts): add Kokoro asset download + voice registry"
```

---

## Task 3: Kokoro backend

**Files:**
- Create: `openvox/tts/kokoro_backend.py`
- Test: `tests/tts/test_kokoro_backend.py`

**Interfaces:**
- Consumes: `TTSBackend`/`TTSResult` (Task 1); `ensure_assets`/`KOKORO_VOICES` (Task 2).
- Produces: `KokoroBackend(device: str = "cuda")` implementing `synthesize(...)`, plus `get_voices() -> list[str]`. On CUDA provider failure it falls back to the CPU provider with a warning.

- [ ] **Step 1: Write the failing integration test**

`tests/tts/test_kokoro_backend.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tts/test_kokoro_backend.py -v -m integration`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `openvox/tts/kokoro_backend.py`**

```python
import logging
import os

import numpy as np

from openvox.tts.backend import TTSBackend, TTSResult
from openvox.tts.models import ensure_assets

log = logging.getLogger(__name__)

class KokoroBackend(TTSBackend):
    def __init__(self, device: str = "cuda") -> None:
        model_path, voices_path = ensure_assets()
        self._kokoro = self._load(model_path, voices_path, device)

    def _load(self, model_path: str, voices_path: str, device: str):
        from kokoro_onnx import Kokoro
        if device == "cuda":
            os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
            try:
                return Kokoro(model_path, voices_path)
            except Exception as exc:
                log.warning("CUDA provider unavailable (%s: %s); falling back to CPU.",
                            type(exc).__name__, exc)
        os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
        return Kokoro(model_path, voices_path)

    def synthesize(self, text: str, voice: str, speed: float) -> TTSResult:
        samples, sample_rate = self._kokoro.create(
            text, voice=voice, speed=speed, lang="en-us")
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        return TTSResult(audio=audio, sample_rate=int(sample_rate))

    def get_voices(self) -> list[str]:
        return self._kokoro.get_voices()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[tts,dev]"` then `pytest tests/tts/test_kokoro_backend.py -v -m integration`
Expected: PASS (downloads the model on first run, then offline; synthesizes on CPU).

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/kokoro_backend.py tests/tts/test_kokoro_backend.py
git commit -m "feat(tts): add Kokoro ONNX backend with CPU fallback"
```

---

## Task 4: TTSEngine façade

**Files:**
- Create: `openvox/tts/engine.py`
- Test: `tests/tts/test_engine.py`

**Interfaces:**
- Consumes: `TTSConfig` (Task 1), `KokoroBackend` (Task 3), `validate_voice`/`voices` (Task 2), `TTSResult`/`AudioDeviceError` (Task 1).
- Produces: `TTSEngine(voice=None, device=None, speed=None, config=None)` with `synthesize(text, voice=None, speed=None) -> TTSResult`, `play(result) -> None`, `say(text, voice=None, speed=None) -> None`, `voices() -> list[str]`. Constructor args override config (a passed config is copied, not mutated). `synthesize` validates the voice and rejects empty text before touching the backend.

- [ ] **Step 1: Write the failing tests**

`tests/tts/test_engine.py`:
```python
import numpy as np
import pytest
from openvox.tts.engine import TTSEngine
from openvox.tts.backend import TTSResult

# Unit tests: patch the backend so no model is loaded.
class _FakeBackend:
    def __init__(self, *a, **k): pass
    def synthesize(self, text, voice, speed):
        return TTSResult(audio=np.zeros(24000, dtype=np.float32), sample_rate=24000)

@pytest.fixture
def patched(monkeypatch):
    import openvox.tts.engine as eng
    monkeypatch.setattr(eng, "KokoroBackend", _FakeBackend)

def test_voices_listed(patched):
    assert "af_heart" in TTSEngine().voices()

def test_empty_text_raises(patched):
    with pytest.raises(ValueError):
        TTSEngine().synthesize("   ")

def test_unknown_voice_raises(patched):
    with pytest.raises(ValueError):
        TTSEngine().synthesize("hello", voice="bogus_voice")

def test_synthesize_returns_result(patched):
    r = TTSEngine(voice="am_michael").synthesize("hello")
    assert r.sample_rate == 24000
    assert len(r.audio) == 24000

def test_config_not_mutated(patched):
    from openvox.tts.config import TTSConfig
    cfg = TTSConfig()
    TTSEngine(voice="am_michael", config=cfg)
    assert cfg.voice == "af_heart"     # caller's config untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tts/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `openvox/tts/engine.py`**

```python
import dataclasses

from openvox.tts.backend import TTSResult, AudioDeviceError
from openvox.tts.config import TTSConfig
from openvox.tts.kokoro_backend import KokoroBackend
from openvox.tts.models import voices as _voices, validate_voice

class TTSEngine:
    def __init__(self, voice: str | None = None, device: str | None = None,
                 speed: float | None = None, config: TTSConfig | None = None) -> None:
        cfg = dataclasses.replace(config) if config is not None else TTSConfig()
        if voice is not None:
            cfg.voice = voice
        if device is not None:
            cfg.device = device
        if speed is not None:
            cfg.speed = speed
        self._config = cfg
        self._backend = KokoroBackend(device=cfg.device)

    def voices(self) -> list[str]:
        return _voices()

    def synthesize(self, text: str, voice: str | None = None,
                   speed: float | None = None) -> TTSResult:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        v = voice if voice is not None else self._config.voice
        validate_voice(v)
        s = speed if speed is not None else self._config.speed
        return self._backend.synthesize(text, v, s)

    def play(self, result: TTSResult) -> None:
        import sounddevice as sd
        try:
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

    def say(self, text: str, voice: str | None = None,
            speed: float | None = None) -> None:
        self.play(self.synthesize(text, voice, speed))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tts/test_engine.py -v`
Expected: PASS (5 tests, no model loaded — backend is patched).

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/engine.py tests/tts/test_engine.py
git commit -m "feat(tts): add TTSEngine facade (synthesize/play/say/voices)"
```

---

## Task 5: Public exports + demo CLI

**Files:**
- Modify: `openvox/tts/__init__.py`
- Create: `openvox/tts/demo.py`
- Test: `tests/tts/test_demo.py`

**Interfaces:**
- Consumes: `TTSEngine` (Task 4), `TTSResult` (Task 1).
- Produces (exported from `openvox.tts`): `TTSEngine`, `TTSResult`. Demo `main(argv=None) -> int` with `build_parser()` (flags `--text` required, `--voice` default `af_heart`, `--device` default `cuda`, `--speed` default `1.0`, `--out`, `--no-play`).

- [ ] **Step 1: Write the failing tests**

`tests/tts/test_demo.py`:
```python
from openvox.tts import demo

def test_build_parser_defaults():
    args = demo.build_parser().parse_args(["--text", "hi"])
    assert args.text == "hi"
    assert args.voice == "af_heart"
    assert args.device == "cuda"
    assert args.speed == 1.0
    assert args.out is None
    assert args.no_play is False

def test_parser_flags():
    args = demo.build_parser().parse_args(
        ["--text", "hi", "--voice", "am_michael", "--device", "cpu",
         "--speed", "1.2", "--out", "o.wav", "--no-play"])
    assert args.voice == "am_michael" and args.device == "cpu"
    assert args.speed == 1.2 and args.out == "o.wav" and args.no_play is True

def test_exports():
    from openvox.tts import TTSEngine, TTSResult
    assert TTSEngine is not None and TTSResult is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tts/test_demo.py -v`
Expected: FAIL (`AttributeError`/`ImportError` — demo/exports missing).

- [ ] **Step 3: Update `openvox/tts/__init__.py`**

```python
"""OpenVox TTS — offline text-to-speech."""
from openvox.tts.engine import TTSEngine
from openvox.tts.backend import TTSResult

__all__ = ["TTSEngine", "TTSResult"]
```

- [ ] **Step 4: Implement `openvox/tts/demo.py`**

```python
import argparse
import sys

from openvox.tts import TTSEngine

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openvox-tts-demo",
                                description="Offline OpenVox text-to-speech.")
    p.add_argument("--text", required=True, help="Text to speak.")
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--device", default="cuda")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", default=None, help="Also save the audio to this WAV path.")
    p.add_argument("--no-play", action="store_true", help="Do not play the audio aloud.")
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = TTSEngine(voice=args.voice, device=args.device, speed=args.speed)
    result = engine.synthesize(args.text)
    if args.out:
        result.save_wav(args.out)
        print(f"Saved {args.out} ({result.duration:.1f}s)")
    if not args.no_play:
        engine.play(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/tts/test_demo.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Manual end-to-end check (on a machine with speakers)**

Run:
```bash
python -m openvox.tts.demo --text "Hello, I am OpenVox, and I run entirely offline." --device cpu --out tts_demo.wav
```
Expected: prints `Saved tts_demo.wav (…s)`, speaks the line aloud, and `tts_demo.wav` is a valid 24 kHz WAV.

- [ ] **Step 7: Run the full suite**

Run: `pytest -q -m "not integration"` then (with the `[tts]` extra installed) `pytest -q -m integration`
Expected: all pass; TTS unit tests green without loading a model.

- [ ] **Step 8: Commit**

```bash
git add openvox/tts/__init__.py openvox/tts/demo.py tests/tts/test_demo.py
git commit -m "feat(tts): add public exports and demo CLI"
```

---

## Task 6: README — mark TTS available

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the status table and TTS section**

In the "Project status & roadmap" table, change the TTS row from `🚧 In development` to `✅ Available`. Replace the "Text-to-Speech (coming next)" section with a real usage section:

````markdown
## 🗣️ Text-to-Speech (available today)

Genuinely human-sounding speech, fully offline, with built-in voices.

```bash
pip install -e ".[tts]"
```

```bash
# Speak a line aloud (and optionally save it):
python -m openvox.tts.demo --text "Hello, I am OpenVox." --out hello.wav
```

```python
from openvox.tts import TTSEngine

engine = TTSEngine(voice="af_heart", device="cuda")   # falls back to CPU
engine.say("This runs entirely offline.")             # synthesize + speak

result = engine.synthesize("Save me to a file.")
result.save_wav("out.wav")

engine.voices()   # list the built-in voices
```

**Demo flags:** `--text` (required) · `--voice` (default `af_heart`) · `--device` (`cuda`/`cpu`) · `--speed` (default `1.0`) · `--out PATH` (save a WAV) · `--no-play` (skip playback).
````

Do not document voices or flags that don't exist (voices come from `openvox/tts/models.py`'s `KOKORO_VOICES`; flags from `demo.py`).

- [ ] **Step 2: Verify accuracy**

Run: `grep -n "TTSEngine\|openvox.tts.demo\|af_heart" README.md` and confirm each documented symbol exists in the code.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: mark TTS available with usage in README"
```

---

## Self-Review Notes (completed)

- **Spec coverage:** package layout (T1-T5) · TTSConfig (T1) · TTSResult+save_wav (T1) · TTSBackend ABC (T1) · Kokoro backend + CUDA→CPU fallback (T3) · asset download to shared cache (T2) · KOKORO_VOICES/voices/validate_voice (T2) · TTSEngine synthesize/play/say/voices + config-copy + voice validation + empty-text guard (T4) · public exports + demo CLI (T5) · error handling: unknown voice/empty text ValueError (T4), AudioDeviceError on play (T4), download failure message (T2), CUDA fallback (T3) · testing incl. no-model unit tests (T1/T2/T4/T5) and integration synth (T3) · deps `[tts]` extra + entry point (T1) · DoD mapped · README (T6).
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type consistency:** `TTSResult(audio, sample_rate)` + `.duration`/`.save_wav`, `TTSBackend.synthesize(text, voice, speed)`, `TTSConfig(device, voice, speed, sample_rate)`, `KokoroBackend(device)`/`.get_voices()`, `ensure_assets()->(model,voices)`, `KOKORO_VOICES`/`voices()`/`validate_voice()`, `TTSEngine(voice,device,speed,config)` with `synthesize/play/say/voices`, demo `build_parser`/`main` — consistent across tasks and match the verified kokoro-onnx API (`Kokoro(model,voices)`, `create(text,voice,speed,lang)->(float32,24000)`, `get_voices()`, `ONNX_PROVIDER`).
- **Intentional spec deviation noted:** stdlib `urllib` download from the GitHub-release URLs instead of `huggingface_hub`; `[tts]` = kokoro-onnx + sounddevice.
