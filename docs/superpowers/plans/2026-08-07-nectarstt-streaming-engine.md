# NectarSTT Streaming STT Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract an importable `nectarstt` Python package with a real-time streaming speech-to-text engine (mic → Silero VAD → faster-whisper on CUDA → live partial + final results with word timestamps).

**Architecture:** A layered package. Audio frame sources (mic or file) feed a `StreamingTranscriber` that gates frames with Silero VAD, periodically re-transcribes the growing speech buffer through a swappable `StreamingBackend` (faster-whisper day 1), and stabilizes partial output with a pure LocalAgreement algorithm. An `STTEngine` façade wires it together and exposes `stream()` and `transcribe_file()`.

**Tech Stack:** Python 3.11+, faster-whisper (CTranslate2), pysilero-vad (ONNX, torch-free), sounddevice, numpy, huggingface_hub, platformdirs, pytest.

## Global Constraints

- Python `>=3.11` (uses stdlib `tomllib`).
- No PyTorch dependency anywhere — VAD uses `pysilero-vad` (onnxruntime + bundled model).
- No hardcoded filesystem paths. All model/cache paths resolve via `platformdirs` or config.
- Backend must sit behind the `StreamingBackend` ABC — no consumer imports `faster_whisper` directly except `faster_whisper_backend.py`.
- Audio contract everywhere: 16000 Hz, mono, float32 in range [-1.0, 1.0], numpy `float32` arrays.
- Default model: `distil-large-v3`; default device: `cuda`; default `compute_type`: `float16`. Tests use `tiny` on CPU (`int8`) for speed and determinism.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- Every task ends with a passing test suite and a commit.

---

## File Structure

- `pyproject.toml` — package metadata, deps, pytest config.
- `nectarstt/__init__.py` — `STTEngine` façade + public exports.
- `nectarstt/config.py` — `Config` dataclass + resolution (defaults → TOML → env).
- `nectarstt/events.py` — `WordTiming`, `PartialResult`, `FinalResult`.
- `nectarstt/models.py` — model-name registry + download-root resolution.
- `nectarstt/audio/sources.py` — `FrameSource` ABC, `MicSource`, `FileSource`.
- `nectarstt/audio/vad.py` — `SileroVAD` wrapper.
- `nectarstt/engine/backend.py` — `StreamingBackend` ABC + `BackendResult`.
- `nectarstt/engine/faster_whisper_backend.py` — `FasterWhisperBackend`.
- `nectarstt/engine/local_agreement.py` — `LocalAgreement`.
- `nectarstt/engine/transcriber.py` — `StreamingTranscriber`.
- `nectarstt/demo.py` — live CLI (`python -m nectarstt.demo`).
- `tests/` — mirrors package; includes `tests/fixtures/` with a short WAV.

---

## Task 1: Project scaffolding & repo cleanup

**Files:**
- Create: `pyproject.toml`
- Create: `nectarstt/__init__.py`, `nectarstt/audio/__init__.py`, `nectarstt/engine/__init__.py`
- Create: `tests/__init__.py`, `tests/test_smoke.py`
- Delete: `Main.py`, `Main-Engine/Source/CodeScripts/`, `Production/`, `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `nectarstt` package (`pip install -e .`) and a working `pytest`.

- [ ] **Step 1: Remove old code from the working tree (preserved in git history)**

```bash
git rm -r Main.py "Main-Engine/Source/CodeScripts" Production requirements.txt
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "nectarstt"
version = "0.1.0"
description = "ElevenLabs-grade, fully offline streaming speech-to-text."
requires-python = ">=3.11"
dependencies = [
    "faster-whisper>=1.0.0",
    "pysilero-vad>=2.0.0",
    "sounddevice>=0.4.6",
    "numpy>=1.24",
    "huggingface_hub>=0.23",
    "platformdirs>=4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
nectarstt-demo = "nectarstt.demo:main"

[tool.setuptools.packages.find]
include = ["nectarstt*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: needs model download / real audio backend"]
```

- [ ] **Step 3: Create empty package init files**

`nectarstt/__init__.py`:
```python
"""NectarSTT — offline streaming speech-to-text."""
__version__ = "0.1.0"
```
Create empty `nectarstt/audio/__init__.py`, `nectarstt/engine/__init__.py`, `tests/__init__.py`.

- [ ] **Step 4: Write the smoke test**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import nectarstt
    assert nectarstt.__version__ == "0.1.0"
```

- [ ] **Step 5: Install and run the smoke test**

Run:
```bash
pip install -e ".[dev]"
pytest tests/test_smoke.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold nectarstt package, remove old GUI/scripts"
```

---

## Task 2: Event dataclasses

**Files:**
- Create: `nectarstt/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces:
  - `WordTiming(word: str, start: float, end: float, probability: float)`
  - `PartialResult(text: str, committed_prefix: str, volatile_tail: str)` with property `is_partial -> bool` (True)
  - `FinalResult(text: str, words: list[WordTiming], start: float, end: float)` with property `is_partial -> bool` (False)

- [ ] **Step 1: Write the failing test**

`tests/test_events.py`:
```python
from nectarstt.events import WordTiming, PartialResult, FinalResult

def test_partial_is_partial():
    p = PartialResult(text="hello wor", committed_prefix="hello", volatile_tail="wor")
    assert p.is_partial is True
    assert p.text == "hello wor"

def test_final_is_not_partial():
    w = [WordTiming(word="hello", start=0.0, end=0.4, probability=0.9)]
    f = FinalResult(text="hello", words=w, start=0.0, end=0.4)
    assert f.is_partial is False
    assert f.words[0].word == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: nectarstt.events`.

- [ ] **Step 3: Implement `nectarstt/events.py`**

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float
    probability: float

@dataclass(frozen=True)
class PartialResult:
    text: str
    committed_prefix: str
    volatile_tail: str

    @property
    def is_partial(self) -> bool:
        return True

@dataclass(frozen=True)
class FinalResult:
    text: str
    words: list[WordTiming] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    @property
    def is_partial(self) -> bool:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_events.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nectarstt/events.py tests/test_events.py
git commit -m "feat: add event dataclasses (PartialResult, FinalResult, WordTiming)"
```

---

## Task 3: LocalAgreement stabilization (pure)

**Files:**
- Create: `nectarstt/engine/local_agreement.py`
- Test: `tests/test_local_agreement.py`

**Interfaces:**
- Produces: `LocalAgreement` with:
  - `update(tokens: list[str]) -> tuple[list[str], list[str]]` returning `(committed, volatile)`.
  - `finalize() -> list[str]` returning all tokens seen in the last hypothesis and resetting internal state for the next utterance.

Behavior: committed tokens grow monotonically as the longest common prefix of the two most recent hypotheses. `volatile` is the remainder of the latest hypothesis after the committed prefix.

- [ ] **Step 1: Write the failing test**

`tests/test_local_agreement.py`:
```python
from nectarstt.engine.local_agreement import LocalAgreement

def test_first_update_commits_nothing():
    la = LocalAgreement()
    committed, volatile = la.update(["the", "quick"])
    assert committed == []
    assert volatile == ["the", "quick"]

def test_common_prefix_becomes_committed():
    la = LocalAgreement()
    la.update(["the", "quick", "brown"])
    committed, volatile = la.update(["the", "quick", "brownish", "fox"])
    assert committed == ["the", "quick"]
    assert volatile == ["brownish", "fox"]

def test_committed_is_monotonic():
    la = LocalAgreement()
    la.update(["the", "quick", "brown"])
    la.update(["the", "quick", "brown", "fox"])   # commits the, quick, brown
    committed, _ = la.update(["the", "quick", "brown", "foxes"])
    assert committed == ["the", "quick", "brown"]

def test_finalize_returns_last_hypothesis_and_resets():
    la = LocalAgreement()
    la.update(["hello", "world"])
    la.update(["hello", "world"])
    final = la.finalize()
    assert final == ["hello", "world"]
    committed, volatile = la.update(["new", "start"])
    assert committed == []
    assert volatile == ["new", "start"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_local_agreement.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `nectarstt/engine/local_agreement.py`**

```python
class LocalAgreement:
    """Commit the longest common prefix of the two most recent hypotheses.

    Committed tokens grow monotonically; the volatile tail is whatever of the
    latest hypothesis follows the committed prefix.
    """

    def __init__(self) -> None:
        self._prev: list[str] = []
        self._committed: list[str] = []

    def update(self, tokens: list[str]) -> tuple[list[str], list[str]]:
        common: list[str] = []
        for a, b in zip(tokens, self._prev):
            if a == b:
                common.append(a)
            else:
                break
        if len(common) > len(self._committed):
            self._committed = common
        self._prev = list(tokens)
        volatile = tokens[len(self._committed):]
        return list(self._committed), list(volatile)

    def finalize(self) -> list[str]:
        result = list(self._prev)
        self._prev = []
        self._committed = []
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_local_agreement.py -v`
Expected: PASS (all 4).

- [ ] **Step 5: Commit**

```bash
git add nectarstt/engine/local_agreement.py tests/test_local_agreement.py
git commit -m "feat: add LocalAgreement partial-result stabilization"
```

---

## Task 4: Config

**Files:**
- Create: `nectarstt/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Config` dataclass with fields: `device: str="cuda"`, `compute_type: str="float16"`, `model: str="distil-large-v3"`, `language: str="en"`, `sample_rate: int=16000`, `vad_threshold: float=0.5`, `min_silence_ms: int=500`, `min_speech_ms: int=200`, `window_interval_ms: int=500`.
  - `Config.load(path: str | None = None, env: Mapping[str,str] | None = None) -> Config` applying precedence defaults → TOML file → env (`NECTARSTT_<FIELD>` uppercased).

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from nectarstt.config import Config

def test_defaults():
    c = Config()
    assert c.device == "cuda"
    assert c.model == "distil-large-v3"
    assert c.sample_rate == 16000

def test_toml_overrides_defaults(tmp_path):
    f = tmp_path / "nectarstt.toml"
    f.write_text('model = "small"\ndevice = "cpu"\n', encoding="utf-8")
    c = Config.load(path=str(f))
    assert c.model == "small"
    assert c.device == "cpu"
    assert c.language == "en"  # untouched default

def test_env_overrides_toml(tmp_path):
    f = tmp_path / "nectarstt.toml"
    f.write_text('model = "small"\n', encoding="utf-8")
    c = Config.load(path=str(f), env={"NECTARSTT_MODEL": "tiny", "NECTARSTT_WINDOW_INTERVAL_MS": "250"})
    assert c.model == "tiny"
    assert c.window_interval_ms == 250  # coerced to int
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `nectarstt/config.py`**

```python
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

@dataclass
class Config:
    device: str = "cuda"
    compute_type: str = "float16"
    model: str = "distil-large-v3"
    language: str = "en"
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_silence_ms: int = 500
    min_speech_ms: int = 200
    window_interval_ms: int = 500

    @classmethod
    def load(cls, path: str | None = None,
             env: Mapping[str, str] | None = None) -> "Config":
        values: dict[str, Any] = {}
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                values.update(tomllib.load(fh))
        env = os.environ if env is None else env
        type_map = {f.name: f.type for f in fields(cls)}
        for f in fields(cls):
            key = f"NECTARSTT_{f.name.upper()}"
            if key in env:
                values[f.name] = env[key]
        coerced = {k: cls._coerce(k, v, type_map) for k, v in values.items()
                   if k in type_map}
        return cls(**coerced)

    @staticmethod
    def _coerce(name: str, value: Any, type_map: dict[str, Any]) -> Any:
        t = type_map[name]
        if t is int or t == "int":
            return int(value)
        if t is float or t == "float":
            return float(value)
        return str(value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all 3).

- [ ] **Step 5: Commit**

```bash
git add nectarstt/config.py tests/test_config.py
git commit -m "feat: add Config with defaults/TOML/env precedence"
```

---

## Task 5: Model registry & download root

**Files:**
- Create: `nectarstt/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `MODEL_ALIASES: dict[str, str]` mapping friendly names to faster-whisper model ids.
  - `resolve_model(name: str) -> str` (raises `ValueError` for unknown names).
  - `download_root() -> str` returning a stable per-user cache dir via `platformdirs`.

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
import pytest
from nectarstt import models

def test_resolve_known_alias():
    assert models.resolve_model("distil-large-v3") == "distil-large-v3"
    assert models.resolve_model("large-v3") == "large-v3"

def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        models.resolve_model("gpt-9")

def test_download_root_is_stable_and_contains_app():
    root = models.download_root()
    assert isinstance(root, str)
    assert "nectarstt" in root.lower()
    assert models.download_root() == root  # deterministic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `nectarstt/models.py`**

```python
import os
from platformdirs import user_cache_dir

# faster-whisper resolves these ids directly against the HF hub.
MODEL_ALIASES: dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "large-v3": "large-v3",
    "distil-large-v3": "distil-large-v3",
}

def resolve_model(name: str) -> str:
    if name not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model '{name}'. Known models: {sorted(MODEL_ALIASES)}"
        )
    return MODEL_ALIASES[name]

def download_root() -> str:
    root = os.path.join(user_cache_dir("nectarstt"), "models")
    os.makedirs(root, exist_ok=True)
    return root
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS (all 3).

- [ ] **Step 5: Commit**

```bash
git add nectarstt/models.py tests/test_models.py
git commit -m "feat: add model registry and cache download root"
```

---

## Task 6: Backend interface

**Files:**
- Create: `nectarstt/engine/backend.py`
- Test: `tests/test_backend_interface.py`

**Interfaces:**
- Produces:
  - `BackendResult(text: str, words: list[WordTiming])` dataclass.
  - `StreamingBackend` ABC with abstract method
    `transcribe(audio: np.ndarray, sample_rate: int, language: str | None, word_timestamps: bool) -> BackendResult`.

- [ ] **Step 1: Write the failing test**

`tests/test_backend_interface.py`:
```python
import numpy as np
import pytest
from nectarstt.engine.backend import StreamingBackend, BackendResult
from nectarstt.events import WordTiming

def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        StreamingBackend()

def test_concrete_subclass_works():
    class Fake(StreamingBackend):
        def transcribe(self, audio, sample_rate, language, word_timestamps):
            return BackendResult(text="ok", words=[WordTiming("ok", 0.0, 0.1, 1.0)])
    r = Fake().transcribe(np.zeros(16000, dtype=np.float32), 16000, "en", True)
    assert r.text == "ok"
    assert r.words[0].word == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_interface.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `nectarstt/engine/backend.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from nectarstt.events import WordTiming

@dataclass(frozen=True)
class BackendResult:
    text: str
    words: list[WordTiming] = field(default_factory=list)

class StreamingBackend(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None,
        word_timestamps: bool,
    ) -> BackendResult:
        """Transcribe a mono float32 audio buffer."""
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backend_interface.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nectarstt/engine/backend.py tests/test_backend_interface.py
git commit -m "feat: add StreamingBackend ABC and BackendResult"
```

---

## Task 7: faster-whisper backend

**Files:**
- Create: `nectarstt/engine/faster_whisper_backend.py`
- Create: `tests/fixtures/hello_world.wav` (see Step 1)
- Test: `tests/test_faster_whisper_backend.py`

**Interfaces:**
- Consumes: `StreamingBackend`, `BackendResult` (Task 6); `resolve_model`, `download_root` (Task 5); `WordTiming` (Task 2).
- Produces: `FasterWhisperBackend(model: str, device: str, compute_type: str)` implementing `transcribe(...)`. On CUDA failure at construction it falls back to `device="cpu"`, `compute_type="int8"` and logs a warning.

- [ ] **Step 1: Create the test fixture WAV**

Generate a spoken "hello world" clip once and commit it. Use any available TTS or a real recording; it must be 16kHz mono. Quick generation via faster-whisper's own dependency stack is not guaranteed, so create it with `sounddevice`-independent tooling. If Piper is available from the old engine, use it; otherwise record 2 seconds of yourself saying "hello world" and save as `tests/fixtures/hello_world.wav`. Verify:

```bash
python -c "import wave; w=wave.open('tests/fixtures/hello_world.wav'); print(w.getframerate(), w.getnchannels())"
```
Expected: `16000 1`.

- [ ] **Step 2: Write the failing integration test**

`tests/test_faster_whisper_backend.py`:
```python
import wave
import numpy as np
import pytest
from nectarstt.engine.faster_whisper_backend import FasterWhisperBackend

pytestmark = pytest.mark.integration

def _load(path):
    with wave.open(path) as w:
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr

def test_transcribes_hello_world():
    audio, sr = _load("tests/fixtures/hello_world.wav")
    backend = FasterWhisperBackend(model="tiny", device="cpu", compute_type="int8")
    result = backend.transcribe(audio, sr, language="en", word_timestamps=True)
    assert "hello" in result.text.lower()
    assert "world" in result.text.lower()
    assert len(result.words) >= 2
    assert result.words[0].end >= result.words[0].start
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_faster_whisper_backend.py -v -m integration`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `nectarstt/engine/faster_whisper_backend.py`**

```python
import logging

import numpy as np
from faster_whisper import WhisperModel

from nectarstt.engine.backend import StreamingBackend, BackendResult
from nectarstt.events import WordTiming
from nectarstt.models import resolve_model, download_root

log = logging.getLogger(__name__)

class FasterWhisperBackend(StreamingBackend):
    def __init__(self, model: str = "distil-large-v3",
                 device: str = "cuda", compute_type: str = "float16") -> None:
        model_id = resolve_model(model)
        try:
            self._model = WhisperModel(
                model_id, device=device, compute_type=compute_type,
                download_root=download_root(),
            )
        except Exception as exc:  # CUDA/cuDNN not available, etc.
            log.warning("Backend '%s' on %s failed (%s); falling back to CPU int8.",
                        model_id, device, exc)
            self._model = WhisperModel(
                model_id, device="cpu", compute_type="int8",
                download_root=download_root(),
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   language: str | None, word_timestamps: bool) -> BackendResult:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        segments, _ = self._model.transcribe(
            audio, language=language, word_timestamps=word_timestamps,
            beam_size=1,
        )
        text_parts: list[str] = []
        words: list[WordTiming] = []
        for seg in segments:
            text_parts.append(seg.text)
            for w in (seg.words or []):
                words.append(WordTiming(
                    word=w.word.strip(), start=w.start, end=w.end,
                    probability=getattr(w, "probability", 1.0),
                ))
        return BackendResult(text="".join(text_parts).strip(), words=words)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_faster_whisper_backend.py -v -m integration`
Expected: PASS (downloads the `tiny` model on first run, then offline).

- [ ] **Step 6: Commit**

```bash
git add nectarstt/engine/faster_whisper_backend.py tests/test_faster_whisper_backend.py tests/fixtures/hello_world.wav
git commit -m "feat: add faster-whisper backend with CPU fallback"
```

---

## Task 8: Frame sources (file-as-mic seam)

**Files:**
- Create: `nectarstt/audio/sources.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `FrameSource` ABC with `frames() -> Iterator[np.ndarray]` (each frame: mono float32) and `close() -> None`.
  - `FileSource(path: str, frame_ms: int = 32, sample_rate: int = 16000, realtime: bool = False)` — yields fixed-size frames from a WAV; pads the final short frame with zeros.
  - `MicSource(sample_rate: int = 16000, frame_ms: int = 32, device: int | None = None)` — sounddevice-backed (thin; validated manually, not in CI).

- [ ] **Step 1: Write the failing test**

`tests/test_sources.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `nectarstt/audio/sources.py`**

```python
import time
import wave
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        ...

    def close(self) -> None:
        pass

class FileSource(FrameSource):
    def __init__(self, path: str, frame_ms: int = 32,
                 sample_rate: int = 16000, realtime: bool = False) -> None:
        self._path = path
        self._frame_len = int(sample_rate * frame_ms / 1000)
        self._sample_rate = sample_rate
        self._realtime = realtime

    def frames(self) -> Iterator[np.ndarray]:
        with wave.open(self._path, "rb") as w:
            raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        step = self._frame_len
        for start in range(0, len(audio), step):
            chunk = audio[start:start + step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            if self._realtime:
                time.sleep(step / self._sample_rate)
            yield chunk

class MicSource(FrameSource):
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 32,
                 device: int | None = None) -> None:
        self._sample_rate = sample_rate
        self._frame_len = int(sample_rate * frame_ms / 1000)
        self._device = device

    def frames(self) -> Iterator[np.ndarray]:
        import queue
        import sounddevice as sd

        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                pass
            q.put(indata[:, 0].copy())

        try:
            with sd.InputStream(samplerate=self._sample_rate, channels=1,
                                dtype="float32", blocksize=self._frame_len,
                                device=self._device, callback=callback):
                while True:
                    yield q.get()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

class AudioDeviceError(RuntimeError):
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sources.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add nectarstt/audio/sources.py tests/test_sources.py
git commit -m "feat: add FrameSource with FileSource (file-as-mic) and MicSource"
```

---

## Task 9: Silero VAD wrapper

**Files:**
- Create: `nectarstt/audio/vad.py`
- Test: `tests/test_vad.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `SileroVAD(threshold: float = 0.5, sample_rate: int = 16000)` with `is_speech(frame: np.ndarray) -> bool` and `reset() -> None`. Internally buffers to the 512-sample window `pysilero-vad` requires.

- [ ] **Step 1: Write the failing test**

`tests/test_vad.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vad.py -v -m integration`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `nectarstt/audio/vad.py`**

```python
import numpy as np
from pysilero_vad import SileroVoiceActivityDetector

_WINDOW = 512  # samples @ 16kHz required by silero

class SileroVAD:
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        if sample_rate != 16000:
            raise ValueError("SileroVAD requires 16000 Hz audio.")
        self._threshold = threshold
        self._detector = SileroVoiceActivityDetector()
        self._buf = np.zeros(0, dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        self._buf = np.concatenate([self._buf, frame.astype(np.float32)])
        speech = False
        while len(self._buf) >= _WINDOW:
            window = self._buf[:_WINDOW]
            self._buf = self._buf[_WINDOW:]
            pcm = (window * 32768.0).astype(np.int16).tobytes()
            if self._detector(pcm) >= self._threshold:
                speech = True
        return speech

    def reset(self) -> None:
        self._detector.reset()
        self._buf = np.zeros(0, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_vad.py -v -m integration`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add nectarstt/audio/vad.py tests/test_vad.py
git commit -m "feat: add torch-free Silero VAD wrapper"
```

---

## Task 10: StreamingTranscriber (the conductor)

**Files:**
- Create: `nectarstt/engine/transcriber.py`
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: `FrameSource` (Task 8), `StreamingBackend`/`BackendResult` (Task 6/7), `LocalAgreement` (Task 3), `PartialResult`/`FinalResult`/`WordTiming` (Task 2). Accepts a VAD-like object exposing `is_speech(frame)->bool` and `reset()` (Task 9), injectable for tests.
- Produces: `StreamingTranscriber(backend, vad, config)` with
  `run(source: FrameSource) -> Iterator[PartialResult | FinalResult]`.

Logic: accumulate consecutive speech frames into a segment buffer. Every `window_interval_ms` of accumulated speech, call `backend.transcribe(buffer, ..., word_timestamps=False)`, tokenize `text.split()`, feed `LocalAgreement.update`, and yield a `PartialResult`. When silence exceeds `min_silence_ms` after speech (and speech exceeded `min_speech_ms`), run a final `backend.transcribe(buffer, ..., word_timestamps=True)`, yield a `FinalResult` with its words, then reset the buffer, VAD, and LocalAgreement.

- [ ] **Step 1: Write the failing test (deterministic, stubbed backend & VAD)**

`tests/test_transcriber.py`:
```python
import numpy as np
from nectarstt.engine.transcriber import StreamingTranscriber
from nectarstt.engine.backend import StreamingBackend, BackendResult
from nectarstt.events import WordTiming, PartialResult, FinalResult
from nectarstt.config import Config

class ListSource:
    def __init__(self, frames): self._frames = frames
    def frames(self): return iter(self._frames)
    def close(self): pass

class ScriptedBackend(StreamingBackend):
    """Returns a growing transcript based on how many speech frames are in the buffer."""
    def transcribe(self, audio, sample_rate, language, word_timestamps):
        n = int(round(len(audio) / 512))  # 1 token per ~frame of speech
        tokens = ["the", "quick", "brown", "fox"][:max(1, min(n, 4))]
        words = [WordTiming(t, i * 0.1, i * 0.1 + 0.1, 1.0) for i, t in enumerate(tokens)]
        return BackendResult(text=" ".join(tokens), words=words if word_timestamps else [])

class ScriptVAD:
    """Speech for the first `speech_frames` frames, then silence."""
    def __init__(self, speech_frames): self._left = speech_frames
    def is_speech(self, frame):
        if self._left > 0:
            self._left -= 1
            return True
        return False
    def reset(self): pass

def test_emits_partials_then_final():
    cfg = Config(window_interval_ms=64, min_silence_ms=64, min_speech_ms=32,
                 sample_rate=16000, device="cpu", compute_type="int8")
    frame = np.zeros(512, dtype=np.float32)          # 32ms frames
    frames = [frame] * 4 + [frame] * 4               # 4 speech, 4 silence
    t = StreamingTranscriber(ScriptedBackend(), ScriptVAD(speech_frames=4), cfg)
    events = list(t.run(ListSource(frames)))
    partials = [e for e in events if isinstance(e, PartialResult)]
    finals = [e for e in events if isinstance(e, FinalResult)]
    assert len(partials) >= 1
    assert len(finals) == 1
    assert finals[0].text == "the quick brown fox"
    assert len(finals[0].words) == 4                 # word timestamps on final only
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcriber.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `nectarstt/engine/transcriber.py`**

```python
from collections.abc import Iterator

import numpy as np

from nectarstt.audio.sources import FrameSource
from nectarstt.config import Config
from nectarstt.engine.backend import StreamingBackend
from nectarstt.engine.local_agreement import LocalAgreement
from nectarstt.events import PartialResult, FinalResult

class StreamingTranscriber:
    def __init__(self, backend: StreamingBackend, vad, config: Config) -> None:
        self._backend = backend
        self._vad = vad
        self._cfg = config

    def run(self, source: FrameSource) -> Iterator:
        cfg = self._cfg
        agree = LocalAgreement()
        buffer: list[np.ndarray] = []
        speech_ms = 0.0
        silence_ms = 0.0
        ms_since_window = 0.0
        had_speech = False

        for frame in source.frames():
            frame_ms = 1000.0 * len(frame) / cfg.sample_rate
            if self._vad.is_speech(frame):
                buffer.append(frame)
                speech_ms += frame_ms
                silence_ms = 0.0
                ms_since_window += frame_ms
                had_speech = True
                if ms_since_window >= cfg.window_interval_ms:
                    ms_since_window = 0.0
                    audio = np.concatenate(buffer)
                    res = self._backend.transcribe(
                        audio, cfg.sample_rate, cfg.language, word_timestamps=False)
                    committed, volatile = agree.update(res.text.split())
                    yield PartialResult(
                        text=res.text,
                        committed_prefix=" ".join(committed),
                        volatile_tail=" ".join(volatile),
                    )
            else:
                if had_speech:
                    silence_ms += frame_ms
                    if (silence_ms >= cfg.min_silence_ms
                            and speech_ms >= cfg.min_speech_ms):
                        yield self._finalize(buffer, agree)
                        buffer, speech_ms, silence_ms = [], 0.0, 0.0
                        ms_since_window, had_speech = 0.0, False

        if buffer and speech_ms >= cfg.min_speech_ms:
            yield self._finalize(buffer, agree)

    def _finalize(self, buffer: list[np.ndarray], agree: LocalAgreement) -> FinalResult:
        cfg = self._cfg
        audio = np.concatenate(buffer)
        res = self._backend.transcribe(
            audio, cfg.sample_rate, cfg.language, word_timestamps=True)
        agree.finalize()
        self._vad.reset()
        start = res.words[0].start if res.words else 0.0
        end = res.words[-1].end if res.words else 0.0
        return FinalResult(text=res.text, words=res.words, start=start, end=end)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcriber.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nectarstt/engine/transcriber.py tests/test_transcriber.py
git commit -m "feat: add StreamingTranscriber orchestration"
```

---

## Task 11: STTEngine façade

**Files:**
- Modify: `nectarstt/__init__.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: all prior components.
- Produces (exported from `nectarstt`):
  - `STTEngine(model=None, device=None, language=None, config: Config | None = None)` — builds `Config`, `FasterWhisperBackend`, `SileroVAD`, `StreamingTranscriber`. Constructor args override config fields when provided.
  - `STTEngine.stream(source: FrameSource | None = None) -> Iterator` — defaults `source` to `MicSource`.
  - `STTEngine.transcribe_file(path: str) -> FinalResult` — batch transcription of a whole WAV via the backend.
  - Re-exports: `Config`, `PartialResult`, `FinalResult`, `WordTiming`, `FileSource`, `MicSource`.

- [ ] **Step 1: Write the failing test**

`tests/test_engine.py`:
```python
import pytest
from nectarstt import STTEngine, FileSource, FinalResult, PartialResult

pytestmark = pytest.mark.integration  # uses the real tiny backend + VAD

def test_stream_over_file_yields_final():
    engine = STTEngine(model="tiny", device="cpu", language="en")
    engine._config.compute_type = "int8"  # ensure CPU-friendly
    events = list(engine.stream(source=FileSource("tests/fixtures/hello_world.wav")))
    finals = [e for e in events if isinstance(e, FinalResult)]
    assert len(finals) >= 1
    joined = " ".join(f.text.lower() for f in finals)
    assert "hello" in joined and "world" in joined

def test_transcribe_file_batch():
    engine = STTEngine(model="tiny", device="cpu", language="en")
    result = engine.transcribe_file("tests/fixtures/hello_world.wav")
    assert isinstance(result, FinalResult)
    assert "hello" in result.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v -m integration`
Expected: FAIL with `ImportError: cannot import name 'STTEngine'`.

- [ ] **Step 3: Implement the façade in `nectarstt/__init__.py`**

```python
"""NectarSTT — offline streaming speech-to-text."""
from collections.abc import Iterator

import numpy as np

from nectarstt.config import Config
from nectarstt.events import WordTiming, PartialResult, FinalResult
from nectarstt.audio.sources import FrameSource, FileSource, MicSource
from nectarstt.audio.vad import SileroVAD
from nectarstt.engine.faster_whisper_backend import FasterWhisperBackend
from nectarstt.engine.transcriber import StreamingTranscriber

__version__ = "0.1.0"
__all__ = ["STTEngine", "Config", "PartialResult", "FinalResult",
           "WordTiming", "FileSource", "MicSource"]

class STTEngine:
    def __init__(self, model: str | None = None, device: str | None = None,
                 language: str | None = None, config: Config | None = None) -> None:
        cfg = config or Config()
        if model is not None:
            cfg.model = model
        if device is not None:
            cfg.device = device
        if language is not None:
            cfg.language = language
        self._config = cfg
        self._backend = FasterWhisperBackend(
            model=cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        self._vad = SileroVAD(threshold=cfg.vad_threshold,
                              sample_rate=cfg.sample_rate)
        self._transcriber = StreamingTranscriber(self._backend, self._vad, cfg)

    def stream(self, source: FrameSource | None = None) -> Iterator:
        if source is None:
            source = MicSource(sample_rate=self._config.sample_rate)
        return self._transcriber.run(source)

    def transcribe_file(self, path: str) -> FinalResult:
        import wave
        with wave.open(path, "rb") as w:
            raw = w.readframes(w.getnframes())
            sr = w.getframerate()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        res = self._backend.transcribe(
            audio, sr, self._config.language, word_timestamps=True)
        start = res.words[0].start if res.words else 0.0
        end = res.words[-1].end if res.words else 0.0
        return FinalResult(text=res.text, words=res.words, start=start, end=end)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v -m integration`
Expected: PASS (both).

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all non-integration tests PASS; integration tests PASS when models are available.

- [ ] **Step 6: Commit**

```bash
git add nectarstt/__init__.py tests/test_engine.py
git commit -m "feat: add STTEngine facade with stream() and transcribe_file()"
```

---

## Task 12: Demo CLI

**Files:**
- Create: `nectarstt/demo.py`
- Test: `tests/test_demo.py`

**Interfaces:**
- Consumes: `STTEngine`, `FileSource` (Task 11).
- Produces: `main(argv: list[str] | None = None) -> int`. Flags: `--model` (default `distil-large-v3`), `--device` (default `cuda`), `--language` (default `en`), `--file PATH` (optional; use `FileSource` instead of mic). Prints partials with a leading `~` on one rewritten line and finals with a leading `✓` on their own line.

- [ ] **Step 1: Write the failing test**

`tests/test_demo.py`:
```python
from nectarstt import demo

def test_build_parser_defaults():
    args = demo.build_parser().parse_args([])
    assert args.model == "distil-large-v3"
    assert args.device == "cuda"
    assert args.language == "en"
    assert args.file is None

def test_parser_accepts_file():
    args = demo.build_parser().parse_args(["--file", "x.wav", "--device", "cpu"])
    assert args.file == "x.wav"
    assert args.device == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `nectarstt/demo.py`**

```python
import argparse
import sys

from nectarstt import STTEngine, FileSource

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nectarstt-demo",
                                description="Live NectarSTT transcription.")
    p.add_argument("--model", default="distil-large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--language", default="en")
    p.add_argument("--file", default=None, help="Transcribe a WAV instead of the mic.")
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = STTEngine(model=args.model, device=args.device, language=args.language)
    source = FileSource(args.file) if args.file else None
    print("Listening… (Ctrl+C to stop)" if source is None else f"Transcribing {args.file}…")
    try:
        for event in engine.stream(source=source):
            if event.is_partial:
                print(f"~ {event.text}", end="\r", flush=True)
            else:
                print(f"\r✓ {event.text}")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_demo.py -v`
Expected: PASS (both).

- [ ] **Step 5: Manual end-to-end check (on the CUDA machine)**

Run:
```bash
python -m nectarstt.demo --file tests/fixtures/hello_world.wav --device cpu --model tiny
```
Expected: prints partial line(s) then `✓ ...hello world...`.

- [ ] **Step 6: Commit**

```bash
git add nectarstt/demo.py tests/test_demo.py
git commit -m "feat: add live demo CLI"
```

---

## Task 13: README refresh

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Rewrite the Installation and Usage sections**

Replace the old zip/Main-Engine instructions with:

````markdown
## Installation
```bash
pip install -e ".[dev]"
```

## Usage
Live mic transcription:
```bash
python -m nectarstt.demo
```

In code:
```python
from nectarstt import STTEngine

engine = STTEngine(model="distil-large-v3", device="cuda", language="en")
for event in engine.stream():
    if event.is_partial:
        print("~", event.text, end="\r")
    else:
        print("✓", event.text)
```

Batch a file:
```python
result = engine.transcribe_file("clip.wav")
print(result.text)
```
````

- [ ] **Step 2: Remove stale claims**

Delete the `pip install nectarstt`, `Main-Engine.zip`, and Piper/eSpeak lines that no longer apply to the STT engine. Keep the roadmap section but point it at `docs/superpowers/specs/` and `docs/superpowers/plans/`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for the nectarstt streaming engine"
```

---

## Self-Review Notes (completed)

- **Spec coverage:** repo restructure (Task 1, 13) · package layout (all) · FrameSource/MicSource/FileSource (Task 8) · SileroVAD ONNX/torch-free (Task 9) · StreamingBackend ABC + FasterWhisperBackend (Task 6/7) · LocalAgreement (Task 3) · StreamingTranscriber (Task 10) · STTEngine + public API (Task 11) · events (Task 2) · config precedence (Task 4) · model registry + cache download (Task 5) · error handling: CPU fallback (Task 7), AudioDeviceError (Task 8), unknown-model ValueError (Task 5) · testing incl. file-as-mic determinism (Task 10) and batch fixture (Task 7) · demo CLI (Task 12) · definition of done (Task 11 full-suite + Task 12 manual run).
- **Placeholder scan:** none — every code step contains runnable code.
- **Type consistency:** `BackendResult(text, words)`, `WordTiming(word, start, end, probability)`, `PartialResult(text, committed_prefix, volatile_tail)`, `FinalResult(text, words, start, end)`, `Config` field names, `resolve_model`/`download_root`, `FrameSource.frames()`, `SileroVAD.is_speech()/reset()`, `LocalAgreement.update()/finalize()`, `StreamingTranscriber.run()`, `STTEngine.stream()/transcribe_file()` are used consistently across tasks.
