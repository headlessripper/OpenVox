# OpenVox Voice Cloning Engine Implementation Plan (Sub-project 2B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline zero-shot voice-cloning engine at `openvox/clone/` that speaks arbitrary text in a voice taken from a short reference clip, via Chatterbox (MIT).

**Architecture:** A `VoiceCloneEngine` façade parallel to the TTS engine. `clone(text, reference_audio)` → `ChatterboxBackend` (Chatterbox: reference clip + text → cloned 24 kHz waveform) → `TTSResult` (reused from `openvox.tts.backend`), which can `save_wav` or `play`. The backend sits behind a `CloneBackend` ABC; torch/Chatterbox load lazily on first clone.

**Tech Stack:** Python 3.11+, chatterbox-tts (PyTorch), sounddevice, numpy, pytest.

## Global Constraints

- Python `>=3.11`. No hardcoded filesystem paths.
- `torch` and `chatterbox` are imported ONLY inside `openvox/clone/chatterbox_backend.py`, and lazily (inside the model-load method) — so `import openvox` and `import openvox.clone` need neither torch nor the `[clone]` extra.
- Reuse `TTSResult` and `AudioDeviceError` from `openvox.tts.backend` (torch-free) — do NOT define a new result type.
- Audio contract: 24000 Hz mono float32 in [-1, 1]; the sample rate is read from `model.sr`, not hardcoded.
- The public knob `cfg` maps to Chatterbox's `generate(..., cfg_weight=…)`; `exaggeration` passes through.
- English-first. Default device `cuda`, resolved to CPU when `torch.cuda.is_available()` is False (pure `_resolve_device` helper).
- Model loads lazily on first `clone()`; input validation (empty text, missing/empty reference) runs BEFORE the load.
- `os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")` is set before the chatterbox import (Windows symlink fix).
- `[clone]` extra = `chatterbox-tts` + `sounddevice`; it stays OUT of the `all` extra (torch is heavy — cloning is opt-in). GPU torch is a documented separate install.
- TDD: failing test first, then implement, then commit. Unit tests must run with no torch/model; only the integration test loads Chatterbox. Playback (`play`/`say`) is validated manually, never in CI.

## File Structure

- `openvox/clone/__init__.py` — exports `VoiceCloneEngine`, `TTSResult` (filled in Task 4).
- `openvox/clone/config.py` — `CloneConfig` dataclass.
- `openvox/clone/backend.py` — `CloneBackend` ABC.
- `openvox/clone/chatterbox_backend.py` — `_resolve_device`, `ChatterboxBackend`.
- `openvox/clone/engine.py` — `VoiceCloneEngine`.
- `openvox/clone/demo.py` — `python -m openvox.clone.demo`.
- `tests/clone/` — mirrors the package.

---

## Task 1: Scaffold + CloneConfig + CloneBackend ABC

**Files:**
- Create: `openvox/clone/__init__.py`, `openvox/clone/config.py`, `openvox/clone/backend.py`, `tests/clone/__init__.py`
- Test: `tests/clone/test_config.py`, `tests/clone/test_backend.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `TTSResult` from `openvox.tts.backend` (exists).
- Produces:
  - `CloneConfig(device="cuda", exaggeration=0.5, cfg=0.5)` dataclass.
  - `CloneBackend` ABC with abstract `clone(text: str, reference_path: str, exaggeration: float, cfg: float) -> TTSResult`.

- [ ] **Step 1: Add the `[clone]` extra + entry point to `pyproject.toml`**

In `[project.optional-dependencies]` add `clone` (do NOT add it to `all`); in `[project.scripts]` add the demo:
```toml
clone = ["chatterbox-tts", "sounddevice>=0.4.6"]
```
```toml
openvox-clone-demo = "openvox.clone.demo:main"
```

- [ ] **Step 2: Create package init files**

`openvox/clone/__init__.py`:
```python
"""OpenVox voice cloning — offline zero-shot voice cloning."""
```
Create empty `tests/clone/__init__.py`.

- [ ] **Step 3: Write the failing tests**

`tests/clone/test_config.py`:
```python
from openvox.clone.config import CloneConfig

def test_defaults():
    c = CloneConfig()
    assert c.device == "cuda"
    assert c.exaggeration == 0.5
    assert c.cfg == 0.5
```

`tests/clone/test_backend.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/clone/ -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5: Implement `openvox/clone/config.py`**

```python
from dataclasses import dataclass

@dataclass
class CloneConfig:
    device: str = "cuda"
    exaggeration: float = 0.5
    cfg: float = 0.5
```

- [ ] **Step 6: Implement `openvox/clone/backend.py`**

```python
from abc import ABC, abstractmethod

from openvox.tts.backend import TTSResult

class CloneBackend(ABC):
    @abstractmethod
    def clone(self, text: str, reference_path: str, exaggeration: float,
              cfg: float) -> TTSResult:
        """Speak text in the voice from the reference clip."""
        raise NotImplementedError
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pip install -e ".[dev]"` then `pytest tests/clone/ -v`
Expected: PASS (3 tests). (No torch / `[clone]` extra needed — the ABC only reuses the torch-free `TTSResult`.)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml openvox/clone/__init__.py openvox/clone/config.py openvox/clone/backend.py tests/clone/
git commit -m "feat(clone): scaffold openvox.clone with CloneConfig and backend ABC"
```

---

## Task 2: Chatterbox backend

**Files:**
- Create: `openvox/clone/chatterbox_backend.py`
- Test: `tests/clone/test_chatterbox_backend.py`

**Interfaces:**
- Consumes: `CloneBackend` (Task 1), `TTSResult` from `openvox.tts.backend`.
- Produces:
  - `_resolve_device(requested: str, cuda_available: bool) -> str` (pure).
  - `ChatterboxBackend(device: str = "cuda")` implementing `clone(...)`, lazy-loading `ChatterboxTTS` on first call.

- [ ] **Step 1: Write the failing tests**

`tests/clone/test_chatterbox_backend.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/clone/test_chatterbox_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `openvox/clone/chatterbox_backend.py`**

```python
import logging
import os

import numpy as np

from openvox.clone.backend import CloneBackend
from openvox.tts.backend import TTSResult

log = logging.getLogger(__name__)

def _resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "cuda" and cuda_available:
        return "cuda"
    return "cpu"

class ChatterboxBackend(CloneBackend):
    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Windows: avoid dangling HF symlinks that break model loading. Set
        # before huggingface_hub is imported (transitively, by chatterbox).
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        import torch
        from chatterbox.tts import ChatterboxTTS

        device = _resolve_device(self._device, torch.cuda.is_available())
        if self._device == "cuda" and device != "cuda":
            log.info("CUDA is not available to torch; loading Chatterbox on CPU "
                     "(slow — install a CUDA torch build for GPU acceleration).")
        self._model = ChatterboxTTS.from_pretrained(device)

    def clone(self, text: str, reference_path: str, exaggeration: float,
              cfg: float) -> TTSResult:
        self._ensure_loaded()
        wav = self._model.generate(
            text, audio_prompt_path=reference_path,
            exaggeration=exaggeration, cfg_weight=cfg)
        audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return TTSResult(audio=audio, sample_rate=int(self._model.sr))
```

- [ ] **Step 4: Run the unit tests (no model) to verify they pass**

Run: `pytest tests/clone/test_chatterbox_backend.py -v -m "not integration"`
Expected: PASS (3 `_resolve_device` tests). Importing the module must not require torch.

- [ ] **Step 5: Run the integration test (installs Chatterbox, downloads the model, clones on CPU)**

Run: `pip install -e ".[clone,dev]"` then `pytest tests/clone/test_chatterbox_backend.py -v -m integration`
Expected: PASS. First run installs chatterbox-tts (pulls torch) and downloads the Chatterbox model (~1–2 GB, one-time); CPU generation can take a minute or more. If the install fails for an environment reason you cannot resolve, report BLOCKED with the exact pip error — do not fake the test.

- [ ] **Step 6: Commit**

```bash
git add openvox/clone/chatterbox_backend.py tests/clone/test_chatterbox_backend.py
git commit -m "feat(clone): add Chatterbox zero-shot cloning backend"
```

---

## Task 3: VoiceCloneEngine façade

**Files:**
- Create: `openvox/clone/engine.py`
- Test: `tests/clone/test_engine.py`

**Interfaces:**
- Consumes: `CloneConfig` (Task 1), `ChatterboxBackend` (Task 2), `TTSResult`/`AudioDeviceError` from `openvox.tts.backend`.
- Produces: `VoiceCloneEngine(device=None, exaggeration=None, cfg=None, config=None)` with `clone(text, reference_audio, exaggeration=None, cfg=None) -> TTSResult`, `play(result)`, `say(text, reference_audio, exaggeration=None, cfg=None)`. `clone` validates non-empty text and an existing, non-empty reference file BEFORE the backend loads; constructor copies the config.

- [ ] **Step 1: Write the failing tests**

`tests/clone/test_engine.py`:
```python
import numpy as np
import pytest
from openvox.clone.engine import VoiceCloneEngine
from openvox.tts.backend import TTSResult

class _FakeBackend:
    def __init__(self, *a, **k): pass
    def clone(self, text, reference_path, exaggeration, cfg):
        return TTSResult(audio=np.zeros(24000, dtype=np.float32), sample_rate=24000)

@pytest.fixture
def patched(monkeypatch):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _FakeBackend)

def test_empty_text_raises(patched, tmp_path):
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    with pytest.raises(ValueError):
        VoiceCloneEngine().clone("   ", str(ref))

def test_missing_reference_raises(patched):
    with pytest.raises(FileNotFoundError):
        VoiceCloneEngine().clone("hello", "does_not_exist.wav")

def test_empty_reference_raises(patched, tmp_path):
    ref = tmp_path / "empty.wav"; ref.write_bytes(b"")
    with pytest.raises(ValueError):
        VoiceCloneEngine().clone("hello", str(ref))

def test_clone_returns_result(patched, tmp_path):
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"x")
    r = VoiceCloneEngine().clone("hello", str(ref))
    assert r.sample_rate == 24000 and len(r.audio) == 24000

def test_config_not_mutated(patched, tmp_path):
    from openvox.clone.config import CloneConfig
    cfg = CloneConfig()
    VoiceCloneEngine(device="cpu", config=cfg)
    assert cfg.device == "cuda"   # caller's config untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/clone/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `openvox/clone/engine.py`**

```python
import dataclasses
import os

from openvox.clone.config import CloneConfig
from openvox.clone.chatterbox_backend import ChatterboxBackend
from openvox.tts.backend import TTSResult, AudioDeviceError

class VoiceCloneEngine:
    def __init__(self, device: str | None = None, exaggeration: float | None = None,
                 cfg: float | None = None, config: CloneConfig | None = None) -> None:
        c = dataclasses.replace(config) if config is not None else CloneConfig()
        if device is not None:
            c.device = device
        if exaggeration is not None:
            c.exaggeration = exaggeration
        if cfg is not None:
            c.cfg = cfg
        self._config = c
        self._backend = ChatterboxBackend(device=c.device)

    def clone(self, text: str, reference_audio: str,
              exaggeration: float | None = None, cfg: float | None = None) -> TTSResult:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(f"reference audio not found: {reference_audio}")
        if os.path.getsize(reference_audio) == 0:
            raise ValueError(f"reference audio is empty: {reference_audio}")
        e = exaggeration if exaggeration is not None else self._config.exaggeration
        g = cfg if cfg is not None else self._config.cfg
        return self._backend.clone(text, reference_audio, e, g)

    def play(self, result: TTSResult) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError(
                'Playback needs the audio extra: pip install "openvox[clone]".') from exc
        try:
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

    def say(self, text: str, reference_audio: str,
            exaggeration: float | None = None, cfg: float | None = None) -> None:
        self.play(self.clone(text, reference_audio, exaggeration, cfg))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/clone/test_engine.py -v`
Expected: PASS (5 tests, backend patched — no torch/model).

- [ ] **Step 5: Commit**

```bash
git add openvox/clone/engine.py tests/clone/test_engine.py
git commit -m "feat(clone): add VoiceCloneEngine facade (clone/play/say)"
```

---

## Task 4: Public exports + demo CLI

**Files:**
- Modify: `openvox/clone/__init__.py`
- Create: `openvox/clone/demo.py`
- Test: `tests/clone/test_demo.py`

**Interfaces:**
- Consumes: `VoiceCloneEngine` (Task 3), `TTSResult`.
- Produces (exported from `openvox.clone`): `VoiceCloneEngine`, `TTSResult`. Demo `main(argv=None) -> int` with `build_parser()` (flags `--text` required, `--ref` required, `--exaggeration` default 0.5, `--cfg` default 0.5, `--device` default cuda, `--out`, `--no-play`).

- [ ] **Step 1: Write the failing tests**

`tests/clone/test_demo.py`:
```python
from openvox.clone import demo

def test_build_parser_defaults():
    args = demo.build_parser().parse_args(["--text", "hi", "--ref", "r.wav"])
    assert args.text == "hi"
    assert args.ref == "r.wav"
    assert args.exaggeration == 0.5
    assert args.cfg == 0.5
    assert args.device == "cuda"
    assert args.out is None
    assert args.no_play is False

def test_parser_flags():
    args = demo.build_parser().parse_args(
        ["--text", "hi", "--ref", "r.wav", "--exaggeration", "0.7",
         "--cfg", "0.3", "--device", "cpu", "--out", "o.wav", "--no-play"])
    assert args.exaggeration == 0.7 and args.cfg == 0.3
    assert args.device == "cpu" and args.out == "o.wav" and args.no_play is True

def test_exports():
    from openvox.clone import VoiceCloneEngine, TTSResult
    assert VoiceCloneEngine is not None and TTSResult is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/clone/test_demo.py -v`
Expected: FAIL (`AttributeError`/`ImportError`).

- [ ] **Step 3: Update `openvox/clone/__init__.py`**

```python
"""OpenVox voice cloning — offline zero-shot voice cloning."""
from openvox.clone.engine import VoiceCloneEngine
from openvox.tts.backend import TTSResult

__all__ = ["VoiceCloneEngine", "TTSResult"]
```

- [ ] **Step 4: Implement `openvox/clone/demo.py`**

```python
import argparse
import sys

from openvox.clone import VoiceCloneEngine

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openvox-clone-demo",
                                description="Offline zero-shot voice cloning.")
    p.add_argument("--text", required=True, help="Text to speak in the cloned voice.")
    p.add_argument("--ref", required=True, help="Reference audio clip of the voice to clone.")
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--cfg", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="Also save the audio to this WAV path.")
    p.add_argument("--no-play", action="store_true", help="Do not play the audio aloud.")
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = VoiceCloneEngine(device=args.device, exaggeration=args.exaggeration, cfg=args.cfg)
    result = engine.clone(args.text, args.ref)
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

Run: `pytest tests/clone/test_demo.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Manual end-to-end check (on the GPU machine, with the `[clone]` extra installed)**

Run:
```bash
python -m openvox.clone.demo --text "This is my cloned voice, running entirely offline." --ref tests/stt/fixtures/hello_world.wav --out clone_demo.wav --no-play
```
Expected: prints `Saved clone_demo.wav (…s)` and the WAV is a valid 24 kHz clip. Delete `clone_demo.wav` afterward; do not commit it.

- [ ] **Step 7: Run the full non-integration suite**

Run: `pytest -q -m "not integration"`
Expected: all pass; the clone unit tests run without torch/model.

- [ ] **Step 8: Commit**

```bash
git add openvox/clone/__init__.py openvox/clone/demo.py tests/clone/test_demo.py
git commit -m "feat(clone): add public exports and demo CLI"
```

---

## Task 5: README — document voice cloning

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the status table, intro, and architecture lines**

- In the "Project status & roadmap" table, change the **Voice cloning** row from `🔭 Planned` to `✅ Available`.
- In the intro paragraph, move voice cloning to available: change "Voice cloning and ultra-low-latency streaming are **in active development**" to "Voice cloning is **available today**; ultra-low-latency streaming is **in active development**".
- In the "Architecture" section, add a bullet: `- **\`openvox.clone\`** — the zero-shot voice-cloning engine (available today).`

- [ ] **Step 2: Add a Voice Cloning section**

Add after the Text-to-Speech section:

````markdown
## 🎭 Voice Cloning (available today)

Zero-shot voice cloning — give a short reference clip and speak any text in that voice, fully offline (via [Chatterbox](https://github.com/resemble-ai/chatterbox), MIT). Every generated clip carries an imperceptible neural watermark for traceability.

```bash
pip install -e ".[clone]"                 # CPU (pulls PyTorch)
# for NVIDIA GPU, also install a CUDA torch build:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```bash
python -m openvox.clone.demo --text "This is my cloned voice." --ref myvoice.mp3 --out cloned.wav
```

```python
from openvox.clone import VoiceCloneEngine

engine = VoiceCloneEngine(device="cuda")            # falls back to CPU
result = engine.clone("Speak this in my voice.", reference_audio="myvoice.mp3")
result.save_wav("cloned.wav")
engine.say("Nice to meet you.", reference_audio="myvoice.mp3")   # clone + speak
```

**Demo flags:** `--text` (required) · `--ref PATH` (required, any audio format) · `--exaggeration` (default 0.5) · `--cfg` (default 0.5) · `--device` (`cuda`/`cpu`) · `--out PATH` · `--no-play`.
````

- [ ] **Step 3: Verify accuracy**

Run: `grep -n "VoiceCloneEngine\|openvox.clone.demo\|\[clone\]" README.md` and confirm each documented symbol/command exists in the code and `pyproject.toml`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document voice cloning in README"
```

---

## Self-Review Notes (completed)

- **Spec coverage:** package layout (T1–T4) · CloneConfig (T1) · CloneBackend ABC (T1) · ChatterboxBackend + `_resolve_device` + lazy load + HF symlink fix + cfg→cfg_weight + tensor→numpy + model.sr (T2) · VoiceCloneEngine clone/play/say + validate-before-load + config copy (T3) · exports + demo CLI (T4) · error handling: empty text/missing ref/empty ref (T3), CUDA→CPU (T2), AudioDeviceError incl. missing-extra ImportError (T3), download failure surfaced (T2) · testing incl. no-torch unit tests (T1/T2/T3/T4) and integration clone (T2) · `[clone]` extra out of `all` + entry point (T1) · DoD mapped · README (T5).
- **Placeholder scan:** none — every code step has runnable code.
- **Type consistency:** `TTSResult(audio, sample_rate)` reused from openvox.tts.backend; `CloneBackend.clone(text, reference_path, exaggeration, cfg)`; `ChatterboxBackend(device)`; `_resolve_device(requested, cuda_available)`; `CloneConfig(device, exaggeration, cfg)`; `VoiceCloneEngine(device, exaggeration, cfg, config)` with `clone/play/say`; demo `build_parser`/`main`. Consistent, and matches the verified chatterbox API (`from_pretrained(device)`, `generate(text, audio_prompt_path=, exaggeration=, cfg_weight=)`, `model.sr`).
