# OpenVox Speech Enhancement Engine Implementation Plan (Sub-project 2B.2a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline speech-restoration engine at `openvox/enhance/` (denoise + restore + bandwidth-extend via resemble-enhance) and have the voice cloner auto-clean its reference clip.

**Architecture:** An `EnhanceEngine` façade over `resemble_enhance` behind an `EnhanceBackend` ABC. The backend applies spike-proven runtime shims (stub the training-only `deepspeed`, `PosixPath`→`WindowsPath`, `HF_HUB_DISABLE_SYMLINKS`) and imports torch/resemble lazily. `VoiceCloneEngine.clone()` auto-enhances the reference by default (cached; degrades gracefully if the enhance deps are absent).

**Tech Stack:** Python 3.11+, resemble-enhance (installed `--no-deps`), torch/torchaudio (from the `[clone]` side), soundfile, numpy, pytest.

## Global Constraints

- Python `>=3.11`. No hardcoded filesystem paths (cache via `openvox._paths.cache_dir`).
- `torch` and `resemble_enhance` imported ONLY inside `openvox/enhance/resemble_backend.py`, and lazily (inside the model-load method) — so `import openvox` and `import openvox.enhance` need neither torch nor the enhance deps.
- Reuse `TTSResult` from `openvox.tts.backend` (do NOT define a new result type). Restoration output is 44100 Hz.
- Runtime shims applied in `resemble_backend` BEFORE importing `resemble_enhance`: `os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS","1")`; stub `deepspeed` + submodules via `sys.modules.setdefault(..., MagicMock())`; on Windows (`os.name == "nt"`) set `pathlib.PosixPath = pathlib.WindowsPath`.
- resemble-enhance inference API (verified by spike): `from resemble_enhance.enhancer.inference import denoise, enhance`; `enhance(dwav, sr, device, nfe=64, solver="midpoint", lambd=0.9, tau=0.5) -> (wav_tensor, new_sr)`; `denoise(dwav, sr, device) -> (wav_tensor, new_sr)`; `dwav` is a mono float32 torch tensor; output sr is 44100.
- Device `cuda` resolves to CPU when `torch.cuda.is_available()` is False (pure `_resolve_device`).
- Input validation (empty array / missing / empty file) runs BEFORE any model load.
- Clone auto-enhance is default-on, cached (key = reference abspath + mtime_ns + size), and NEVER breaks cloning — any enhancement failure/absence degrades to the raw reference with a log notice.
- `[enhance]` extra = light deps only (`matplotlib`, `omegaconf`, `pandas`, `celluloid`, `resampy`, `soundfile`, `rich`, `tabulate`); `resemble-enhance` is installed separately `--no-deps`. `[enhance]` stays OUT of `all`. Two-step install documented.
- TDD; unit tests run with no torch/resemble; only the integration test loads resemble-enhance. Playback validated manually.

## File Structure

- `openvox/enhance/__init__.py` — exports `EnhanceEngine`, `TTSResult` (Task 4).
- `openvox/enhance/config.py` — `EnhanceConfig`.
- `openvox/enhance/backend.py` — `EnhanceBackend` ABC.
- `openvox/enhance/resemble_backend.py` — `_resolve_device`, `_install_shims`, `ResembleEnhanceBackend`.
- `openvox/enhance/engine.py` — `EnhanceEngine`.
- `openvox/enhance/demo.py` — `python -m openvox.enhance.demo`.
- `openvox/clone/{config,engine,demo}.py` — auto-enhance integration (Task 5).
- `tests/enhance/` — mirrors the package.

---

## Task 1: Scaffold + EnhanceConfig + EnhanceBackend ABC

**Files:**
- Create: `openvox/enhance/__init__.py`, `openvox/enhance/config.py`, `openvox/enhance/backend.py`, `tests/enhance/__init__.py`
- Test: `tests/enhance/test_config.py`, `tests/enhance/test_backend.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `TTSResult` from `openvox.tts.backend`.
- Produces: `EnhanceConfig(device="cuda", nfe=64, solver="midpoint", lambd=0.9, tau=0.5, denoise_only=False)`; `EnhanceBackend` ABC with abstract `enhance(audio, sample_rate: int) -> TTSResult`.

- [ ] **Step 1: Add the `[enhance]` extra + entry point to `pyproject.toml`**

In `[project.optional-dependencies]` add `enhance` (do NOT add it to `all`); in `[project.scripts]` add the demo:
```toml
enhance = ["matplotlib", "omegaconf", "pandas", "celluloid", "resampy", "soundfile", "rich", "tabulate"]
```
```toml
openvox-enhance-demo = "openvox.enhance.demo:main"
```

- [ ] **Step 2: Create package init files**

`openvox/enhance/__init__.py`:
```python
"""OpenVox speech enhancement — offline denoise + restoration."""
```
Create empty `tests/enhance/__init__.py`.

- [ ] **Step 3: Write the failing tests**

`tests/enhance/test_config.py`:
```python
from openvox.enhance.config import EnhanceConfig

def test_defaults():
    c = EnhanceConfig()
    assert c.device == "cuda"
    assert c.nfe == 64
    assert c.solver == "midpoint"
    assert c.lambd == 0.9
    assert c.tau == 0.5
    assert c.denoise_only is False
```

`tests/enhance/test_backend.py`:
```python
import numpy as np
import pytest
from openvox.enhance.backend import EnhanceBackend
from openvox.tts.backend import TTSResult

def test_backend_abc_not_instantiable():
    with pytest.raises(TypeError):
        EnhanceBackend()

def test_concrete_backend_ok():
    class Fake(EnhanceBackend):
        def enhance(self, audio, sample_rate):
            return TTSResult(audio=np.zeros(10, dtype=np.float32), sample_rate=44100)
    r = Fake().enhance(np.zeros(4, dtype=np.float32), 16000)
    assert r.sample_rate == 44100
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/enhance/ -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 5: Implement `openvox/enhance/config.py`**

```python
from dataclasses import dataclass

@dataclass
class EnhanceConfig:
    device: str = "cuda"
    nfe: int = 64
    solver: str = "midpoint"
    lambd: float = 0.9
    tau: float = 0.5
    denoise_only: bool = False
```

- [ ] **Step 6: Implement `openvox/enhance/backend.py`**

```python
from abc import ABC, abstractmethod

import numpy as np

from openvox.tts.backend import TTSResult

class EnhanceBackend(ABC):
    @abstractmethod
    def enhance(self, audio: np.ndarray, sample_rate: int) -> TTSResult:
        """Restore/denoise mono float32 audio; returns the cleaned result."""
        raise NotImplementedError
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pip install -e ".[dev]"` then `pytest tests/enhance/ -v`
Expected: PASS (3 tests). No torch / enhance extra needed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml openvox/enhance/__init__.py openvox/enhance/config.py openvox/enhance/backend.py tests/enhance/
git commit -m "feat(enhance): scaffold openvox.enhance with EnhanceConfig and backend ABC"
```

---

## Task 2: Resemble-enhance backend

**Files:**
- Create: `openvox/enhance/resemble_backend.py`
- Test: `tests/enhance/test_resemble_backend.py`

**Interfaces:**
- Consumes: `EnhanceBackend` (Task 1), `EnhanceConfig` (Task 1), `TTSResult`.
- Produces: `_resolve_device(requested, cuda_available) -> str`; `ResembleEnhanceBackend(config: EnhanceConfig | None = None)` implementing `enhance(audio, sample_rate) -> TTSResult`, lazy-loading resemble-enhance with the shims.

- [ ] **Step 1: Write the failing tests**

`tests/enhance/test_resemble_backend.py`:
```python
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
```

- [ ] **Step 2: Run the unit tests to verify they fail**

Run: `pytest tests/enhance/test_resemble_backend.py -v -m "not integration"`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `openvox/enhance/resemble_backend.py`**

```python
import logging
import os
import sys

import numpy as np

from openvox.enhance.backend import EnhanceBackend
from openvox.enhance.config import EnhanceConfig
from openvox.tts.backend import TTSResult

log = logging.getLogger(__name__)

def _resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "cuda" and cuda_available:
        return "cuda"
    return "cpu"

def _install_shims() -> None:
    """Make resemble-enhance's inference importable/loadable without its bad pins.

    resemble-enhance's pip metadata pins an old torch and requires deepspeed
    (training-only) — the inference path merely imports the training modules. We
    install it --no-deps and, at load time: keep HF cache symlink-free on
    Windows; stub the training-only deepspeed imports; and (Windows only) map
    PosixPath -> WindowsPath so the checkpoint's embedded PosixPath deserializes.
    Verified by a feasibility spike on torch 2.6 / Windows.
    """
    from unittest.mock import MagicMock

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    for name in ("deepspeed", "deepspeed.accelerator", "deepspeed.runtime",
                 "deepspeed.runtime.engine", "deepspeed.runtime.utils"):
        sys.modules.setdefault(name, MagicMock())
    if os.name == "nt":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath

class ResembleEnhanceBackend(EnhanceBackend):
    def __init__(self, config: EnhanceConfig | None = None) -> None:
        self._cfg = config or EnhanceConfig()
        self._device = None
        self._enhance = None
        self._denoise = None

    def _ensure_loaded(self) -> None:
        if self._device is not None:
            return
        _install_shims()
        import torch
        from resemble_enhance.enhancer.inference import denoise, enhance

        self._torch = torch
        self._enhance = enhance
        self._denoise = denoise
        self._device = _resolve_device(self._cfg.device, torch.cuda.is_available())
        if self._cfg.device == "cuda" and self._device != "cuda":
            log.info("CUDA is not available to torch; enhancing on CPU (slower).")

    def enhance(self, audio: np.ndarray, sample_rate: int) -> TTSResult:
        self._ensure_loaded()
        dwav = self._torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        if self._cfg.denoise_only:
            wav, new_sr = self._denoise(dwav, sample_rate, self._device)
        else:
            wav, new_sr = self._enhance(
                dwav, sample_rate, self._device,
                nfe=self._cfg.nfe, solver=self._cfg.solver,
                lambd=self._cfg.lambd, tau=self._cfg.tau)
        out = wav.detach().cpu().numpy().astype(np.float32)
        return TTSResult(audio=out, sample_rate=int(new_sr))
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `pytest tests/enhance/test_resemble_backend.py -v -m "not integration"`
Expected: PASS (3 `_resolve_device` tests). Importing the module must not require torch/resemble.

- [ ] **Step 5: Run the integration test (installs resemble-enhance, restores real audio)**

Run:
```bash
pip install -e ".[enhance]"
pip install resemble-enhance --no-deps
pytest tests/enhance/test_resemble_backend.py -v -m integration
```
Expected: PASS. First run downloads the resemble-enhance model (one-time); CPU restoration of ~2 s takes tens of seconds. If the install fails for an environment reason you cannot resolve, report BLOCKED with the exact error — do not fake the test. (`torch`/`torchaudio` are already present from the `[clone]` extra installed earlier.)

- [ ] **Step 6: Commit**

```bash
git add openvox/enhance/resemble_backend.py tests/enhance/test_resemble_backend.py
git commit -m "feat(enhance): add resemble-enhance backend with runtime shims"
```

---

## Task 3: EnhanceEngine façade

**Files:**
- Create: `openvox/enhance/engine.py`
- Test: `tests/enhance/test_engine.py`

**Interfaces:**
- Consumes: `EnhanceConfig` (Task 1), `ResembleEnhanceBackend` (Task 2), `TTSResult`.
- Produces: `EnhanceEngine(device=None, denoise_only=None, config=None)` with `enhance(audio, sample_rate) -> TTSResult` and `enhance_file(path) -> TTSResult`. Validates input before the backend loads; copies its config.

- [ ] **Step 1: Write the failing tests**

`tests/enhance/test_engine.py`:
```python
import numpy as np
import pytest
from openvox.enhance.engine import EnhanceEngine
from openvox.tts.backend import TTSResult

class _FakeBackend:
    def __init__(self, *a, **k): pass
    def enhance(self, audio, sample_rate):
        return TTSResult(audio=np.zeros(44100, dtype=np.float32), sample_rate=44100)

@pytest.fixture
def patched(monkeypatch):
    import openvox.enhance.engine as eng
    monkeypatch.setattr(eng, "ResembleEnhanceBackend", _FakeBackend)

def test_empty_array_raises(patched):
    with pytest.raises(ValueError):
        EnhanceEngine().enhance(np.zeros(0, dtype=np.float32), 16000)

def test_missing_file_raises(patched):
    with pytest.raises(FileNotFoundError):
        EnhanceEngine().enhance_file("does_not_exist.wav")

def test_empty_file_raises(patched, tmp_path):
    p = tmp_path / "empty.wav"; p.write_bytes(b"")
    with pytest.raises(ValueError):
        EnhanceEngine().enhance_file(str(p))

def test_enhance_returns_result(patched):
    r = EnhanceEngine().enhance(np.ones(1000, dtype=np.float32), 16000)
    assert r.sample_rate == 44100 and len(r.audio) == 44100

def test_config_not_mutated(patched):
    from openvox.enhance.config import EnhanceConfig
    cfg = EnhanceConfig()
    EnhanceEngine(device="cpu", denoise_only=True, config=cfg)
    assert cfg.device == "cuda" and cfg.denoise_only is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/enhance/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `openvox/enhance/engine.py`**

```python
import dataclasses
import os

import numpy as np

from openvox.enhance.config import EnhanceConfig
from openvox.enhance.resemble_backend import ResembleEnhanceBackend
from openvox.tts.backend import TTSResult

class EnhanceEngine:
    def __init__(self, device: str | None = None, denoise_only: bool | None = None,
                 config: EnhanceConfig | None = None) -> None:
        cfg = dataclasses.replace(config) if config is not None else EnhanceConfig()
        if device is not None:
            cfg.device = device
        if denoise_only is not None:
            cfg.denoise_only = denoise_only
        self._config = cfg
        self._backend = ResembleEnhanceBackend(cfg)

    def enhance(self, audio: np.ndarray, sample_rate: int) -> TTSResult:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            raise ValueError("audio must be a non-empty array")
        return self._backend.enhance(arr, sample_rate)

    def enhance_file(self, path: str) -> TTSResult:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"audio file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"audio file is empty: {path}")
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.enhance(audio, sr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/enhance/test_engine.py -v`
Expected: PASS (5 tests, backend patched — no torch/model).

- [ ] **Step 5: Commit**

```bash
git add openvox/enhance/engine.py tests/enhance/test_engine.py
git commit -m "feat(enhance): add EnhanceEngine facade (enhance/enhance_file)"
```

---

## Task 4: Public exports + demo CLI + import guard

**Files:**
- Modify: `openvox/enhance/__init__.py`
- Create: `openvox/enhance/demo.py`
- Test: `tests/enhance/test_demo.py`, `tests/enhance/test_import_lean.py`

**Interfaces:**
- Consumes: `EnhanceEngine` (Task 3), `TTSResult`.
- Produces (exported from `openvox.enhance`): `EnhanceEngine`, `TTSResult`. Demo `main(argv=None) -> int` with `build_parser()` (flags `--in` required, `--out` required, `--device` default cuda, `--denoise-only` store_true, `--nfe` default 64).

- [ ] **Step 1: Write the failing tests**

`tests/enhance/test_demo.py`:
```python
from openvox.enhance import demo

def test_build_parser_defaults():
    a = demo.build_parser().parse_args(["--in", "a.wav", "--out", "b.wav"])
    assert a.input == "a.wav"
    assert a.output == "b.wav"
    assert a.device == "cuda"
    assert a.denoise_only is False
    assert a.nfe == 64

def test_parser_flags():
    a = demo.build_parser().parse_args(
        ["--in", "a.wav", "--out", "b.wav", "--device", "cpu", "--denoise-only", "--nfe", "32"])
    assert a.device == "cpu" and a.denoise_only is True and a.nfe == 32

def test_exports():
    from openvox.enhance import EnhanceEngine, TTSResult
    assert EnhanceEngine is not None and TTSResult is not None
```

`tests/enhance/test_import_lean.py`:
```python
import subprocess
import sys

def test_import_openvox_enhance_without_heavy_deps():
    code = (
        "import sys\n"
        "for m in ('torch', 'resemble_enhance'):\n"
        "    sys.modules[m] = None\n"
        "import openvox.enhance\n"
        "from openvox.enhance import EnhanceEngine\n"
        "eng = EnhanceEngine(device='cpu')\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/enhance/test_demo.py tests/enhance/test_import_lean.py -v`
Expected: FAIL (`AttributeError`/`ImportError`).

- [ ] **Step 3: Update `openvox/enhance/__init__.py`**

```python
"""OpenVox speech enhancement — offline denoise + restoration."""
from openvox.enhance.engine import EnhanceEngine
from openvox.tts.backend import TTSResult

__all__ = ["EnhanceEngine", "TTSResult"]
```

- [ ] **Step 4: Implement `openvox/enhance/demo.py`**

```python
import argparse
import sys

from openvox.enhance import EnhanceEngine

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openvox-enhance-demo",
                                description="Offline speech denoise + restoration.")
    p.add_argument("--in", dest="input", required=True, help="Input audio file.")
    p.add_argument("--out", dest="output", required=True, help="Output WAV path.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoise-only", action="store_true",
                   help="Denoise without full restoration/bandwidth extension.")
    p.add_argument("--nfe", type=int, default=64)
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = EnhanceEngine(device=args.device, denoise_only=args.denoise_only)
    engine._config.nfe = args.nfe
    result = engine.enhance_file(args.input)
    result.save_wav(args.output)
    print(f"Saved {args.output} ({result.duration:.1f}s @ {result.sample_rate} Hz)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/enhance/test_demo.py tests/enhance/test_import_lean.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add openvox/enhance/__init__.py openvox/enhance/demo.py tests/enhance/test_demo.py tests/enhance/test_import_lean.py
git commit -m "feat(enhance): add public exports, demo CLI, and torch-free import guard"
```

---

## Task 5: Auto-enhance the clone reference

**Files:**
- Modify: `openvox/clone/config.py`, `openvox/clone/engine.py`, `openvox/clone/demo.py`
- Test: `tests/clone/test_enhance_integration.py`

**Interfaces:**
- Consumes: `EnhanceEngine` (lazily, from `openvox.enhance`), `openvox._paths.cache_dir`.
- Produces: `CloneConfig.enhance: bool = True`; `VoiceCloneEngine.clone(text, reference_audio, exaggeration=None, cfg=None, enhance=None)`; clone demo `--enhance` (default on) / `--no-enhance`.

- [ ] **Step 1: Write the failing tests**

`tests/clone/test_enhance_integration.py`:
```python
import numpy as np
import pytest
from openvox.clone.engine import VoiceCloneEngine
from openvox.tts.backend import TTSResult

class _RecordingBackend:
    def __init__(self, *a, **k): self.ref = None
    def clone(self, text, reference_path, exaggeration, cfg):
        self.ref = reference_path
        return TTSResult(audio=np.zeros(10, dtype=np.float32), sample_rate=24000)

def _wav(path):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.zeros(16000, dtype=np.int16)).tobytes())

def test_enhance_false_uses_raw_reference(monkeypatch, tmp_path):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _RecordingBackend)
    ref = tmp_path / "ref.wav"; _wav(ref)
    e = VoiceCloneEngine()
    e.clone("hi", str(ref), enhance=False)
    assert e._backend.ref == str(ref)              # raw reference passed

def test_enhance_true_uses_cleaned_reference(monkeypatch, tmp_path):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _RecordingBackend)

    class _FakeEnhance:
        def __init__(self, *a, **k): pass
        def enhance_file(self, path):
            return TTSResult(audio=np.zeros(44100, dtype=np.float32), sample_rate=44100)
    import openvox.enhance
    monkeypatch.setattr(openvox.enhance, "EnhanceEngine", _FakeEnhance)

    ref = tmp_path / "ref.wav"; _wav(ref)
    e = VoiceCloneEngine()
    e.clone("hi", str(ref), enhance=True)
    assert e._backend.ref != str(ref)              # a cleaned (cached) path
    assert e._backend.ref.endswith(".wav")

def test_enhance_unavailable_degrades_to_raw(monkeypatch, tmp_path):
    import openvox.clone.engine as eng
    monkeypatch.setattr(eng, "ChatterboxBackend", _RecordingBackend)

    class _Boom:
        def __init__(self, *a, **k): raise RuntimeError("enhance deps missing")
    import openvox.enhance
    monkeypatch.setattr(openvox.enhance, "EnhanceEngine", _Boom)

    ref = tmp_path / "ref.wav"; _wav(ref)
    e = VoiceCloneEngine()
    e.clone("hi", str(ref), enhance=True)          # must not raise
    assert e._backend.ref == str(ref)              # degraded to raw reference
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/clone/test_enhance_integration.py -v`
Expected: FAIL (`clone()` has no `enhance` param / no auto-enhance).

- [ ] **Step 3: Add `enhance` to `CloneConfig`**

In `openvox/clone/config.py`, add the field:
```python
    enhance: bool = True
```

- [ ] **Step 4: Wire auto-enhance into `openvox/clone/engine.py`**

Add `import logging`, `import os`, `import hashlib`, and a module `log = logging.getLogger(__name__)` if not present. In `VoiceCloneEngine.__init__`, add `self._enhancer = None`. Give `clone` an `enhance` parameter and route the reference through enhancement:

```python
    def clone(self, text: str, reference_audio: str,
              exaggeration: float | None = None, cfg: float | None = None,
              enhance: bool | None = None) -> TTSResult:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(f"reference audio not found: {reference_audio}")
        if os.path.getsize(reference_audio) == 0:
            raise ValueError(f"reference audio is empty: {reference_audio}")
        do_enhance = enhance if enhance is not None else self._config.enhance
        ref_path = self._enhanced_reference(reference_audio) if do_enhance else reference_audio
        e = exaggeration if exaggeration is not None else self._config.exaggeration
        g = cfg if cfg is not None else self._config.cfg
        return self._backend.clone(text, ref_path, e, g)

    def _enhanced_reference(self, reference_audio: str) -> str:
        """Return a cached, cleaned copy of the reference; fall back to the raw
        path if enhancement is unavailable or fails (never breaks cloning)."""
        from openvox._paths import cache_dir
        st = os.stat(reference_audio)
        key = hashlib.sha1(
            f"{os.path.abspath(reference_audio)}|{st.st_mtime_ns}|{st.st_size}".encode()
        ).hexdigest()
        out = os.path.join(cache_dir("enhance/cache"), key + ".wav")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
        try:
            from openvox.enhance import EnhanceEngine
            if self._enhancer is None:
                self._enhancer = EnhanceEngine(device=self._config.device)
            self._enhancer.enhance_file(reference_audio).save_wav(out)
            return out
        except Exception as exc:
            log.info("Reference enhancement unavailable (%s); cloning from the raw "
                     "reference. Install with: pip install -e \".[enhance]\" && "
                     "pip install resemble-enhance --no-deps", exc)
            return reference_audio
```

(Keep the existing `play`/`say` methods unchanged.)

- [ ] **Step 5: Add the `--enhance/--no-enhance` flag to the clone demo**

In `openvox/clone/demo.py` `build_parser`, add a mutually-exclusive default-on toggle, and pass it through in `main`:
```python
    p.add_argument("--no-enhance", dest="enhance", action="store_false",
                   help="Do not auto-enhance the reference clip before cloning.")
    p.set_defaults(enhance=True)
```
In `main`, pass it: `engine.clone(args.text, args.ref, enhance=args.enhance)`.

- [ ] **Step 6: Run the tests + the full non-integration suite**

Run: `pytest tests/clone/test_enhance_integration.py -v` then `pytest -q -m "not integration"`
Expected: the 3 new tests pass; the full non-integration suite is green (existing clone tests still pass — a malformed/degenerate reference simply degrades to raw).

- [ ] **Step 7: Commit**

```bash
git add openvox/clone/config.py openvox/clone/engine.py openvox/clone/demo.py tests/clone/test_enhance_integration.py
git commit -m "feat(clone): auto-enhance the reference clip (cached, graceful)"
```

---

## Task 6: README — document speech enhancement

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update status table + architecture + add an Enhancement section**

- In the "Project status & roadmap" table, add a row: `| 🧼 **Speech enhancement** — denoise + restore + bandwidth-extend a poor recording | ✅ Available |`.
- In the "Architecture" section, add a bullet: `- **\`openvox.enhance\`** — the offline speech-restoration engine (available today).`
- Add a section after Voice Cloning:

````markdown
## 🧼 Speech Enhancement (available today)

Restore a poorly-recorded clip — denoise, enhance, and extend bandwidth (e.g. 16 kHz → 44.1 kHz) — fully offline (via [resemble-enhance](https://github.com/resemble-ai/resemble-enhance), MIT). The voice cloner uses it **automatically** to clean the reference clip before cloning (pass `--no-enhance` to skip).

Install is two steps (resemble-enhance ships incompatible pins, so it goes in `--no-deps`):

```bash
pip install -e ".[enhance]"
pip install resemble-enhance --no-deps
```

```bash
python -m openvox.enhance.demo --in poor.wav --out clean.wav
```

```python
from openvox.enhance import EnhanceEngine

engine = EnhanceEngine(device="cuda")            # falls back to CPU
result = engine.enhance_file("poor.wav")         # denoise + restore -> 44.1 kHz
result.save_wav("clean.wav")
```

**Demo flags:** `--in PATH` (required) · `--out PATH` (required) · `--device` (`cuda`/`cpu`) · `--denoise-only` · `--nfe` (default 64).
````

- [ ] **Step 2: Verify accuracy**

Run: `grep -n "EnhanceEngine\|openvox.enhance.demo\|\[enhance\]" README.md` and confirm each documented symbol/command exists in the code and `pyproject.toml`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document speech enhancement in README"
```

---

## Self-Review Notes (completed)

- **Spec coverage:** package layout (T1–T4) · EnhanceConfig (T1) · EnhanceBackend ABC (T1) · ResembleEnhanceBackend + `_resolve_device` + `_install_shims` (deepspeed stub / PosixPath / HF symlink) + lazy load + enhance/denoise (T2) · EnhanceEngine enhance/enhance_file + validate-before-load + config copy (T3) · exports + demo + import-guard (T4) · clone auto-enhance default-on + cache + graceful degradation + `--enhance/--no-enhance` (T5) · error handling: empty/missing input (T3), CUDA→CPU (T2), enhance-unavailable degrades in clone (T5) · testing incl. no-torch unit tests (T1/T2/T3/T4) + integration restore (T2) + clone-integration units (T5) · `[enhance]` extra out of `all` + two-step install (T1, T6) · DoD mapped · README (T6).
- **Placeholder scan:** none — every code step has runnable code.
- **Type consistency:** `TTSResult(audio, sample_rate)` reused; `EnhanceBackend.enhance(audio, sample_rate) -> TTSResult`; `_resolve_device(requested, cuda_available)`; `ResembleEnhanceBackend(config)`; `EnhanceConfig(device, nfe, solver, lambd, tau, denoise_only)`; `EnhanceEngine(device, denoise_only, config)` with `enhance`/`enhance_file`; `CloneConfig.enhance`; `VoiceCloneEngine.clone(..., enhance=None)` + `_enhanced_reference`; demo `--in/--out` -> `args.input/args.output`. Consistent, and matches the spike-verified resemble-enhance API (`enhance(dwav, sr, device, nfe, solver, lambd, tau)`, `denoise(dwav, sr, device)`, output 44100).
