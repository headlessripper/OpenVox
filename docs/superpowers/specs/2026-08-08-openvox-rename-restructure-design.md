# OpenVox — Rename & Restructure (nectarstt → openvox) — Design

**Date:** 2026-08-08
**Status:** Approved (design), pending spec review
**Scope:** Prerequisite refactor before the TTS engine. Renames the project
`nectarstt` → `openvox` and restructures it into an all-in-one engine with
`openvox/stt/` (the existing engine, moved) and `openvox/tts/` (to be built
next). No behavior changes — this is a move + rename + dependency reorg.

---

## 1. Motivation

NectarSTT is becoming an all-in-one offline voice engine (STT + TTS + later
voice cloning / streaming). The package is renamed to **OpenVox** and
restructured so the two engines live side by side under one package while each
stays independently importable and independently installable (lean for edge /
robot deployment). This must happen *before* the TTS engine is built so TTS
lands in the final structure rather than being renamed later.

## 2. Decisions locked during brainstorming

| Decision | Choice |
| --- | --- |
| Package/distribution name | `nectarstt` → `openvox` |
| Structure | `openvox/stt/` (move existing) + `openvox/tts/` (next sub-project) |
| Public API shape | Modular submodule imports: `from openvox.stt import STTEngine` |
| Dependency model | Per-engine optional extras: `openvox[stt]`, `openvox[stt-demo]`, `openvox[all]` |
| Shared layer | `openvox/_paths.py` (single cache root under `openvox/`) |
| Behavior | Byte-for-byte identical — move/rename only, no logic changes |

## 3. Scope

### In scope

- Rename the import/distribution package `nectarstt` → `openvox`.
- Move the entire existing STT package into `openvox/stt/`.
- Extract a shared `openvox/_paths.py` (cache dir helper) used by STT models
  (and later TTS).
- Reorganize dependencies into per-engine optional extras.
- Rename the console entry point `nectarstt-demo` → `openvox-stt-demo`.
- Move and re-import the test suite under `tests/stt/`; keep it green.
- Rewrite the README for the new name/structure/install.

### Out of scope (explicitly deferred)

- Building the TTS engine (`openvox/tts/`) — the next sub-project. `openvox/tts/`
  is NOT created in this refactor; it is added when TTS is built.
- Any change to STT behavior, algorithms, model choices, or test assertions.
- Rewriting the historical design/plan docs under `docs/superpowers/` — those
  are dated snapshots and keep their original `nectarstt` naming.
- Publishing to PyPI.

## 4. Target structure

```
openvox/
  __init__.py               # __version__ only — no heavy engine imports
  _paths.py                 # cache_dir(sub) -> user_cache_dir("openvox")/sub
  stt/
    __init__.py             # exports STTEngine, Config, PartialResult,
                            #   FinalResult, WordTiming, FileSource,
                            #   MicSource, ArraySource
    config.py
    events.py
    models.py               # download_root() delegates to _paths.cache_dir("stt/models")
    audio/
      __init__.py
      sources.py
      vad.py
    engine/
      __init__.py
      backend.py
      faster_whisper_backend.py
      local_agreement.py
      transcriber.py
    demo.py                 # python -m openvox.stt.demo
  # (openvox/tts/ is added in the next sub-project, not here)
tests/
  stt/
    __init__.py
    fixtures/hello_world.wav
    test_smoke.py test_events.py test_local_agreement.py test_config.py
    test_models.py test_backend_interface.py test_faster_whisper_backend.py
    test_sources.py test_vad.py test_transcriber.py test_engine.py test_demo.py
pyproject.toml
```

## 5. Mechanical changes

### 5.1 Module moves & import rewrites

- Move every module from `nectarstt/` to `openvox/stt/` preserving relative
  layout. The current `nectarstt/__init__.py` (the `STTEngine` façade) becomes
  `openvox/stt/__init__.py`.
- Rewrite every internal import `from nectarstt.X` / `import nectarstt.X` to
  `from openvox.stt.X` / `import openvox.stt.X`.
- Behavior, docstrings, and logic stay identical. The
  `os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")` line at the top of
  `faster_whisper_backend.py` is preserved verbatim and in the same position.

### 5.2 Shared paths helper

Create `openvox/_paths.py`:

```python
import os
from platformdirs import user_cache_dir

def cache_dir(sub: str = "") -> str:
    root = user_cache_dir("openvox")
    path = os.path.join(root, sub) if sub else root
    os.makedirs(path, exist_ok=True)
    return path
```

`openvox/stt/models.py`'s `download_root()` becomes:

```python
from openvox._paths import cache_dir

def download_root() -> str:
    return cache_dir("stt/models")
```

`resolve_model()` and `MODEL_ALIASES` are unchanged. The cache location moves
from `user_cache_dir("nectarstt")/models` to `user_cache_dir("openvox")/stt/
models`; STT models re-download once into the new path (one-time, harmless).

### 5.3 Top-level package init

`openvox/__init__.py`:

```python
"""OpenVox — all-in-one offline voice engine."""
__version__ = "0.1.0"
```

No engine imports here, so `import openvox` pulls nothing heavy; engines are
reached via `openvox.stt` (and later `openvox.tts`).

### 5.4 pyproject.toml

```toml
[project]
name = "openvox"
version = "0.1.0"
description = "All-in-one, fully offline voice engine (speech-to-text; text-to-speech next)."
requires-python = ">=3.11"
dependencies = ["numpy>=1.24", "platformdirs>=4.0"]

[project.optional-dependencies]
stt      = ["faster-whisper>=1.0.0", "pysilero-vad>=2.0.0",
            "sounddevice>=0.4.6", "huggingface_hub>=0.23"]
stt-demo = ["av>=11"]
dev      = ["pytest>=8.0"]
all      = ["openvox[stt,stt-demo]"]

[project.scripts]
openvox-stt-demo = "openvox.stt.demo:main"

[tool.setuptools.packages.find]
include = ["openvox*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: needs model download / real audio backend"]
```

Note: `numpy` moves to the shared base (both engines need it); the remaining
STT runtime deps move to the `stt` extra. The old `demo` extra is renamed
`stt-demo`.

### 5.5 Tests

- Move `tests/*.py` → `tests/stt/`; add `tests/stt/__init__.py`.
- Move `tests/fixtures/hello_world.wav` → `tests/stt/fixtures/hello_world.wav`.
- Rewrite imports `nectarstt` → `openvox.stt` in every test.
- Update fixture path string references from `"tests/fixtures/hello_world.wav"`
  to `"tests/stt/fixtures/hello_world.wav"` (used in `test_faster_whisper_backend.py`
  and `test_engine.py`).
- Test assertions and logic otherwise unchanged.

### 5.6 README

Rewrite install (`pip install -e ".[stt,stt-demo]"`, add `dev` for tests),
usage (`python -m openvox.stt.demo`), and code examples
(`from openvox.stt import STTEngine`). Update the roadmap section wording to
reflect the all-in-one OpenVox framing. The `docs/superpowers/` links stay.

## 6. Error handling

No new error paths. Existing behavior (CUDA→CPU fallback, `AudioDeviceError`,
`ValueError` on unknown model / malformed WAV) is preserved unchanged.

## 7. Testing

- The move is verified by the existing suite: `pip install -e ".[stt,stt-demo,dev]"`
  then `pytest` must show all 48 tests passing under the new layout (39
  non-integration + integration).
- A grep gate: no `nectarstt` identifier remains in `openvox/`, `tests/`,
  `pyproject.toml`, or `README.md` (historical `docs/superpowers/` snapshots
  excepted).
- `python -c "import openvox"` must succeed without importing faster-whisper /
  onnxruntime (lean top-level import).

## 8. Definition of done

1. `from openvox.stt import STTEngine` works; `import openvox` pulls nothing heavy.
2. `python -m openvox.stt.demo --file …` works exactly as before (any-format via
   `stt-demo` extra; clean `✓`-only output; `--full` for partials).
3. `pip install -e ".[stt,stt-demo,dev]"` then `pytest` → all 48 tests green.
4. No `nectarstt` references remain in code, tests, `pyproject.toml`, or README
   (historical docs excepted); `openvox-stt-demo` entry point resolves.
