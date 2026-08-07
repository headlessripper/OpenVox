# OpenVox Rename & Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the package `nectarstt` → `openvox` and restructure it into an all-in-one engine with the existing STT engine moved under `openvox/stt/`, with no behavior change.

**Architecture:** A pure move + rename + dependency reorg. The entire existing `nectarstt/` package moves to `openvox/stt/`; a tiny shared `openvox/_paths.py` provides the cache root; dependencies split into per-engine optional extras. The full existing test suite must stay green under the new layout.

**Tech Stack:** Python 3.11+, setuptools, pytest. (No new runtime libraries.)

## Global Constraints

- Package/distribution name is `openvox`; import package is `openvox`.
- The existing STT engine lives under `openvox/stt/`; public API is `from openvox.stt import STTEngine`.
- `import openvox` must NOT import faster-whisper / onnxruntime / sounddevice (lean top-level).
- No STT behavior, algorithm, model, or test-assertion changes — move/rename only.
- The `os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")` line stays verbatim and in the same position at the top of `openvox/stt/engine/faster_whisper_backend.py`.
- Cache root moves to `platformdirs.user_cache_dir("openvox")`; STT models under `.../openvox/stt/models`.
- No `nectarstt`/`NectarSTT` reference may remain in `openvox/`, `tests/`, `pyproject.toml`, or `README.md` (historical `docs/superpowers/` snapshots are exempt).
- The full suite (48 tests) must pass after the move: `pip install -e ".[stt,stt-demo,dev]"` then `pytest`.

---

## File Structure

- `openvox/__init__.py` — `__version__` only; no engine imports.
- `openvox/_paths.py` — `cache_dir(sub)` shared cache helper.
- `openvox/stt/**` — the entire former `nectarstt/**`, imports rewritten.
- `openvox/stt/models.py` — `download_root()` delegates to `_paths.cache_dir("stt/models")`.
- `pyproject.toml` — name `openvox`, per-engine extras, `openvox-stt-demo` entry point.
- `tests/stt/**` — the former `tests/**`, imports rewritten, fixture path updated.
- `README.md` — rewritten for the new name/structure/install.

---

## Task 1: Rename & move the package + tests (atomic, ends green)

This is a single atomic task: a half-done rename leaves the suite red, so the move, the import rewrite, and the test move all land together and the task ends with the full suite passing.

**Files:**
- Move: `nectarstt/` → `openvox/stt/` (all modules)
- Move: `tests/*.py` → `tests/stt/`, `tests/fixtures/` → `tests/stt/fixtures/`
- Create: `openvox/__init__.py`, `openvox/_paths.py`, `tests/stt/__init__.py`
- Modify: `openvox/stt/models.py`, `openvox/stt/demo.py`, `openvox/stt/__init__.py` (docstring), `tests/stt/test_smoke.py`, `tests/stt/test_models.py`, `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: `openvox` package with `openvox.stt.STTEngine` (and the STT public API), `openvox._paths.cache_dir(sub: str) -> str`. Entry point `openvox-stt-demo`.

- [ ] **Step 1: Move the package and tests with git (preserves history)**

```bash
mkdir -p openvox
git mv nectarstt openvox/stt
mkdir -p tests/stt
git mv tests/fixtures tests/stt/fixtures
# move every test module (they are all at tests/ top level)
for f in tests/*.py; do git mv "$f" tests/stt/; done
```

- [ ] **Step 2: Blanket-rewrite the lowercase `nectarstt` import/identifier references**

Every `from nectarstt...`, `import nectarstt`, and `monkeypatch.setattr(nectarstt, ...)` reference uses the lowercase module name and becomes `openvox.stt`:

```bash
grep -rl --include=*.py 'nectarstt' openvox tests | xargs sed -i 's/nectarstt/openvox.stt/g'
```

This is correct for all dotted imports and the `monkeypatch.setattr(openvox.stt, ...)` attribute reference in `test_engine.py`. It also (incorrectly) rewrites three spots that the next steps fix: the demo `prog` string, `models.py`'s cache name, and `test_models.py`'s assertion.

- [ ] **Step 3: Create `openvox/__init__.py` (lean top-level)**

```python
"""OpenVox — all-in-one offline voice engine."""
__version__ = "0.1.0"
```

- [ ] **Step 4: Create `openvox/_paths.py`**

```python
import os

from platformdirs import user_cache_dir

def cache_dir(sub: str = "") -> str:
    """Return a stable per-user cache directory under the OpenVox root.

    ``cache_dir("stt/models")`` -> ``<user cache>/openvox/stt/models``.
    The directory (including parents) is created if missing.
    """
    root = user_cache_dir("openvox")
    path = os.path.join(root, sub) if sub else root
    os.makedirs(path, exist_ok=True)
    return path
```

- [ ] **Step 5: Rewrite `openvox/stt/models.py`'s `download_root()` to use the shared cache**

After Step 2, `models.py` contains a broken `user_cache_dir("openvox.stt")`. Replace the top import and the `download_root` function so the file reads:

```python
from openvox._paths import cache_dir

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
    return cache_dir("stt/models")
```

(Remove the now-unused `import os` / `from platformdirs import user_cache_dir` lines if present. Keep `MODEL_ALIASES` and `resolve_model` exactly as they were.)

- [ ] **Step 6: Fix the demo entry-point prog string**

In `openvox/stt/demo.py`, Step 2 turned `prog="nectarstt-demo"` into `prog="openvox.stt-demo"`. Set it to the hyphenated entry-point name:

```python
    p = argparse.ArgumentParser(prog="openvox-stt-demo",
                                description="Live OpenVox STT transcription.")
```

- [ ] **Step 7: Fix the brand string in the STT package docstring**

In `openvox/stt/__init__.py`, change the module docstring `"""NectarSTT — offline streaming speech-to-text."""` to:

```python
"""OpenVox STT — offline streaming speech-to-text."""
```

Keep the `__version__ = "0.1.0"` line and all imports (now `from openvox.stt...`) intact.

- [ ] **Step 8: Rewrite `tests/stt/test_smoke.py`**

Step 2 left it importing `openvox.stt`; make it assert the new top-level version and the STT subpackage import explicitly:

```python
def test_package_imports():
    import openvox
    assert openvox.__version__ == "0.1.0"
    from openvox.stt import STTEngine
    assert STTEngine is not None
```

- [ ] **Step 9: Fix `tests/stt/test_models.py`'s cache assertion**

Step 2 turned `assert "nectarstt" in root.lower()` into `assert "openvox.stt" in root.lower()`, which is false (the path uses separators, not a dot). Change that one line to:

```python
    assert "openvox" in root.lower()
```

- [ ] **Step 10: Update the fixture path strings in the tests**

Two integration tests reference the fixture by literal path. Update both files:

```bash
sed -i 's#tests/fixtures/hello_world.wav#tests/stt/fixtures/hello_world.wav#g' \
    tests/stt/test_faster_whisper_backend.py tests/stt/test_engine.py
```

- [ ] **Step 11: Create `tests/stt/__init__.py`**

```bash
: > tests/stt/__init__.py
```

- [ ] **Step 12: Rewrite `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "openvox"
version = "0.1.0"
description = "All-in-one, fully offline voice engine (speech-to-text; text-to-speech next)."
requires-python = ">=3.11"
dependencies = ["numpy>=1.24", "platformdirs>=4.0"]

[project.optional-dependencies]
stt      = ["faster-whisper>=1.0.0", "pysilero-vad>=2.0.0", "sounddevice>=0.4.6", "huggingface_hub>=0.23"]
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

- [ ] **Step 13: Reinstall and run the full suite**

Run:
```bash
pip install -e ".[stt,stt-demo,dev]"
pytest -q
```
Expected: `48 passed` (39 non-integration + 9 integration). If any test still references `nectarstt`, fix the import to `openvox.stt` and re-run.

- [ ] **Step 14: Verify the lean top-level import and the grep gate**

Run:
```bash
python -c "import sys, openvox; assert 'faster_whisper' not in sys.modules and 'onnxruntime' not in sys.modules; print('lean import OK', openvox.__version__)"
grep -rin "nectarstt" openvox tests pyproject.toml && echo "FAIL: nectarstt refs remain" || echo "grep gate OK (no nectarstt refs)"
```
Expected: `lean import OK 0.1.0` and `grep gate OK (no nectarstt refs)`.

- [ ] **Step 15: Commit**

```bash
git add -A
git commit -m "refactor: rename nectarstt -> openvox, move STT engine to openvox/stt"
```

---

## Task 2: README refresh + final verification

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Rewrite the install / usage / examples for OpenVox**

Update `README.md` so every command and import matches the new package. Replace the Installation and Usage sections' commands/imports:

- Install: `pip install -e ".[stt,stt-demo]"` (add `dev` for tests: `.[stt,stt-demo,dev]`).
- Demo: `python -m openvox.stt.demo` (and the `--full` / `--file` examples unchanged in behavior).
- Code example import: `from openvox.stt import STTEngine`.
- Batch example import: `from openvox.stt import STTEngine`.

Reframe the intro/title to OpenVox (all-in-one offline voice engine; STT today, TTS next). Keep the `docs/superpowers/specs/` and `plans/` roadmap links. Preserve the existing feature bullets (streaming, sliding-window, any-format demo, model sizes) — only the name/commands change.

- [ ] **Step 2: Grep gate on the README**

Run:
```bash
grep -in "nectarstt" README.md && echo "FAIL: nectarstt refs remain in README" || echo "README grep gate OK"
```
Expected: `README grep gate OK`.

- [ ] **Step 3: Final full-suite verification**

Run:
```bash
pytest -q
```
Expected: `48 passed`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README for the OpenVox rename"
```

---

## Self-Review Notes (completed)

- **Spec coverage:** rename package (T1 S1-S3, S12) · move STT to openvox/stt (T1 S1-S2) · shared `_paths.py` (T1 S4) + models delegation (T1 S5) · lean top-level `__init__` (T1 S3, verified S14) · per-engine extras + `openvox-stt-demo` entry point (T1 S6, S12) · tests moved + re-imported + fixture paths (T1 S1-S2, S8-S11) · behavior unchanged incl. HF_HUB line (Global Constraints; no logic edits) · README rewrite (T2 S1) · grep gates for no-nectarstt-refs (T1 S14, T2 S2) · full suite green (T1 S13, T2 S3) · DoD all mapped.
- **Placeholder scan:** none — every step is a concrete command or exact file content.
- **Type/name consistency:** `cache_dir(sub)`, `download_root()`, `resolve_model()`, `MODEL_ALIASES`, `openvox.stt.STTEngine`, `openvox.__version__`, entry point `openvox-stt-demo`, extras `stt`/`stt-demo`/`dev`/`all` are used consistently across tasks and match the spec.
- **Known post-sed exceptions handled:** demo `prog` string (S6), `models.py` cache name (S5), `test_models.py` assertion (S9), capitalized brand docstring (S7), `test_smoke.py` version target (S8), fixture path strings (S10).
