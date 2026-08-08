# OpenVox — Speech Enhancement Engine (Sub-project 2B.2a) — Design

**Date:** 2026-08-08
**Status:** Approved (design), proceeding to implementation plan
**Scope:** On-the-fly speech restoration — denoise, restore, and bandwidth-extend
a poorly-recorded clip — as a standalone `openvox.enhance` engine that the voice
cloner uses automatically to clean its reference. Per-voice fine-tuning (2B.2b)
is a separate later sub-project.

---

## 1. Goal & positioning

Voice clones are only as good as the reference clip. This engine cleans a poor
reference — removing noise, restoring detail, and extending bandwidth (e.g.
16 kHz → 44.1 kHz) — on the fly, so a clone made from a phone recording sounds
far closer to a studio one. It's a general, reusable speech-restoration
capability (`openvox.enhance`) that the cloner applies automatically, and that
other engines could use later.

## 2. Decisions locked during brainstorming

| Decision | Choice |
| --- | --- |
| First mechanism | Reference-audio enhancement (fine-tuning = 2B.2b, later) |
| Model | **resemble-enhance** (Resemble AI) — MIT license (verified) |
| Feasibility | Runs on the existing torch 2.6 without breaking Chatterbox, via runtime shims (proven by spike) |
| Structure | Standalone `openvox/enhance/` engine, consumed by the cloner |
| Clone integration | Auto-enhance the reference by default; toggle + cache; graceful degradation |
| Packaging | Documented two-step install (`[enhance]` light deps + `resemble-enhance --no-deps`) |
| Hardware (day 1) | Windows x86 + CUDA (torch), CPU fallback |

**License verified:** resemble-enhance is MIT (code + weights). **Feasibility
spike result:** its inference runs on torch 2.6 / Windows with these runtime
shims (all applied in our backend, none touching the working env): install
`--no-deps`; stub the training-only `deepspeed` import; add light deps
(`matplotlib`, `omegaconf`, `pandas`, `celluloid`, `resampy`, `soundfile`);
patch `pathlib.PosixPath = WindowsPath` so the checkpoint deserializes on
Windows. Verified it denoised + restored a 16 kHz clip to 44.1 kHz. Chatterbox
remained intact (torch unchanged at 2.6.0).

## 3. Scope

### In scope

- `openvox/enhance/` with an `EnhanceEngine` façade.
- resemble-enhance behind an `EnhanceBackend` ABC, with the proven shims.
- `enhance(audio, sample_rate)` / `enhance_file(path) -> TTSResult`.
- Automatic reference enhancement in `VoiceCloneEngine.clone()` (default on,
  toggle, cache, graceful degradation).
- A demo CLI (`python -m openvox.enhance.demo`) and a `--enhance/--no-enhance`
  flag on the clone demo.
- A `[enhance]` optional extra + documented two-step install.
- Tests (unit without torch/resemble; one integration that really restores).

### Out of scope (deferred)

- Per-voice fine-tuning (2B.2b).
- Vendoring resemble-enhance's inference (a possible later packaging cleanup;
  the two-step `--no-deps` install is used now).
- De-reverb/de-clip beyond what resemble-enhance already does; custom DSP
  stages.
- Enhancing STT input or TTS output (the engine is reusable for it later, but
  only the clone integration is built now).

## 4. Architecture

### Package layout

```
openvox/enhance/
  __init__.py           # exports EnhanceEngine, TTSResult
  config.py             # EnhanceConfig (device, nfe, solver, lambd, tau, denoise_only)
  backend.py            # EnhanceBackend ABC
  resemble_backend.py   # ResembleEnhanceBackend — the ONLY file importing resemble_enhance/torch
  engine.py             # EnhanceEngine: enhance / enhance_file
  demo.py               # python -m openvox.enhance.demo
tests/enhance/
```

### Components (each independently testable)

- **EnhanceBackend (ABC)** + **ResembleEnhanceBackend** — `enhance(audio:
  np.ndarray, sample_rate: int) -> TTSResult`. Only `resemble_backend.py`
  imports `resemble_enhance`/`torch`, lazily inside the model-load method,
  after applying the shims. A pure `_resolve_device(requested, cuda_available)`
  helper. Reuses `TTSResult` from `openvox.tts.backend` (audio float32 +
  `sample_rate` + `save_wav`; restoration output is 44.1 kHz).
- **EnhanceEngine** — façade: `enhance(audio, sample_rate)`,
  `enhance_file(path)`; validates input before the lazy load; copies its
  config.
- **Clone integration** (`openvox/clone/engine.py`): `clone()` gains an
  `enhance` flag (default from `CloneConfig.enhance=True`). When on, it lazily
  builds an `EnhanceEngine`, cleans the reference, caches the cleaned WAV under
  the shared `openvox` cache (keyed by the reference's abspath + mtime + size),
  and passes the cleaned path to Chatterbox. If the enhance deps are missing (or
  enhancement raises), it logs a one-line notice and proceeds with the raw
  reference. The `openvox.enhance` import stays lazy inside `clone()`, so
  `import openvox.clone` remains torch-free (the existing guard test holds).

`import openvox` and `import openvox.enhance` stay lean (no torch/resemble at
import).

## 5. Public API

```python
from openvox.enhance import EnhanceEngine

eng = EnhanceEngine(device="cuda")              # falls back to CPU
result = eng.enhance_file("poor.wav")           # -> TTSResult (float32, 44100 Hz)
result.save_wav("clean.wav")
result = eng.enhance(audio_np, sample_rate=16000)
EnhanceEngine(denoise_only=True).enhance_file("noisy.wav")   # faster, denoise-only
```

```python
from openvox.clone import VoiceCloneEngine

eng = VoiceCloneEngine(device="cuda")
eng.clone("Hello.", reference_audio="poor.wav")                 # auto-cleans the reference
eng.clone("Hello.", reference_audio="poor.wav", enhance=False)  # raw reference
```

## 6. Data flow

### Enhancement

```
audio (array or file) ─▶ ResembleEnhanceBackend
   1. lazy shims: HF_HUB_DISABLE_SYMLINKS; stub deepspeed(+submodules) via MagicMock;
      pathlib.PosixPath = pathlib.WindowsPath
   2. import torch + resemble_enhance.enhancer.inference; resolve device; load model (cached)
   3. denoise_only ? denoise(dwav, sr, device) : enhance(dwav, sr, device, nfe, solver, lambd, tau)
      -> (wav @ 44100, sr)
   ▼
TTSResult(audio float32, 44100)  ->  save_wav
```

`dwav` is a mono float32 torch tensor; `enhance`/`denoise` come from
`resemble_enhance.enhancer.inference` and return `(waveform, sample_rate)`.

### Clone auto-enhance

```
clone(text, ref, enhance=True):
   cleaned = cache lookup by (abspath + mtime + size) under openvox cache
   if miss and enhance available:
       cleaned = EnhanceEngine.enhance_file(ref) -> save_wav(cache_path)
   if enhance unavailable / errors:
       log one-line notice; cleaned = ref (raw)          # graceful, never breaks
   ChatterboxBackend.clone(text, cleaned_or_raw_ref, exaggeration, cfg)
```

## 7. Configuration

`EnhanceConfig`: `device="cuda"`, `nfe=64`, `solver="midpoint"`, `lambd=0.9`,
`tau=0.5`, `denoise_only=False`. `EnhanceEngine(device=…, denoise_only=…,
config=…)` overrides fields when provided; a passed config is copied, not
mutated. `CloneConfig` gains `enhance: bool = True`.

## 8. Error handling

- `device="cuda"` but no CUDA in torch → CPU with an info notice
  (`_resolve_device`).
- resemble-enhance not installed → `EnhanceEngine` raises a clear error naming
  the two-step install; the clone path catches this (or any enhancement error)
  and degrades to the raw reference with a one-line notice.
- Missing/empty input file, or an empty/zero-length array → `FileNotFoundError`
  / `ValueError` before any model load.
- Shims applied in `resemble_backend` before importing `resemble_enhance`.

## 9. Testing

- **Unit (no torch/resemble):** `EnhanceConfig` defaults; `_resolve_device`
  (cuda→cuda when available, cuda→cpu when not, cpu→cpu); `enhance_file` on a
  missing/empty file → error; an import-guard subprocess test (`import
  openvox.enhance` + construct `EnhanceEngine` with `torch`/`resemble_enhance`
  blocked → succeeds).
- **Clone-integration unit (patched, no torch):** `clone(enhance=True)` builds
  the enhance engine and passes the cleaned reference; `clone(enhance=False)`
  does not enhance; if the enhance engine raises `ImportError`, `clone` proceeds
  with the raw reference (graceful degradation asserted).
- **Integration (heavy, real):** `ResembleEnhanceBackend.enhance` on a 16 kHz
  clip → `TTSResult` float32, `sample_rate == 44100`, length greater than the
  input (upsampled), non-silent (RMS above a floor).
- Demo `build_parser` unit tests (enhance demo + the clone demo's
  `--enhance/--no-enhance`).
- `play`/`say` unaffected; manual checks only.

## 10. Dependencies

New `[enhance]` extra = `matplotlib`, `omegaconf`, `pandas`, `celluloid`,
`resampy`, `soundfile`, `rich`, `tabulate` (torch/torchaudio come from the
`[clone]` side; resemble-enhance uses `torchaudio.load`). Documented two-step:
`pip install -e ".[enhance]"` then `pip install resemble-enhance --no-deps`.
`[enhance]` stays **out of `all`**. Demo entry point `openvox-enhance-demo`.

## 11. Demo CLI

`python -m openvox.enhance.demo` (entry point `openvox-enhance-demo`): `--in
PATH` (required), `--out PATH` (required), `--device` (default cuda),
`--denoise-only`, `--nfe` (default 64). Restores `--in` and writes `--out`.
The clone demo gains `--enhance` (default on) / `--no-enhance`.

## 12. Definition of done

1. `from openvox.enhance import EnhanceEngine; enhance_file(poor)` → a cleaned
   44.1 kHz `TTSResult`, fully offline (after a one-time model download).
2. `python -m openvox.enhance.demo --in poor.wav --out clean.wav` restores it.
3. `VoiceCloneEngine.clone()` auto-enhances the reference by default (cached),
   `--no-enhance`/`enhance=False` skips, and it degrades gracefully when the
   enhance deps are absent.
4. Unit tests green with no torch/resemble; integration verifies real
   enhancement; `import openvox` / `import openvox.enhance` stay lean.
