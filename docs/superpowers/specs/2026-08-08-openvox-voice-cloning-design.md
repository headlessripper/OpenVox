# OpenVox — Voice Cloning Engine (Sub-project 2B: Zero-shot instant cloning) — Design

**Date:** 2026-08-08
**Status:** Approved (design), proceeding to implementation plan
**Scope:** Zero-shot "instant" voice cloning — speak arbitrary text in a voice
taken from a short reference clip, no training, fully offline. Fine-tuned
"professional" cloning and a voice-profile library are later sub-projects.

---

## 1. Goal & positioning

OpenVox should clone a voice from a short reference clip and speak any text in
that voice — ElevenLabs' "Instant Voice Clone", but 100% local. This is the
headline differentiator for the "surpass the cloud" horizon: a private,
on-device voice clone with no upload. It follows the same façade + swappable
backend + optional-extra shape as the STT and TTS engines.

## 2. Decisions locked during brainstorming

| Decision | Choice |
| --- | --- |
| Cloning mode | Zero-shot / instant (reference clip → speak any text, no training) |
| Model | **Chatterbox** (Resemble AI) — MIT license (code + weights), commercial-OK |
| Runtime | PyTorch (chatterbox-tts); isolated in a new `[clone]` extra |
| Hardware (day 1) | Windows x86 + CUDA (torch), with CPU fallback |
| Language | English-first |
| Deliverable | `VoiceCloneEngine` (clone + save_wav + play) + demo CLI |
| Placement | `openvox/clone/` subpackage; not imported by top-level |

**License verified:** Chatterbox is MIT (code and the ResembleAI/chatterbox
model weights). Every generated clip carries Resemble's imperceptible "Perth"
neural watermark — always on; we disclose it as a responsible-cloning feature.

## 3. Scope

### In scope

- `openvox/clone/` with a `VoiceCloneEngine` façade.
- Zero-shot cloning via Chatterbox behind a `CloneBackend` ABC.
- `clone(text, reference_audio, exaggeration, cfg) -> TTSResult`,
  `save_wav`/`play`/`say`.
- A demo CLI (`python -m openvox.clone.demo`).
- A `[clone]` optional extra (chatterbox-tts + sounddevice).
- Tests (unit without torch/model; one integration that really clones).

### Out of scope (deferred)

- Fine-tuned / "professional" cloning and any training pipeline (2B.2).
- A saved voice-profile library / named cloned voices (later).
- Non-English cloning (Chatterbox multilingual variant) — later.
- Removing or toggling the Chatterbox watermark (it is intrinsic; not touched).

## 4. Architecture

### Package layout

```
openvox/clone/
  __init__.py            # exports VoiceCloneEngine, TTSResult
  config.py              # CloneConfig (device, exaggeration, cfg)
  backend.py             # CloneBackend ABC
  chatterbox_backend.py  # ChatterboxBackend — the only file importing chatterbox/torch
  engine.py              # VoiceCloneEngine: clone / save_wav / play / say
  demo.py                # python -m openvox.clone.demo
tests/clone/
```

### Components (each independently testable)

- **CloneBackend (ABC)** + **ChatterboxBackend** — `clone(text,
  reference_path, exaggeration, cfg) -> TTSResult`. Only
  `chatterbox_backend.py` imports `chatterbox`/`torch`, and it does so lazily
  (inside the model-load method), so `import openvox.clone` needs neither torch
  nor the `[clone]` extra. The ABC is the seam for a future cloning model
  (e.g. OpenVoice).
- **VoiceCloneEngine** — the façade: `clone(text, reference_audio,
  exaggeration=None, cfg=None)`, `save_wav` (via the result), `play`, `say`.
  Reuses `TTSResult` and `AudioDeviceError` from `openvox.tts.backend` (a
  lightweight, torch-free import — DRY rather than a duplicate result type).
- **Reference audio** — a file path passed straight to Chatterbox
  (`audio_prompt_path`), which decodes common formats (wav/mp3/flac) via
  librosa. The engine only validates the file exists and is non-empty.

### Lazy loading

`ChatterboxTTS` is heavy, so it loads on the **first `clone()`** call, cached
on the engine — not in the constructor. Input validation (empty text, missing
reference) happens in `VoiceCloneEngine.clone()` *before* the load, so those
paths are testable with no torch and no download, and engine construction is
cheap.

## 5. Public API

```python
from openvox.clone import VoiceCloneEngine

engine = VoiceCloneEngine(device="cuda")   # falls back to CPU if torch has no CUDA

result = engine.clone("Hello, this is my cloned voice.", reference_audio="myvoice.mp3")
result.save_wav("cloned.wav")
engine.play(result)

engine.say("Nice to meet you.", reference_audio="myvoice.mp3")   # clone + speak

engine.clone(text, reference_audio="ref.wav", exaggeration=0.7, cfg=0.3)  # expressiveness knobs
```

## 6. Data flow

```
text + reference_path ──▶ ChatterboxBackend
   1. lazy-load ChatterboxTTS.from_pretrained(device)   [once, cached]
      device = "cuda" only if torch.cuda.is_available(), else "cpu"
   2. model.generate(text, audio_prompt_path=reference_path,
                     exaggeration=…, cfg_weight=…)  ->  torch tensor (1, N) float32
   3. audio = tensor.squeeze(0).cpu().numpy().astype(float32);  sr = int(model.sr)  # 24000
   4. -> TTSResult(audio, sr)
   ▼
TTSResult  ──►  save_wav(path)  |  engine.play(result)
```

Note the public knob `cfg` maps to Chatterbox's `cfg_weight`; `exaggeration`
passes through unchanged. The sample rate is read from `model.sr` (24000), not
hardcoded. The returned waveform already contains the Perth watermark.

## 7. Configuration

`CloneConfig` dataclass: `device="cuda"`, `exaggeration=0.5`, `cfg=0.5`.
`VoiceCloneEngine(device=…, exaggeration=…, cfg=…, config=…)` overrides fields
when provided (a passed config is copied, not mutated).

## 8. Error handling

- `device="cuda"` but `torch.cuda.is_available()` is False → resolve to CPU
  with an info-level log; do not crash. (Pure `_resolve_device(requested,
  cuda_available)` helper.)
- Empty/whitespace text → `ValueError` (before load).
- Reference file missing or empty → `FileNotFoundError`/`ValueError` (before
  load).
- Model download failure → actionable message. No output device on `play()` →
  `AudioDeviceError`.
- `HF_HUB_DISABLE_SYMLINKS=1` set before the chatterbox import (Windows
  symlink fix, reusing the STT lesson).

## 9. Testing

- **Unit (no torch/model):** `CloneConfig` defaults; `_resolve_device`
  (cuda→cuda when available, cuda→cpu when not, cpu→cpu); `clone("")` →
  `ValueError`; `clone(text, "missing.wav")` → error (validation precedes the
  lazy load); demo `build_parser` defaults/flags.
- **Integration (torch + Chatterbox download, GPU/CPU):** `clone("hello
  world", reference_audio="tests/stt/fixtures/hello_world.wav")` → `TTSResult`
  float32, `sample_rate == 24000`, length > 0, non-silent (RMS above a floor).
  Marked `integration`.
- `play()`/`say()` validated manually (needs speakers), not in CI.

## 10. Dependencies

New `[clone]` extra: `chatterbox-tts` (pulls torch, torchaudio, transformers,
librosa), `sounddevice`. `numpy` is in the shared base. **GPU** is a documented
separate step (torch's CUDA wheels):
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`.
`[clone]` stays **out of `all`** (torch is heavy — cloning is opt-in). Demo
entry point `openvox-clone-demo`.

## 11. Demo CLI

`python -m openvox.clone.demo` (entry point `openvox-clone-demo`):
`--text` (required), `--ref PATH` (required, any format), `--exaggeration`
(default 0.5), `--cfg` (default 0.5), `--device` (default cuda), `--out PATH`
(save a WAV), `--no-play` (skip playback). Default: clone `--text` in the
`--ref` voice and speak it aloud.

## 12. Definition of done

1. `from openvox.clone import VoiceCloneEngine; engine.clone(text,
   reference_audio)` speaks the text in the reference voice, fully offline
   (after a one-time model download).
2. `python -m openvox.clone.demo --text "…" --ref clip.wav` plays it; `--out`
   saves a 24 kHz WAV.
3. Validation + CUDA→CPU fallback work; unit tests green with no torch/model;
   integration verifies real cloning.
4. `import openvox` / `import openvox.clone` stay lean; cloning needs no STT/TTS
   model import.
