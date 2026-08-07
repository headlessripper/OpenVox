# OpenVox — TTS Engine (Sub-project 2A: Naturalness) — Design

**Date:** 2026-08-08
**Status:** Approved (design), proceeding to implementation plan
**Scope:** The first text-to-speech deliverable: high-quality, natural,
built-in voices, fully offline. Voice cloning (2B) and ultra-low-latency
streaming (2C) are later sub-projects.

---

## 1. Goal & positioning

OpenVox's TTS engine must sound **genuinely human — a decisive step past
classic robotic offline voices (Piper/eSpeak)** — while running fully
offline. This first deliverable proves that quality with a set of built-in
voices and batch synthesis + playback. It mirrors the STT engine's shape
(façade + swappable backend + shared cache), so the two halves of OpenVox
feel like one system.

## 2. Decisions locked during brainstorming

| Decision | Choice |
| --- | --- |
| First capability | Naturalness / voice quality (cloning + streaming later) |
| Model | Integrate **Kokoro** (82M, Apache-2.0) behind a swappable `TTSBackend` ABC |
| Runtime | Torch-free **ONNX** (`kokoro-onnx` + misaki + espeakng-loader; no system install) |
| Hardware target (day 1) | Windows x86 + CUDA, with CPU fallback |
| Language | English-first |
| Deliverable | Batch synthesis library + demo CLI, **with playback** |
| Placement | `openvox/tts/` subpackage; not imported by top-level; new `[tts]` extra |

## 3. Scope

### In scope

- `openvox/tts/` subpackage with a `TTSEngine` façade parallel to `STTEngine`.
- Kokoro synthesis via `kokoro-onnx` behind a `TTSBackend` ABC.
- `synthesize(text, voice, speed) -> TTSResult`, `TTSResult.save_wav(path)`,
  `TTSEngine.play(result)`, `TTSEngine.say(text)`, `TTSEngine.voices()`.
- Model + voices asset download to the shared `openvox` cache; offline after.
- A demo CLI (`python -m openvox.tts.demo`) that synthesizes and speaks.
- A `[tts]` optional dependency extra; `all` extended to include it.
- Tests (unit + integration) that run without speakers.

### Out of scope (deferred)

- Voice cloning (Sub-project 2B) and streaming/low-latency synthesis (2C).
- TOML/env config loading for TTS (a plain `TTSConfig` dataclass suffices now).
- Non-English UX, SSML, prosody controls beyond `speed`.
- A torch-based Kokoro backend (the ABC leaves room; not built now).

## 4. Architecture

### Package layout

```
openvox/tts/
  __init__.py            # exports TTSEngine, TTSResult
  config.py              # TTSConfig (device, voice, speed, sample_rate=24000)
  backend.py             # TTSBackend ABC + TTSResult
  kokoro_backend.py      # KokoroBackend — kokoro-onnx, CUDA->CPU provider fallback
  models.py              # asset download to openvox._paths.cache_dir("tts/models");
                         #   KOKORO_VOICES registry; voices(); validate_voice()
  engine.py              # TTSEngine façade: synthesize / save_wav / play / say / voices
  demo.py                # python -m openvox.tts.demo
tests/tts/
```

### Component responsibilities (each independently testable)

- **TTSBackend (ABC)** + **KokoroBackend** — `synthesize(text, voice, speed)
  -> TTSResult`. Only `kokoro_backend.py` imports `kokoro_onnx`. The ABC is
  the swap point for a future StyleTTS2 / custom backend.
- **TTSResult** — `audio: np.ndarray` (float32, mono, 24000 Hz),
  `sample_rate: int`, a `duration` property, and `save_wav(path)`
  (self-contained; writes a 24 kHz mono 16-bit WAV, no engine required).
- **models.py** — downloads the Kokoro ONNX model + voices file to
  `openvox._paths.cache_dir("tts/models")` via `huggingface_hub`; exposes a
  static `KOKORO_VOICES` registry, `voices() -> list[str]`, and
  `validate_voice(name)` (raises `ValueError` without loading the model).
- **TTSConfig** — dataclass: `device="cuda"`, `voice="af_heart"`,
  `speed=1.0`, `sample_rate=24000`.
- **TTSEngine** — the façade: builds the backend, `synthesize()`,
  `play(result)`, `say(text)`, `voices()`. Validates the voice before
  synthesis.

`import openvox` stays lean; TTS is reached via `openvox.tts` and pulls only
`[tts]` dependencies.

## 5. Public API

```python
from openvox.tts import TTSEngine

engine = TTSEngine(voice="af_heart", device="cuda")   # device auto-falls back to CPU

result = engine.synthesize("Hello, I am OpenVox.")     # TTSResult(audio float32, sample_rate 24000)
result.save_wav("out.wav")
engine.play(result)                                    # speak (blocking)
engine.say("This runs entirely offline.")              # synthesize + play

engine.voices()                                        # -> ["af_heart", "af_bella", "am_michael", ...]
engine.synthesize("Faster voice.", voice="am_michael", speed=1.15)
```

## 6. Data flow

```
text ──▶ KokoroBackend
           1. phonemize (misaki + espeakng-loader; g2p, pip-only, no system install)
           2. Kokoro ONNX inference with the selected voice embedding
              (onnxruntime; CUDA provider -> CPU provider fallback)
           3. 24 kHz mono float32 waveform
           ▼
        TTSResult(audio, 24000)
           ├── save_wav(path)        -> 24 kHz mono 16-bit WAV
           └── engine.play(result)   -> sounddevice output stream (blocking)
```

`TTSEngine.synthesize()` validates the voice (unknown -> `ValueError` listing
available voices), calls the backend, returns a `TTSResult`. `play()` streams
the array to the default output device via `sounddevice`. `say()` is
`play(synthesize(...))`.

## 7. Error handling

- CUDA execution provider unavailable -> warn, fall back to the CPU provider
  (Kokoro runs faster-than-real-time on CPU); do not crash.
- Unknown voice -> `ValueError` listing available voices.
- Empty / whitespace-only text -> `ValueError`.
- Asset download failure (offline first run) -> actionable message naming the
  missing asset and how to pre-fetch.
- No output device on `play()` -> a clear `AudioDeviceError`.

## 8. Configuration

`TTSConfig` dataclass with the defaults in §4. `TTSEngine(voice=…, device=…,
speed=…, config=…)` overrides fields when provided (a passed `config` is
copied, not mutated — matching the STT engine's fix).

## 9. Model & voice management

- Assets: the Kokoro ONNX model (~310 MB) and the voices file, downloaded via
  `huggingface_hub` to `openvox._paths.cache_dir("tts/models")` on first use;
  fully offline thereafter.
- `KOKORO_VOICES` is a static registry of the built-in voice names.
  `voices()` returns them; `validate_voice(name)` raises `ValueError` for an
  unknown name — both without loading the model. Integration tests confirm the
  downloaded voices file actually contains the registry's names.

## 10. Testing

- **Unit (no model):** `TTSConfig` defaults; `voices()` non-empty and contains
  a known voice; unknown voice -> `ValueError`; empty text -> `ValueError`;
  `TTSResult.save_wav()` writes a valid 24 kHz mono 16-bit WAV from a synthetic
  array and `duration` is correct; demo `build_parser` defaults + flags.
- **Integration (downloads model, CPU):** `KokoroBackend.synthesize("hello
  world")` -> float32 audio, `sample_rate == 24000`, length > 0, non-silent
  (RMS above a floor); `TTSEngine` synthesize -> `save_wav` roundtrip.
- `play()` / `say()` stay thin and are validated manually (needs speakers),
  not in CI.

## 11. Dependencies

New `[tts]` extra: `kokoro-onnx` (brings `onnxruntime`, `misaki`,
`espeakng-loader`), `sounddevice` (playback), `huggingface_hub` (asset
download). `numpy` is already in the shared base. `all` becomes
`openvox[stt,stt-demo,tts]`.

## 12. Demo CLI

`python -m openvox.tts.demo` (entry point `openvox-tts-demo`):
- `--text` (required), `--voice` (default `af_heart`), `--device` (default
  `cuda`), `--speed` (default `1.0`), `--out PATH` (also save a WAV),
  `--no-play` (skip playback).
- Default behavior: synthesize `--text` and speak it aloud.

## 13. Definition of done

1. `from openvox.tts import TTSEngine; TTSEngine().say("…")` speaks a genuinely
   human voice, fully offline (after a one-time model download).
2. `python -m openvox.tts.demo --text "…"` speaks it; `--out` saves a 24 kHz WAV.
3. Voice selection works; unknown voice -> a clear error.
4. Test suite green; `import openvox` stays lean; TTS needs no STT import.
