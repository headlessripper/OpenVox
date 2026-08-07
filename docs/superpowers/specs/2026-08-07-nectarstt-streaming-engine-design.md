# NectarSTT — Streaming STT Engine (Sub-project 1) — Design

**Date:** 2026-08-07
**Status:** Approved (design), pending spec review
**Scope:** Horizon 1, Sub-project 1 of the "ElevenLabs-grade, fully offline" roadmap.

---

## 1. Vision & positioning

NectarSTT aims to be an ElevenLabs-grade voice stack that runs **fully offline**, targeting
robotics and other sectors that cannot (or should not) call the cloud: air-gapped, privacy-
sensitive, latency-critical, zero-marginal-cost deployments.

We do not out-train the cloud; we do **excellent inference engineering around the best open
weights**, plus deep integration (streaming, robust VAD, wake word, command grammars, ROS)
that a cloud API structurally cannot match.

### Roadmap horizons (context — not all in scope here)

- **Horizon 1 — Parity of plumbing:** extractable library, streaming STT, streaming TTS with
  barge-in, headless daemon + ROS 2 node.
- **Horizon 2 — Parity of quality:** model ladder, GPU/Jetson/ARM backends, word timestamps,
  punctuation, diarization, command-grammar biasing, wake word.
- **Horizon 3 — Surpass:** on-device voice cloning, sub-100ms full-duplex loop, adaptive
  on-device fine-tuning (operator voice, motor-noise profile, domain vocab), multimodal
  grounding (mic-array direction-of-arrival + beamforming).
- **Horizon 4 — Platform:** voice-model hub, eval harness proving WER/latency beats cloud on
  robotics audio, hardened cross-platform SDK.

**This document covers Sub-project 1 only: the streaming STT engine.**

## 2. Scope of Sub-project 1

### Decisions locked during brainstorming

| Decision | Choice |
| --- | --- |
| Primary hardware target (day 1) | Windows x86 + CUDA (dev machine); Jetson/ARM later |
| Deliverable | STT engine only (TTS is Sub-project 2) |
| STT backend | faster-whisper (CTranslate2), behind a swappable interface |
| Language | English-first on a multilingual-capable model |
| GUI | Remove the old GUI; new interface is a later sub-project |
| Streaming strategy | Sliding-window re-transcription + LocalAgreement (Approach B) |

### In scope

- Convert the repo into an importable `nectarstt` Python package.
- Streaming pipeline: mic → Silero VAD → faster-whisper (CUDA) → live partial + final
  results with word timestamps.
- Swappable `StreamingBackend` interface (so whisper.cpp / Parakeet slot in later).
- Config system with no hardcoded paths; model auto-download to an OS cache dir.
- A demo CLI (`python -m nectarstt.demo`).
- Test suite that runs without a microphone (file-as-mic seam).
- Remove old GUI (`Main.py`), old `CodeScripts` scripts, and `Production/` (kept in git history).

### Out of scope (explicitly deferred)

- TTS (Sub-project 2 — Piper alternative or custom engine TBD).
- Any GUI / new interface.
- Non-CUDA backends (whisper.cpp for Pi, Parakeet for Jetson) — interface is ready, impls later.
- Diarization, wake word, command-grammar biasing, punctuation control — Horizon 2.
- Multilingual UX (language selection/auto-detect surfaced to users) — architecture is ready.

## 3. Architecture

### Repo restructure

Convert from "script + zipped binary blob" to a proper package:

- Remove from working tree (preserved in git history): `Main.py`, `Main-Engine/Source/
  CodeScripts/`, `Production/`.
- `requirements.txt` → `pyproject.toml`. Drop unused deps (PyQt6, edge-tts, redis, gunicorn,
  ipython, Sphinx, etc.). STT package deps only.
- STT models are downloaded on demand; the `Main-Engine.zip` model-blob approach is retired
  for STT.

### Package layout

```
nectarstt/
  __init__.py          # public API surface (STTEngine)
  config.py            # Config dataclass, defaults, TOML/env overrides — no hardcoded paths
  models.py            # model registry + HF-hub downloader -> OS cache dir
  events.py            # PartialResult, FinalResult, WordTiming dataclasses
  audio/
    sources.py         # FrameSource ABC; MicSource (sounddevice) + FileSource (file-as-mic seam)
    vad.py             # SileroVAD wrapper (ONNX runtime, no torch)
  engine/
    backend.py         # StreamingBackend ABC + BackendResult
    faster_whisper_backend.py
    local_agreement.py # LocalAgreement stabilization (pure, testable)
    transcriber.py     # StreamingTranscriber: orchestrates VAD + windowing + agreement
  demo.py              # python -m nectarstt.demo — live partial/final CLI
tests/
pyproject.toml
```

### Component responsibilities

Each unit has one purpose, a defined interface, and is testable in isolation.

- **FrameSource (ABC)** — yields 16kHz mono float32 frames. `MicSource` owns the `sounddevice`
  input device and pushes frames to a thread-safe queue; `FileSource` yields frames from a WAV
  in simulated real-time. Neither knows about VAD or Whisper.
- **SileroVAD** — frame in → speech / not-speech out. ONNX build (no PyTorch dependency).
  Replaces the crude energy threshold of the old code.
- **StreamingBackend (ABC)** + **FasterWhisperBackend** — audio buffer in → text + word
  timings out (`BackendResult`). The abstract interface is the swap point for future backends.
- **LocalAgreement** — pure function: given consecutive hypotheses, returns the stable
  committed prefix vs. the volatile tail. No I/O.
- **StreamingTranscriber** — the conductor: pulls frames from a `FrameSource`, gates them with
  VAD, fires a re-transcribe every `window_interval_ms`, runs LocalAgreement, emits partials,
  finalizes on silence.
- **STTEngine** — public façade wiring the above together; exposes `stream()` and
  `transcribe_file()`.

## 4. Public API

```python
from nectarstt import STTEngine

engine = STTEngine(model="distil-large-v3", device="cuda", language="en")

# Live streaming over the mic (blocking generator):
for event in engine.stream():
    if event.is_partial:
        print("~", event.text, end="\r")   # firms up as you speak
    else:
        print("✓", event.text)              # final, with event.words[] timestamps

# Batch (also the backbone of deterministic tests):
result = engine.transcribe_file("clip.wav")
```

### Event types (`events.py`)

- `PartialResult(text, committed_prefix, volatile_tail)` — evolving hypothesis; `is_partial=True`.
- `FinalResult(text, words: list[WordTiming], start, end)` — stabilized utterance; `is_partial=False`.
- `WordTiming(word, start, end, probability)`.

## 5. Data flow (streaming loop)

```
FrameSource (mic or file) ──frames──▶ [thread-safe queue]
                                            │
                                  StreamingTranscriber
                                            │
                        ┌───────────────────┼─────────────────────┐
                     SileroVAD        window timer            silence?
                     (speech?)             │                    │
                        │           backend.transcribe(    on end-of-speech:
                 gate frames into      speech_buffer)       final pass →
                 current segment          │                 FinalResult(words)
                        └──────────▶ LocalAgreement ──▶ PartialResult   │
                                                                     reset buffer
```

1. Frames land in a queue from the active `FrameSource`.
2. The transcriber pulls frames and asks Silero if it is speech; speech frames accumulate into
   the current segment.
3. A `window_interval_ms` (~500ms) timer triggers `backend.transcribe(segment_so_far)`.
4. LocalAgreement diffs this hypothesis against the previous one and emits a `PartialResult`
   (stable prefix + volatile tail).
5. On VAD silence past `min_silence_ms`, a final pass runs with `word_timestamps=True`, emits
   `FinalResult`, and clears the buffer for the next utterance.

**File-as-mic seam:** `StreamingTranscriber` reads from the abstract `FrameSource`, not from
`sounddevice` directly, so a WAV feeder makes the whole streaming pipeline deterministically
testable without a microphone.

## 6. Configuration (`config.py`)

`Config` dataclass resolved from defaults → optional `nectarstt.toml` → environment overrides.
No hardcoded paths.

Defaults:

```
device="cuda", compute_type="float16", model="distil-large-v3", language="en",
sample_rate=16000, vad_threshold=0.5, min_silence_ms=500, min_speech_ms=200,
window_interval_ms=500
```

## 7. Model management (`models.py`)

Registry mapping friendly names (`tiny`, `base`, `small`, `distil-large-v3`, `large-v3`) to
Hugging Face repos (`Systran/faster-whisper-*`). On first use, `huggingface_hub` downloads to
an OS cache dir (`platformdirs`); every subsequent run is fully offline. Replaces the
`Main-Engine.zip` blob for STT models — no manual extraction, no GitHub size-limit hacks.

## 8. Error handling

Fail loud and actionable; never let one bad segment kill the stream.

- No CUDA available → warn, fall back to CPU (`compute_type="int8"`); do not crash.
- Mic device missing/busy → clear `AudioDeviceError` naming the device.
- Model download failure (offline first run) → actionable message: which model, how to pre-fetch.
- Backend inference error on a segment → logged, that segment dropped, stream continues.

## 9. Testing (TDD — tests written first)

- **LocalAgreement** — pure unit tests: hypothesis sequences → assert committed prefix vs.
  volatile tail. Deterministic, no models.
- **SileroVAD** — fixture WAV with known speech/silence regions → assert boundaries.
- **FasterWhisperBackend** — `transcribe_file()` on a short bundled WAV fixture → assert
  expected transcript (batch = deterministic).
- **Streaming integration** — drive the pipeline via the file-as-mic `FileSource`; assert
  partials converge monotonically and the final matches the known transcript. No mic in CI.
- **Config/models** — resolution precedence and registry lookups.
- Mic capture stays thin; validated manually via the demo.

## 10. Dependencies (STT package)

`faster-whisper`, `onnxruntime` (+ Silero VAD ONNX model), `sounddevice`, `numpy`,
`huggingface_hub`, `platformdirs`. Dev: `pytest`. (PyTorch intentionally avoided by using the
ONNX Silero build.)

## 11. Definition of done

1. `from nectarstt import STTEngine; engine.stream()` yields live partials + finals on the CUDA
   machine.
2. `python -m nectarstt.demo` shows real-time transcription.
3. Old GUI/scripts/`Production/` removed; clean `pyproject.toml`; test suite green.
4. First run auto-downloads the model; subsequent runs fully offline.
