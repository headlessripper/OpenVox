# OpenVox — Streaming TTS with Barge-in + Pluggable Voices (Sub-project 2C) — Design

**Date:** 2026-08-11
**Status:** Approved (design), proceeding to implementation plan
**Scope:** Add real-time **streaming** synthesis and **barge-in** (`stop()`) to the
TTS engine, and make the engine's **voice pluggable** so a cloned `.ovx` profile is
a valid voice alongside the Kokoro built-ins — behind one streaming interface,
honest about the latency difference between the two backends.

---

## 1. Goal & positioning

The TTS engine synthesizes whole utterances today. For robots and interactive
agents that is too rigid: speech should start almost immediately and be
**interruptible** the instant the user speaks. 2C adds a streaming interface that
emits audio in segments as they are synthesized, and a `stop()` barge-in primitive
that cuts playback within one audio block. The same interface accepts either a
fast Kokoro built-in voice or a cloned `.ovx` profile, so a user can stream in a
custom voice — accepting that a clone's first audio arrives seconds later than
Kokoro's.

## 2. Decisions locked during brainstorming

| Decision | Choice |
| --- | --- |
| Barge-in semantics | **Programmatic `stop()` now**, designed so an STT-driven full-duplex loop can call it later (full-duplex is a separate future sub-project) |
| Cloned voices | **Unified streaming interface**: `voice=` is a Kokoro name OR an `.ovx` profile; Kokoro streams ultra-low-latency, clones stream at sentence granularity with higher first-audio latency; same `stop()` for both |
| Placement | **Extend `openvox.tts`** (not a new package) — streaming is a property of the TTS engine; voice pluggability belongs where voices live |
| Streaming model | **Segment-based**: split text into speakable segments; synthesize each to an audio chunk; emit/play in order; barge-in cuts the current chunk and stops further segments |
| Backend identity | `resolve_voice` is the single seam that hides which backend serves a voice (also the future hook for brand-abstraction / hiding Kokoro/Chatterbox naming) |
| Torch-free import | `import openvox.tts` stays torch-free; the cloned-voice path imports torch lazily, only when an `.ovx` voice is used |

**Explicitly out of scope (deferred):** the full-duplex STT-driven auto-barge-in
loop; an LLM plug-and-play adapter (STT → LLM → streaming TTS) for building voice
agents; hiding third-party backend names behind OpenVox branding; sub-segment
(token/frame-level) Kokoro streaming; echo cancellation. These are future asks
noted in the roadmap; 2C only builds the programmatic streaming + `stop()`
foundation they will sit on.

## 3. Scope

### In scope

- Text segmentation (`segment.py`): `split_text(text) -> list[str]` — a pure-Python
  sentence/clause splitter, the streaming + barge-in granularity.
- Streaming backend protocol: `TTSBackend.stream_segments(segments, voice, speed)
  -> Iterator[TTSResult]` (default implementation loops `synthesize` per segment).
- Voice resolution (`voices.py`): `resolve_voice(voice) -> (backend, voice_id)` —
  a built-in name → `KokoroBackend`; an `.ovx` path or `VoiceProfile` →
  `ClonedVoiceBackend`.
- Cloned-voice backend (`cloned_backend.py`): wraps the existing clone/enroll path
  (`clone_from_profile` per segment); the only new torch entry point, imported
  lazily.
- Real-time player + barge-in (`stream.py`): `SpeechHandle` with `.stop()`,
  `.wait()`, `.done`; a background producer thread + a bounded queue + a
  sounddevice `OutputStream`.
- `TTSEngine.stream(text, voice, speed) -> Iterator[TTSResult]` (library form) and
  `TTSEngine.say_stream(text, voice, speed) -> SpeechHandle` (playback + barge-in).
- Config additions (`segment_max_chars`, `stream_queue_size`).
- Demo flags: `--stream`, `.ovx` voice support, `--interrupt-after SECONDS`.
- Tests: unit (no torch, no audio device) + opt-in heavy integration.

### Out of scope (deferred)

- Full-duplex STT-driven barge-in; LLM adapter; backend-name hiding; token/frame
  Kokoro streaming; echo cancellation; network/streaming transport. (See §2.)

## 4. Architecture

### Package layout (extends `openvox/tts/`)

```
openvox/tts/
  backend.py        # TTSBackend gains stream_segments(segments, voice, speed) -> Iterator[TTSResult]
                    #   default: yield synthesize(seg, voice, speed) for each segment
  segment.py        # NEW: split_text(text, max_chars) -> list[str]
  voices.py         # NEW: resolve_voice(voice) -> (TTSBackend, str) ; is_profile_voice(voice) -> bool
  cloned_backend.py # NEW: ClonedVoiceBackend(profile) — clone_from_profile per segment; lazy torch
  stream.py         # NEW: SpeechHandle + _StreamPlayer (producer thread + queue + OutputStream)
  engine.py         # TTSEngine gains .stream() and .say_stream()
  config.py         # TTSConfig gains segment_max_chars, stream_queue_size
  demo.py           # --stream, .ovx voice, --interrupt-after
tests/tts/
```

### Components (each independently testable)

- **`split_text(text, max_chars)`** — pure Python. Splits on sentence terminators
  (`.`/`?`/`!` followed by space/EOL), then further splits any segment longer than
  `max_chars` on clause boundaries (`,`/`;`/`:`) and finally on whitespace, never
  mid-word. Collapses whitespace; drops empty segments. Deterministic.
- **`TTSBackend.stream_segments`** — a concrete default method on the ABC that
  yields `synthesize(seg, voice, speed)` per segment. `KokoroBackend` inherits it
  (segments are small → low latency). Keeps the ABC's existing `synthesize`
  contract intact.
- **`resolve_voice(voice)`** — if `voice` is a `VoiceProfile` or a path ending
  `.ovx` (and the file exists), returns a `ClonedVoiceBackend` for it; otherwise
  validates the built-in name and returns the engine's `KokoroBackend`. `voice_id`
  is the Kokoro voice name (built-ins) or a sentinel for cloned voices. This is the
  single seam where backend identity is decided.
- **`ClonedVoiceBackend`** — constructed from a `VoiceProfile` (or `.ovx` path it
  loads via `openvox.enroll.VoiceProfile.load`). `synthesize`/`stream_segments`
  call the clone engine's `clone_from_profile` per segment → `TTSResult` @ 24 kHz.
  The only new file importing torch/chatterbox, lazily inside its methods. Reuses
  `openvox.clone.chatterbox_backend.ChatterboxBackend.clone_from_profile`.
- **`SpeechHandle`** — returned by `say_stream()`. Holds a `threading.Event`
  (`_stop`) and a `threading.Event`/flag (`_done`). `stop()` sets `_stop`, flushes
  the player queue, and aborts the output stream. `wait(timeout=None)` joins the
  producer and waits for drain. `done` reflects completion or stop. Idempotent;
  callable from another thread (the future STT loop).
- **`_StreamPlayer`** — owns the producer thread and the sounddevice
  `OutputStream`. The producer pulls segments from `backend.stream_segments`,
  checks `_stop` before each, and pushes chunks onto a bounded `queue.Queue`
  (size `stream_queue_size`); the OutputStream callback drains the queue to the
  device. On `_stop`: producer breaks, queue is cleared, stream is aborted (not
  drained). An in-flight cloned segment already being generated finishes on the
  thread and is discarded (never enqueued).
- **`TTSEngine.stream(text, voice, speed)`** — validates input, resolves the
  voice, splits the text, returns a generator yielding `TTSResult` chunks from
  `backend.stream_segments`. No audio device; the caller consumes chunks. Barge-in
  here = caller stops iterating.
- **`TTSEngine.say_stream(text, voice, speed)`** — builds a `_StreamPlayer` over
  the same producer and starts playback on a background thread; returns a
  `SpeechHandle`.

`import openvox` and `import openvox.tts` stay torch-free: only
`cloned_backend.py` imports torch/chatterbox, lazily inside methods; the existing
subprocess import-guard test continues to hold.

## 5. Public API

```python
from openvox.tts import TTSEngine

eng = TTSEngine(voice="af_heart", device="cuda")

# Library streaming (consume chunks yourself):
for chunk in eng.stream("Streamed as it is synthesized."):
    handle_audio(chunk.audio, chunk.sample_rate)          # chunk: TTSResult

# Playback with barge-in:
h = eng.say_stream("I can be interrupted at any moment.")
h.stop()        # cut audio within ~one audio block
h.wait()        # block until done/stopped
h.done          # bool

# Cloned voice — same interface:
eng.say_stream("Now in a cloned voice.", voice="alice.ovx")
eng.stream("Cloned and streamed.", voice=my_voice_profile)   # VoiceProfile also accepted

# Unchanged whole-utterance API (additive change, no breakage):
eng.synthesize("Whole utterance.").save_wav("out.wav")
eng.say("Whole utterance, spoken.")
```

## 6. Data flow

### Library streaming (`stream`)

```
stream(text, voice, speed):
  validate text/speed
  backend, voice_id = resolve_voice(voice)       # Kokoro or ClonedVoiceBackend (lazy torch)
  segments = split_text(text, segment_max_chars)
  for seg in segments:
      yield backend.synthesize(seg, voice_id, speed)   # via stream_segments
```

### Playback with barge-in (`say_stream`)

```
say_stream(text, voice, speed) -> SpeechHandle:
  backend, voice_id = resolve_voice(voice); segments = split_text(...)
  player = _StreamPlayer(queue_size=stream_queue_size)
  start OutputStream(callback drains queue -> device)
  start producer thread:
      for seg in segments:
          if handle._stop.is_set(): break
          chunk = backend.synthesize(seg, voice_id, speed)   # in-flight chunk on stop -> discarded
          if handle._stop.is_set(): break
          queue.put(chunk.audio)         # blocks when full (backpressure)
      mark producer finished
  return handle

handle.stop():  set _stop ; flush queue ; stream.abort() ; mark done
handle.wait():  join producer ; wait for stream drain (unless stopped) ; mark done
```

## 7. Configuration

`TTSConfig` gains: `segment_max_chars: int = 160` (soft cap; the splitter breaks
longer segments on clause/space boundaries) and `stream_queue_size: int = 8`
(playback buffer depth in chunks). Existing fields (`voice`, `device`, `speed`)
unchanged. `TTSEngine(...)` keeps copying its config (`dataclasses.replace`).

## 8. Error handling

- Empty/whitespace `text` → `ValueError` (matches `synthesize`); `speed <= 0` →
  `ValueError`.
- `voice` is an `.ovx` path that does not exist → `FileNotFoundError` from
  `resolve_voice` before any synthesis.
- `voice` is an unknown built-in name → `ValueError` (via `validate_voice`).
- A cloned voice requested but the clone/enroll deps are absent → a clear error
  naming the `[clone]`/`[enroll]` install, raised from `ClonedVoiceBackend`
  construction (lazy import failure), before playback starts.
- Playback with no audio device (headless) → `AudioDeviceError` from `say_stream`
  (as `say` does today); `stream()` never touches the device.
- `stop()` and `wait()` are idempotent and safe after natural completion.
- An exception on the producer thread is captured and re-raised from `wait()`
  (the stream stops); it never crashes the process silently.

## 9. Testing

- **Unit (no torch, no audio device):**
  - `split_text`: sentence boundaries, long-segment clause splitting, never
    mid-word, whitespace collapse, empty/whitespace input → `[]`.
  - `resolve_voice`: built-in name → the Kokoro backend + name; `.ovx` path →
    `ClonedVoiceBackend` (lazily; assert torch not imported for the name path);
    missing `.ovx` → `FileNotFoundError`; a `VoiceProfile` instance → cloned
    backend.
  - `stream_segments` default: a fake backend records one `synthesize` call per
    segment, in order.
  - `SpeechHandle`/`_StreamPlayer` with an **injected fake player/backend** (no
    real device): `stop()` halts the producer and sets `done`; `stop()` before
    start and after completion are idempotent; a fake backend that blocks on the
    2nd segment proves `stop()` returns promptly and the in-flight/next chunk is
    discarded; a producer-thread exception surfaces from `wait()`.
  - Import-guard subprocess test: `import openvox.tts` + construct `TTSEngine`
    with `torch`/`chatterbox` blocked → succeeds (cloned path stays lazy).
- **Integration (heavy, real, opt-in `@pytest.mark.integration`):**
  - Kokoro `stream("two sentences.")` yields ≥2 non-silent chunks.
  - A cloned `.ovx` voice `stream(...)` yields ≥1 non-silent 24 kHz chunk.
  - `say_stream(...).stop()` shortly after start returns quickly and leaves
    `done` true (guarded/skipped when no output device is available).
- Demo `build_parser` unit tests for `--stream`, `--voice`, `--interrupt-after`.

## 10. Dependencies & packaging

No new extra. Streaming core + Kokoro voices use `[tts]` (`kokoro-onnx`,
`sounddevice` — already present). Cloned voices in streaming reuse the existing
`[clone]`/`[enroll]` extras, imported lazily only when an `.ovx` voice is used.
`import openvox.tts` stays torch-free. No new entry points (demo is the existing
`openvox-tts-demo`).

## 11. Demo CLI

Extend `python -m openvox.tts.demo`: `--stream` (use `say_stream` instead of
`say`), `--voice` accepts an `.ovx` path as well as a built-in name, and
`--interrupt-after SECONDS` (with `--stream`) calls `handle.stop()` after N
seconds to demonstrate barge-in, then reports that playback was cut. Existing
flags (`--text`, `--out`, `--speed`, `--no-play`, `--device`) unchanged.

## 12. Definition of done

1. `TTSEngine.stream(text, voice)` yields `TTSResult` chunks for both a Kokoro
   voice and an `.ovx` profile.
2. `TTSEngine.say_stream(text, voice)` plays with a working `stop()` that cuts
   audio within ~one audio block; `wait()`/`done` behave; safe from another
   thread.
3. `synthesize()`/`say()` are unchanged; `import openvox` / `import openvox.tts`
   stay torch-free.
4. `python -m openvox.tts.demo --stream --interrupt-after 1.5` demonstrates
   barge-in; `--voice alice.ovx` streams a cloned voice.
5. Unit tests green without torch or an audio device; integration proves real
   Kokoro low-latency streaming, a cloned-voice stream, and an instant interrupt.
