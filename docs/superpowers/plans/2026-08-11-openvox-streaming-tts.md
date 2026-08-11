# OpenVox Streaming TTS + Barge-in (2C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-time streaming synthesis and a `stop()` barge-in primitive to `openvox.tts`, and make `voice=` accept a cloned `.ovx` profile as well as a Kokoro built-in — one streaming interface for both.

**Architecture:** Text is split into speakable segments; each segment is synthesized to a `TTSResult` chunk and emitted/played in order. `stream()` is a generator of chunks; `say_stream()` plays them on a background producer thread through a queue + sounddevice `OutputStream` and returns a `SpeechHandle` whose `stop()` flushes-and-aborts within one audio block. `resolve_voice()` maps a built-in name to the Kokoro backend and an `.ovx`/`VoiceProfile` to a lazy-torch `ClonedVoiceBackend`.

**Tech Stack:** Python ≥3.11, numpy, kokoro-onnx (ONNX, torch-free), sounddevice; Chatterbox (torch, via the existing `[clone]`/`[enroll]` path) only when a cloned voice is used.

## Global Constraints

- Python ≥ 3.11. Working dir `C:\Users\nateg\Documents\NectarSTT`.
- `import openvox` and `import openvox.tts` MUST stay torch-free — only `openvox/tts/cloned_backend.py` may import torch/chatterbox, and only lazily inside methods. Enforced by the existing/extended subprocess import-guard test.
- `segment.py`, `voices.py` are pure Python (no torch, no sounddevice at module top).
- Reuse `TTSResult` from `openvox.tts.backend`; do not redefine it. Both backends emit 24 kHz mono float32.
- `TTSConfig` is a plain dataclass; the engine copies it with `dataclasses.replace` (never mutates a passed config).
- Cloned voices reuse `openvox.clone.chatterbox_backend.ChatterboxBackend.clone_from_profile(text, conditionals, exaggeration, cfg)` and `openvox.enroll.VoiceProfile.load(path)` — both already exist on `main`.
- The whole-utterance API (`synthesize`, `say`) must remain byte-for-byte behavior-compatible — this sub-project is purely additive.
- No new optional-dependency extra and no new script entry point.
- Commit message prefix `feat(tts):` (or `test(tts):` for the integration task).

---

### Task 1: Text segmentation (`split_text`)

**Files:**
- Create: `openvox/tts/segment.py`
- Test: `tests/tts/test_segment.py`

**Interfaces:**
- Produces: `split_text(text: str, max_chars: int = 160) -> list[str]` — splits on sentence terminators (`.`/`?`/`!` runs followed by whitespace/end), keeps the terminator with its sentence, then sub-splits any segment longer than `max_chars` on clause punctuation (`,`/`;`/`:`) and finally on whitespace, never mid-word. Collapses internal whitespace; drops empty/whitespace-only segments. Returns `[]` for empty/whitespace input.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_segment.py
from openvox.tts.segment import split_text

def test_splits_on_sentence_boundaries():
    segs = split_text("Hello there. How are you? I am fine!")
    assert segs == ["Hello there.", "How are you?", "I am fine!"]

def test_empty_and_whitespace():
    assert split_text("") == []
    assert split_text("   \n\t ") == []

def test_collapses_whitespace():
    assert split_text("Hello    world.") == ["Hello world."]

def test_long_segment_split_on_clauses_never_midword():
    text = "alpha beta gamma, delta epsilon zeta, eta theta iota kappa"
    segs = split_text(text, max_chars=20)
    assert all(len(s) <= 20 for s in segs)
    # every original word survives intact, in order (commas kept for prosody,
    # stripped here only to compare the word sequence)
    assert " ".join(segs).replace(",", "").split() == text.replace(",", "").split()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_segment.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.tts.segment`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/tts/segment.py
import re

_SENTENCE = re.compile(r'.+?(?:[.!?]+(?=\s|$)|$)', re.DOTALL)
_CLAUSE = re.compile(r'.+?(?:[,;:](?=\s|$)|$)', re.DOTALL)

def _norm(s: str) -> str:
    return " ".join(s.split())

def _pack_words(text: str, max_chars: int) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)
    return out

def split_text(text: str, max_chars: int = 160) -> list[str]:
    segments: list[str] = []
    for sent_match in _SENTENCE.finditer(text or ""):
        sentence = _norm(sent_match.group())
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            segments.append(sentence)
            continue
        # too long: split on clauses, then hard-pack words if a clause is still long
        for clause_match in _CLAUSE.finditer(sentence):
            clause = _norm(clause_match.group())
            if not clause:
                continue
            if len(clause) <= max_chars:
                segments.append(clause)
            else:
                segments.extend(_pack_words(clause, max_chars))
    return segments
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_segment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/segment.py tests/tts/test_segment.py
git commit -m "feat(tts): add sentence/clause text segmentation for streaming"
```

---

### Task 2: Streaming backend protocol + config fields

**Files:**
- Modify: `openvox/tts/backend.py` (add `stream_segments` to `TTSBackend`)
- Modify: `openvox/tts/config.py` (add `segment_max_chars`, `stream_queue_size`)
- Test: `tests/tts/test_backend_stream.py`, `tests/tts/test_config.py` (extend if it exists, else create)

**Interfaces:**
- Produces: `TTSBackend.stream_segments(self, segments: list[str], voice: str, speed: float) -> Iterator[TTSResult]` — a concrete default method that yields `self.synthesize(seg, voice, speed)` for each segment, in order. `TTSConfig` gains `segment_max_chars: int = 160` and `stream_queue_size: int = 8`.
- Consumes: `TTSResult`, `synthesize` (existing).

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_backend_stream.py
import numpy as np
from openvox.tts.backend import TTSBackend, TTSResult

class FakeBackend(TTSBackend):
    def __init__(self):
        self.calls = []
    def synthesize(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        return TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)

def test_stream_segments_yields_one_result_per_segment_in_order():
    be = FakeBackend()
    out = list(be.stream_segments(["a", "b", "c"], voice="af_heart", speed=1.0))
    assert [r.sample_rate for r in out] == [24000, 24000, 24000]
    assert be.calls == [("a", "af_heart", 1.0), ("b", "af_heart", 1.0), ("c", "af_heart", 1.0)]
```

```python
# tests/tts/test_config.py   (create if missing; otherwise add this test)
from openvox.tts.config import TTSConfig

def test_streaming_config_defaults():
    c = TTSConfig()
    assert c.segment_max_chars == 160
    assert c.stream_queue_size == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_backend_stream.py tests/tts/test_config.py -v`
Expected: FAIL (`stream_segments` missing / attribute error on config fields)

- [ ] **Step 3: Write minimal implementation**

In `openvox/tts/backend.py`, add the import and the default method to `TTSBackend`:

```python
from collections.abc import Iterator
```

```python
    def stream_segments(self, segments: list[str], voice: str,
                        speed: float) -> Iterator[TTSResult]:
        """Yield one synthesized chunk per segment, in order.

        Default implementation loops synthesize(); backends may override for
        finer-grained streaming."""
        for seg in segments:
            yield self.synthesize(seg, voice, speed)
```

In `openvox/tts/config.py`, add the two fields to `TTSConfig`:

```python
    segment_max_chars: int = 160   # soft cap; the splitter breaks longer segments
    stream_queue_size: int = 8     # streaming playback buffer depth (chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_backend_stream.py tests/tts/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/backend.py openvox/tts/config.py tests/tts/test_backend_stream.py tests/tts/test_config.py
git commit -m "feat(tts): add stream_segments backend protocol + streaming config"
```

---

### Task 3: Cloned-voice backend (`ClonedVoiceBackend`)

**Files:**
- Create: `openvox/tts/cloned_backend.py`
- Test: `tests/tts/test_cloned_backend.py`

**Interfaces:**
- Produces: `CLONED_VOICE_ID: str = "__cloned__"`; `ClonedVoiceBackend(profile, device="cuda", exaggeration=0.5, cfg=0.5)` implementing `TTSBackend`. `synthesize(text, voice, speed) -> TTSResult` clones one segment via the existing clone backend (ignores `voice`/`speed` — a clone's identity comes from the profile and Chatterbox has no speed knob). Inherits `stream_segments`. Lazily imports torch/chatterbox and `openvox.enroll` inside `_ensure_loaded`.
- Consumes: `openvox.enroll.VoiceProfile.load`, `openvox.clone.chatterbox_backend.ChatterboxBackend.clone_from_profile`, `TTSResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_cloned_backend.py
import numpy as np
from openvox.tts.cloned_backend import ClonedVoiceBackend, CLONED_VOICE_ID
from openvox.tts.backend import TTSResult

class _FakeConds:
    pass

class _FakeProfile:
    conditionals = _FakeConds()

def test_synthesize_clones_each_segment(monkeypatch):
    calls = {}
    class _FakeChatBackend:
        def __init__(self, device):
            calls["device"] = device
        def clone_from_profile(self, text, conditionals, exaggeration, cfg):
            calls.setdefault("texts", []).append(text)
            calls["exaggeration"] = exaggeration
            return TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)
    # patch the lazy import target
    import openvox.clone.chatterbox_backend as cb
    monkeypatch.setattr(cb, "ChatterboxBackend", _FakeChatBackend)

    be = ClonedVoiceBackend(_FakeProfile(), device="cpu", exaggeration=0.7)
    out = list(be.stream_segments(["one", "two"], voice=CLONED_VOICE_ID, speed=1.0))
    assert len(out) == 2 and out[0].sample_rate == 24000
    assert calls["texts"] == ["one", "two"]
    assert calls["device"] == "cpu" and calls["exaggeration"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_cloned_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.tts.cloned_backend`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/tts/cloned_backend.py
import os

from openvox.tts.backend import TTSBackend, TTSResult

CLONED_VOICE_ID = "__cloned__"

class ClonedVoiceBackend(TTSBackend):
    """A TTS backend whose voice is a cloned .ovx profile (via Chatterbox).

    Ignores the ``voice``/``speed`` arguments: identity comes from the profile
    and Chatterbox has no speed control. Imports torch/chatterbox lazily."""

    def __init__(self, profile, device: str = "cuda",
                 exaggeration: float = 0.5, cfg: float = 0.5) -> None:
        self._profile = profile      # a VoiceProfile or an .ovx path
        self._device = device
        self._exaggeration = exaggeration
        self._cfg = cfg
        self._backend = None
        self._conds = None

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        prof = self._profile
        if hasattr(prof, "conditionals"):
            conds = prof.conditionals
        else:
            from openvox.enroll import VoiceProfile
            conds = VoiceProfile.load(os.fspath(prof)).conditionals
        from openvox.clone.chatterbox_backend import ChatterboxBackend
        self._backend = ChatterboxBackend(device=self._device)
        self._conds = conds

    def synthesize(self, text: str, voice: str, speed: float) -> TTSResult:
        self._ensure_loaded()
        return self._backend.clone_from_profile(
            text, self._conds, self._exaggeration, self._cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_cloned_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/cloned_backend.py tests/tts/test_cloned_backend.py
git commit -m "feat(tts): add ClonedVoiceBackend (stream a cloned .ovx voice)"
```

---

### Task 4: Voice resolution (`resolve_voice`)

**Files:**
- Create: `openvox/tts/voices.py`
- Test: `tests/tts/test_voices.py`

**Interfaces:**
- Produces:
  - `is_profile_voice(voice) -> bool` — True if `voice` is a `VoiceProfile`-like object (has a `conditionals` attribute) or a `str`/`os.PathLike` ending in `.ovx`.
  - `resolve_voice(voice, kokoro_backend, device) -> tuple[TTSBackend, str]` — a profile voice → `(ClonedVoiceBackend(voice, device), CLONED_VOICE_ID)`; a string `.ovx` path that does not exist → `FileNotFoundError`; a built-in name → validate via `openvox.tts.models.validate_voice` then `(kokoro_backend, name)`.
- Consumes: `ClonedVoiceBackend`, `CLONED_VOICE_ID` (Task 3); `validate_voice` (`openvox.tts.models`).

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_voices.py
import pytest
from openvox.tts.voices import is_profile_voice, resolve_voice
from openvox.tts.cloned_backend import ClonedVoiceBackend, CLONED_VOICE_ID

class _Sentinel:
    """stand-in for a KokoroBackend"""

class _FakeProfile:
    conditionals = object()

def test_is_profile_voice():
    assert is_profile_voice("alice.ovx") is True
    assert is_profile_voice(_FakeProfile()) is True
    assert is_profile_voice("af_heart") is False
    assert is_profile_voice(None) is False

def test_resolve_builtin_name_returns_kokoro():
    kok = _Sentinel()
    backend, vid = resolve_voice("af_heart", kok, device="cpu")
    assert backend is kok and vid == "af_heart"

def test_resolve_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_voice("not_a_voice", _Sentinel(), device="cpu")

def test_resolve_profile_instance_returns_cloned_backend():
    backend, vid = resolve_voice(_FakeProfile(), _Sentinel(), device="cpu")
    assert isinstance(backend, ClonedVoiceBackend) and vid == CLONED_VOICE_ID

def test_resolve_missing_ovx_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_voice(str(tmp_path / "nope.ovx"), _Sentinel(), device="cpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_voices.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.tts.voices`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/tts/voices.py
import os

from openvox.tts.cloned_backend import ClonedVoiceBackend, CLONED_VOICE_ID
from openvox.tts.models import validate_voice

def is_profile_voice(voice) -> bool:
    if hasattr(voice, "conditionals"):
        return True
    if isinstance(voice, (str, os.PathLike)):
        return os.fspath(voice).endswith(".ovx")
    return False

def resolve_voice(voice, kokoro_backend, device: str):
    """Return (backend, voice_id): the Kokoro backend for a built-in name, or a
    ClonedVoiceBackend for an .ovx profile."""
    if is_profile_voice(voice):
        if isinstance(voice, (str, os.PathLike)) and not os.path.isfile(voice):
            raise FileNotFoundError(f"voice profile not found: {voice}")
        return ClonedVoiceBackend(voice, device=device), CLONED_VOICE_ID
    validate_voice(voice)
    return kokoro_backend, voice
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_voices.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/voices.py tests/tts/test_voices.py
git commit -m "feat(tts): add resolve_voice (Kokoro name or .ovx profile)"
```

---

### Task 5: SpeechHandle + streaming player (`stream.py`)

**Files:**
- Create: `openvox/tts/stream.py`
- Test: `tests/tts/test_stream.py`

**Interfaces:**
- Produces:
  - `SpeechHandle(player)` — `.start(backend, voice_id, speed, segments)` spawns a producer thread that pulls chunks from `backend.stream_segments(segments, voice_id, speed)`, checks the stop flag before and after each `synthesize`, and calls `player.put(chunk.audio, chunk.sample_rate)`; on normal completion calls `player.finish()`. `.stop()` sets the stop flag, calls `player.abort()`, marks done (idempotent). `.wait(timeout=None)` joins the producer, calls `player.wait_drain()` unless stopped, then re-raises any producer exception. `.done -> bool`.
  - `_StreamPlayer(sample_rate, queue_size)` — real sounddevice player with `start()`, `put(audio, sample_rate)`, `finish()`, `abort()`, `wait_drain(timeout=None)`, `close()`. Imports sounddevice lazily in `start()`.
- Consumes: nothing from earlier tasks directly (the engine wires backends in); `TTSResult` chunks are duck-typed (`.audio`, `.sample_rate`).

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_stream.py
import threading
import numpy as np
import pytest
from openvox.tts.stream import SpeechHandle
from openvox.tts.backend import TTSResult

class FakePlayer:
    def __init__(self):
        self.puts, self.finished, self.aborted, self.drained = [], False, False, False
    def start(self): pass
    def put(self, audio, sample_rate): self.puts.append((len(audio), sample_rate))
    def finish(self): self.finished = True
    def abort(self): self.aborted = True
    def wait_drain(self, timeout=None): self.drained = True
    def close(self): pass

class ListBackend:
    def __init__(self, n): self._n = n
    def stream_segments(self, segments, voice, speed):
        for _ in segments:
            yield TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)

def test_full_playthrough_puts_all_chunks_and_finishes():
    p = FakePlayer()
    h = SpeechHandle(p)
    h.start(ListBackend(3), "af_heart", 1.0, ["a", "b", "c"])
    h.wait()
    assert len(p.puts) == 3 and p.finished is True and h.done is True

def test_stop_is_idempotent_and_sets_done():
    p = FakePlayer()
    h = SpeechHandle(p)
    h.stop(); h.stop()
    assert p.aborted is True and h.done is True

def test_stop_mid_stream_discards_inflight_and_returns_promptly():
    release = threading.Event()
    started = threading.Event()
    class BlockingBackend:
        def stream_segments(self, segments, voice, speed):
            yield TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)  # 1st
            started.set()
            release.wait(5)                                                          # 2nd blocks
            yield TTSResult(audio=np.ones(4, dtype=np.float32), sample_rate=24000)
    p = FakePlayer()
    h = SpeechHandle(p)
    h.start(BlockingBackend(), "af_heart", 1.0, ["a", "b"])
    assert started.wait(5)
    h.stop()                       # returns promptly while 2nd segment is blocked
    assert h.done is True and p.aborted is True
    release.set()
    h.wait()
    assert len(p.puts) == 1        # in-flight/next chunk never enqueued

def test_producer_exception_surfaces_from_wait():
    class BoomBackend:
        def stream_segments(self, segments, voice, speed):
            raise RuntimeError("boom")
            yield  # pragma: no cover
    h = SpeechHandle(FakePlayer())
    h.start(BoomBackend(), "af_heart", 1.0, ["a"])
    with pytest.raises(RuntimeError, match="boom"):
        h.wait()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_stream.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.tts.stream`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/tts/stream.py
import queue
import threading

class SpeechHandle:
    """Controls a streaming synthesis+playback session. stop() is the barge-in
    primitive; safe to call from another thread."""

    def __init__(self, player) -> None:
        self._player = player
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread = None
        self._error = None

    def start(self, backend, voice_id: str, speed: float, segments: list[str]) -> None:
        self._player.start()
        self._thread = threading.Thread(
            target=self._produce, args=(backend, voice_id, speed, segments),
            daemon=True)
        self._thread.start()

    def _produce(self, backend, voice_id, speed, segments) -> None:
        try:
            for chunk in backend.stream_segments(segments, voice_id, speed):
                if self._stop.is_set():
                    return
                self._player.put(chunk.audio, chunk.sample_rate)
                if self._stop.is_set():
                    return
            if not self._stop.is_set():
                self._player.finish()
        except Exception as exc:  # surfaced from wait()
            self._error = exc
        finally:
            self._done.set()

    def stop(self) -> None:
        self._stop.set()
        self._player.abort()
        self._done.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
        if not self._stop.is_set():
            self._player.wait_drain(timeout)
        if self._error is not None:
            raise self._error

    @property
    def done(self) -> bool:
        return self._done.is_set()


class _StreamPlayer:
    """Real-time player: a consumer thread drains a queue to a sounddevice
    OutputStream. put() applies backpressure; abort() cuts audio immediately."""

    def __init__(self, sample_rate: int, queue_size: int) -> None:
        self._sr = sample_rate
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._abort = threading.Event()
        self._finished = threading.Event()
        self._stream = None
        self._consumer = None

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=self._sr, channels=1, dtype="float32")
        self._stream.start()
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._consumer.start()

    def _consume(self) -> None:
        import numpy as np
        while not self._abort.is_set():
            try:
                audio = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._finished.is_set():
                    break
                continue
            if self._abort.is_set():
                break
            self._stream.write(np.ascontiguousarray(audio, dtype="float32"))

    def put(self, audio, sample_rate) -> None:
        # backpressure, but stay responsive to abort so a blocked producer frees
        while not self._abort.is_set():
            try:
                self._queue.put(audio, timeout=0.1)
                return
            except queue.Full:
                continue

    def finish(self) -> None:
        self._finished.set()

    def abort(self) -> None:
        self._abort.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass

    def wait_drain(self, timeout: float | None = None) -> None:
        if self._consumer is not None:
            self._consumer.join(timeout)
        self.close()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_stream.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/stream.py tests/tts/test_stream.py
git commit -m "feat(tts): add SpeechHandle + streaming player with stop() barge-in"
```

---

### Task 6: Engine wiring (`stream` + `say_stream`)

**Files:**
- Modify: `openvox/tts/engine.py` (add `stream`, `say_stream`)
- Modify: `openvox/tts/__init__.py` (export `SpeechHandle`)
- Test: `tests/tts/test_engine_stream.py`, `tests/tts/test_import_lean.py` (extend the guard if present, else create)

**Interfaces:**
- Produces:
  - `TTSEngine.stream(text, voice=None, speed=None) -> Iterator[TTSResult]` — validates text/speed, resolves the voice, splits the text, yields chunks from `backend.stream_segments`.
  - `TTSEngine.say_stream(text, voice=None, speed=None) -> SpeechHandle` — builds a `_StreamPlayer(self._config.sample_rate, self._config.stream_queue_size)`, a `SpeechHandle`, and starts it.
- Consumes: `split_text` (Task 1), `resolve_voice` (Task 4), `SpeechHandle`, `_StreamPlayer` (Task 5), existing `KokoroBackend`, `TTSConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_engine_stream.py
import numpy as np
import pytest
from openvox.tts import TTSEngine, TTSResult
from openvox.tts.engine import TTSEngine as _Eng

class _FakeKokoro:
    def synthesize(self, text, voice, speed):
        return TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)
    def stream_segments(self, segments, voice, speed):
        for _ in segments:
            yield self.synthesize("", voice, speed)

def _engine():
    eng = _Eng.__new__(_Eng)          # bypass real Kokoro asset load
    from openvox.tts.config import TTSConfig
    eng._config = TTSConfig(device="cpu")
    eng._backend = _FakeKokoro()
    return eng

def test_stream_yields_one_chunk_per_segment():
    eng = _engine()
    chunks = list(eng.stream("Hello there. How are you?", voice="af_heart"))
    assert len(chunks) == 2 and all(c.sample_rate == 24000 for c in chunks)

def test_stream_rejects_empty_text():
    with pytest.raises(ValueError):
        list(_engine().stream("   "))

def test_say_stream_returns_handle_that_completes(monkeypatch):
    # inject a fake player so no audio device is needed
    import openvox.tts.engine as eng_mod
    from openvox.tts.stream import SpeechHandle
    class FakePlayer:
        def start(self): pass
        def put(self, a, sr): pass
        def finish(self): pass
        def abort(self): pass
        def wait_drain(self, timeout=None): pass
        def close(self): pass
    monkeypatch.setattr(eng_mod, "_StreamPlayer", lambda sr, qs: FakePlayer())
    h = _engine().say_stream("Hello there.", voice="af_heart")
    h.wait()
    assert h.done is True
```

Also add to the import-guard test (create `tests/tts/test_import_lean.py` if missing):

```python
# tests/tts/test_import_lean.py
import subprocess, sys

def test_import_openvox_tts_torch_free():
    code = (
        "import sys\n"
        "for m in ('torch', 'chatterbox'):\n"
        "    sys.modules[m] = None\n"
        "import openvox.tts\n"
        "from openvox.tts import TTSEngine, SpeechHandle\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_engine_stream.py tests/tts/test_import_lean.py -v`
Expected: FAIL (`stream`/`say_stream`/`SpeechHandle` missing)

- [ ] **Step 3: Write minimal implementation**

In `openvox/tts/engine.py`, add imports and the two methods to `TTSEngine`:

```python
from collections.abc import Iterator

from openvox.tts.segment import split_text
from openvox.tts.voices import resolve_voice
from openvox.tts.stream import SpeechHandle, _StreamPlayer
```

```python
    def stream(self, text: str, voice: str | None = None,
               speed: float | None = None) -> Iterator[TTSResult]:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        v = voice if voice is not None else self._config.voice
        s = speed if speed is not None else self._config.speed
        if s <= 0:
            raise ValueError("speed must be > 0")
        backend, voice_id = resolve_voice(v, self._backend, self._config.device)
        segments = split_text(text, self._config.segment_max_chars)
        return backend.stream_segments(segments, voice_id, s)

    def say_stream(self, text: str, voice: str | None = None,
                   speed: float | None = None) -> SpeechHandle:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        v = voice if voice is not None else self._config.voice
        s = speed if speed is not None else self._config.speed
        if s <= 0:
            raise ValueError("speed must be > 0")
        backend, voice_id = resolve_voice(v, self._backend, self._config.device)
        segments = split_text(text, self._config.segment_max_chars)
        player = _StreamPlayer(self._config.sample_rate, self._config.stream_queue_size)
        handle = SpeechHandle(player)
        handle.start(backend, voice_id, s, segments)
        return handle
```

In `openvox/tts/__init__.py`, export `SpeechHandle`:

```python
"""OpenVox TTS — offline text-to-speech."""
from openvox.tts.engine import TTSEngine
from openvox.tts.backend import TTSResult
from openvox.tts.stream import SpeechHandle

__all__ = ["TTSEngine", "TTSResult", "SpeechHandle"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_engine_stream.py tests/tts/test_import_lean.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/engine.py openvox/tts/__init__.py tests/tts/test_engine_stream.py tests/tts/test_import_lean.py
git commit -m "feat(tts): add TTSEngine.stream and say_stream with barge-in"
```

---

### Task 7: Demo streaming flags

**Files:**
- Modify: `openvox/tts/demo.py` (add `--stream`, `--interrupt-after`; `--voice` accepts an `.ovx` path)
- Test: `tests/tts/test_demo.py` (create if missing)

**Interfaces:**
- Produces: `build_parser()` gains `--stream` (store_true) and `--interrupt-after` (float, default None). `main` uses `say_stream` when `--stream` is set, and if `--interrupt-after N` is given, sleeps N seconds then calls `handle.stop()` and prints that playback was cut; otherwise `handle.wait()`.
- Consumes: `TTSEngine.say_stream` (Task 6).

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_demo.py
from openvox.tts.demo import build_parser

def test_stream_flags_default_off():
    args = build_parser().parse_args(["--text", "hi"])
    assert args.stream is False
    assert args.interrupt_after is None

def test_stream_flags_parse():
    args = build_parser().parse_args(
        ["--text", "hi", "--stream", "--interrupt-after", "1.5", "--voice", "alice.ovx"])
    assert args.stream is True
    assert args.interrupt_after == 1.5
    assert args.voice == "alice.ovx"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_demo.py -v`
Expected: FAIL (`--stream`/`--interrupt-after` unknown)

- [ ] **Step 3: Write minimal implementation**

In `openvox/tts/demo.py`, add the two flags in `build_parser` (after `--no-play`):

```python
    p.add_argument("--stream", action="store_true",
                   help="Stream with barge-in support (uses say_stream).")
    p.add_argument("--interrupt-after", type=float, default=None,
                   help="With --stream: call stop() after this many seconds to demo barge-in.")
```

Replace the play section of `main` so streaming is honored (keep the `--out`/`synthesize` path for the non-stream case):

```python
def main(argv: list[str] | None = None) -> int:
    import time
    args = build_parser().parse_args(argv)
    engine = TTSEngine(voice=args.voice, device=args.device, speed=args.speed)
    if args.stream and not args.no_play:
        handle = engine.say_stream(args.text)
        if args.interrupt_after is not None:
            time.sleep(args.interrupt_after)
            handle.stop()
            print(f"Interrupted after {args.interrupt_after}s (barge-in).")
        else:
            handle.wait()
        return 0
    result = engine.synthesize(args.text)
    if args.out:
        result.save_wav(args.out)
        print(f"Saved {args.out} ({result.duration:.1f}s)")
    if not args.no_play:
        engine.play(result)
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_demo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/demo.py tests/tts/test_demo.py
git commit -m "feat(tts): add --stream and --interrupt-after demo flags"
```

---

### Task 8: Real end-to-end integration test

**Files:**
- Test: `tests/tts/test_integration_stream.py`

**Interfaces:**
- Consumes: the full stack. Marked `@pytest.mark.integration`; downloads the Kokoro model and really synthesizes. No audio device is required (it uses `stream()`, not `say_stream()`).

- [ ] **Step 1: Write the test**

```python
# tests/tts/test_integration_stream.py
import numpy as np
import pytest

pytestmark = pytest.mark.integration

def test_kokoro_stream_yields_multiple_nonsilent_chunks():
    from openvox.tts import TTSEngine
    eng = TTSEngine(device="cpu")   # CPU is fine; Kokoro is fast
    chunks = list(eng.stream("Hello there. How are you today?", voice="af_heart"))
    assert len(chunks) >= 2
    for c in chunks:
        assert c.sample_rate == 24000
        assert c.audio.size > 0
    assert any(float(np.sqrt(np.mean(c.audio ** 2))) > 1e-4 for c in chunks)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/tts/test_integration_stream.py -v -m integration`
Expected: PASS (downloads the Kokoro model once, then synthesizes two chunks).

- [ ] **Step 3: Commit**

```bash
git add tests/tts/test_integration_stream.py
git commit -m "test(tts): end-to-end Kokoro streaming integration"
```

---

## Self-Review

**Spec coverage:**
- §3 `split_text` → Task 1. `stream_segments` protocol + config → Task 2. `ClonedVoiceBackend` → Task 3. `resolve_voice`/`is_profile_voice` → Task 4. `SpeechHandle` + player + barge-in → Task 5. `TTSEngine.stream`/`say_stream` + `SpeechHandle` export → Task 6. Demo `--stream`/`--interrupt-after`/`.ovx` voice → Task 7. Integration → Task 8. ✓
- §7 config (`segment_max_chars=160`, `stream_queue_size=8`) → Task 2. ✓
- §8 error handling: empty text / speed≤0 (Task 6 stream/say_stream); missing `.ovx` → `FileNotFoundError` (Task 4); unknown name → `ValueError` (Task 4); cloned-deps-absent surfaces on `ClonedVoiceBackend` load (Task 3, lazy import raises naturally); producer exception via `wait()` (Task 5); `stop()`/`wait()` idempotent (Task 5). ✓
- §4 boundaries: only `cloned_backend.py` imports torch (lazy); `segment`/`voices` pure; `import openvox.tts` torch-free guard (Task 6). ✓
- §10 no new extra / entry point — confirmed none added. ✓

**Placeholder scan:** No TBD/TODO; every step has concrete code and tests. ✓

**Type consistency:** `stream_segments(segments, voice, speed) -> Iterator[TTSResult]` identical across Tasks 2, 3, 5, 6. `resolve_voice(voice, kokoro_backend, device) -> (backend, voice_id)` consistent Tasks 4 and 6. `CLONED_VOICE_ID` defined Task 3, used Tasks 3/4. `SpeechHandle(player)` + `.start(backend, voice_id, speed, segments)`/`.stop()`/`.wait()`/`.done` consistent Tasks 5 and 6. `_StreamPlayer(sample_rate, queue_size)` consistent Tasks 5 and 6. Chunks are duck-typed `.audio`/`.sample_rate` (a `TTSResult`) everywhere. ✓

**Note on §8 cloned-deps error message:** Task 3's lazy imports raise a plain `ImportError`/`ModuleNotFoundError` if `[clone]`/`[enroll]` are absent. The spec asks for a "clear error naming the install." If a reviewer wants the friendlier message, wrap the lazy imports in `_ensure_loaded` with a `try/except ImportError` re-raising a `RuntimeError` naming `pip install "openvox[clone]"` — but since torch/chatterbox are already required for any clone workflow and were installed for 2B, the plain import error is acceptable for v1; flagged here so it is a conscious choice, not an oversight.
