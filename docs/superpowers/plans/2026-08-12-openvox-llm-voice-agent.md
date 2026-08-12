# OpenVox LLM Voice Agent (Sub-project 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `openvox.agent`, a bring-your-own-LLM voice-assistant loop (mic to STT to your LLM to streaming TTS) with VAD barge-in, plus the clean adapters it is built on, bundling no model.

**Architecture:** The enabler is teaching `TTSEngine.say_stream` to accept a text-chunk iterator (a streaming LLM), aggregated into sentences by `openvox.tts.segment.iter_sentences` so speech starts before the model finishes. `openvox.agent` is thin glue: `listen_once` consumes the STT event stream to one finalized utterance; `speak_stream` forwards to `say_stream`; `VoiceAgent` runs the turn loop, and a threaded VAD monitor calls the 2C `SpeechHandle.stop()` for barge-in.

**Tech Stack:** Python >=3.11, `openvox.stt` (faster-whisper + Silero VAD, torch-free), `openvox.tts` (Kokoro ONNX + 2C streaming, torch-free), stdlib `urllib` for the optional local-LLM helpers.

## Global Constraints

- Python >=3.11. Working dir `C:\Users\nateg\Documents\NectarSTT`.
- `import openvox`, `import openvox.agent`, and `import openvox.tts` MUST stay torch-free. Enforced by subprocess import-guard tests. A cloned `.ovx` voice still loads torch lazily inside the TTS cloned backend only.
- No new third-party dependency. The local-LLM helpers use only stdlib `urllib`.
- Reuse: `SpeechHandle` / `_StreamPlayer` from `openvox.tts.stream`; `resolve_voice` from `openvox.tts.voices`; `split_text` from `openvox.tts.segment`; `TTSResult` from `openvox.tts.backend`; STT `PartialResult`/`FinalResult` (`.is_partial`, `.text`) and `STTEngine.stream(source=None)`.
- Do NOT write any em-dash (U+2014) or en-dash (U+2013) character anywhere, including comments and docstrings. Use a hyphen, comma, or parentheses.
- Config objects are plain dataclasses copied with `dataclasses.replace`, never mutated in place.
- The whole-utterance and 2C streaming TTS behavior stays backward compatible (additive changes only).
- Commit prefixes: `feat(tts):` for tasks touching `openvox/tts`, `feat(agent):` / `test(agent):` otherwise.

---

### Task 1: Streaming sentence aggregator (`iter_sentences`)

**Files:**
- Modify: `openvox/tts/segment.py` (add `iter_sentences`)
- Test: `tests/tts/test_iter_sentences.py`

**Interfaces:**
- Produces: `iter_sentences(chunks: Iterable[str], max_chars: int = 160) -> Iterator[str]` - consumes a stream of text chunks lazily and yields complete sentences as soon as a terminator (`.`/`?`/`!` followed by whitespace) appears, splits an over-length run with no terminator at the last space before `max_chars`, and yields the trailing remainder when the input ends. Never splits mid-word.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_iter_sentences.py
from openvox.tts.segment import iter_sentences

def test_emits_sentences_as_terminators_arrive():
    chunks = ["Hello ", "there. How ", "are you? Good"]
    assert list(iter_sentences(chunks)) == ["Hello there.", "How are you?", "Good"]

def test_buffers_until_terminator():
    # no terminator until the very end -> one flushed remainder
    assert list(iter_sentences(["one ", "two ", "three"])) == ["one two three"]

def test_long_run_without_terminator_splits_at_space():
    out = list(iter_sentences(["alpha beta gamma delta epsilon zeta"], max_chars=16))
    assert all(len(s) <= 16 for s in out)
    assert " ".join(out).split() == "alpha beta gamma delta epsilon zeta".split()

def test_empty_input():
    assert list(iter_sentences([])) == []
    assert list(iter_sentences(["   "])) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_iter_sentences.py -v`
Expected: FAIL with `ImportError: cannot import name 'iter_sentences'`

- [ ] **Step 3: Write minimal implementation**

Add to `openvox/tts/segment.py` (keep the existing `split_text`; add the import if missing):

```python
import re
from collections.abc import Iterable, Iterator

_TERM = re.compile(r'[.!?](?=\s)')

def iter_sentences(chunks: Iterable[str], max_chars: int = 160) -> Iterator[str]:
    """Aggregate a stream of text chunks into complete sentences, lazily.

    Yields a sentence as soon as a terminator (. ? !) followed by whitespace is
    seen; splits an over-length terminator-free run at the last space before
    max_chars; yields the trailing remainder when the input ends."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        while True:
            m = _TERM.search(buf)
            if m:
                i = m.end()
                sent = buf[:i].strip()
                buf = buf[i:].lstrip()
                if sent:
                    yield sent
                continue
            if len(buf) >= max_chars:
                cut = buf.rfind(' ', 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                sent = buf[:cut].strip()
                buf = buf[cut:].lstrip()
                if sent:
                    yield sent
                continue
            break
    tail = buf.strip()
    if tail:
        yield tail
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_iter_sentences.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/segment.py tests/tts/test_iter_sentences.py
git commit -m "feat(tts): add iter_sentences streaming sentence aggregator"
```

---

### Task 2: `say_stream` accepts a text-chunk iterator

**Files:**
- Modify: `openvox/tts/engine.py` (`say_stream` accepts `str | Iterable[str]`)
- Test: `tests/tts/test_say_stream_iter.py`

**Interfaces:**
- Consumes: `iter_sentences` (Task 1); existing `resolve_voice`, `split_text`, `SpeechHandle`, `_StreamPlayer`.
- Produces: `TTSEngine.say_stream(text: str | Iterable[str], voice=None, speed=None) -> SpeechHandle`. A string keeps the current behavior (`split_text`); a non-string iterable of text chunks is aggregated with `iter_sentences` into a lazy segment generator passed to the streaming producer, so audio starts before the iterator is exhausted.

- [ ] **Step 1: Write the failing test**

```python
# tests/tts/test_say_stream_iter.py
import numpy as np
import pytest
from openvox.tts import TTSResult
from openvox.tts.engine import TTSEngine
from openvox.tts.config import TTSConfig

class _FakeKokoro:
    def __init__(self): self.segs = []
    def synthesize(self, text, voice, speed):
        self.segs.append(text)
        return TTSResult(audio=np.zeros(4, dtype=np.float32), sample_rate=24000)
    def stream_segments(self, segments, voice, speed):
        for seg in segments:
            yield self.synthesize(seg, voice, speed)

def _engine():
    eng = TTSEngine.__new__(TTSEngine)
    eng._config = TTSConfig(device="cpu")
    eng._backend = _FakeKokoro()
    return eng

def test_say_stream_accepts_chunk_iterator(monkeypatch):
    import openvox.tts.engine as em
    captured = {}
    class FakePlayer:
        def start(self): pass
        def put(self, a, sr): pass
        def finish(self): pass
        def abort(self): pass
        def wait_drain(self, timeout=None): pass
        def close(self): pass
    monkeypatch.setattr(em, "_StreamPlayer", lambda sr, qs: FakePlayer())
    eng = _engine()
    chunks = iter(["Hello there. ", "How are you? ", "Bye"])
    h = eng.say_stream(chunks)      # an iterator, not a string
    h.wait()
    # the fake backend saw sentence-segmented text, in order
    assert eng._backend.segs == ["Hello there.", "How are you?", "Bye"]

def test_say_stream_string_unchanged(monkeypatch):
    import openvox.tts.engine as em
    class FakePlayer:
        def start(self): pass
        def put(self, a, sr): pass
        def finish(self): pass
        def abort(self): pass
        def wait_drain(self, timeout=None): pass
        def close(self): pass
    monkeypatch.setattr(em, "_StreamPlayer", lambda sr, qs: FakePlayer())
    eng = _engine()
    eng.say_stream("One sentence. Two sentences.").wait()
    assert eng._backend.segs == ["One sentence.", "Two sentences."]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tts/test_say_stream_iter.py -v`
Expected: FAIL (say_stream rejects a non-string / iterates a str's chars)

- [ ] **Step 3: Write minimal implementation**

In `openvox/tts/engine.py`, add the import and rewrite `say_stream`:

```python
from openvox.tts.segment import split_text, iter_sentences
```

```python
    def say_stream(self, text, voice=None, speed=None):
        v = voice if voice is not None else self._config.voice
        s = speed if speed is not None else self._config.speed
        if s <= 0:
            raise ValueError("speed must be > 0")
        backend, voice_id = resolve_voice(v, self._backend, self._config.device)
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("text must be a non-empty string")
            segments = split_text(text, self._config.segment_max_chars)
        else:
            # a stream of text chunks (e.g. a streaming LLM): aggregate lazily
            segments = iter_sentences(text, self._config.segment_max_chars)
        player = _StreamPlayer(self._config.sample_rate, self._config.stream_queue_size)
        handle = SpeechHandle(player)
        handle.start(backend, voice_id, s, segments)
        return handle
```

(Confirm `resolve_voice`, `SpeechHandle`, `_StreamPlayer`, and the `split_text` import already present from 2C; only add `iter_sentences` to the segment import.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tts/test_say_stream_iter.py tests/tts/test_engine_stream.py -v`
Expected: PASS (the new iterator behavior and the existing string streaming both work)

- [ ] **Step 5: Commit**

```bash
git add openvox/tts/engine.py tests/tts/test_say_stream_iter.py
git commit -m "feat(tts): say_stream accepts a text-chunk iterator for streaming LLM output"
```

---

### Task 3: Turn + ConversationHistory

**Files:**
- Create: `openvox/agent/__init__.py`
- Create: `openvox/agent/turn.py`
- Test: `tests/agent/__init__.py`, `tests/agent/test_turn.py`

**Interfaces:**
- Produces: `Turn(role: str, text: str)` (a dataclass). `ConversationHistory(max_turns: int = 12)` - `append(turn)` keeps only the last `max_turns`; supports `__iter__`, `__len__`, `__getitem__`. Iterating yields `Turn`s oldest-first.
- Consumes: nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_turn.py
from openvox.agent.turn import Turn, ConversationHistory

def test_turn_fields():
    t = Turn("user", "hello")
    assert t.role == "user" and t.text == "hello"

def test_history_caps_and_orders():
    h = ConversationHistory(max_turns=3)
    for i in range(5):
        h.append(Turn("user", str(i)))
    assert len(h) == 3
    assert [t.text for t in h] == ["2", "3", "4"]   # last 3, oldest-first
    assert h[-1].text == "4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agent/test_turn.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.agent.turn`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/agent/turn.py
from dataclasses import dataclass

@dataclass
class Turn:
    role: str      # "user" or "assistant"
    text: str

class ConversationHistory:
    def __init__(self, max_turns: int = 12) -> None:
        self._max = max(1, max_turns)
        self._turns: list[Turn] = []

    def append(self, turn: Turn) -> None:
        self._turns.append(turn)
        if len(self._turns) > self._max:
            self._turns = self._turns[-self._max:]

    def __iter__(self):
        return iter(self._turns)

    def __len__(self):
        return len(self._turns)

    def __getitem__(self, i):
        return self._turns[i]
```

```python
# openvox/agent/__init__.py  (Task 3 version; expanded in later tasks)
"""OpenVox voice-agent: mic -> STT -> your LLM -> streaming TTS."""
from openvox.agent.turn import Turn, ConversationHistory

__all__ = ["Turn", "ConversationHistory"]
```

Also create an empty `tests/agent/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/agent/test_turn.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/agent/__init__.py openvox/agent/turn.py tests/agent/__init__.py tests/agent/test_turn.py
git commit -m "feat(agent): add Turn and capped ConversationHistory"
```

---

### Task 4: Adapters (`listen_once`, `speak_stream`)

**Files:**
- Create: `openvox/agent/adapters.py`
- Test: `tests/agent/test_adapters.py`

**Interfaces:**
- Produces:
  - `listen_once(stt, *, source=None) -> str` - iterate `stt.stream(source)` and return the text of the first non-empty `FinalResult` (an event with `is_partial == False`); return `""` if the stream ends first.
  - `speak_stream(tts, text_or_iter, *, voice=None) -> SpeechHandle` - forward to `tts.say_stream(text_or_iter, voice=voice)` (which handles both a string and a chunk iterator after Task 2).
- Consumes: STT event objects with `.is_partial` and `.text`; `TTSEngine.say_stream` (Task 2).

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_adapters.py
from openvox.agent.adapters import listen_once, speak_stream

class _Ev:
    def __init__(self, partial, text): self.is_partial = partial; self.text = text

class _FakeSTT:
    def __init__(self, events): self._events = events
    def stream(self, source=None):
        for e in self._events: yield e

def test_listen_once_returns_first_final():
    stt = _FakeSTT([_Ev(True, "hel"), _Ev(True, "hello"), _Ev(False, "hello there"), _Ev(False, "ignored")])
    assert listen_once(stt) == "hello there"

def test_listen_once_skips_empty_final():
    stt = _FakeSTT([_Ev(False, "   "), _Ev(False, "real text")])
    assert listen_once(stt) == "real text"

def test_speak_stream_forwards_to_say_stream():
    captured = {}
    class _FakeTTS:
        def say_stream(self, text, voice=None):
            captured["text"] = text; captured["voice"] = voice
            return "HANDLE"
    assert speak_stream(_FakeTTS(), "hi", voice="af_heart") == "HANDLE"
    assert captured == {"text": "hi", "voice": "af_heart"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agent/test_adapters.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.agent.adapters`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/agent/adapters.py
def listen_once(stt, *, source=None) -> str:
    """Consume the STT stream until one finalized (non-empty) utterance and
    return its text. Returns '' if the stream ends without a final."""
    for event in stt.stream(source):
        if not event.is_partial and event.text and event.text.strip():
            return event.text.strip()
    return ""

def speak_stream(tts, text_or_iter, *, voice=None):
    """Speak a string or a stream of text chunks; returns the TTS SpeechHandle
    (with stop()/wait()) so the caller controls playback and barge-in."""
    return tts.say_stream(text_or_iter, voice=voice)
```

Add to `openvox/agent/__init__.py`:

```python
from openvox.agent.adapters import listen_once, speak_stream
```
and extend `__all__` with `"listen_once", "speak_stream"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/agent/test_adapters.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/agent/adapters.py openvox/agent/__init__.py tests/agent/test_adapters.py
git commit -m "feat(agent): add listen_once and speak_stream adapters"
```

---

### Task 5: AgentConfig + VoiceAgent base loop (strict turn-taking)

**Files:**
- Create: `openvox/agent/config.py`
- Create: `openvox/agent/agent.py`
- Test: `tests/agent/test_agent_loop.py`

**Interfaces:**
- Produces:
  - `AgentConfig` (dataclass): `barge_in: bool = True`, `barge_in_debounce_s: float = 0.3`, `history_max_turns: int = 12`, `greeting: str | None = None`, `stop_phrase: str | None = None`, `on_error: str = "continue"`.
  - `VoiceAgent(llm, stt, tts, *, voice=None, barge_in=None, config=None, on_user_text=None, on_agent_text=None, on_state=None)`. `run()` drives the loop; `_speak_and_watch(reply) -> (spoken_text, interrupt_text_or_None)`. Non-callable `llm` raises `ValueError`. In this task `_watch_barge_in` is a placeholder raising `NotImplementedError` (Task 6 fills it); the base loop is validated with `barge_in=False`.
- Consumes: `listen_once`, `speak_stream` (Task 4); `Turn`, `ConversationHistory` (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_agent_loop.py
import pytest
from openvox.agent.agent import VoiceAgent
from openvox.agent.config import AgentConfig

class _Ev:
    def __init__(self, text): self.is_partial = False; self.text = text

class _ScriptSTT:
    """Yields one final per listen_once call, from a script; '' ends nothing."""
    def __init__(self, utterances): self._u = list(utterances)
    def stream(self, source=None):
        if self._u: yield _Ev(self._u.pop(0))

class _FakeHandle:
    def __init__(self): self.stopped=False; self.waited=False
    def stop(self): self.stopped=True
    def wait(self, timeout=None): self.waited=True
    @property
    def done(self): return True

class _FakeTTS:
    def __init__(self): self.said=[]
    def say_stream(self, text, voice=None):
        self.said.append(text if isinstance(text,str) else "".join(text)); return _FakeHandle()

def _agent(utterances, llm, **kw):
    return VoiceAgent(llm=llm, stt=_ScriptSTT(utterances), tts=_FakeTTS(),
                      config=AgentConfig(barge_in=False, **kw))

def test_rejects_non_callable_llm():
    with pytest.raises(ValueError):
        VoiceAgent(llm="not-callable", stt=_ScriptSTT([]), tts=_FakeTTS())

def test_one_turn_updates_history_and_speaks():
    seen = {}
    def llm(text, history): seen["history_len"]=len(history); return "reply to "+text
    ag = _agent(["hello", "bye"], llm, stop_phrase="bye")
    ag.run()
    assert ag._tts.said == ["reply to hello"]
    assert [t.text for t in ag._history] == ["hello", "reply to hello"]  # "bye" ends before append
    assert seen["history_len"] == 1   # user turn present when llm called

def test_stop_phrase_ends_without_calling_llm_on_it():
    calls = []
    def llm(text, history): calls.append(text); return "x"
    ag = _agent(["quit"], llm, stop_phrase="quit")
    ag.run()
    assert calls == []                 # stop phrase never reaches the llm

def test_on_error_continue_survives_a_raising_turn():
    def llm(text, history):
        if text == "boom": raise RuntimeError("nope")
        return "ok"
    ag = _agent(["boom", "hi", "bye"], llm, stop_phrase="bye", on_error="continue")
    ag.run()   # must not raise
    assert ag._tts.said == ["ok"]      # the boom turn was skipped, hi spoke
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agent/test_agent_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.agent.agent`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/agent/config.py
from dataclasses import dataclass

@dataclass
class AgentConfig:
    barge_in: bool = True
    barge_in_debounce_s: float = 0.3
    history_max_turns: int = 12
    greeting: str | None = None
    stop_phrase: str | None = None
    on_error: str = "continue"   # "continue" or "raise"
```

```python
# openvox/agent/agent.py
import dataclasses
import logging

from openvox.agent.adapters import listen_once, speak_stream
from openvox.agent.config import AgentConfig
from openvox.agent.turn import Turn, ConversationHistory

log = logging.getLogger(__name__)

class _Tee:
    """Wrap a text-chunk iterator, accumulating everything pulled through it."""
    def __init__(self, it): self._it = it; self.text = ""
    def __iter__(self):
        for c in self._it:
            self.text += c
            yield c

class VoiceAgent:
    def __init__(self, llm, stt, tts, *, voice=None, barge_in=None,
                 config: AgentConfig | None = None,
                 on_user_text=None, on_agent_text=None, on_state=None) -> None:
        if not callable(llm):
            raise ValueError("llm must be callable: respond(user_text, history) -> str | Iterator[str]")
        cfg = dataclasses.replace(config) if config is not None else AgentConfig()
        if barge_in is not None:
            cfg.barge_in = barge_in
        self._llm = llm
        self._stt = stt
        self._tts = tts
        self._voice = voice
        self._cfg = cfg
        self._history = ConversationHistory(cfg.history_max_turns)
        self._on_user_text = on_user_text
        self._on_agent_text = on_agent_text
        self._on_state = on_state

    def _state(self, s):
        if self._on_state:
            self._on_state(s)

    def _speak_and_watch(self, reply):
        """Speak reply (str or chunk iterator); return (spoken_text, interrupt|None)."""
        if isinstance(reply, str):
            handle = speak_stream(self._tts, reply, voice=self._voice)
            tee = None
            base_text = reply
        else:
            tee = _Tee(reply)
            handle = speak_stream(self._tts, tee, voice=self._voice)
            base_text = None
        if self._cfg.barge_in:
            interrupt = self._watch_barge_in(handle)
        else:
            handle.wait()
            interrupt = None
        spoken = base_text if base_text is not None else (tee.text if tee else "")
        return spoken, interrupt

    def _watch_barge_in(self, handle):
        raise NotImplementedError("barge-in monitor is added in Task 6")

    def run(self):
        cfg = self._cfg
        if cfg.greeting:
            speak_stream(self._tts, cfg.greeting, voice=self._voice).wait()
        pending = None
        try:
            while True:
                self._state("listening")
                user_text = pending if pending is not None else listen_once(self._stt)
                pending = None
                if not user_text:
                    continue
                if cfg.stop_phrase and user_text.strip().lower() == cfg.stop_phrase.strip().lower():
                    break
                if self._on_user_text:
                    self._on_user_text(user_text)
                self._history.append(Turn("user", user_text))
                self._state("thinking")
                try:
                    reply = self._llm(user_text, list(self._history))
                except Exception as exc:
                    if cfg.on_error == "raise":
                        raise
                    log.warning("LLM turn failed (%s); continuing.", exc)
                    continue
                self._state("speaking")
                spoken, interrupt = self._speak_and_watch(reply)
                if self._on_agent_text and spoken:
                    self._on_agent_text(spoken)
                self._history.append(Turn("assistant", spoken))
                if interrupt:
                    self._state("interrupted")
                    pending = interrupt
        except KeyboardInterrupt:
            pass
```

Add to `openvox/agent/__init__.py`: `from openvox.agent.agent import VoiceAgent` and `from openvox.agent.config import AgentConfig`; extend `__all__` with `"VoiceAgent", "AgentConfig"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/agent/test_agent_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/agent/config.py openvox/agent/agent.py openvox/agent/__init__.py tests/agent/test_agent_loop.py
git commit -m "feat(agent): add AgentConfig and VoiceAgent turn loop (strict)"
```

---

### Task 6: VAD barge-in monitor

**Files:**
- Modify: `openvox/agent/agent.py` (implement `_watch_barge_in`)
- Test: `tests/agent/test_barge_in.py`

**Interfaces:**
- Consumes: a `SpeechHandle` (`.stop()`, `.wait()`); `stt.stream(source)` yielding partial/final events.
- Produces: `VoiceAgent._watch_barge_in(handle) -> str | None` - starts a daemon thread that consumes a fresh `stt.stream()`; on the first non-empty partial it sets an interrupted flag and calls `handle.stop()`; it keeps consuming until the first non-empty final, which it returns as the interrupting utterance. It calls `handle.wait()`; if no interruption occurred (playback finished), it signals the watcher to stop and returns `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_barge_in.py
import threading
from openvox.agent.agent import VoiceAgent
from openvox.agent.config import AgentConfig

class _Ev:
    def __init__(self, partial, text): self.is_partial=partial; self.text=text

class _Handle:
    """wait() blocks until stop() or release() is called."""
    def __init__(self): self._e=threading.Event(); self.stopped=False
    def stop(self): self.stopped=True; self._e.set()
    def wait(self, timeout=None): self._e.wait(timeout)
    def release(self): self._e.set()

def _agent(stt):
    return VoiceAgent(llm=lambda t,h: "x", stt=stt, tts=object(),
                      config=AgentConfig(barge_in=True))

def test_barge_in_stops_playback_and_returns_utterance():
    # the mic stream sees the user start talking, then a final
    class STT:
        def stream(self, source=None):
            yield _Ev(True, "wai")       # partial -> speech started
            yield _Ev(False, "wait stop")  # final -> the interrupting utterance
    ag = _agent(STT())
    h = _Handle()
    out = ag._watch_barge_in(h)
    assert h.stopped is True
    assert out == "wait stop"

def test_no_speech_returns_none_and_does_not_stop():
    # empty mic stream; playback finishes on its own
    class STT:
        def stream(self, source=None):
            if False: yield   # empty generator
    ag = _agent(STT())
    h = _Handle()
    # playback ends shortly after the watcher starts
    threading.Timer(0.05, h.release).start()
    out = ag._watch_barge_in(h)
    assert out is None
    assert h.stopped is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agent/test_barge_in.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

Replace `_watch_barge_in` in `openvox/agent/agent.py`:

```python
    def _watch_barge_in(self, handle):
        import threading
        interrupted = threading.Event()
        stop_watch = threading.Event()
        result = {}

        def watch():
            try:
                for event in self._stt.stream(None):
                    if stop_watch.is_set():
                        break
                    txt = (event.text or "").strip()
                    if not txt:
                        continue
                    if event.is_partial:
                        if not interrupted.is_set():
                            interrupted.set()
                            handle.stop()     # 2C barge-in: cut within one audio block
                    else:
                        result["text"] = txt   # the interrupting utterance
                        break
            except Exception as exc:            # never let the watcher crash the turn
                log.debug("barge-in watcher stopped: %s", exc)

        th = threading.Thread(target=watch, daemon=True)
        th.start()
        handle.wait()                           # returns when playback ends OR was stopped
        if interrupted.is_set():
            th.join(timeout=3.0)                 # let it capture the interrupting final
            return result.get("text", "")
        stop_watch.set()
        self._close_source_best_effort()
        th.join(timeout=1.0)
        return None

    def _close_source_best_effort(self):
        # Best effort: unblock a mic read the watcher may be parked on. The STT
        # mic source is opened inside stt.stream(); if the backend exposes a way
        # to interrupt it in future, call it here. For now the watcher thread is
        # a daemon and ends with the process if it cannot be unblocked sooner.
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/agent/test_barge_in.py tests/agent/test_agent_loop.py -v`
Expected: PASS (barge-in path works; the strict-mode tests still pass)

- [ ] **Step 5: Commit**

```bash
git add openvox/agent/agent.py tests/agent/test_barge_in.py
git commit -m "feat(agent): add VAD barge-in monitor (stop playback, carry the interrupting utterance)"
```

---

### Task 7: Local-LLM helpers (`ollama`, `openai_compatible`)

**Files:**
- Create: `openvox/agent/llm.py`
- Test: `tests/agent/test_llm_helpers.py`

**Interfaces:**
- Produces:
  - `ollama(model: str, host: str = "http://localhost:11434") -> callable` returning `respond(user_text, history) -> Iterator[str]` that streams from Ollama's `/api/chat` (newline-delimited JSON, each line `{"message": {"content": "..."}}`).
  - `openai_compatible(base_url: str, model: str = "local", api_key: str | None = None) -> callable` returning `respond(user_text, history) -> Iterator[str]` that streams from a local OpenAI-compatible `/chat/completions` (SSE lines `data: {"choices":[{"delta":{"content":"..."}}]}`, terminated by `data: [DONE]`).
  - Both build a messages list from `history` (each `Turn.role`/`Turn.text`) plus the new user text, use stdlib `urllib.request`, and raise a clear error if the endpoint is unreachable.
- Consumes: `Turn` (duck-typed `.role`/`.text`).

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_llm_helpers.py
import io, json
import pytest
from openvox.agent.llm import ollama, openai_compatible
from openvox.agent.turn import Turn

class _FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False

def test_ollama_streams_content(monkeypatch):
    lines = [json.dumps({"message": {"content": c}}) for c in ["Hel", "lo", "!"]]
    body = ("\n".join(lines) + "\n").encode()
    def fake_urlopen(req, timeout=None):
        assert req.full_url.endswith("/api/chat")
        payload = json.loads(req.data)
        assert payload["model"] == "m" and payload["stream"] is True
        assert payload["messages"][-1] == {"role": "user", "content": "hi"}
        return _FakeResp(body)
    monkeypatch.setattr("openvox.agent.llm.urllib.request.urlopen", fake_urlopen)
    respond = ollama(model="m")
    assert "".join(respond("hi", [Turn("user", "prev"), Turn("assistant", "yo")])) == "Hello!"

def test_openai_compatible_streams_sse(monkeypatch):
    chunks = ["Hi", " there"]
    sse = "".join("data: " + json.dumps({"choices":[{"delta":{"content":c}}]}) + "\n\n" for c in chunks)
    sse += "data: [DONE]\n\n"
    def fake_urlopen(req, timeout=None):
        assert req.full_url.endswith("/chat/completions")
        return _FakeResp(sse.encode())
    monkeypatch.setattr("openvox.agent.llm.urllib.request.urlopen", fake_urlopen)
    respond = openai_compatible(base_url="http://localhost:8080/v1")
    assert "".join(respond("hi", [])) == "Hi there"

def test_unreachable_endpoint_raises(monkeypatch):
    def boom(req, timeout=None): raise OSError("connection refused")
    monkeypatch.setattr("openvox.agent.llm.urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="reach"):
        list(ollama(model="m")("hi", []))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agent/test_llm_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: openvox.agent.llm`

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/agent/llm.py
import json
import urllib.request

def _messages(history, user_text):
    msgs = []
    for turn in history:
        role = "assistant" if turn.role == "assistant" else "user"
        msgs.append({"role": role, "content": turn.text})
    msgs.append({"role": "user", "content": user_text})
    return msgs

def _post(url, payload, headers, who):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req, timeout=120)
    except OSError as exc:
        raise RuntimeError(f"could not reach {who} at {url}: {exc}. Is the local server running?") from exc

def ollama(model: str, host: str = "http://localhost:11434"):
    url = host.rstrip("/") + "/api/chat"
    def respond(user_text, history):
        payload = {"model": model, "stream": True, "messages": _messages(history, user_text)}
        with _post(url, payload, {}, "Ollama") as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                obj = json.loads(line)
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    yield piece
    return respond

def openai_compatible(base_url: str, model: str = "local", api_key: str | None = None):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    def respond(user_text, history):
        payload = {"model": model, "stream": True, "messages": _messages(history, user_text)}
        with _post(url, payload, headers, "the OpenAI-compatible server") as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content", "")
                if piece:
                    yield piece
    return respond
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/agent/test_llm_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/agent/llm.py tests/agent/test_llm_helpers.py
git commit -m "feat(agent): add local Ollama and OpenAI-compatible LLM helpers"
```

---

### Task 8: Demo CLI + packaging + import guard

**Files:**
- Create: `openvox/agent/demo.py`
- Modify: `pyproject.toml` (`[agent]` extra + `openvox-agent-demo` entry point)
- Test: `tests/agent/test_demo.py`, `tests/agent/test_import_lean.py`

**Interfaces:**
- Produces: `openvox/agent/demo.py` with `build_parser()` and `main(argv=None) -> int`. Flags: `--llm` (`echo`/`ollama`/`openai`, default `echo`), `--model`, `--base-url`, `--voice`, `--stt-model` (default `base`), `--no-barge-in`. The `echo` LLM returns a short acknowledgement so the demo runs with no model. `main` builds STT + TTS + the chosen LLM and calls `VoiceAgent(...).run()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_demo.py
from openvox.agent.demo import build_parser, _echo_llm

def test_parser_defaults():
    a = build_parser().parse_args([])
    assert a.llm == "echo" and a.stt_model == "base" and a.barge_in is True

def test_parser_flags():
    a = build_parser().parse_args(["--llm","ollama","--model","llama3.2","--voice","alice.ovx","--no-barge-in"])
    assert a.llm == "ollama" and a.model == "llama3.2" and a.voice == "alice.ovx" and a.barge_in is False

def test_echo_llm_responds():
    assert isinstance(_echo_llm("hello", []), str)
    assert "hello" in _echo_llm("hello", [])
```

```python
# tests/agent/test_import_lean.py
import subprocess, sys

def test_import_openvox_agent_torch_free():
    code = (
        "import sys\n"
        "for m in ('torch','chatterbox'):\n"
        "    sys.modules[m]=None\n"
        "import openvox.agent\n"
        "from openvox.agent import VoiceAgent, listen_once, speak_stream, Turn\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable,"-c",code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/agent/test_demo.py tests/agent/test_import_lean.py -v`
Expected: FAIL (`openvox.agent.demo` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# openvox/agent/demo.py
import argparse
import sys

from openvox.agent import VoiceAgent

def _echo_llm(user_text, history):
    return f"You said: {user_text}. I am OpenVox, running entirely on this machine."

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openvox-agent-demo",
                                description="Offline voice assistant: mic -> STT -> your LLM -> streaming TTS.")
    p.add_argument("--llm", choices=["echo", "ollama", "openai"], default="echo")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--base-url", dest="base_url", default="http://localhost:8080/v1")
    p.add_argument("--voice", default="af_heart", help="A built-in voice name or an .ovx profile path.")
    p.add_argument("--stt-model", dest="stt_model", default="base")
    p.add_argument("--no-barge-in", dest="barge_in", action="store_false")
    p.set_defaults(barge_in=True)
    return p

def _make_llm(args):
    if args.llm == "ollama":
        from openvox.agent.llm import ollama
        return ollama(model=args.model)
    if args.llm == "openai":
        from openvox.agent.llm import openai_compatible
        return openai_compatible(base_url=args.base_url, model=args.model)
    return _echo_llm

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from openvox.stt import STTEngine
    from openvox.tts import TTSEngine
    stt = STTEngine(model=args.stt_model)
    tts = TTSEngine(voice=args.voice)
    agent = VoiceAgent(llm=_make_llm(args), stt=stt, tts=tts,
                       voice=args.voice, barge_in=args.barge_in)
    print("OpenVox assistant ready. Speak; Ctrl-C to quit.")
    agent.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Add to `pyproject.toml` `[project.optional-dependencies]`:

```toml
agent    = ["openvox[stt,tts]"]
```

and to `[project.scripts]`:

```toml
openvox-agent-demo = "openvox.agent.demo:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/agent/test_demo.py tests/agent/test_import_lean.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add openvox/agent/demo.py pyproject.toml tests/agent/test_demo.py tests/agent/test_import_lean.py
git commit -m "feat(agent): add voice-assistant demo CLI, [agent] extra, import guard"
```

---

### Task 9: Real end-to-end integration test

**Files:**
- Test: `tests/agent/test_integration.py`

**Interfaces:**
- Consumes: `openvox.stt.STTEngine.transcribe_file`, `openvox.tts.segment.iter_sentences`. Marked `@pytest.mark.integration`; uses real STT on a local WAV (no mic, no output device).

- [ ] **Step 1: Write the test**

```python
# tests/agent/test_integration.py
import glob
import pytest

pytestmark = pytest.mark.integration

WAVS = sorted(glob.glob("*.wav"))

@pytest.mark.skipif(not WAVS, reason="no local wav to transcribe")
def test_stt_to_stub_llm_to_sentences():
    from openvox.stt import STTEngine
    from openvox.tts.segment import iter_sentences

    text = STTEngine(model="base", device="cpu").transcribe_file(WAVS[0]).text
    assert isinstance(text, str)

    # a stub streaming LLM: echo the transcript back as a two-sentence reply
    def stub_stream():
        for chunk in [f"You said {text[:40]}. ", "This is OpenVox speaking locally."]:
            yield chunk

    sentences = list(iter_sentences(stub_stream()))
    assert len(sentences) >= 2
    assert all(s.strip() for s in sentences)
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/agent/test_integration.py -v -m integration`
Expected: PASS on a machine with a local `.wav` (downloads the STT `base` model once). SKIP if none present.

- [ ] **Step 3: Commit**

```bash
git add tests/agent/test_integration.py
git commit -m "test(agent): end-to-end STT -> streaming aggregation integration"
```

---

## Self-Review

**Spec coverage:**
- Spec sentence aggregator -> Task 1 (`iter_sentences`, placed in `openvox.tts.segment` so `say_stream` reuses it; the spec's `openvox/agent/sentence.py` is intentionally consolidated here to avoid duplication and to make `say_stream(iterable)` a real public capability). Streaming bridge (speak while generating) -> Tasks 1+2. `Turn`/history -> Task 3. Adapters -> Task 4. `VoiceAgent` loop + config + callbacks + greeting + stop_phrase + on_error -> Task 5. VAD barge-in -> Task 6. Local-LLM helpers -> Task 7. Demo + `[agent]` extra + entry point + import guard -> Task 8. Integration -> Task 9. ✓
- Spec §7 config fields -> Task 5 (`barge_in`, `barge_in_debounce_s`, `history_max_turns`, `greeting`, `stop_phrase`, `on_error`; `sentence_max_chars` is served by the TTS `segment_max_chars` via `say_stream`, so it is not duplicated on `AgentConfig`). ✓
- Spec §8 errors: non-callable llm (Task 5), llm raises mid-turn (Task 5 `on_error`), unreachable endpoint (Task 7), stop_phrase / KeyboardInterrupt end cleanly (Task 5). No-device errors surface from the underlying engines through `listen_once`/`speak_stream`. ✓
- Spec §4 torch-free boundary: `openvox.agent` imports only `openvox.stt`/`openvox.tts` (torch-free) and stdlib; import-guard (Task 8). ✓

**Placeholder scan:** No TBD/TODO; every step has concrete code and tests. The Task 5 `_watch_barge_in` placeholder raising `NotImplementedError` is deliberate and filled in Task 6 (the base-loop tests use `barge_in=False`, so it is never hit there). ✓

**Type consistency:** `iter_sentences(chunks, max_chars)` identical in Tasks 1, 2, 9. `say_stream(text: str | Iterable[str], voice, speed)` consistent Tasks 2 and 4. `listen_once(stt, source=None) -> str` and `speak_stream(tts, text_or_iter, voice=None) -> SpeechHandle` consistent Tasks 4, 5, 6. `Turn(role, text)` / `ConversationHistory(max_turns)` consistent Tasks 3, 5, 7. `respond(user_text, history) -> str | Iterator[str]` is the contract used by Tasks 5, 7, 8. `_watch_barge_in(handle) -> str | None` consistent Tasks 5 (stub) and 6. ✓

**Deviations from spec, noted deliberately:** (1) the sentence aggregator lives in `openvox.tts.segment` (not `openvox/agent/sentence.py`) so `say_stream` owns the streaming capability and there is no duplication; (2) `sentence_max_chars` is not repeated on `AgentConfig` since `say_stream` uses the TTS config's `segment_max_chars`. Both keep the agent thin glue over the TTS engine, matching the spec's architecture intent.
