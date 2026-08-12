# OpenVox - LLM Plug-and-Play Voice Agent (Sub-project 3) - Design

**Date:** 2026-08-12
**Status:** Approved (design), proceeding to implementation plan
**Scope:** A drop-in route that turns OpenVox's STT + TTS into a voice-assistant
loop: mic to speech-to-text to your LLM to streaming text-to-speech, with VAD
barge-in. Ships both a batteries-included `VoiceAgent` and the clean adapters it
is built on. Bundles no model of any kind; the LLM is bring-your-own.

---

## 1. Goal and positioning

People building voice assistants and agent services need the voice I/O, not
another model. OpenVox already has streaming STT (mic to text with neural VAD)
and 2C streaming TTS with a thread-safe `stop()` barge-in. This sub-project wires
them into a conversation loop that a user plugs their own LLM into, and exposes
the same wiring as low-level building blocks for people who already have an agent
framework. It stays fully offline-capable: STT and TTS run on device, and the LLM
is injected (a local runner such as Ollama or llama.cpp, or any callable).

## 2. Decisions locked during brainstorming

| Decision | Choice |
| --- | --- |
| Shape | **Both**: a batteries-included `VoiceAgent` loop built on clean adapters (`listen_once`, `speak_stream`) |
| LLM contract | **Bring-your-own, streaming-first**: a callable `respond(user_text, history) -> str \| Iterator[str]`; no LLM bundled |
| Local-LLM convenience | Thin optional helpers for **local** runners (Ollama, OpenAI-compatible), stdlib `urllib` only, zero new deps |
| Streaming bridge | LLM text chunks are buffered into sentences and spoken as they form (speak while generating) |
| Turn-taking (v1) | **VAD barge-in now**: sustained user speech during playback triggers `handle.stop()` and a new turn; `barge_in=False` for strict turn-taking |
| Echo handling | Deferred: v1 mitigates with a debounce + energy threshold and documents headset / push-to-talk; true acoustic echo cancellation (full-duplex) is a later sub-project |
| Placement | New `openvox.agent` engine (pure glue over stt + tts) |
| Torch-free import | `import openvox.agent` stays torch-free (stt + Kokoro tts are torch-free at import; a cloned `.ovx` voice loads torch lazily) |

**Out of scope (deferred):** acoustic echo cancellation / true full-duplex;
wake-word activation; bundling or downloading an LLM; multi-speaker diarization
in the loop; remote (non-local) LLM adapters as first-class (the helpers target
local endpoints, though any callable the user writes can do whatever it wants).

## 3. Scope

### In scope

- `openvox/agent/` with:
  - `Turn(role, text)` and a small capped `ConversationHistory`.
  - A streaming **sentence aggregator**: LLM text chunks in, complete sentences out.
  - Adapters: `listen_once(stt, ...) -> str` and
    `speak_stream(tts, text_or_iter, voice=...) -> SpeechHandle`.
  - `VoiceAgent`: the mic to STT to LLM to streaming-TTS loop with VAD barge-in,
    lifecycle callbacks, and history management.
  - Optional thin local-LLM helpers (`ollama`, `openai_compatible`), stdlib only.
  - A demo CLI with an `echo` stub LLM so it runs without a model.
- A `[agent]` optional extra and a demo entry point.
- Tests: unit (no mic / audio / torch / LLM) plus one opt-in integration.

### Out of scope (deferred)

- Echo cancellation / full-duplex; wake-word; bundled LLM; diarization; a GUI.

## 4. Architecture

### Package layout

```
openvox/agent/
  __init__.py       # exports VoiceAgent, Turn, listen_once, speak_stream
  turn.py           # Turn(role, text) + ConversationHistory (capped)
  sentence.py       # SentenceAggregator: feed(chunk) -> list[str]; flush() -> list[str]
  adapters.py       # listen_once(stt, ...) -> str ; speak_stream(tts, text_or_iter, voice) -> SpeechHandle
  agent.py          # VoiceAgent + AgentConfig
  llm.py            # optional local-LLM helpers (ollama, openai_compatible), stdlib urllib
  demo.py           # python -m openvox.agent.demo (echo stub LLM by default)
tests/agent/
```

### Components (each independently testable)

- **SentenceAggregator (`sentence.py`)** - pure Python, no deps. `feed(chunk: str)
  -> list[str]` appends to an internal buffer and returns any complete sentences
  now available (split on `.`/`?`/`!` at a boundary, or when the buffer exceeds
  `max_chars`, reusing the same splitting rules as `openvox.tts.segment`);
  `flush() -> list[str]` returns whatever remains. This is what lets speech start
  before the LLM has finished.
- **Turn / ConversationHistory (`turn.py`)** - `Turn(role: str, text: str)`;
  `ConversationHistory(max_turns)` is an append-with-cap list passed to the LLM.
- **Adapters (`adapters.py`)** - the low-level building blocks:
  - `listen_once(stt, *, timeout=None) -> str`: consume `stt.stream()` until an
    end-of-utterance final (VAD silence) and return the finalized text.
  - `speak_stream(tts, text_or_iter, *, voice=None) -> SpeechHandle`: if given a
    string, call `tts.say_stream(text, voice=voice)`; if given a text-chunk
    iterator (a streaming LLM), run it through a `SentenceAggregator` and feed
    each finished sentence to a streaming `say_stream` session, returning the 2C
    `SpeechHandle` so the caller keeps `stop()`/`wait()`.
- **VoiceAgent (`agent.py`)** - the loop, built on the adapters. `run()` repeats:
  `listen_once` -> append user turn -> `speak_stream(respond(user_text, history))`
  -> during playback, a listener watches the STT VAD; on sustained user speech it
  calls `handle.stop()` and carries the interrupting utterance into the next turn;
  otherwise `handle.wait()` -> append the assistant turn. Fires `on_user_text`,
  `on_agent_text`, `on_state` callbacks. `barge_in=False` skips the listener and
  always `wait()`s. An optional `greeting` is spoken on start; an optional
  `stop_phrase` ends the loop; `on_error="continue"` keeps the session alive when
  a turn raises.
- **LLM helpers (`llm.py`)** - `ollama(model, host="http://localhost:11434")` and
  `openai_compatible(base_url, model="local", api_key=None)` each return a
  `respond(user_text, history) -> Iterator[str]` that streams from a **local**
  endpoint via stdlib `urllib`. No third-party dependency; entirely optional.

`import openvox` and `import openvox.agent` stay torch-free: the module composes
`openvox.stt` and `openvox.tts` (both torch-free at import); a cloned `.ovx` voice
only loads torch lazily inside the TTS cloned backend, as in 2C.

## 5. Public API

```python
from openvox.agent import VoiceAgent
from openvox.stt import STTEngine
from openvox.tts import TTSEngine

def respond(user_text, history):        # bring-your-own LLM
    for chunk in my_llm.stream(user_text):
        yield chunk                      # or: return "a whole string"

agent = VoiceAgent(
    llm=respond,
    stt=STTEngine(model="base"),
    tts=TTSEngine(voice="af_heart"),     # or voice="alice.ovx"
    barge_in=True,
)
agent.run()                              # blocks until stop_phrase / KeyboardInterrupt

# Low-level building blocks:
from openvox.agent import listen_once, speak_stream
text = listen_once(stt)
speak_stream(tts, respond(text, []), voice="af_heart").wait()

# Optional local-LLM helpers:
from openvox.agent.llm import ollama, openai_compatible
VoiceAgent(llm=ollama(model="llama3.2"), stt=stt, tts=tts).run()
```

## 6. Data flow

```
run() loop (barge_in=True):
  user_text = listen_once(stt)                       # mic -> STT finals (VAD end-of-utterance)
  history.append(Turn("user", user_text))
  reply = respond(user_text, history)                # str or Iterator[str]
  handle = speak_stream(tts, reply, voice)           # chunks -> SentenceAggregator -> say_stream
      producer: for chunk in reply:
                  for sentence in aggregator.feed(chunk): enqueue sentence -> TTS
                for sentence in aggregator.flush():   enqueue -> TTS
  meanwhile (barge_in): listener watches stt VAD
      if user speaks > barge_in_debounce_s: handle.stop(); capture utterance; next turn
  else: handle.wait()
  history.append(Turn("assistant", spoken_text))
```

## 7. Configuration

`AgentConfig`: `barge_in=True`, `barge_in_debounce_s=0.3`, `history_max_turns=12`,
`sentence_max_chars=160`, `greeting=None`, `stop_phrase=None`,
`on_error="continue"` (or `"raise"`). `VoiceAgent(llm, stt, tts, voice=None,
barge_in=None, config=None)` overrides fields when provided; a passed config is
copied, not mutated.

## 8. Error handling

- No `llm` provided (not callable) -> `ValueError` at construction.
- LLM raises mid-turn -> with `on_error="continue"`, log a one-line notice and
  return to listening; with `"raise"`, propagate.
- No microphone / output device -> `AudioDeviceError` surfaced from
  `listen_once` / `speak_stream` (as the underlying engines already do).
- A local-LLM helper endpoint is unreachable -> a clear error naming the host/URL.
- `stop_phrase` match (case-insensitive, trimmed) ends `run()` cleanly.
- `KeyboardInterrupt` ends `run()` cleanly (stops any active playback).

## 9. Testing

- **Unit (no mic, no audio device, no torch, no LLM):**
  - `SentenceAggregator`: chunk sequences emit sentences at terminators; partial
    text stays buffered; `flush()` returns the remainder; overlong buffer splits
    at `max_chars`.
  - `Turn` / `ConversationHistory`: append order and capping at `max_turns`.
  - `listen_once` with a fake STT (scripted partial then final events) returns the
    final text.
  - `speak_stream`: a string calls `say_stream(text, voice)`; a chunk iterator is
    aggregated into sentences and each fed to the fake TTS, returning its handle.
  - `VoiceAgent` with fakes (fake stt / llm / tts + a fake `SpeechHandle`): one
    full turn updates history and fires callbacks; the barge-in path (fake VAD
    signals mid-speech) calls `handle.stop()` and starts a new turn with the
    interrupting utterance; `barge_in=False` calls `wait()` not `stop()`;
    `stop_phrase` ends `run()`; `on_error="continue"` survives a raising turn.
  - `llm.py` helpers build the request and parse a streamed response against a
    monkeypatched `urllib` (no network); an unreachable endpoint raises clearly.
  - Import-guard subprocess test: `import openvox.agent` with `torch`/`chatterbox`
    blocked succeeds.
- **Integration (heavy, opt-in `@pytest.mark.integration`):** real STT
  `transcribe_file` on a known WAV -> a stub streaming LLM -> `SentenceAggregator`,
  asserting non-empty sentences flow end-to-end without an audio device. (The real
  mic loop and device playback are exercised by the demo and 2C's TTS integration.)

## 10. Dependencies and packaging

New `[agent]` extra = `openvox[stt,tts]`. The LLM helpers use only stdlib
`urllib`, so no third-party dependency is added. `[agent]` stays out of `all`
only if `all` would otherwise pull audio extras it does not already include; since
`all` already contains `stt`, `stt-demo`, and `tts`, `[agent]` may be folded in or
kept separate at implementation time (kept separate by default for clarity). New
demo entry point `openvox-agent-demo`.

## 11. Demo CLI

`python -m openvox.agent.demo` (entry point `openvox-agent-demo`) runs a real
voice assistant. Flags: `--llm` (`ollama`/`openai`/`echo`, default `echo` so it
runs with no model), `--model`, `--base-url`, `--voice` (a built-in name or an
`.ovx` profile), `--stt-model` (default `base`), `--no-barge-in`. The `echo` LLM
simply speaks back a short acknowledgement, enough to demonstrate the full loop
and barge-in without any model installed.

## 12. Definition of done

1. `VoiceAgent(llm, stt, tts).run()` runs a working mic to STT to LLM to
   streaming-TTS assistant with VAD barge-in.
2. `listen_once` and `speak_stream` are usable as standalone building blocks.
3. Streaming LLM output is spoken sentence-by-sentence as it generates.
4. `import openvox` / `import openvox.agent` stay torch-free; unit tests pass with
   no mic, audio device, torch, or LLM.
5. The optional local-LLM helpers stream from a local endpoint; the demo runs a
   real assistant, defaulting to an `echo` stub that needs no model.
