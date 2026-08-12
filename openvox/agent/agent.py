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
        import threading
        import time
        from openvox.stt.audio.sources import MicSource
        interrupted = threading.Event()
        result = {}
        source = MicSource()   # explicit, so we can close it to unblock the watcher

        def watch():
            first_partial = None
            try:
                for event in self._stt.stream(source):
                    txt = (event.text or "").strip()
                    if not txt:
                        continue
                    if event.is_partial:
                        if first_partial is None:
                            first_partial = time.monotonic()
                        if (not interrupted.is_set()
                                and time.monotonic() - first_partial >= self._cfg.barge_in_debounce_s):
                            interrupted.set()
                            handle.stop()   # 2C barge-in: cut within one audio block
                    else:
                        result["text"] = txt   # the interrupting utterance
                        break
            except Exception as exc:           # never let the watcher crash the turn
                log.debug("barge-in watcher stopped: %s", exc)

        th = threading.Thread(target=watch, daemon=True)
        th.start()
        handle.wait()                           # returns when playback ends or was stopped
        if interrupted.is_set():
            th.join(timeout=3.0)                 # let it capture the interrupting final
        self._close_source(source)               # ALWAYS release the mic + unblock the watcher
        th.join(timeout=1.0)
        return result.get("text") if interrupted.is_set() else None

    def _close_source(self, source):
        try:
            source.close()
        except Exception:
            pass

    def run(self):
        cfg = self._cfg
        pending = None
        try:
            if cfg.greeting:
                speak_stream(self._tts, cfg.greeting, voice=self._voice).wait()
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
                    self._state("speaking")
                    spoken, interrupt = self._speak_and_watch(reply)
                except Exception as exc:
                    if cfg.on_error == "raise":
                        raise
                    log.warning("Turn failed (%s); continuing.", exc)
                    continue
                if self._on_agent_text and spoken:
                    self._on_agent_text(spoken)
                self._history.append(Turn("assistant", spoken))
                if interrupt:
                    self._state("interrupted")
                    pending = interrupt
        except KeyboardInterrupt:
            pass
