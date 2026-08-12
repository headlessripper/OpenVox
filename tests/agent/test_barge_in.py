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

def _agent(stt, **kw):
    return VoiceAgent(llm=lambda t,h: "x", stt=stt, tts=object(),
                      config=AgentConfig(barge_in=True, **kw))

def test_barge_in_stops_playback_and_returns_utterance():
    # the mic stream sees the user start talking, then a final
    class STT:
        def stream(self, source=None):
            yield _Ev(True, "wai")       # partial -> speech started
            yield _Ev(False, "wait stop")  # final -> the interrupting utterance
    ag = _agent(STT(), barge_in_debounce_s=0.0)
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

def test_debounce_suppresses_brief_blip():
    # a single instant partial immediately followed by a final should NOT
    # fire handle.stop() when the debounce window is large
    class STT:
        def stream(self, source=None):
            yield _Ev(True, "u")           # partial -> too brief to count
            yield _Ev(False, "u")          # final arrives before debounce elapses
    ag = _agent(STT(), barge_in_debounce_s=5.0)
    h = _Handle()
    threading.Timer(0.05, h.release).start()
    out = ag._watch_barge_in(h)
    assert h.stopped is False
    assert out is None
