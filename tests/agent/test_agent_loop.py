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

class _RaisingHandle:
    """A handle whose wait() consumes the (streaming) text and re-raises,
    simulating a streaming LLM whose exception surfaces during playback."""
    def __init__(self, text_iter): self._it = text_iter; self.stopped = False
    def stop(self): self.stopped = True
    def wait(self, timeout=None):
        for _ in self._it:
            pass
    @property
    def done(self): return True

class _DeferredRaiseTTS:
    """say_stream() succeeds immediately; the exception only surfaces when
    the returned handle's wait() iterates the (streaming) text."""
    def __init__(self): self.said=[]
    def say_stream(self, text, voice=None):
        if isinstance(text, str):
            self.said.append(text)
            return _FakeHandle()
        return _RaisingHandle(text)

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

def test_on_error_continue_survives_a_streaming_llm_raising_during_speak():
    def boom_chunks():
        yield "par"
        raise RuntimeError("stream broke")

    def llm(text, history):
        if text == "boom": return boom_chunks()
        return "ok"
    ag = VoiceAgent(llm=llm, stt=_ScriptSTT(["boom", "hi", "bye"]), tts=_DeferredRaiseTTS(),
                     config=AgentConfig(barge_in=False, stop_phrase="bye", on_error="continue"))
    ag.run()   # must not raise even though the exception surfaces in wait()
    assert ag._tts.said == ["ok"]      # the boom turn was skipped, hi spoke normally
