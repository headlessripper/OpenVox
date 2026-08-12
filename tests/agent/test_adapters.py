from openvox.agent.adapters import listen_once, speak_stream


class _Ev:
    def __init__(self, partial, text):
        self.is_partial = partial
        self.text = text


class _FakeSTT:
    def __init__(self, events):
        self._events = events

    def stream(self, source=None):
        for e in self._events:
            yield e


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
            captured["text"] = text
            captured["voice"] = voice
            return "HANDLE"

    assert speak_stream(_FakeTTS(), "hi", voice="af_heart") == "HANDLE"
    assert captured == {"text": "hi", "voice": "af_heart"}
