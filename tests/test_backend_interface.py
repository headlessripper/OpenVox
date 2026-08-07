import numpy as np
import pytest
from nectarstt.engine.backend import StreamingBackend, BackendResult
from nectarstt.events import WordTiming

def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        StreamingBackend()

def test_concrete_subclass_works():
    class Fake(StreamingBackend):
        def transcribe(self, audio, sample_rate, language, word_timestamps):
            return BackendResult(text="ok", words=[WordTiming("ok", 0.0, 0.1, 1.0)])
    r = Fake().transcribe(np.zeros(16000, dtype=np.float32), 16000, "en", True)
    assert r.text == "ok"
    assert r.words[0].word == "ok"
