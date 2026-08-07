from openvox.stt.events import WordTiming, PartialResult, FinalResult

def test_partial_is_partial():
    p = PartialResult(text="hello wor", committed_prefix="hello", volatile_tail="wor")
    assert p.is_partial is True
    assert p.text == "hello wor"

def test_final_is_not_partial():
    w = [WordTiming(word="hello", start=0.0, end=0.4, probability=0.9)]
    f = FinalResult(text="hello", words=w, start=0.0, end=0.4)
    assert f.is_partial is False
    assert f.words[0].word == "hello"
