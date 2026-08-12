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
