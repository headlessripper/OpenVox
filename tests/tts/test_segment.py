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
