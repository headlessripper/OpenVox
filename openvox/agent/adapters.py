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
