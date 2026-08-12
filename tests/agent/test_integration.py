# tests/agent/test_integration.py
import math
import struct
import wave

import pytest

pytestmark = pytest.mark.integration

def _make_16k_wav(path, seconds=1.0, sr=16000, freq=180.0):
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq * t / sr)))
            for t in range(n)
        )
        w.writeframes(frames)

def test_stt_to_stub_llm_to_sentences(tmp_path):
    from openvox.stt import STTEngine
    from openvox.tts.segment import iter_sentences

    wav = tmp_path / "probe.wav"
    _make_16k_wav(wav)
    text = STTEngine(model="base", device="cpu").transcribe_file(str(wav)).text
    assert isinstance(text, str)   # may be empty for a tone; only the type matters here

    def stub_stream():
        for chunk in ["You said something. ", "This is OpenVox speaking locally."]:
            yield chunk

    sentences = list(iter_sentences(stub_stream()))
    assert len(sentences) >= 2
    assert all(s.strip() for s in sentences)
