# tests/agent/test_integration.py
import glob
import pytest

pytestmark = pytest.mark.integration

WAVS = sorted(glob.glob("*.wav"))

@pytest.mark.skipif(not WAVS, reason="no local wav to transcribe")
def test_stt_to_stub_llm_to_sentences():
    from openvox.stt import STTEngine
    from openvox.tts.segment import iter_sentences

    text = STTEngine(model="base", device="cpu").transcribe_file(WAVS[0]).text
    assert isinstance(text, str)

    # a stub streaming LLM: echo the transcript back as a two-sentence reply
    def stub_stream():
        for chunk in [f"You said {text[:40]}. ", "This is OpenVox speaking locally."]:
            yield chunk

    sentences = list(iter_sentences(stub_stream()))
    assert len(sentences) >= 2
    assert all(s.strip() for s in sentences)
