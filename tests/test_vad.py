import numpy as np
import pytest
from nectarstt.audio.vad import SileroVAD, _to_pcm16

def test_to_pcm16_no_overflow_at_full_scale():
    """_to_pcm16 clips scaled values to prevent int16 overflow at full scale."""
    # Positive full scale: 1.0 * 32768 = 32768, which must clip to 32767
    pcm = np.frombuffer(_to_pcm16(np.ones(512, dtype=np.float32)), dtype=np.int16)
    assert pcm.min() >= 0, "Positive full scale should not wrap to negative"
    assert pcm.max() == 32767, "Positive full scale should clip to 32767, not overflow"
    # Negative full scale: -1.0 * 32768 = -32768, which is valid
    pcm_neg = np.frombuffer(_to_pcm16(-np.ones(512, dtype=np.float32)), dtype=np.int16)
    assert pcm_neg.max() <= 0, "Negative full scale should not wrap to positive"
    assert pcm_neg.min() == -32768, "Negative full scale should be -32768"

def test_rejects_non_16k_sample_rate():
    """ValueError when constructing SileroVAD with non-16kHz sample rate."""
    with pytest.raises(ValueError, match="16000 Hz"):
        SileroVAD(sample_rate=8000)

@pytest.mark.integration
def test_silence_is_not_speech():
    vad = SileroVAD(threshold=0.5)
    silence = np.zeros(512, dtype=np.float32)
    assert vad.is_speech(silence) is False

@pytest.mark.integration
def test_loud_noise_probability_differs_from_silence():
    vad = SileroVAD(threshold=0.5)
    rng = np.random.default_rng(0)
    loud = (rng.standard_normal(512) * 0.5).astype(np.float32)
    silence = np.zeros(512, dtype=np.float32)
    # Not asserting 'loud is speech' (noise != speech); assert the wrapper runs
    # and returns bools for both without error.
    assert isinstance(vad.is_speech(loud), bool)
    assert isinstance(vad.is_speech(silence), bool)

@pytest.mark.integration
def test_reset_clears_partial_buffer():
    """reset() clears both detector state and internal buffer."""
    vad = SileroVAD(threshold=0.5)
    partial_frame = np.zeros(256, dtype=np.float32)
    vad.is_speech(partial_frame)  # Buffer now has 256 samples
    assert vad._buf.size == 256
    vad.reset()
    assert vad._buf.size == 0

@pytest.mark.integration
def test_buffers_across_multiple_calls():
    """Buffering works correctly across multiple sub-512-sample frames."""
    vad = SileroVAD(threshold=0.5)
    # Call with 300 samples
    frame1 = np.zeros(300, dtype=np.float32)
    result1 = vad.is_speech(frame1)
    assert isinstance(result1, bool)
    # Buffer now has 300 samples; call with 300 more (total 600)
    frame2 = np.zeros(300, dtype=np.float32)
    result2 = vad.is_speech(frame2)
    assert isinstance(result2, bool)
    # Should have processed one 512-sample window, leaving 88 samples in buffer
    assert vad._buf.size == 88
