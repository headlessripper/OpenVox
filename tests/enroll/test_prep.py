import numpy as np
from openvox.enroll.prep import rms, passes_gate, segment


def test_rms_and_gate():
    loud = 0.5 * np.ones(16000, dtype=np.float32)
    quiet = 1e-4 * np.ones(16000, dtype=np.float32)
    assert rms(loud) > 0.4
    assert passes_gate(loud, 16000, min_rms=0.01, min_dur_s=0.5) is True
    assert passes_gate(quiet, 16000, min_rms=0.01, min_dur_s=0.5) is False   # too quiet
    assert passes_gate(loud[:4000], 16000, min_rms=0.01, min_dur_s=0.5) is False  # too short


def test_segment_splits_long_and_drops_silence():
    sr = 16000
    short = 0.5 * np.ones(sr * 3, dtype=np.float32)
    assert len(segment(short, sr, max_clip_s=12.0, min_rms=0.01)) == 1
    # 20s: 12s loud + 8s silence -> two windows, silent one dropped
    long = np.concatenate([0.5 * np.ones(sr * 12, dtype=np.float32),
                           np.zeros(sr * 8, dtype=np.float32)])
    segs = segment(long, sr, max_clip_s=12.0, min_rms=0.01)
    assert len(segs) == 1
