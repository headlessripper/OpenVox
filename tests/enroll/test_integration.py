# tests/enroll/test_integration.py
import glob
import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REF_CANDIDATES = sorted(glob.glob("reference_hq.wav") + glob.glob("*.wav"))

@pytest.mark.skipif(not REF_CANDIDATES, reason="no local reference wav to enroll from")
def test_enroll_then_clone_from_profile(tmp_path):
    from openvox.enroll import VoiceEnrollEngine, VoiceProfile
    from openvox.clone import VoiceCloneEngine

    clips = REF_CANDIDATES[:2] or REF_CANDIDATES
    eng = VoiceEnrollEngine(device="cuda", quality="fast")
    profile = eng.enroll(clips)
    assert 0.0 < profile.score <= 1.0
    assert profile.metadata["stage"] in ("A", "B")

    out = tmp_path / "voice.ovx"
    profile.save(str(out))
    reloaded = VoiceProfile.load(str(out))
    assert reloaded.metadata["stage"] == profile.metadata["stage"]

    clone = VoiceCloneEngine(device="cuda")
    result = clone.clone("This is a profile-based clone.", profile=reloaded)
    assert result.sample_rate == 24000
    assert result.audio.size > 0
    assert float(np.sqrt(np.mean(result.audio ** 2))) > 1e-4   # non-silent
