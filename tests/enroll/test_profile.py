import pytest
from openvox.enroll.profile import VoiceProfile

def test_payload_roundtrip_is_torch_free():
    sentinel = object()  # stands in for a Conditionals object
    p = VoiceProfile(conditionals=sentinel, score=0.87, metadata={"stage": "B"})
    restored = VoiceProfile._from_payload(p._to_payload())
    assert restored.conditionals is sentinel
    assert restored.score == 0.87
    assert restored.metadata == {"stage": "B"}

def test_from_payload_rejects_bad_schema():
    bad = {"schema": 999, "conditionals": object(), "score": 0.0, "metadata": {}}
    with pytest.raises(ValueError, match="schema"):
        VoiceProfile._from_payload(bad)
