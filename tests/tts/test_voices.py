import pytest
from openvox.tts.voices import is_profile_voice, resolve_voice
from openvox.tts.cloned_backend import ClonedVoiceBackend, CLONED_VOICE_ID

class _Sentinel:
    """stand-in for a KokoroBackend"""

class _FakeProfile:
    conditionals = object()

def test_is_profile_voice():
    assert is_profile_voice("alice.ovx") is True
    assert is_profile_voice(_FakeProfile()) is True
    assert is_profile_voice("af_heart") is False
    assert is_profile_voice(None) is False

def test_resolve_builtin_name_returns_kokoro():
    kok = _Sentinel()
    backend, vid = resolve_voice("af_heart", kok, device="cpu")
    assert backend is kok and vid == "af_heart"

def test_resolve_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_voice("not_a_voice", _Sentinel(), device="cpu")

def test_resolve_profile_instance_returns_cloned_backend():
    backend, vid = resolve_voice(_FakeProfile(), _Sentinel(), device="cpu")
    assert isinstance(backend, ClonedVoiceBackend) and vid == CLONED_VOICE_ID

def test_resolve_missing_ovx_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_voice(str(tmp_path / "nope.ovx"), _Sentinel(), device="cpu")
