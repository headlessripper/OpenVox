import os

from openvox.tts.cloned_backend import ClonedVoiceBackend, CLONED_VOICE_ID
from openvox.tts.models import validate_voice

def is_profile_voice(voice) -> bool:
    if hasattr(voice, "conditionals"):
        return True
    if isinstance(voice, (str, os.PathLike)):
        return os.fspath(voice).endswith(".ovx")
    return False

def resolve_voice(voice, kokoro_backend, device: str):
    """Return (backend, voice_id): the Kokoro backend for a built-in name, or a
    ClonedVoiceBackend for an .ovx profile."""
    if is_profile_voice(voice):
        if isinstance(voice, (str, os.PathLike)) and not os.path.isfile(voice):
            raise FileNotFoundError(f"voice profile not found: {voice}")
        return ClonedVoiceBackend(voice, device=device), CLONED_VOICE_ID
    validate_voice(voice)
    return kokoro_backend, voice
