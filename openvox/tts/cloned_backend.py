import os

from openvox.tts.backend import TTSBackend, TTSResult

CLONED_VOICE_ID = "__cloned__"

class ClonedVoiceBackend(TTSBackend):
    """A TTS backend whose voice is a cloned .ovx profile (via Chatterbox).

    Ignores the ``voice``/``speed`` arguments: identity comes from the profile
    and Chatterbox has no speed control. Imports torch/chatterbox lazily."""

    def __init__(self, profile, device: str = "cuda",
                 exaggeration: float = 0.5, cfg: float = 0.5) -> None:
        self._profile = profile      # a VoiceProfile or an .ovx path
        self._device = device
        self._exaggeration = exaggeration
        self._cfg = cfg
        self._backend = None
        self._conds = None

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        prof = self._profile
        if hasattr(prof, "conditionals"):
            conds = prof.conditionals
        else:
            from openvox.enroll import VoiceProfile
            conds = VoiceProfile.load(os.fspath(prof)).conditionals
        from openvox.clone.chatterbox_backend import ChatterboxBackend
        self._backend = ChatterboxBackend(device=self._device)
        self._conds = conds

    def synthesize(self, text: str, voice: str, speed: float) -> TTSResult:
        self._ensure_loaded()
        return self._backend.clone_from_profile(
            text, self._conds, self._exaggeration, self._cfg)
