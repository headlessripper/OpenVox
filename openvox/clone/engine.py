import dataclasses
import os

from openvox.clone.config import CloneConfig
from openvox.clone.chatterbox_backend import ChatterboxBackend
from openvox.tts.backend import TTSResult, AudioDeviceError

class VoiceCloneEngine:
    def __init__(self, device: str | None = None, exaggeration: float | None = None,
                 cfg: float | None = None, config: CloneConfig | None = None) -> None:
        c = dataclasses.replace(config) if config is not None else CloneConfig()
        if device is not None:
            c.device = device
        if exaggeration is not None:
            c.exaggeration = exaggeration
        if cfg is not None:
            c.cfg = cfg
        self._config = c
        self._backend = ChatterboxBackend(device=c.device)

    def clone(self, text: str, reference_audio: str,
              exaggeration: float | None = None, cfg: float | None = None) -> TTSResult:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(f"reference audio not found: {reference_audio}")
        if os.path.getsize(reference_audio) == 0:
            raise ValueError(f"reference audio is empty: {reference_audio}")
        e = exaggeration if exaggeration is not None else self._config.exaggeration
        g = cfg if cfg is not None else self._config.cfg
        return self._backend.clone(text, reference_audio, e, g)

    def play(self, result: TTSResult) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError(
                'Playback needs the audio extra: pip install "openvox[clone]".') from exc
        try:
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

    def say(self, text: str, reference_audio: str,
            exaggeration: float | None = None, cfg: float | None = None) -> None:
        self.play(self.clone(text, reference_audio, exaggeration, cfg))
