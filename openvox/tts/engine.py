import dataclasses

from openvox.tts.backend import TTSResult, AudioDeviceError
from openvox.tts.config import TTSConfig
from openvox.tts.kokoro_backend import KokoroBackend
from openvox.tts.models import voices as _voices, validate_voice

class TTSEngine:
    def __init__(self, voice: str | None = None, device: str | None = None,
                 speed: float | None = None, config: TTSConfig | None = None) -> None:
        cfg = dataclasses.replace(config) if config is not None else TTSConfig()
        if voice is not None:
            cfg.voice = voice
        if device is not None:
            cfg.device = device
        if speed is not None:
            cfg.speed = speed
        self._config = cfg
        self._backend = KokoroBackend(device=cfg.device)

    def voices(self) -> list[str]:
        return _voices()

    def synthesize(self, text: str, voice: str | None = None,
                   speed: float | None = None) -> TTSResult:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        v = voice if voice is not None else self._config.voice
        validate_voice(v)
        s = speed if speed is not None else self._config.speed
        return self._backend.synthesize(text, v, s)

    def play(self, result: TTSResult) -> None:
        import sounddevice as sd
        try:
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

    def say(self, text: str, voice: str | None = None,
            speed: float | None = None) -> None:
        self.play(self.synthesize(text, voice, speed))
