import dataclasses
from collections.abc import Iterable, Iterator

from openvox.tts.backend import TTSResult, AudioDeviceError
from openvox.tts.config import TTSConfig
from openvox.tts.kokoro_backend import KokoroBackend
from openvox.tts.models import voices as _voices, validate_voice
from openvox.tts.segment import split_text, iter_sentences
from openvox.tts.voices import resolve_voice
from openvox.tts.stream import SpeechHandle, _StreamPlayer

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
        if s <= 0:
            raise ValueError("speed must be > 0")
        return self._backend.synthesize(text, v, s)

    def play(self, result: TTSResult) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError("Playback needs the audio extra: pip install \"openvox[tts]\".") from exc
        try:
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

    def say(self, text: str, voice: str | None = None,
            speed: float | None = None) -> None:
        self.play(self.synthesize(text, voice, speed))

    def stream(self, text: str, voice: str | None = None,
               speed: float | None = None) -> Iterator[TTSResult]:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        v = voice if voice is not None else self._config.voice
        s = speed if speed is not None else self._config.speed
        if s <= 0:
            raise ValueError("speed must be > 0")
        backend, voice_id = resolve_voice(v, self._backend, self._config.device)
        segments = split_text(text, self._config.segment_max_chars)
        return backend.stream_segments(segments, voice_id, s)

    def say_stream(self, text: str | Iterable[str], voice: str | None = None,
                   speed: float | None = None) -> SpeechHandle:
        v = voice if voice is not None else self._config.voice
        s = speed if speed is not None else self._config.speed
        if s <= 0:
            raise ValueError("speed must be > 0")
        backend, voice_id = resolve_voice(v, self._backend, self._config.device)
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("text must be a non-empty string")
            segments = split_text(text, self._config.segment_max_chars)
        else:
            # a stream of text chunks (e.g. a streaming LLM): aggregate lazily
            segments = iter_sentences(text, self._config.segment_max_chars)
        player = _StreamPlayer(self._config.sample_rate, self._config.stream_queue_size)
        handle = SpeechHandle(player)
        handle.start(backend, voice_id, s, segments)
        return handle
