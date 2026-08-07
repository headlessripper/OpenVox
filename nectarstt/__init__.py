"""NectarSTT — offline streaming speech-to-text."""
import dataclasses
from collections.abc import Iterator

import numpy as np

from nectarstt.config import Config
from nectarstt.events import WordTiming, PartialResult, FinalResult
from nectarstt.audio.sources import FrameSource, FileSource, MicSource, ArraySource
from nectarstt.audio.vad import SileroVAD
from nectarstt.engine.faster_whisper_backend import FasterWhisperBackend
from nectarstt.engine.transcriber import StreamingTranscriber

__version__ = "0.1.0"
__all__ = ["STTEngine", "Config", "PartialResult", "FinalResult",
           "WordTiming", "FileSource", "MicSource", "ArraySource"]

class STTEngine:
    def __init__(self, model: str | None = None, device: str | None = None,
                 language: str | None = None, config: Config | None = None) -> None:
        cfg = dataclasses.replace(config) if config is not None else Config()
        if model is not None:
            cfg.model = model
        if device is not None:
            cfg.device = device
        if language is not None:
            cfg.language = language
        self._config = cfg
        self._backend = FasterWhisperBackend(
            model=cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        self._vad = SileroVAD(threshold=cfg.vad_threshold,
                              sample_rate=cfg.sample_rate)
        self._transcriber = StreamingTranscriber(self._backend, self._vad, cfg)

    def stream(self, source: FrameSource | None = None) -> Iterator:
        if source is None:
            source = MicSource(sample_rate=self._config.sample_rate)
        return self._transcriber.run(source)

    def transcribe_file(self, path: str) -> FinalResult:
        source = FileSource(path, sample_rate=self._config.sample_rate)
        audio = np.concatenate(list(source.frames()))
        res = self._backend.transcribe(
            audio, self._config.sample_rate, self._config.language, word_timestamps=True)
        start = res.words[0].start if res.words else 0.0
        end = res.words[-1].end if res.words else 0.0
        return FinalResult(text=res.text, words=res.words, start=start, end=end)
