from collections.abc import Iterator

import numpy as np

from nectarstt.audio.sources import FrameSource
from nectarstt.config import Config
from nectarstt.engine.backend import StreamingBackend
from nectarstt.engine.local_agreement import LocalAgreement
from nectarstt.events import PartialResult, FinalResult

class StreamingTranscriber:
    def __init__(self, backend: StreamingBackend, vad, config: Config) -> None:
        self._backend = backend
        self._vad = vad
        self._cfg = config

    def run(self, source: FrameSource) -> Iterator:
        cfg = self._cfg
        agree = LocalAgreement()
        buffer: list[np.ndarray] = []
        speech_ms = 0.0
        silence_ms = 0.0
        ms_since_window = 0.0
        had_speech = False

        for frame in source.frames():
            frame_ms = 1000.0 * len(frame) / cfg.sample_rate
            if self._vad.is_speech(frame):
                buffer.append(frame)
                speech_ms += frame_ms
                silence_ms = 0.0
                ms_since_window += frame_ms
                had_speech = True
                if ms_since_window >= cfg.window_interval_ms:
                    ms_since_window = 0.0
                    audio = np.concatenate(buffer)
                    res = self._backend.transcribe(
                        audio, cfg.sample_rate, cfg.language, word_timestamps=False)
                    committed, volatile = agree.update(res.text.split())
                    yield PartialResult(
                        text=res.text,
                        committed_prefix=" ".join(committed),
                        volatile_tail=" ".join(volatile),
                    )
            else:
                if had_speech:
                    silence_ms += frame_ms
                    if (silence_ms >= cfg.min_silence_ms
                            and speech_ms >= cfg.min_speech_ms):
                        yield self._finalize(buffer, agree)
                        buffer, speech_ms, silence_ms = [], 0.0, 0.0
                        ms_since_window, had_speech = 0.0, False

        if buffer and speech_ms >= cfg.min_speech_ms:
            yield self._finalize(buffer, agree)

    def _finalize(self, buffer: list[np.ndarray], agree: LocalAgreement) -> FinalResult:
        cfg = self._cfg
        audio = np.concatenate(buffer)
        res = self._backend.transcribe(
            audio, cfg.sample_rate, cfg.language, word_timestamps=True)
        agree.finalize()
        self._vad.reset()
        start = res.words[0].start if res.words else 0.0
        end = res.words[-1].end if res.words else 0.0
        return FinalResult(text=res.text, words=res.words, start=start, end=end)
