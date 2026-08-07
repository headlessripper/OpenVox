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

    def run(self, source: FrameSource) -> Iterator[PartialResult | FinalResult]:
        cfg = self._cfg
        agree = LocalAgreement()
        buffer: list[np.ndarray] = []
        speech_ms = 0.0
        silence_ms = 0.0
        ms_since_window = 0.0
        had_speech = False

        try:
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
                        window_samples = int(cfg.partial_window_s * cfg.sample_rate)
                        audio = self._tail_window(buffer, window_samples)
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
                        if silence_ms >= cfg.min_silence_ms:
                            if speech_ms >= cfg.min_speech_ms:
                                yield self._finalize(buffer, agree)
                            else:
                                # Sub-threshold speech blip: discard silently so a
                                # later real utterance doesn't get concatenated
                                # onto this stale buffer.
                                self._discard(agree)
                            buffer, speech_ms, silence_ms = [], 0.0, 0.0
                            ms_since_window, had_speech = 0.0, False

            if buffer and speech_ms >= cfg.min_speech_ms:
                yield self._finalize(buffer, agree)
        finally:
            source.close()

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

    def _discard(self, agree: LocalAgreement) -> None:
        """Drop a sub-threshold speech blip without emitting any result."""
        agree.finalize()
        self._vad.reset()

    @staticmethod
    def _tail_window(buffer: list[np.ndarray], window_samples: int) -> np.ndarray:
        """Concatenate only the trailing frames covering ``window_samples``.

        Partial passes re-transcribe on every window tick, so scanning the
        whole growing buffer each time would cost O(n^2) over a long utterance.
        Bounding the context to a trailing window keeps each partial pass O(1)
        in the utterance length. Gathering from the end (rather than
        concatenating the full buffer then slicing) keeps the gather itself
        bounded too. If the buffer is shorter than the window, all of it is
        used.
        """
        tail: list[np.ndarray] = []
        total = 0
        for frame in reversed(buffer):
            tail.append(frame)
            total += len(frame)
            if total >= window_samples:
                break
        tail.reverse()
        return np.concatenate(tail)
