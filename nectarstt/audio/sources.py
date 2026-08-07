import time
import wave
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        ...

    def close(self) -> None:
        pass

class FileSource(FrameSource):
    def __init__(self, path: str, frame_ms: int = 32,
                 sample_rate: int = 16000, realtime: bool = False) -> None:
        self._path = path
        self._frame_len = int(sample_rate * frame_ms / 1000)
        self._sample_rate = sample_rate
        self._realtime = realtime

    def frames(self) -> Iterator[np.ndarray]:
        with wave.open(self._path, "rb") as w:
            raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        step = self._frame_len
        for start in range(0, len(audio), step):
            chunk = audio[start:start + step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            if self._realtime:
                time.sleep(step / self._sample_rate)
            yield chunk

class MicSource(FrameSource):
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 32,
                 device: int | None = None) -> None:
        self._sample_rate = sample_rate
        self._frame_len = int(sample_rate * frame_ms / 1000)
        self._device = device

    def frames(self) -> Iterator[np.ndarray]:
        import queue
        import sounddevice as sd

        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                pass
            q.put(indata[:, 0].copy())

        try:
            with sd.InputStream(samplerate=self._sample_rate, channels=1,
                                dtype="float32", blocksize=self._frame_len,
                                device=self._device, callback=callback):
                while True:
                    yield q.get()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

class AudioDeviceError(RuntimeError):
    pass
