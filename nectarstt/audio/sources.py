import queue
import threading
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
            # Validate WAV format
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()

            if channels != 1:
                raise ValueError(
                    f"FileSource expects {self._sample_rate} Hz mono 16-bit WAV; "
                    f"got {framerate} Hz, {channels} ch, {sampwidth} bytes/sample"
                )
            if sampwidth != 2:
                raise ValueError(
                    f"FileSource expects {self._sample_rate} Hz mono 16-bit WAV; "
                    f"got {framerate} Hz, {channels} ch, {sampwidth} bytes/sample"
                )
            if framerate != self._sample_rate:
                raise ValueError(
                    f"FileSource expects {self._sample_rate} Hz mono 16-bit WAV; "
                    f"got {framerate} Hz, {channels} ch, {sampwidth} bytes/sample"
                )

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
        self._stop = threading.Event()

    def frames(self) -> Iterator[np.ndarray]:
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
                while not self._stop.is_set():
                    try:
                        yield q.get(timeout=0.1)
                    except queue.Empty:
                        continue
        except sd.PortAudioError as exc:
            raise AudioDeviceError(str(exc)) from exc

    def close(self) -> None:
        self._stop.set()

class AudioDeviceError(RuntimeError):
    pass
