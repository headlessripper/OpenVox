import queue
import threading
import time
import wave
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

def _iter_frames(audio: np.ndarray, frame_len: int, sample_rate: int,
                 realtime: bool) -> Iterator[np.ndarray]:
    """Yield fixed-size float32 frames from a mono audio array.

    The final short frame is zero-padded to ``frame_len``. When ``realtime``
    is set, sleeps one frame-duration between yields to simulate a live source.
    """
    for start in range(0, len(audio), frame_len):
        chunk = audio[start:start + frame_len]
        if len(chunk) < frame_len:
            chunk = np.pad(chunk, (0, frame_len - len(chunk)))
        if realtime:
            time.sleep(frame_len / sample_rate)
        yield chunk

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
        yield from _iter_frames(audio, self._frame_len, self._sample_rate,
                                self._realtime)

class ArraySource(FrameSource):
    """A frame source over an in-memory mono float32 array.

    Used to feed already-decoded/resampled audio (e.g. the demo's PyAV loader
    output) into the engine without going through a WAV file. The array is
    assumed to be mono, ``sample_rate`` Hz, float32 in [-1, 1].
    """

    def __init__(self, audio: np.ndarray, frame_ms: int = 32,
                 sample_rate: int = 16000, realtime: bool = False) -> None:
        self._audio = np.ascontiguousarray(audio, dtype=np.float32)
        self._frame_len = int(sample_rate * frame_ms / 1000)
        self._sample_rate = sample_rate
        self._realtime = realtime

    def frames(self) -> Iterator[np.ndarray]:
        yield from _iter_frames(self._audio, self._frame_len,
                                self._sample_rate, self._realtime)

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
