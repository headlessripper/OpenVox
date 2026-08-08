import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

class AudioDeviceError(RuntimeError):
    pass

@dataclass(eq=False)
class TTSResult:
    audio: np.ndarray          # mono float32 in [-1, 1]
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    def save_wav(self, path: str) -> None:
        # Scale-then-clip so a +1.0 sample maps to 32767, not a wrapped -32768.
        pcm = np.clip(np.asarray(self.audio, dtype=np.float32) * 32768.0,
                      -32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())

class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float) -> TTSResult:
        """Synthesize speech for text with the given voice and speed."""
        raise NotImplementedError
