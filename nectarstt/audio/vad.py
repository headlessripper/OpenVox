import numpy as np
from pysilero_vad import SileroVoiceActivityDetector

_WINDOW = 512  # samples @ 16kHz required by silero

class SileroVAD:
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        if sample_rate != 16000:
            raise ValueError("SileroVAD requires 16000 Hz audio.")
        self._threshold = threshold
        self._detector = SileroVoiceActivityDetector()
        self._buf = np.zeros(0, dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        self._buf = np.concatenate([self._buf, frame.astype(np.float32)])
        speech = False
        while len(self._buf) >= _WINDOW:
            window = self._buf[:_WINDOW]
            self._buf = self._buf[_WINDOW:]
            pcm = (window * 32768.0).astype(np.int16).tobytes()
            if self._detector(pcm) >= self._threshold:
                speech = True
        return speech

    def reset(self) -> None:
        self._detector.reset()
        self._buf = np.zeros(0, dtype=np.float32)
