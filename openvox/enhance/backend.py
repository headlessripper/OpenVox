from abc import ABC, abstractmethod

import numpy as np

from openvox.tts.backend import TTSResult

class EnhanceBackend(ABC):
    @abstractmethod
    def enhance(self, audio: np.ndarray, sample_rate: int) -> TTSResult:
        """Restore/denoise mono float32 audio; returns the cleaned result."""
        raise NotImplementedError
