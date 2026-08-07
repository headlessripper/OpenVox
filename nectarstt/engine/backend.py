from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from nectarstt.events import WordTiming

@dataclass(frozen=True)
class BackendResult:
    text: str
    words: list[WordTiming] = field(default_factory=list)

class StreamingBackend(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None,
        word_timestamps: bool,
    ) -> BackendResult:
        """Transcribe a mono float32 audio buffer."""
        raise NotImplementedError
