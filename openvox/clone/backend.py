from abc import ABC, abstractmethod

from openvox.tts.backend import TTSResult

class CloneBackend(ABC):
    @abstractmethod
    def clone(self, text: str, reference_path: str, exaggeration: float,
              cfg: float) -> TTSResult:
        """Speak text in the voice from the reference clip."""
        raise NotImplementedError
