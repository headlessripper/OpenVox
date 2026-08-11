from abc import ABC, abstractmethod

import numpy as np


class EnrollBackend(ABC):
    @abstractmethod
    def embed_clips(self, wavs_16k: list) -> np.ndarray:
        """Voice-encoder embeddings for a list of 16 kHz mono clips -> (N, 256)."""
        raise NotImplementedError

    @abstractmethod
    def reference_from_clip(self, clip_16k, clip_24k):
        """Build the s3gen reference + cond-prompt tokens for one clip."""
        raise NotImplementedError

    @abstractmethod
    def make_conditionals(self, speaker_emb, reference, exaggeration):
        """Assemble a Chatterbox Conditionals from a speaker embedding + reference."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, conditionals, text, seed):
        """Generate speech for text under conditionals -> (wav float32, sample_rate)."""
        raise NotImplementedError

    @abstractmethod
    def ve_embed(self, wav, sr) -> np.ndarray:
        """Voice-encoder embedding of a generated waveform -> (256,)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cuda_available(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_sr(self) -> int:
        raise NotImplementedError
