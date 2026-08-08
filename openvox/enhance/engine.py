import dataclasses
import os

import numpy as np

from openvox.enhance.config import EnhanceConfig
from openvox.enhance.resemble_backend import ResembleEnhanceBackend
from openvox.tts.backend import TTSResult

class EnhanceEngine:
    def __init__(self, device: str | None = None, denoise_only: bool | None = None,
                 config: EnhanceConfig | None = None) -> None:
        cfg = dataclasses.replace(config) if config is not None else EnhanceConfig()
        if device is not None:
            cfg.device = device
        if denoise_only is not None:
            cfg.denoise_only = denoise_only
        self._config = cfg
        self._backend = ResembleEnhanceBackend(cfg)

    def enhance(self, audio: np.ndarray, sample_rate: int) -> TTSResult:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            raise ValueError("audio must be a non-empty array")
        return self._backend.enhance(arr, sample_rate)

    def enhance_file(self, path: str) -> TTSResult:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"audio file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"audio file is empty: {path}")
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.enhance(audio, sr)
