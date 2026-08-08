import logging
import os

import numpy as np

from openvox.tts.backend import TTSBackend, TTSResult
from openvox.tts.models import ensure_assets

log = logging.getLogger(__name__)

class KokoroBackend(TTSBackend):
    def __init__(self, device: str = "cuda") -> None:
        model_path, voices_path = ensure_assets()
        self._kokoro = self._load(model_path, voices_path, device)

    def _load(self, model_path: str, voices_path: str, device: str):
        from kokoro_onnx import Kokoro
        if device == "cuda":
            os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
            try:
                return Kokoro(model_path, voices_path)
            except Exception as exc:
                log.warning("CUDA provider unavailable (%s: %s); falling back to CPU.",
                            type(exc).__name__, exc)
        os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
        return Kokoro(model_path, voices_path)

    def synthesize(self, text: str, voice: str, speed: float) -> TTSResult:
        samples, sample_rate = self._kokoro.create(
            text, voice=voice, speed=speed, lang="en-us")
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        return TTSResult(audio=audio, sample_rate=int(sample_rate))

    def get_voices(self) -> list[str]:
        return self._kokoro.get_voices()
