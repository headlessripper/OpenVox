import logging
import os

import numpy as np

from openvox.tts.backend import TTSBackend, TTSResult
from openvox.tts.models import ensure_assets

log = logging.getLogger(__name__)

def _choose_provider(device: str, available: list[str]) -> str:
    """Pick the ONNX execution provider to request.

    Only request the CUDA provider when onnxruntime actually offers it —
    otherwise onnxruntime prints a warning and silently falls back to CPU. A
    CPU-only onnxruntime (the default wheel) advertises only Azure + CPU, so a
    blind CUDA request there is noisy and misleading. Returns a provider that
    is guaranteed to be in `available`.
    """
    if device == "cuda" and "CUDAExecutionProvider" in available:
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"

class KokoroBackend(TTSBackend):
    def __init__(self, device: str = "cuda") -> None:
        model_path, voices_path = ensure_assets()
        self._kokoro = self._load(model_path, voices_path, device)

    def _load(self, model_path: str, voices_path: str, device: str):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        provider = _choose_provider(device, ort.get_available_providers())
        if device == "cuda" and provider != "CUDAExecutionProvider":
            log.info(
                "CUDA execution provider not available in onnxruntime; using CPU. "
                "Install onnxruntime-gpu (or kokoro-onnx[gpu]) for GPU acceleration."
            )
        os.environ["ONNX_PROVIDER"] = provider
        try:
            return Kokoro(model_path, voices_path)
        except Exception as exc:
            if provider == "CPUExecutionProvider":
                raise
            log.warning("GPU backend load failed (%s: %s); falling back to CPU.",
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
