import logging
import os

import numpy as np

from openvox.tts.backend import TTSBackend, TTSResult
from openvox.tts.models import ensure_assets

log = logging.getLogger(__name__)

_dll_dir_handles: list = []

def _prepare_cuda_libs(ort) -> None:
    """Make onnxruntime-gpu's CUDA/cuDNN/cuFFT DLLs loadable at runtime.

    The nvidia-*-cu12 pip wheels (the [tts-gpu] extra) install their DLLs under
    ``site-packages/nvidia/<lib>/bin``. onnxruntime's ``preload_dlls`` loads the
    core CUDA/cuDNN DLLs, but cuDNN lazily loads sublibraries and cuFFT is a
    separate library — the OS loader resolves those via the DLL search path.
    Add every nvidia wheel bin dir to that path (Windows) so a CUDA session can
    both build and run. Best-effort and harmless when the wheels or the API are
    absent (e.g. the CPU-only [tts] install).
    """
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    try:
        import glob
        import nvidia
        base = list(nvidia.__path__)[0]
    except Exception:
        return
    bindirs = glob.glob(os.path.join(base, "*", "bin"))
    if not bindirs:
        return
    if hasattr(os, "add_dll_directory"):
        for bindir in bindirs:
            try:
                _dll_dir_handles.append(os.add_dll_directory(bindir))
            except OSError:
                pass
    # cuDNN loads its sublibraries (cudnn_engines_*, etc.) and cuFFT by name via
    # the OS search path, which add_dll_directory does not cover — prepend the
    # wheel bin dirs to PATH so those resolve.
    os.environ["PATH"] = os.pathsep.join(bindirs) + os.pathsep + os.environ.get("PATH", "")

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

        # Quiet onnxruntime's non-error logging — the CUDA provider emits benign
        # performance notices (Memcpy nodes, CPU-assigned shape ops) at WARNING
        # level. Genuine ERROR/FATAL messages still surface.
        try:
            ort.set_default_logger_severity(3)  # 3 = ERROR
        except Exception:
            pass

        # onnxruntime-gpu needs the CUDA 12 / cuDNN 9 / cuFFT runtime DLLs on the
        # load path (see _prepare_cuda_libs) before a CUDA session can build/run.
        if device == "cuda":
            _prepare_cuda_libs(ort)

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
