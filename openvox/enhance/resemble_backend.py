import logging
import os
import sys

import numpy as np

from openvox.enhance.backend import EnhanceBackend
from openvox.enhance.config import EnhanceConfig
from openvox.tts.backend import TTSResult

log = logging.getLogger(__name__)

def _resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "cuda" and cuda_available:
        return "cuda"
    return "cpu"

def _install_shims() -> None:
    """Make resemble-enhance's inference importable/loadable without its bad pins.

    resemble-enhance's pip metadata pins an old torch and requires deepspeed
    (training-only) — the inference path merely imports the training modules. We
    install it --no-deps and, at load time: keep HF cache symlink-free on
    Windows; stub the training-only deepspeed imports; and (Windows only) map
    PosixPath -> WindowsPath so the checkpoint's embedded PosixPath deserializes.
    Verified by a feasibility spike on torch 2.6 / Windows.
    """
    from unittest.mock import MagicMock

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    for name in ("deepspeed", "deepspeed.accelerator", "deepspeed.runtime",
                 "deepspeed.runtime.engine", "deepspeed.runtime.utils"):
        sys.modules.setdefault(name, MagicMock())
    if os.name == "nt":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath

class ResembleEnhanceBackend(EnhanceBackend):
    def __init__(self, config: EnhanceConfig | None = None) -> None:
        self._cfg = config or EnhanceConfig()
        self._device = None
        self._enhance = None
        self._denoise = None

    def _ensure_loaded(self) -> None:
        if self._device is not None:
            return
        _install_shims()
        import torch
        from resemble_enhance.enhancer.inference import denoise, enhance

        self._torch = torch
        self._enhance = enhance
        self._denoise = denoise
        self._device = _resolve_device(self._cfg.device, torch.cuda.is_available())
        if self._cfg.device == "cuda" and self._device != "cuda":
            log.info("CUDA is not available to torch; enhancing on CPU (slower).")

    def enhance(self, audio: np.ndarray, sample_rate: int) -> TTSResult:
        self._ensure_loaded()
        dwav = self._torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        if self._cfg.denoise_only:
            wav, new_sr = self._denoise(dwav, sample_rate, self._device)
        else:
            wav, new_sr = self._enhance(
                dwav, sample_rate, self._device,
                nfe=self._cfg.nfe, solver=self._cfg.solver,
                lambd=self._cfg.lambd, tau=self._cfg.tau)
        out = wav.detach().cpu().numpy().astype(np.float32)
        return TTSResult(audio=out, sample_rate=int(new_sr))
