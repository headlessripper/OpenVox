import logging
import os

import numpy as np

from openvox.clone.backend import CloneBackend
from openvox.tts.backend import TTSResult

log = logging.getLogger(__name__)

def _quiet_libraries() -> None:
    """Silence noisy third-party warnings/logging so clone output stays clean.

    Chatterbox pulls in torch/diffusers/transformers, which emit FutureWarnings,
    a mel-normalization WARNING, an attention notice, and a stray PerthNet print
    on every run. None are actionable for a caller; suppress them (genuine
    ERROR-level messages still surface).
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    for name in ("chatterbox", "diffusers", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as _hf_log
        _hf_log.set_verbosity_error()
    except Exception:
        pass
    try:
        from diffusers.utils import logging as _df_log
        _df_log.set_verbosity_error()
    except Exception:
        pass

def _resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "cuda" and cuda_available:
        return "cuda"
    return "cpu"

class ChatterboxBackend(CloneBackend):
    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Windows: avoid dangling HF symlinks that break model loading. Set
        # before huggingface_hub is imported (transitively, by chatterbox).
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        _quiet_libraries()
        import contextlib
        import io

        import torch
        from chatterbox.tts import ChatterboxTTS

        device = _resolve_device(self._device, torch.cuda.is_available())
        if self._device == "cuda" and device != "cuda":
            log.info("CUDA is not available to torch; loading Chatterbox on CPU "
                     "(slow — install a CUDA torch build for GPU acceleration).")
        # from_pretrained prints a stray "loaded PerthNet ..." line to stdout;
        # swallow it (exceptions still propagate — only stdout is redirected).
        with contextlib.redirect_stdout(io.StringIO()):
            self._model = ChatterboxTTS.from_pretrained(device)

    def clone(self, text: str, reference_path: str, exaggeration: float,
              cfg: float) -> TTSResult:
        self._ensure_loaded()
        wav = self._model.generate(
            text, audio_prompt_path=reference_path,
            exaggeration=exaggeration, cfg_weight=cfg)
        audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return TTSResult(audio=audio, sample_rate=int(self._model.sr))
