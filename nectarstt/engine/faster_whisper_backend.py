import os

# On Windows without Developer Mode, huggingface_hub falls back to symlinks that
# ctranslate2 cannot open ("Unable to open file 'model.bin'"). Forcing real file
# copies keeps model loading robust across platforms. Must be set before the
# faster_whisper (and thus huggingface_hub) import below.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import logging

import numpy as np
from faster_whisper import WhisperModel

from nectarstt.engine.backend import StreamingBackend, BackendResult
from nectarstt.events import WordTiming
from nectarstt.models import resolve_model, download_root

log = logging.getLogger(__name__)

class FasterWhisperBackend(StreamingBackend):
    def __init__(self, model: str = "distil-large-v3",
                 device: str = "cuda", compute_type: str = "float16") -> None:
        model_id = resolve_model(model)
        try:
            self._model = WhisperModel(
                model_id, device=device, compute_type=compute_type,
                download_root=download_root(),
            )
        except Exception as exc:  # CUDA/cuDNN not available, etc.
            log.error(
                "Primary backend init failed for '%s' on %s (%s: %s); "
                "falling back to CPU int8.",
                model_id, device, type(exc).__name__, exc,
            )
            self._model = WhisperModel(
                model_id, device="cpu", compute_type="int8",
                download_root=download_root(),
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int,
                   language: str | None, word_timestamps: bool) -> BackendResult:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        segments, _ = self._model.transcribe(
            audio, language=language, word_timestamps=word_timestamps,
            beam_size=1,
        )
        text_parts: list[str] = []
        words: list[WordTiming] = []
        for seg in segments:
            text_parts.append(seg.text)
            for w in (seg.words or []):
                words.append(WordTiming(
                    word=w.word.strip(), start=w.start, end=w.end,
                    probability=getattr(w, "probability", 1.0),
                ))
        return BackendResult(text="".join(text_parts).strip(), words=words)
