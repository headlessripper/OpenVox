import os
import shutil
import urllib.request

from openvox._paths import cache_dir

_MODEL_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
              "model-files-v1.0/kokoro-v1.0.onnx")
_VOICES_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
               "model-files-v1.0/voices-v1.0.bin")

# Built-in English voices (American af_/am_, British bf_/bm_) shipped with
# Kokoro v1.0. Integration tests cross-check this against the model's own list.
KOKORO_VOICES: frozenset[str] = frozenset({
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
})

def voices() -> list[str]:
    return sorted(KOKORO_VOICES)

def validate_voice(name: str) -> None:
    if name not in KOKORO_VOICES:
        raise ValueError(
            f"Unknown voice '{name}'. Available voices: {voices()}"
        )

def ensure_assets() -> tuple[str, str]:
    root = cache_dir("tts/models")
    model_path = os.path.join(root, "kokoro-v1.0.onnx")
    voices_path = os.path.join(root, "voices-v1.0.bin")
    _download_if_missing(_MODEL_URL, model_path, min_bytes=100_000_000)
    _download_if_missing(_VOICES_URL, voices_path, min_bytes=1_000_000)
    return model_path, voices_path

def _download_if_missing(url: str, dest: str, min_bytes: int = 0) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        size = os.path.getsize(tmp)
        if size < min_bytes:
            os.remove(tmp)
            raise RuntimeError(
                f"Downloaded asset from {url} is too small ({size} bytes < {min_bytes}); "
                "likely a truncated or error response."
            )
        os.replace(tmp, dest)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Failed to download TTS asset from {url} to {dest}: {exc}. "
            "Check your connection, or pre-place the file at that path."
        ) from exc
