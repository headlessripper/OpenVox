from openvox._paths import cache_dir

# faster-whisper resolves these ids directly against the HF hub.
MODEL_ALIASES: dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "large-v3": "large-v3",
    "distil-large-v3": "distil-large-v3",
}

def resolve_model(name: str) -> str:
    if name not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model '{name}'. Known models: {sorted(MODEL_ALIASES)}"
        )
    return MODEL_ALIASES[name]

def download_root() -> str:
    return cache_dir("stt/models")
