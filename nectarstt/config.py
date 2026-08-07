import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

@dataclass
class Config:
    device: str = "cuda"
    compute_type: str = "float16"
    model: str = "distil-large-v3"
    language: str = "en"
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_silence_ms: int = 500
    min_speech_ms: int = 200
    window_interval_ms: int = 500

    @classmethod
    def load(cls, path: str | None = None,
             env: Mapping[str, str] | None = None) -> "Config":
        values: dict[str, Any] = {}
        if path and os.path.exists(path):
            with open(path, "rb") as fh:
                values.update(tomllib.load(fh))
        env = os.environ if env is None else env
        type_map = {f.name: f.type for f in fields(cls)}
        for f in fields(cls):
            key = f"NECTARSTT_{f.name.upper()}"
            if key in env:
                values[f.name] = env[key]
        coerced = {k: cls._coerce(k, v, type_map) for k, v in values.items()
                   if k in type_map}
        return cls(**coerced)

    @staticmethod
    def _coerce(name: str, value: Any, type_map: dict[str, Any]) -> Any:
        t = type_map[name]
        if t is int:
            return int(value)
        if t is float:
            return float(value)
        return str(value)
