from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VoiceProfile:
    SCHEMA_VERSION = 1

    conditionals: Any  # Chatterbox Conditionals (opaque here)
    score: float
    metadata: dict = field(default_factory=dict)

    def _to_payload(self) -> dict:
        return {
            "schema": self.SCHEMA_VERSION,
            "conditionals": self.conditionals,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def _from_payload(cls, payload: dict) -> "VoiceProfile":
        schema = payload.get("schema")
        if schema != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported .ovx schema {schema!r} (expected {cls.SCHEMA_VERSION}); "
                "regenerate the profile with the current version"
            )
        return cls(
            conditionals=payload["conditionals"],
            score=payload["score"],
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        import torch

        tmp = os.fspath(path) + ".tmp"
        torch.save(self._to_payload(), tmp)
        os.replace(tmp, os.fspath(path))

    @classmethod
    def load(cls, path: str) -> "VoiceProfile":
        import torch

        payload = torch.load(os.fspath(path), map_location="cpu", weights_only=False)
        return cls._from_payload(payload)
