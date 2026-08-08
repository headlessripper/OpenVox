from dataclasses import dataclass

_QUALITY_EVALS = {"fast": 15, "balanced": 40, "thorough": 100}


def evals_for_quality(quality: str) -> int:
    try:
        return _QUALITY_EVALS[quality]
    except KeyError:
        raise ValueError(
            f"unknown quality {quality!r}; choose from {sorted(_QUALITY_EVALS)}"
        ) from None


@dataclass
class EnrollConfig:
    device: str = "cuda"
    quality: str = "balanced"
    max_evals: int | None = None
    probes: list[str] | None = None
    enhance_clips: bool = True
    min_clips: int = 1
    outlier_threshold: float = 0.6
    realizability_lambda: float = 0.05
    exaggeration: float = 0.5
    seed: int = 0
    accept_margin: float = 0.005
    min_rms: float = 0.01
    min_dur_s: float = 1.0
    max_clip_s: float = 12.0
