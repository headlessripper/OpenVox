from dataclasses import dataclass

@dataclass
class EnhanceConfig:
    device: str = "cuda"
    nfe: int = 64
    solver: str = "midpoint"
    lambd: float = 0.9
    tau: float = 0.5
    denoise_only: bool = False
