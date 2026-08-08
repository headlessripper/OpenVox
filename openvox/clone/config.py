from dataclasses import dataclass

@dataclass
class CloneConfig:
    device: str = "cuda"
    exaggeration: float = 0.5
    cfg: float = 0.5
    enhance: bool = True
