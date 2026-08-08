from dataclasses import dataclass

@dataclass
class TTSConfig:
    device: str = "cuda"
    voice: str = "af_heart"
    speed: float = 1.0
    sample_rate: int = 24000  # Kokoro's fixed 24 kHz output; informational only, not used to reconfigure the model
