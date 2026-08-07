from dataclasses import dataclass, field

@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float
    probability: float

@dataclass(frozen=True)
class PartialResult:
    text: str
    committed_prefix: str
    volatile_tail: str

    @property
    def is_partial(self) -> bool:
        return True

@dataclass(frozen=True)
class FinalResult:
    text: str
    words: list[WordTiming] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    @property
    def is_partial(self) -> bool:
        return False
