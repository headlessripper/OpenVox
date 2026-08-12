from dataclasses import dataclass


@dataclass
class AgentConfig:
    barge_in: bool = True
    barge_in_debounce_s: float = 0.3
    history_max_turns: int = 12
    greeting: str | None = None
    stop_phrase: str | None = None
    on_error: str = "continue"   # "continue" or "raise"
