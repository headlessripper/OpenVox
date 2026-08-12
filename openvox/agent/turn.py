from dataclasses import dataclass


@dataclass
class Turn:
    role: str      # "user" or "assistant"
    text: str


class ConversationHistory:
    def __init__(self, max_turns: int = 12) -> None:
        self._max = max(1, max_turns)
        self._turns: list[Turn] = []

    def append(self, turn: Turn) -> None:
        self._turns.append(turn)
        if len(self._turns) > self._max:
            self._turns = self._turns[-self._max:]

    def __iter__(self):
        return iter(self._turns)

    def __len__(self):
        return len(self._turns)

    def __getitem__(self, i):
        return self._turns[i]
