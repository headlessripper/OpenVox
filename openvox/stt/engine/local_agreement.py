class LocalAgreement:
    """Commit the longest common prefix of the two most recent hypotheses.

    Committed tokens grow monotonically; the volatile tail is whatever of the
    latest hypothesis follows the committed prefix.
    """

    def __init__(self) -> None:
        self._prev: list[str] = []
        self._committed: list[str] = []

    def update(self, tokens: list[str]) -> tuple[list[str], list[str]]:
        common: list[str] = []
        for a, b in zip(tokens, self._prev):
            if a == b:
                common.append(a)
            else:
                break
        if len(common) > len(self._committed):
            self._committed = common
        self._prev = list(tokens)
        volatile = tokens[len(self._committed):]
        return list(self._committed), list(volatile)

    def finalize(self) -> list[str]:
        result = list(self._prev)
        self._prev = []
        self._committed = []
        return result
