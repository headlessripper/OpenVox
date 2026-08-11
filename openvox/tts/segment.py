import re

_SENTENCE = re.compile(r'.+?(?:[.!?]+(?=\s|$)|$)', re.DOTALL)
_CLAUSE = re.compile(r'.+?(?:[,;:](?=\s|$)|$)', re.DOTALL)


def _norm(s: str) -> str:
    return " ".join(s.split())


def _pack_words(text: str, max_chars: int) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)
    return out


def split_text(text: str, max_chars: int = 160) -> list[str]:
    segments: list[str] = []
    for sent_match in _SENTENCE.finditer(text or ""):
        sentence = _norm(sent_match.group())
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            segments.append(sentence)
            continue
        # too long: split on clauses, then hard-pack words if a clause is still long
        for clause_match in _CLAUSE.finditer(sentence):
            clause = _norm(clause_match.group())
            if not clause:
                continue
            if len(clause) <= max_chars:
                segments.append(clause)
            else:
                segments.extend(_pack_words(clause, max_chars))
    return segments
