import re
from collections.abc import Iterable, Iterator

_SENTENCE = re.compile(r'.+?(?:[.!?]+(?=\s|$)|$)', re.DOTALL)
_CLAUSE = re.compile(r'.+?(?:[,;:](?=\s|$)|$)', re.DOTALL)
_TERM = re.compile(r'[.!?](?=\s)')


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


def iter_sentences(chunks: Iterable[str], max_chars: int = 160) -> Iterator[str]:
    """Aggregate a stream of text chunks into complete sentences, lazily.

    Yields a sentence as soon as a terminator (. ? !) followed by whitespace is
    seen; splits an over-length terminator-free run at the last space before
    max_chars; yields the trailing remainder when the input ends."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        while True:
            m = _TERM.search(buf)
            if m:
                i = m.end()
                sent = buf[:i].strip()
                buf = buf[i:].lstrip()
                if sent:
                    yield sent
                continue
            if len(buf) >= max_chars:
                cut = buf.rfind(' ', 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                sent = buf[:cut].strip()
                buf = buf[cut:].lstrip()
                if sent:
                    yield sent
                continue
            break
    tail = buf.strip()
    if tail:
        yield tail
