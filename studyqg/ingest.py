"""Stage 1: load study material (PDF / text), clean it and split it into passages."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_REPLACEMENTS = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ", "­": "",
}
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Passage:
    pid: int
    sentences: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def read_pdf(path: str | Path) -> str:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover
        import fitz

    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text("text") for page in doc)


def load_text(source: str | Path) -> str:
    """`source` is a path to a .pdf / .txt / .md file, or raw text."""
    s = str(source)
    try:
        p = Path(s)
        is_file = len(s) < 4096 and p.is_file()
    except (OSError, ValueError):
        is_file = False
    if not is_file:
        return s
    if p.suffix.lower() == ".pdf":
        return read_pdf(p)
    return p.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Normalise quotes/ligatures, undo PDF line wrapping, drop junk lines."""
    for k, v in _REPLACEMENTS.items():
        text = text.replace(k, v)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # de-hyphenate wrapped words
    text = re.sub(r"[ \t]+", " ", text)
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        alpha = sum(ch.isalpha() for ch in s)
        if alpha < 3 or alpha < 0.4 * len(s):          # page numbers, rules, equations
            continue
        kept.append(s)
    text = "\n".join(kept)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # join wrapped lines inside a paragraph
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_sentences(text: str, nlp=None) -> list[str]:
    if nlp is not None:
        sents = [s.text.strip() for s in nlp(text).sents]
    else:
        sents = [s.strip() for s in _SENT_RE.split(text)]
    return [s for s in sents if s]


def chunk_passages(text: str, max_words: int = 120, nlp=None, min_words: int = 25) -> list[Passage]:
    """Group sentences into passages of at most `max_words`, never splitting a sentence."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    passages: list[Passage] = []
    buf: list[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words
        if buf:
            passages.append(Passage(pid=len(passages), sentences=buf))
        buf, buf_words = [], 0

    for para in paragraphs:
        for sent in split_sentences(para, nlp):
            w = len(sent.split())
            if w < 4:
                continue
            if buf_words + w > max_words and buf:
                flush()
            buf.append(sent)
            buf_words += w
        if buf_words >= 0.6 * max_words:   # prefer paragraph boundaries
            flush()
    flush()

    # merge a tiny trailing passage into the previous one
    if len(passages) >= 2 and passages[-1].n_words < min_words:
        last = passages.pop()
        passages[-1].sentences.extend(last.sentences)
    return passages
