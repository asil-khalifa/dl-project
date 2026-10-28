"""Stage 5a: distractor (wrong option) generation for multiple-choice questions."""
from __future__ import annotations

import random
import re

from .answer_extraction import Candidate

_NUM_RE = re.compile(r"^(-?\d[\d,]*)(\.\d+)?(%?)$")
_GROUP = {"GPE": "PLACE", "LOC": "PLACE", "FAC": "PLACE", "PERSON": "PERSON", "NORP": "PERSON",
          "ORG": "NAMED", "PRODUCT": "NAMED", "WORK_OF_ART": "NAMED", "EVENT": "NAMED", "LAW": "NAMED", "LANGUAGE": "NAMED",
          "DATE": "DATE", "TIME": "DATE", "NUMBER": "NUMBER", "CARDINAL": "NUMBER", "QUANTITY": "NUMBER",
          "PERCENT": "NUMBER", "MONEY": "NUMBER", "NOUN_CHUNK": "NOUN"}


_YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


def proper(s: str) -> bool:
    """True for proper-noun-like spans (every content word capitalised or containing a digit)."""
    words = [w for w in s.split() if w.lower() not in ("of", "the", "and", "for", "de", "in")]
    return bool(words) and all(w[:1].isupper() or any(ch.isdigit() for ch in w) for w in words)


def group_of(kind: str) -> str:
    return _GROUP.get(kind, "NOUN")


def shape(s: str) -> tuple[bool, bool, int]:
    """(has capitalised words, has digits, word-count bucket): distractors should match the answer's shape."""
    words = s.split()
    caps = any(w[:1].isupper() for w in words)
    digit = any(ch.isdigit() for ch in s)
    n = len(words)
    return caps, digit, 0 if n == 1 else (1 if n <= 3 else 2)


class DistractorGenerator:
    """Three sources, tried in order: perturbed numbers, same-type spans from the same
    document, WordNet co-hyponyms (sibling concepts). Falls back to any other span."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, answer: str, kind: str, pool: list[Candidate], n: int = 3) -> list[str]:
        out: list[str] = []
        a_norm = answer.lower().strip()

        a_shape = shape(answer)

        def ok(x: str) -> bool:
            xn = x.lower().strip()
            if not xn or xn == a_norm or xn in a_norm or a_norm in xn or x[-1] in ".,;:":
                return False
            return all(xn != o.lower() for o in out)

        def similar_shape(x: str) -> bool:
            s = shape(x)
            return s[0] == a_shape[0] and s[1] == a_shape[1] and abs(s[2] - a_shape[2]) <= 1

        def take(items):
            for x in items:
                if len(out) >= n:
                    return
                if ok(x):
                    out.append(x)

        if group_of(kind) == "DATE":
            take(self._date_variants(answer))
        if _NUM_RE.match(answer.replace(" ", "")):
            take(self._numeric(answer))
        same = [c.answer for c in pool if c.kind == kind and similar_shape(c.answer)]
        self.rng.shuffle(same)
        take(same)
        if len(out) < n:
            grp = [c.answer for c in pool if group_of(c.kind) == group_of(kind) and similar_shape(c.answer)]
            self.rng.shuffle(grp)
            take(grp)
        if len(out) < n and group_of(kind) in ("NOUN", "NAMED"):
            take(self._match_case(answer, w) for w in self._wordnet(answer))
        if len(out) < n:
            named = group_of(kind) in ("PLACE", "PERSON", "NAMED")
            others = [c.answer for c in pool if group_of(c.kind) != "NUMBER" and similar_shape(c.answer)
                      and (proper(c.answer) if named else True)]
            self.rng.shuffle(others)
            take(others)
        return out[:n]

    # ---- sources ---------------------------------------------------------------------------
    def _date_variants(self, answer: str) -> list[str]:
        """Shift the year and/or swap the month of a date-like answer."""
        out: list[str] = []
        m = _YEAR_RE.search(answer)
        if m:
            y = int(m.group(1))
            for d in self.rng.sample([1, 2, 3, 4, 5, 7, 10, 15], 3):
                out.append(answer[:m.start()] + str(y - d) + answer[m.end():])
                out.append(answer[:m.start()] + str(y + d) + answer[m.end():])
        for i, mo in enumerate(_MONTHS):
            if mo in answer:
                for j in self.rng.sample([k for k in range(12) if k != i], 3):
                    out.append(answer.replace(mo, _MONTHS[j]))
                break
        self.rng.shuffle(out)
        return out

    def _numeric(self, answer: str) -> list[str]:
        m = _NUM_RE.match(answer.replace(" ", ""))
        if not m:
            return []
        raw, dec, pct = m.groups()
        try:
            val = float(raw.replace(",", "") + (dec or ""))
        except ValueError:
            return []
        is_int = dec is None
        use_commas = "," in raw
        cands: list[float] = []
        if is_int and 1500 <= val <= 2100:                     # looks like a year
            for d in self.rng.sample(range(1, 16), 6):
                cands += [val - d, val + d]
        else:
            for f in (0.5, 2.0, 0.8, 1.25, 1.5, 0.75, 3.0, 0.1):
                cands.append(val * f)
            cands += [val + 1, val - 1, val + 10, val - 10]
        out = []
        for c in cands:
            if c <= 0 and val > 0:
                continue
            if is_int:
                c = int(round(c))
                s = f"{c:,}" if use_commas else str(c)
            else:
                s = f"{c:.{len(dec) - 1}f}"
            s += pct
            if s not in out and s != answer:
                out.append(s)
        return out

    @staticmethod
    def _wordnet(phrase: str) -> list[str]:
        try:
            from nltk.corpus import wordnet as wn

            head = phrase.split()[-1].lower()
            sibs: list[str] = []
            for syn in wn.synsets(head, pos=wn.NOUN)[:2]:
                for hyper in syn.hypernyms():
                    for hypo in hyper.hyponyms():
                        name = hypo.lemma_names()[0].replace("_", " ")
                        if name.lower() != head and name not in sibs:
                            sibs.append(name)
            random.Random(hash(phrase) & 0xFFFF).shuffle(sibs)
            return sibs[:10]
        except Exception:  # WordNet data missing
            return []

    @staticmethod
    def _match_case(answer: str, word: str) -> str:
        if answer.istitle():
            return word.title()
        if answer.isupper():
            return word.upper()
        return word.lower()
