"""Stage 2: choose what to ask about (candidate answer spans) with spaCy."""
from __future__ import annotations

import re
from dataclasses import dataclass

GOOD_ENT = {
    "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "CARDINAL", "PERCENT", "MONEY",
    "QUANTITY", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "NORP", "FAC", "PRODUCT",
}
_BAD_HEADS = {"it", "this", "that", "these", "those", "they", "he", "she", "we", "you", "i",
              "there", "which", "who", "what", "something", "one", "example", "way", "thing"}
_NUM_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?%?$")


@dataclass
class Candidate:
    answer: str
    sentence: str
    sent_idx: int
    kind: str          # spaCy entity label, NOUN_CHUNK or NUMBER
    score: float
    source: str        # "ner" | "chunk" | "neural"
    passage_id: int = -1


def load_nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:  # pragma: no cover
        raise RuntimeError("spaCy model missing. Run: python -m spacy download en_core_web_sm") from e


class HeuristicAnswerExtractor:
    """Named entities + noun chunks, scored by type and position.

    The document-level salience bonus (how often a term recurs) is added by the pipeline,
    which knows the whole document.
    """

    def __init__(self, nlp=None):
        self.nlp = nlp or load_nlp()

    def kind_of(self, phrase: str) -> str:
        if _NUM_RE.match(phrase.replace(" ", "")):
            return "NUMBER"
        doc = self.nlp(phrase)
        for ent in doc.ents:
            if ent.text.strip() == phrase.strip() and ent.label_ in GOOD_ENT:
                return ent.label_
        return "NOUN_CHUNK"

    def __call__(self, sentences: list[str], max_per_sentence: int = 3) -> list[Candidate]:
        out: list[Candidate] = []
        for si, doc in enumerate(self.nlp.pipe(sentences)):
            sent = sentences[si]
            local: dict[str, Candidate] = {}

            for ent in doc.ents:
                if ent.label_ not in GOOD_ENT:
                    continue
                txt = ent.text.strip(" .,;:")
                if len(txt) < 2:
                    continue
                kind = "NUMBER" if _NUM_RE.match(txt.replace(" ", "")) else ent.label_
                c = Candidate(txt, sent, si, kind, 1.0, "ner")
                if txt.lower() not in local:
                    local[txt.lower()] = c

            for chunk in doc.noun_chunks:
                root = chunk.root
                if root.pos_ in ("PRON", "DET") or root.lemma_.lower() in _BAD_HEADS:
                    continue
                span = chunk
                while len(span) and span[0].pos_ in ("DET", "PRON"):
                    span = span[1:]
                txt = span.text.strip(" .,;:")
                n = len(txt.split())
                if len(txt) < 3 or n > 6 or txt.lower() in local:
                    continue
                score = 0.55 + (0.15 if 2 <= n <= 4 else 0.0) + (0.1 if root.pos_ == "PROPN" else 0.0)
                if root.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr"):
                    score += 0.05
                local[txt.lower()] = Candidate(txt, sent, si, "NOUN_CHUNK", score, "chunk")

            ranked = sorted(local.values(), key=lambda c: -c.score)[:max_per_sentence]
            out.extend(ranked)
        return out
