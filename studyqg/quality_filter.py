"""Stage 4: round-trip consistency filter with an extractive QA model.

Implemented directly on top of AutoModelForQuestionAnswering (start/end span logits), because the
`question-answering` pipeline was removed in Transformers 5.
"""
from __future__ import annotations

import re
import string
from collections import Counter

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from .config import QA_MODEL, default_threads


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lower-case, strip punctuation, articles and extra spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


class RoundTripFilter:
    """Ask a QA model the generated question; keep it only if the QA answer matches the target.

    Idea from Alberti et al. (2019), "Synthetic QA corpora generation with roundtrip consistency".
    """

    def __init__(self, model_name: str = QA_MODEL, device: str | None = None, max_length: int = 384,
                 stride: int = 128, max_answer_len: int = 30):
        self.name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(default_threads())
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        self.max_length, self.stride, self.max_answer_len = max_length, stride, max_answer_len

    @torch.inference_mode()
    def answer_batch(self, questions: list[str], contexts: list[str], batch_size: int = 16) -> list[tuple[str, float]]:
        """Best answer span and its probability (p_start * p_end) for each (question, context)."""
        best: list[tuple[str, float]] = [("", 0.0)] * len(questions)
        for i in range(0, len(questions), batch_size):
            qs, cs = questions[i:i + batch_size], contexts[i:i + batch_size]
            enc = self.tok(qs, cs, truncation="only_second", max_length=self.max_length, stride=self.stride,
                           return_overflowing_tokens=True, return_offsets_mapping=True, padding=True,
                           return_tensors="pt")
            sample_map = enc.pop("overflow_to_sample_mapping").tolist()
            offsets = enc.pop("offset_mapping").tolist()
            out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
            start_logits, end_logits = out.start_logits.cpu(), out.end_logits.cpu()
            for f, si in enumerate(sample_map):
                ctx = torch.tensor([sid == 1 for sid in enc.sequence_ids(f)])
                p_s = torch.softmax(start_logits[f].masked_fill(~ctx, -1e9), -1)
                p_e = torch.softmax(end_logits[f].masked_fill(~ctx, -1e9), -1)
                n = p_s.shape[0]
                span = p_s[:, None] * p_e[None, :]
                valid = torch.triu(torch.ones(n, n, dtype=torch.bool)) & ~torch.triu(
                    torch.ones(n, n, dtype=torch.bool), diagonal=self.max_answer_len)
                span = span.masked_fill(~valid, 0.0)
                s, e = divmod(int(span.argmax()), n)
                score = float(span[s, e])
                if score > best[i + si][1]:
                    text = cs[si][offsets[f][s][0]:offsets[f][e][1]].strip()
                    best[i + si] = (text, score)
        return best

    def score(self, question: str, context: str, expected: str) -> dict:
        return self.score_batch([question], [context], [expected])[0]

    def score_batch(self, questions: list[str], contexts: list[str], expecteds: list[str],
                    batch_size: int = 16) -> list[dict]:
        if not questions:
            return []
        res = []
        for (pred, conf), exp in zip(self.answer_batch(questions, contexts, batch_size), expecteds):
            res.append({"predicted": pred, "confidence": conf, "f1": token_f1(pred, exp), "exact": exact_match(pred, exp)})
        return res
