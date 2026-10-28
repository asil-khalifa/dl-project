"""Stage 3: T5 question generation (and optional answer extraction) in highlight format."""
from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import AE_PREFIX, DEFAULT_QG_MODEL, HL, QG_PREFIX, default_threads


class QuestionGenerator:
    def __init__(self, model_name_or_path: str = DEFAULT_QG_MODEL, device: str | None = None,
                 num_threads: int | None = None):
        self.name = str(model_name_or_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(num_threads or default_threads())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device).eval()
        if HL not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [HL]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        # lmqg "*-qg-ae" checkpoints were trained on both tasks.
        self.supports_answer_extraction = self.name.endswith("-ae") or "qg-ae" in self.name

    # ---- input formatting -------------------------------------------------------------
    @staticmethod
    def highlight(passage: str, answer: str, sentence: str | None = None) -> str | None:
        """Wrap the answer span with <hl> tokens (first occurrence inside `sentence` if given)."""
        lo, hi = 0, len(passage)
        if sentence and sentence in passage:
            lo = passage.index(sentence)
            hi = lo + len(sentence)
        idx = passage.find(answer, lo, hi)
        if idx == -1:
            m = re.search(re.escape(answer), passage[lo:hi], re.IGNORECASE) or \
                re.search(re.escape(answer), passage, re.IGNORECASE)
            if not m:
                return None
            idx = m.start() + (lo if m.string is not passage else 0)
        end = idx + len(answer)
        return f"{passage[:idx]}{HL} {passage[idx:end]} {HL}{passage[end:]}"

    # ---- generation ----------------------------------------------------------------------
    @torch.inference_mode()
    def _generate(self, texts: list[str], max_input_len: int = 512, max_output_len: int = 64,
                  num_beams: int = 4, batch_size: int = 8, **kw) -> list[str]:
        outs: list[str] = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], max_length=max_input_len, truncation=True,
                                 padding=True, return_tensors="pt").to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_output_len, num_beams=num_beams,
                                      early_stopping=True, no_repeat_ngram_size=3, **kw)
            outs.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return [o.strip() for o in outs]

    def generate_questions(self, highlighted: list[str], **kw) -> list[str]:
        if not highlighted:
            return []
        return self._generate([QG_PREFIX + h for h in highlighted], **kw)

    def generate_question(self, passage: str, answer: str, sentence: str | None = None, **kw) -> str | None:
        h = self.highlight(passage, answer, sentence)
        return None if h is None else self.generate_questions([h], **kw)[0]

    def extract_answers(self, passage: str, sentences: list[str], **kw) -> list[tuple[int, str]]:
        """T5 'extract answers:' mode: highlight each sentence, read back the key span."""
        if not self.supports_answer_extraction:
            return []
        inputs, idxs = [], []
        for i, s in enumerate(sentences):
            j = passage.find(s)
            if j == -1:
                continue
            inputs.append(AE_PREFIX + passage[:j] + f"{HL} {s} {HL}" + passage[j + len(s):])
            idxs.append(i)
        kw.setdefault("max_output_len", 32)
        outs = self._generate(inputs, **kw)
        found: list[tuple[int, str]] = []
        for i, o in zip(idxs, outs):
            sent = sentences[i]
            for piece in [o] + re.split(r"\s*[,;|]\s*", o):
                piece = piece.strip(" .,;:!?\"'()")
                if len(piece) >= 2 and re.search(re.escape(piece), sent, re.IGNORECASE):
                    m = re.search(re.escape(piece), sent, re.IGNORECASE)
                    found.append((i, sent[m.start():m.end()]))   # keep the document's casing
                    break
        return found
