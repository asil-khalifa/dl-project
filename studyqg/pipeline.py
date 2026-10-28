"""End-to-end StudyQG pipeline: study material -> graded, typed questions."""
from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from .answer_extraction import Candidate, HeuristicAnswerExtractor, load_nlp
from .config import DEFAULT_QG_MODEL, QA_MODEL, GenConfig
from .distractors import DistractorGenerator
from .ingest import Passage, chunk_passages, clean_text, load_text
from .quality_filter import normalize_answer

BLANK = "______"
TYPE_NAMES = {"short": "Short answer", "mcq": "Multiple choice", "fill": "Fill in the blank", "tf": "True / False"}


@dataclass
class Question:
    qid: int
    qtype: str
    question: str
    answer: str
    options: list[str] = field(default_factory=list)
    passage_id: int = 0
    source_sentence: str = ""
    quality: Optional[float] = None       # round-trip token F1 in [0, 1]
    qa_confidence: Optional[float] = None
    qa_answer: str = ""
    bloom: str = "Remember"
    answer_kind: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    questions: list[Question]
    passages: list[Passage]
    stats: dict

    def to_records(self) -> list[dict]:
        return [q.to_dict() for q in self.questions]

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        payload = {"stats": self.stats, "questions": self.to_records()}
        s = json.dumps(payload, indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    def to_markdown(self, with_answers: bool = True) -> str:
        lines = []
        for q in self.questions:
            lines.append(f"### Q{q.qid}. {TYPE_NAMES[q.qtype]} ({q.bloom})")
            lines.append(f"**{q.question}**")
            if q.qtype == "mcq":
                for letter, opt in zip("ABCD", q.options):
                    lines.append(f"- {letter}. {opt}")
            if with_answers:
                if q.qtype == "mcq":
                    letter = "ABCD"[q.options.index(q.answer)]
                    ans = f"{letter}. {q.answer}"
                else:
                    ans = q.answer
                qual = f" (quality {q.quality:.2f})" if q.quality is not None else ""
                lines.append(f"\n*Answer:* {ans}{qual}")
                lines.append(f"*Source:* \"{q.source_sentence}\"")
            lines.append("")
        return "\n".join(lines)


def bloom_level(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(compare|difference|differ|contrast|distinguish|relationship between)\b", q):
        return "Analyze"
    if re.search(r"\b(calculate|compute|how would|how could|apply|predict)\b", q):
        return "Apply"
    if re.search(r"\b(why|how does|how do|how did|how is|how are|how can|explain|what happens|"
                 r"what is the (purpose|role|effect|function|result|advantage|benefit)|what does .* do)\b", q):
        return "Understand"
    return "Remember"


def replace_ci(text: str, old: str, new: str) -> Optional[str]:
    m = re.search(re.escape(old), text, re.IGNORECASE)
    return None if not m else text[: m.start()] + new + text[m.end():]


class StudyQG:
    """Study material -> questions. Heavy models load lazily on first use."""

    def __init__(self, qg_model: str = DEFAULT_QG_MODEL, qa_model: str = QA_MODEL,
                 use_qa_filter: bool = True, use_neural_answers: bool = True,
                 device: str | None = None, num_threads: int | None = None, verbose: bool = True):
        self.qg_model_name = qg_model
        self.qa_model_name = qa_model
        self.use_qa_filter = use_qa_filter
        self.use_neural_answers = use_neural_answers
        self.device = device
        self.num_threads = num_threads
        self.verbose = verbose
        self._nlp = self._qg = self._qa = None

    # ---- lazy components ---------------------------------------------------------------------
    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = load_nlp()
        return self._nlp

    @property
    def qg(self):
        if self._qg is None:
            from .question_generator import QuestionGenerator

            self.log(f"Loading question generator: {self.qg_model_name}")
            self._qg = QuestionGenerator(self.qg_model_name, device=self.device, num_threads=self.num_threads)
        return self._qg

    @property
    def qa(self):
        if self._qa is None:
            from .quality_filter import RoundTripFilter

            self.log(f"Loading QA filter: {self.qa_model_name}")
            self._qa = RoundTripFilter(self.qa_model_name, device=self.device)
        return self._qa

    def log(self, msg: str):
        if self.verbose:
            print(f"[StudyQG] {msg}", flush=True)

    # ---- main entry point ----------------------------------------------------------------------
    def run(self, source, cfg: GenConfig | None = None) -> GenerationResult:
        cfg = cfg or GenConfig()
        rng = random.Random(cfg.seed)
        t0 = time.time()

        # 1. ingest
        text = clean_text(load_text(source))
        if len(text.split()) < 20:
            raise ValueError("Too little text to generate questions from (need at least ~20 words).")
        passages = chunk_passages(text, max_words=cfg.chunk_words, nlp=self.nlp)
        self.log(f"[1/5] {len(text.split())} words -> {len(passages)} passages")

        # 2. candidate answers (heuristic + neural), with document-level salience
        extractor = HeuristicAnswerExtractor(self.nlp)
        low_text = text.lower()
        per_passage: dict[int, list[Candidate]] = {}
        all_cands: list[Candidate] = []
        for p in passages:
            cands = extractor(p.sentences, max_per_sentence=cfg.max_candidates_per_sentence)
            if self.use_neural_answers and self.qg.supports_answer_extraction:
                for si, ans in self.qg.extract_answers(p.text, p.sentences, max_input_len=cfg.max_input_len):
                    cands.append(Candidate(ans, p.sentences[si], si, extractor.kind_of(ans), 1.3, "neural"))
            merged: dict[str, Candidate] = {}
            for c in cands:
                c.passage_id = p.pid
                c.score += 0.15 * math.log1p(max(low_text.count(c.answer.lower()) - 1, 0))
                key = c.answer.lower()
                if key not in merged or c.score > merged[key].score:
                    merged[key] = c
            per_passage[p.pid] = list(merged.values())
            all_cands.extend(merged.values())
        k = max(cfg.max_questions, math.ceil(cfg.max_questions * cfg.oversample))
        chosen = self._select(per_passage, k)
        self.log(f"[2/5] {len(all_cands)} candidate answers -> {len(chosen)} selected for generation")

        # 3. question generation
        pmap = {p.pid: p for p in passages}
        inputs, kept = [], []
        for c in chosen:
            h = self.qg.highlight(pmap[c.passage_id].text, c.answer, c.sentence)
            if h:
                inputs.append(h)
                kept.append(c)
        raw = self.qg.generate_questions(inputs, num_beams=cfg.num_beams, max_input_len=cfg.max_input_len,
                                         max_output_len=cfg.max_output_len)
        items: list[tuple[Candidate, str]] = []
        seen: list[str] = []
        for c, q in zip(kept, raw):
            q = self._tidy_question(q)
            if q is None or any(SequenceMatcher(None, q.lower(), s).ratio() > 0.9 for s in seen):
                continue
            if normalize_answer(c.answer) and normalize_answer(c.answer) in normalize_answer(q):
                continue  # the question gives the answer away
            seen.append(q.lower())
            items.append((c, q))
        self.log(f"[3/5] {len(items)} well-formed, distinct questions generated")

        # 4. round-trip quality filter
        if self.use_qa_filter and items:
            scores = self.qa.score_batch([q for _, q in items], [pmap[c.passage_id].text for c, _ in items],
                                         [c.answer for c, _ in items])
            scored = [(c, q, s) for (c, q), s in zip(items, scores)]
            passed = [x for x in scored if x[2]["f1"] >= cfg.min_quality]
            passed.sort(key=lambda x: (x[2]["f1"], x[2]["confidence"], x[0].score), reverse=True)
            self.log(f"[4/5] {len(passed)}/{len(scored)} passed the QA round-trip filter (F1 >= {cfg.min_quality})")
        else:
            passed = [(c, q, None) for c, q in items]
            passed.sort(key=lambda x: x[0].score, reverse=True)
            self.log("[4/5] QA filter disabled")
        final = passed[: cfg.max_questions]
        final.sort(key=lambda x: (x[0].passage_id, x[0].sent_idx))

        # 5. format into question types
        types = [t for t in cfg.question_types if t in TYPE_NAMES] or ["short"]
        distract = DistractorGenerator(seed=cfg.seed)
        questions = [self._build(i + 1, types[i % len(types)], c, q, s, all_cands, distract, rng)
                     for i, (c, q, s) in enumerate(final)]
        elapsed = time.time() - t0
        quals = [q.quality for q in questions if q.quality is not None]
        stats = {
            "n_words": len(text.split()), "n_passages": len(passages), "n_candidates": len(all_cands),
            "n_selected": len(chosen), "n_generated": len(items),
            "n_passed_filter": len(passed) if self.use_qa_filter else None,
            "n_final": len(questions), "mean_quality": round(sum(quals) / len(quals), 3) if quals else None,
            "by_type": {t: sum(q.qtype == t for q in questions) for t in types},
            "qg_model": self.qg_model_name, "qa_model": self.qa_model_name if self.use_qa_filter else None,
            "elapsed_s": round(elapsed, 1),
        }
        self.log(f"[5/5] {len(questions)} questions ready in {elapsed:.1f}s")
        return GenerationResult(questions, passages, stats)

    # ---- helpers -----------------------------------------------------------------------------------
    @staticmethod
    def _select(per_passage: dict[int, list[Candidate]], k: int) -> list[Candidate]:
        """Round-robin over passages (coverage), best-scored and one-per-sentence first."""
        queues: dict[int, list[Candidate]] = {}
        for pid, cands in per_passage.items():
            ranked = sorted(cands, key=lambda c: -c.score)
            first, rest, seen_sent = [], [], set()
            for c in ranked:
                (rest if c.sent_idx in seen_sent else first).append(c)
                seen_sent.add(c.sent_idx)
            queues[pid] = first + rest
        chosen, used = [], set()
        while len(chosen) < k and any(queues.values()):
            for pid in sorted(queues):
                q = queues[pid]
                while q:
                    c = q.pop(0)
                    if c.answer.lower() in used:
                        continue
                    used.add(c.answer.lower())
                    chosen.append(c)
                    break
                if len(chosen) >= k:
                    break
        return chosen

    @staticmethod
    def _tidy_question(q: str) -> Optional[str]:
        q = re.sub(r"\s+", " ", q).strip().strip('"')
        if not q or len(q.split()) < 3:
            return None
        q = q[0].upper() + q[1:]
        if not q.endswith("?"):
            q = q.rstrip(".!") + "?"
        return q

    @staticmethod
    def _build(qid: int, qtype: str, c: Candidate, q: str, s: Optional[dict], pool: list[Candidate],
               distract: DistractorGenerator, rng: random.Random) -> Question:
        base = dict(qid=qid, passage_id=c.passage_id, source_sentence=c.sentence, answer_kind=c.kind,
                    bloom=bloom_level(q))
        if s:
            base.update(quality=round(s["f1"], 3), qa_confidence=round(s["confidence"], 3), qa_answer=s["predicted"])
        if qtype == "mcq":
            ds = distract.generate(c.answer, c.kind, pool, n=3)
            if len(ds) >= 2:
                options = [c.answer] + ds
                rng.shuffle(options)
                return Question(qtype="mcq", question=q, answer=c.answer, options=options, **base)
            qtype = "short"
        if qtype == "fill":
            blank = replace_ci(c.sentence, c.answer, BLANK)
            if blank and len(c.sentence.split()) >= 6:
                return Question(qtype="fill", question=f"Fill in the blank: {blank}", answer=c.answer, **base)
            qtype = "short"
        if qtype == "tf":
            ds = distract.generate(c.answer, c.kind, pool, n=1)
            if ds and rng.random() < 0.5:
                false_stmt = replace_ci(c.sentence, c.answer, ds[0])
                if false_stmt:
                    return Question(qtype="tf", question=f"True or False: {false_stmt}",
                                    answer=f"False (correct: {c.answer})", options=["True", "False"], **base)
            return Question(qtype="tf", question=f"True or False: {c.sentence}", answer="True",
                            options=["True", "False"], **base)
        return Question(qtype="short", question=q, answer=c.answer, **base)
