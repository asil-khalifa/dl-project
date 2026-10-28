"""Central configuration for StudyQG (CSPE72 Team 3)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def default_threads() -> int:
    """CPU threads for PyTorch. Half the logical CPUs is fastest on hyper-threaded machines
    (16 threads ran 10x slower than 8 on our 16-logical-core laptop). Override: STUDYQG_THREADS."""
    env = os.environ.get("STUDYQG_THREADS")
    if env:
        return max(1, int(env))
    n = os.cpu_count() or 4
    return max(1, n // 2) if n >= 8 else n

# Public checkpoint used by default in the demo: T5-base fine-tuned on the full SQuAD
# for question generation (QG) and answer extraction (AE) by Ushio et al. (2022).
DEFAULT_QG_MODEL = "lmqg/t5-base-squad-qg-ae"
# Smaller public checkpoint: faster on CPU, slightly lower quality.
SMALL_QG_MODEL = "lmqg/t5-small-squad-qg-ae"
# Our own checkpoint, produced by scripts/train.py (t5-small fine-tuned on a SQuAD subset).
OUR_QG_MODEL = "models/t5-small-studyqg"
# Extractive QA model used by the round-trip quality filter.
QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"

HL = "<hl>"                        # highlight token that marks the answer span
QG_PREFIX = "generate question: "  # T5 task prefix for question generation
AE_PREFIX = "extract answers: "    # T5 task prefix for answer extraction

# Labels shown in the Gradio app -> model id / path.
MODEL_CHOICES = {
    "T5-base (lmqg, full SQuAD) - best quality": DEFAULT_QG_MODEL,
    "T5-small (lmqg, full SQuAD) - faster": SMALL_QG_MODEL,
    "T5-small (ours, scripts/train.py)": OUR_QG_MODEL,
}

QUESTION_TYPES = ("short", "mcq", "fill", "tf")


@dataclass
class GenConfig:
    """Knobs for one generation run."""

    max_questions: int = 10
    question_types: tuple[str, ...] = QUESTION_TYPES
    chunk_words: int = 120            # passage size fed to T5
    num_beams: int = 4
    max_input_len: int = 512
    max_output_len: int = 64
    min_quality: float = 0.5          # round-trip token-F1 threshold
    oversample: float = 1.8           # generate this many x max_questions, then filter
    max_candidates_per_sentence: int = 3
    seed: int = 42
