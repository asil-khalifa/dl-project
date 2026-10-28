"""Fast unit tests (no model downloads)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studyqg.answer_extraction import Candidate
from studyqg.distractors import DistractorGenerator
from studyqg.ingest import chunk_passages, clean_text
from studyqg.pipeline import bloom_level, replace_ci
from studyqg.quality_filter import exact_match, token_f1


def test_clean_text_joins_wrapped_lines_and_drops_page_numbers():
    raw = "Photosyn-\nthesis happens in chloro-\nplasts.\n12\n\nSecond paragraph here."
    out = clean_text(raw)
    assert "Photosynthesis happens in chloroplasts." in out
    assert "\n12" not in out
    assert "\n\nSecond paragraph" in out


def test_chunk_passages_respects_word_limit():
    text = " ".join(f"Sentence number {i} is here and it is fairly long indeed." for i in range(40))
    passages = chunk_passages(text, max_words=60)
    assert len(passages) > 1
    assert all(p.n_words <= 72 for p in passages)  # one sentence may overflow slightly
    assert " ".join(s for p in passages for s in p.sentences).startswith("Sentence number 0")


def test_token_f1_and_exact_match():
    assert token_f1("the Calvin cycle", "Calvin cycle") == 1.0
    assert exact_match("The Calvin Cycle.", "calvin cycle")
    assert token_f1("stroma", "thylakoid membranes") == 0.0
    assert 0 < token_f1("thylakoid", "thylakoid membranes") < 1


def test_replace_ci_and_bloom():
    assert replace_ci("ISRO was founded in 1969.", "1969", "____") == "ISRO was founded in ____."
    assert replace_ci("no match here", "xyz", "_") is None
    assert bloom_level("Why does ReLU reduce vanishing gradients?") == "Understand"
    assert bloom_level("Who proposed the perceptron?") == "Remember"


def test_numeric_distractors_are_distinct_numbers():
    dg = DistractorGenerator(seed=1)
    ds = dg.generate("1969", "NUMBER", pool=[], n=3)
    assert len(ds) == 3 and "1969" not in ds and all(d.isdigit() for d in ds)


def test_same_type_distractors_come_from_pool():
    pool = [Candidate("Vikram Sarabhai", "s", 0, "PERSON", 1, "ner"), Candidate("Melvin Calvin", "s", 0, "PERSON", 1, "ner"),
            Candidate("Jan Ingenhousz", "s", 0, "PERSON", 1, "ner"), Candidate("Bengaluru", "s", 0, "GPE", 1, "ner")]
    ds = DistractorGenerator(seed=1).generate("Yann LeCun", "PERSON", pool, n=3)
    assert set(ds) == {"Vikram Sarabhai", "Melvin Calvin", "Jan Ingenhousz"}
