"""StudyQG demo: Gradio web app. Run:  python app/app.py  (opens http://localhost:7860)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from studyqg import GenConfig, StudyQG  # noqa: E402
from studyqg.config import MODEL_CHOICES, OUR_QG_MODEL  # noqa: E402
from studyqg.pipeline import TYPE_NAMES  # noqa: E402

SAMPLES = {p.stem.replace("_", " ").title(): p for p in sorted((ROOT / "data" / "sample").glob("*.txt"))}
TYPE_LABELS = {"Short answer": "short", "Multiple choice": "mcq", "Fill in the blank": "fill", "True / False": "tf"}
_engines: dict = {}


def model_labels() -> list[str]:
    return [label for label, mid in MODEL_CHOICES.items() if mid != OUR_QG_MODEL or (ROOT / mid).exists()]


def get_engine(model_label: str, use_filter: bool) -> StudyQG:
    mid = MODEL_CHOICES[model_label]
    path = str(ROOT / mid) if (ROOT / mid).exists() else mid
    key = (path, use_filter)
    if key not in _engines:
        _engines[key] = StudyQG(qg_model=path, use_qa_filter=use_filter)
    return _engines[key]


def load_sample(name: str) -> str:
    return SAMPLES[name].read_text(encoding="utf-8") if name in SAMPLES else ""


def generate(file, text, model_label, n, type_labels, min_quality, use_filter, beams, progress=gr.Progress()):
    source = file if file else (text or "").strip()
    if not source:
        raise gr.Error("Upload a PDF/TXT file or paste some study material first.")
    types = tuple(TYPE_LABELS[t] for t in type_labels) or ("short",)
    progress(0.05, desc="Loading models (first run downloads them)...")
    engine = get_engine(model_label, use_filter)
    progress(0.2, desc="Reading material, selecting answers, generating questions...")
    t0 = time.time()
    res = engine.run(source, GenConfig(max_questions=int(n), question_types=types,
                                       min_quality=float(min_quality), num_beams=int(beams)))
    elapsed = time.time() - t0
    rows = []
    for q in res.questions:
        rows.append({"#": q.qid, "Type": TYPE_NAMES[q.qtype], "Question": q.question,
                     "Options": " | ".join(f"{l}. {o}" for l, o in zip("ABCD", q.options)) if q.qtype == "mcq" else "",
                     "Answer": q.answer, "Quality": "" if q.quality is None else f"{q.quality:.2f}",
                     "Bloom": q.bloom, "Source sentence": q.source_sentence})
    df = pd.DataFrame(rows, columns=["#", "Type", "Question", "Options", "Answer", "Quality", "Bloom", "Source sentence"])
    s = res.stats
    filt = f", {s['n_passed_filter']}/{s['n_generated']} passed the QA round-trip filter" if s["n_passed_filter"] is not None else ""
    stats_md = (f"**{s['n_final']} questions** from {s['n_words']} words / {s['n_passages']} passages. "
                f"{s['n_candidates']} candidate answers, {s['n_generated']} questions generated{filt}. "
                f"Mean quality {s['mean_quality']}. Model `{s['qg_model']}`. {elapsed:.1f} s.")
    tmp = Path(tempfile.mkdtemp(prefix="studyqg_"))
    json_path, csv_path = tmp / "questions.json", tmp / "questions.csv"
    res.to_json(json_path)
    df.to_csv(csv_path, index=False)
    return df, res.to_markdown(), stats_md, str(json_path), str(csv_path)


with gr.Blocks(title="StudyQG - Automatic Question Generation") as demo:
    gr.Markdown(
        "# StudyQG: Automatic Question Generation from Study Material\n"
        "CSPE72 Deep Learning Techniques, Team 3 (Asil Khalifa, L Hitesh Reddy, P Kamal Nag). "
        "Pipeline: PDF/text ingestion -> answer selection (spaCy + T5) -> T5 question generation "
        "-> DistilBERT round-trip quality filter -> short / MCQ / fill-in-the-blank / true-false items."
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="Upload study material (.pdf / .txt / .md)", file_types=[".pdf", ".txt", ".md"], type="filepath")
            with gr.Row():
                sample_dd = gr.Dropdown(choices=list(SAMPLES), value=next(iter(SAMPLES), None), label="...or load a sample")
                sample_btn = gr.Button("Load sample", size="sm")
            text_in = gr.Textbox(label="...or paste text", lines=12, placeholder="Paste a few paragraphs of notes here")
            model_dd = gr.Dropdown(choices=model_labels(), value=model_labels()[0], label="Question generator")
            n_sl = gr.Slider(1, 30, value=10, step=1, label="Number of questions")
            types_cg = gr.CheckboxGroup(choices=list(TYPE_LABELS), value=list(TYPE_LABELS), label="Question types (cycled)")
            with gr.Accordion("Advanced", open=False):
                filt_cb = gr.Checkbox(value=True, label="Round-trip QA quality filter")
                minq_sl = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Minimum quality (token F1)")
                beams_sl = gr.Slider(1, 8, value=4, step=1, label="Beam size")
            go_btn = gr.Button("Generate questions", variant="primary")
        with gr.Column(scale=2):
            stats_out = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Table"):
                    table_out = gr.Dataframe(wrap=True, interactive=False)
                with gr.Tab("Quiz view"):
                    quiz_out = gr.Markdown()
            with gr.Row():
                json_out = gr.File(label="Download JSON")
                csv_out = gr.File(label="Download CSV")

    sample_btn.click(load_sample, inputs=sample_dd, outputs=text_in)
    go_btn.click(generate, inputs=[file_in, text_in, model_dd, n_sl, types_cg, minq_sl, filt_cb, beams_sl],
                 outputs=[table_out, quiz_out, stats_out, json_out, csv_out])

if __name__ == "__main__":
    demo.queue().launch(server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)),
                        share=os.environ.get("SHARE", "0") == "1")
