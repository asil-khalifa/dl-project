"""Evaluate a question-generation model on the SQuAD validation subset: BLEU-4, ROUGE-L, METEOR."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import sacrebleu  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402
from nltk.translate.meteor_score import meteor_score  # noqa: E402
from rouge_score import rouge_scorer  # noqa: E402

from studyqg.quality_filter import exact_match  # noqa: E402
from studyqg.question_generator import QuestionGenerator  # noqa: E402


def compute_metrics(preds: list[str], refs: list[str]) -> dict:
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = sum(scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / len(preds)
    met = sum(meteor_score([word_tokenize(r)], word_tokenize(p)) for p, r in zip(preds, refs)) / len(preds)
    em = sum(exact_match(p, r) for p, r in zip(preds, refs)) / len(preds)
    return {"bleu4": round(bleu.score, 2), "bleu1": round(bleu.precisions[0], 2),
            "rougeL": round(100 * rl, 2), "meteor": round(100 * met, 2), "exact_match": round(100 * em, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF hub id or local path")
    ap.add_argument("--name", required=True, help="label used in results files")
    ap.add_argument("--data", default="data/squad_hl/validation.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.data).open(encoding="utf-8")][: args.limit]
    model_path = str(ROOT / args.model) if (ROOT / args.model).exists() else args.model
    qg = QuestionGenerator(model_path)
    t0 = time.time()
    preds = qg.generate_questions([r["highlighted"] for r in rows], num_beams=args.num_beams,
                                  batch_size=args.batch_size)
    secs = time.time() - t0
    refs = [r["target"] for r in rows]
    m = compute_metrics(preds, refs)
    m.update({"name": args.name, "model": args.model, "n": len(rows), "num_beams": args.num_beams,
              "seconds_per_question": round(secs / len(rows), 3)})
    print(json.dumps(m, indent=2))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{args.name}.json").write_text(json.dumps(
        {**m, "examples": [{"highlighted": r["highlighted"], "answer": r["answer"], "reference": ref, "prediction": p}
                           for r, ref, p in list(zip(rows, refs, preds))[:25]]}, indent=2, ensure_ascii=False))
    with (out / f"preds_{args.name}.jsonl").open("w", encoding="utf-8") as f:
        for r, ref, p in zip(rows, refs, preds):
            f.write(json.dumps({"id": r["id"], "answer": r["answer"], "reference": ref, "prediction": p},
                               ensure_ascii=False) + "\n")

    csv = out / "metrics.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    df = df[df["name"] != args.name] if len(df) else df
    df = pd.concat([df, pd.DataFrame([m])], ignore_index=True)
    cols = ["name", "model", "n", "num_beams", "bleu1", "bleu4", "rougeL", "meteor", "exact_match", "seconds_per_question"]
    df[cols].to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
