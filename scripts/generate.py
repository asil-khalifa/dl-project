"""Command-line demo: study material in, questions out (JSON + Markdown quiz)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studyqg import DEFAULT_QG_MODEL, GenConfig, StudyQG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="PDF / TXT / MD file, or raw text")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--types", default="short,mcq,fill,tf")
    ap.add_argument("--model", default=DEFAULT_QG_MODEL)
    ap.add_argument("--min_quality", type=float, default=0.5)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--out", default="results/questions.json")
    args = ap.parse_args()

    model = str(ROOT / args.model) if (ROOT / args.model).exists() else args.model
    engine = StudyQG(qg_model=model, use_qa_filter=not args.no_filter)
    cfg = GenConfig(max_questions=args.n, question_types=tuple(args.types.split(",")),
                    min_quality=args.min_quality, num_beams=args.beams)
    res = engine.run(args.input, cfg)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_json(out)
    out.with_suffix(".md").write_text(res.to_markdown(), encoding="utf-8")
    print()
    print(res.to_markdown())
    print("stats:", res.stats)
    print(f"saved {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
