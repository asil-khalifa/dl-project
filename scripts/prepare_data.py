"""Build highlight-format training data from SQuAD v1.1.

Each example:  source = "generate question: <context with <hl> answer <hl>>",  target = question.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

HL = "<hl>"
PREFIX = "generate question: "


def crop_context(ctx: str, start: int, end: int, max_chars: int = 1500) -> tuple[str, int, int]:
    """Keep long contexts within max_chars around the answer, cutting at sentence boundaries."""
    if len(ctx) <= max_chars:
        return ctx, start, end
    half = max_chars // 2
    lo = max(0, start - half)
    hi = min(len(ctx), end + half)
    cut = ctx.rfind(". ", lo, start)
    if cut != -1:
        lo = cut + 2
    cut = ctx.find(". ", end, hi)
    if cut != -1:
        hi = cut + 1
    return ctx[lo:hi], start - lo, end - lo


def convert(ex: dict) -> dict | None:
    ans = ex["answers"]["text"][0].strip()
    start = ex["answers"]["answer_start"][0]
    ctx = ex["context"]
    end = start + len(ex["answers"]["text"][0])
    if not ans or ctx[start:end].strip() != ans:
        return None
    ctx, start, end = crop_context(ctx, start, end)
    highlighted = f"{ctx[:start]}{HL} {ctx[start:end]} {HL}{ctx[end:]}"
    return {"id": ex["id"], "highlighted": highlighted, "source": PREFIX + highlighted,
            "target": ex["question"].strip(), "answer": ans, "context": ctx}


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_size", type=int, default=8000)
    ap.add_argument("--val_size", type=int, default=300)
    ap.add_argument("--out", default="data/squad_hl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("rajpurkar/squad")
    rng = random.Random(args.seed)
    out = Path(args.out)
    for split, size, name in (("train", args.train_size, "train"), ("validation", args.val_size, "validation")):
        idx = list(range(len(ds[split])))
        rng.shuffle(idx)
        rows, seen_ctx = [], set()
        for i in idx:
            ex = ds[split][i]
            if split == "validation" and ex["context"] in seen_ctx:
                continue  # one question per paragraph in the eval set -> more diverse
            r = convert(ex)
            if r is None:
                continue
            seen_ctx.add(ex["context"])
            rows.append(r)
            if size and len(rows) >= size:
                break
        write_jsonl(rows, out / f"{name}.jsonl")
        print(f"{name}: {len(rows)} examples -> {out / (name + '.jsonl')}")
    print("example source:", rows[0]["source"][:200], "...")
    print("example target:", rows[0]["target"])


if __name__ == "__main__":
    main()
