"""Fine-tune T5 for answer-aware question generation (highlight format) on SQuAD.

Plain PyTorch loop (no Trainer) so every step is visible: tokenise -> forward with labels
(teacher forcing, cross-entropy) -> backward -> AdamW step -> linear LR decay.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studyqg.config import default_threads  # noqa: E402

HL = "<hl>"


class JsonlDataset(Dataset):
    def __init__(self, path: Path, limit: int = 0):
        self.rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(tok, max_source_len: int, max_target_len: int):
    def collate(batch):
        enc = tok([b["source"] for b in batch], max_length=max_source_len, truncation=True,
                  padding=True, return_tensors="pt")
        labels = tok(text_target=[b["target"] for b in batch], max_length=max_target_len,
                     truncation=True, padding=True, return_tensors="pt")["input_ids"]
        labels[labels == tok.pad_token_id] = -100
        enc["labels"] = labels
        return enc
    return collate


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item() * batch["labels"].shape[0]
        n += batch["labels"].shape[0]
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google-t5/t5-small")
    ap.add_argument("--data", default="data/squad_hl")
    ap.add_argument("--out", default="models/t5-small-studyqg")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max_source_len", type=int, default=384)
    ap.add_argument("--max_target_len", type=int, default=48)
    ap.add_argument("--max_train", type=int, default=0, help="use only the first N training rows")
    ap.add_argument("--max_val", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0, help="0 = half the logical CPUs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_path", default="results/train_log.json")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads or default_threads())
    print(f"device={device} threads={torch.get_num_threads()} base={args.base_model}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    if HL not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [HL]})
        model.resize_token_embeddings(len(tok))
    model.to(device).train()
    n_params = sum(p.numel() for p in model.parameters())

    data = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    train_ds = JsonlDataset(data / "train.jsonl", args.max_train)
    val_ds = JsonlDataset(data / "validation.jsonl", args.max_val)
    collate = make_collate(tok, args.max_source_len, args.max_target_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    print(f"train={len(train_ds)} val={len(val_ds)} params={n_params/1e6:.1f}M steps={total_steps}")

    log = {"config": vars(args), "n_params": n_params, "steps": [], "loss": [], "lr": [],
           "epoch_val_loss": [], "initial_val_loss": None}
    log["initial_val_loss"] = evaluate_loss(model, val_dl, device)
    print(f"initial val loss = {log['initial_val_loss']:.4f}")

    step, running, t0 = 0, [], time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running.append(loss.item() * args.grad_accum)
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == total_steps:
                    mean = sum(running) / len(running)
                    running = []
                    log["steps"].append(step)
                    log["loss"].append(mean)
                    log["lr"].append(sched.get_last_lr()[0])
                    el = time.time() - t0
                    eta = el / step * (total_steps - step)
                    print(f"epoch {epoch+1} step {step}/{total_steps} loss {mean:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} elapsed {el/60:.1f}m eta {eta/60:.1f}m", flush=True)
        vl = evaluate_loss(model, val_dl, device)
        log["epoch_val_loss"].append(vl)
        print(f"== epoch {epoch+1} done: val loss {vl:.4f}", flush=True)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log["elapsed_s"] = time.time() - t0
    log_path = ROOT / args.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"saved model to {out}; log to {log_path}; {log['elapsed_s']/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
