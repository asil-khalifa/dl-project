"""Plots for the report: training curve and metric comparison -> results/*.png."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#c9c8c2"
LABELS = {
    "t5-small-zero-shot": "t5-small\nzero-shot",
    "t5-small-ours": "t5-small\nours\n8k ex., CPU",
    "t5-small-lmqg": "t5-small\nlmqg\nfull SQuAD",
    "t5-base-lmqg": "t5-base\nlmqg\nfull SQuAD",
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def training_curve():
    p = RES / "train_log.json"
    if not p.exists():
        print("no train_log.json, skipping training curve")
        return
    log = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150, facecolor=SURFACE)
    style(ax)
    ax.plot(log["steps"], log["loss"], color=BLUE, lw=2, solid_joinstyle="round", solid_capstyle="round",
            label=f"training loss (mean of {log['config']['log_every']} steps)")
    n_ep = max(len(log["epoch_val_loss"]), 1)
    per = log["steps"][-1] / n_ep
    xs = [0] + [per * (i + 1) for i in range(len(log["epoch_val_loss"]))]
    ys = [log["initial_val_loss"]] + log["epoch_val_loss"]
    ax.plot(xs, ys, color=ORANGE, lw=0, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2,
            label="validation loss (before training, epoch end)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("optimizer step (batch size %d)" % log["config"]["batch_size"], color=INK2, fontsize=9)
    ax.set_ylabel("cross-entropy loss", color=INK2, fontsize=9)
    base = log["config"]["base_model"].split("/")[-1]
    ax.set_title(f"Fine-tuning {base} on SQuAD: validation loss falls from {ys[0]:.2f} to {ys[-1]:.2f}",
                 loc="left", fontsize=11, color=INK, fontweight="semibold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "training_curve.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote results/training_curve.png")


def metrics_chart():
    p = RES / "metrics.csv"
    if not p.exists():
        print("no metrics.csv, skipping metrics chart")
        return
    df = pd.read_csv(p)
    df = df[df["name"].isin(LABELS)]
    if df.empty:
        print("metrics.csv has no known models yet")
        return
    order = [n for n in LABELS if n in set(df["name"])]
    df = df.set_index("name").loc[order].reset_index()
    metrics = [("bleu4", "BLEU-4"), ("rougeL", "ROUGE-L"), ("meteor", "METEOR")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), dpi=150, facecolor=SURFACE)
    for ax, (col, title) in zip(axes, metrics):
        style(ax)
        colors = [BLUE if n == "t5-small-ours" else MUTED for n in df["name"]]
        bars = ax.bar(range(len(df)), df[col], width=0.32, color=colors, linewidth=0)
        for b, v in zip(bars, df[col]):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([LABELS[n] for n in df["name"]], fontsize=8, color=INK2, linespacing=1.3)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="semibold")
        ax.set_ylim(0, max(df[col].max() * 1.25, 1))
    n = int(df["n"].iloc[0])
    fig.suptitle(f"Question quality on {n} SQuAD validation paragraphs (higher is better); our fine-tuned model in blue",
                 x=0.01, ha="left", fontsize=11, color=INK, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RES / "metrics.png", facecolor=SURFACE)
    plt.close(fig)
    print("wrote results/metrics.png")


if __name__ == "__main__":
    training_curve()
    metrics_chart()
