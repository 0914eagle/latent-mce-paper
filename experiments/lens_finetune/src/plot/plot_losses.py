"""Train / validation loss of the lens SFT runs on the other datasets (one panel per run), in the style of the
Cancer-Myth figure: per-step train loss (faint) and its 10-step mean, validation loss at each evaluation, step 0 = the
base lens (dashed). Writes results/figures/sft_losses_other.png.
Usage (repo root, a venv with matplotlib): python -m experiments.lens_finetune.src.plot.plot_losses
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results"
RUNS = [  # (results subdir, adapter, title)
    ("hotpotqa", "fold0", "HotpotQA · default (lr 2e-5, top 2, 1 epoch) · 3-fold f0"),
    ("hotpotqa_split", "s1_lr1e-4_k1_e2", "HotpotQA · setting ① (lr 1e-4, top 1, 2 epochs) · own split"),
    ("hotpotqa", "lr5e-5_k2_e3", "HotpotQA · setting ② (lr 5e-5, top 2, 3 epochs) · 3-fold f0"),
    ("synqa2_split", "s1_lr1e-4_k1_e2", "SynQA2 · setting ① · own split"),
    ("hover", "fold0", "HoVer · default · 3-fold f0"),
    ("hover_split", "s1_lr1e-4_k1_e2", "HoVer · setting ① · own split"),
    ("medcalc_split", "s1_lr1e-4_k1_e2", "MedCalc-Bench · setting ① · own split"),
]


DATASETS = {"hotpotqa": "HotpotQA", "hover": "HoVer", "synqa2": "SynQA2", "medcalc": "MedCalc-Bench"}


def panel(ax, sub, ad, title):
    f = R / sub / "data" / "adapters" / ad / "train_log.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
    tr = [(r["step"], r["train_loss"]) for r in rows if "train_loss" in r]
    va = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    if tr:
        xs, ys = zip(*tr)
        ax.plot(xs, ys, color="#2f6fdb", alpha=0.3, lw=0.8)
        sm = [sum(ys[max(0, i - 9): i + 1]) / len(ys[max(0, i - 9): i + 1]) for i in range(len(ys))]
        ax.plot(xs, sm, color="#2f6fdb", lw=2, label="train loss (10-step mean; faint = per step)")
    if va:
        xv, yv = zip(*va)
        ax.plot(xv, yv, "--o", color="#e8673a", lw=2, ms=6, label="validation loss")
        ax.axhline(yv[0], color="k", ls="--", lw=1)
        ax.text(0.97, yv[0], "step 0 = base lens", ha="right", va="bottom", transform=ax.get_yaxis_transform(), fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("optimizer step (16 examples each)")
    ax.grid(alpha=0.3)


def main():
    out = R / "figures"; out.mkdir(exist_ok=True)
    for key, name in DATASETS.items():
        runs = [(sub, ad, t.split(" · ", 1)[1]) for sub, ad, t in RUNS if sub.split("_")[0] == key]
        fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 4.6), sharey=True, squeeze=False)
        for ax, (sub, ad, title) in zip(axes[0], runs):
            panel(ax, sub, ad, title)
        axes[0, 0].set_ylabel("cross-entropy on target tokens")
        axes[0, 0].legend(loc="lower left", fontsize=9)
        fig.suptitle(f"Oracle-lens SFT on {name}: validation loss never drops below the base lens", x=0.01, ha="left", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"sft_losses_{key}.png", dpi=110)
        plt.close(fig)
        print(out / f"sft_losses_{key}.png")


if __name__ == "__main__":
    main()
