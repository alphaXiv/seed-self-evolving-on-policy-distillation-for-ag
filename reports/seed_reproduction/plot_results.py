from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).parent
DATA = json.loads((ROOT / "data" / "results.json").read_text())
IMAGES = ROOT / "images"
IMAGES.mkdir(exist_ok=True)

COLORS = {"grpo": "#65758b", "seed": "#ec5f45"}
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "figure.dpi": 180})


def save_final_split() -> None:
    fixed = DATA["bounded_training"]["fixed_evaluation"]
    labels = ["Seen", "Unseen"]
    grpo = [fixed["seen"]["grpo"]["rate"], fixed["unseen"]["grpo"]["rate"]]
    seed = [fixed["seen"]["seed_deterministic"]["rate"], fixed["unseen"]["seed_deterministic"]["rate"]]
    x = [0, 1]
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    b1 = ax.bar([v - width / 2 for v in x], grpo, width, color=COLORS["grpo"], label="Outcome-only GRPO")
    b2 = ax.bar([v + width / 2 for v in x], seed, width, color=COLORS["seed"], label="SEED, public fallback")
    for bars in (b1, b2):
        ax.bar_label(bars, labels=[f"{v.get_height():.1%}" for v in bars], padding=3, fontsize=10)
    for idx, (a, b) in enumerate(zip(grpo, seed)):
        ax.text(idx, max(a, b) + 0.035, f"Δ {b-a:+.1%}", ha="center", color="#9f2f1d", weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.34)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("ALFWorld task success")
    ax.set_title("Bounded fixed-split evaluation (36 tasks per split)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "final_split_success.png", bbox_inches="tight")
    plt.close(fig)


def save_training_curve() -> None:
    bounded = DATA["bounded_training"]
    x = bounded["updates"]
    grpo = bounded["trainer_validation"]["grpo"]
    seed = bounded["trainer_validation"]["seed_deterministic"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(x, grpo, marker="o", linewidth=2.3, color=COLORS["grpo"], label="Outcome-only GRPO")
    ax.plot(x, seed, marker="o", linewidth=2.3, color=COLORS["seed"], label="SEED, public fallback")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 0.27)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("RL update")
    ax.set_ylabel("48-task trainer-validation success")
    ax.set_title("Early-training curve: the auxiliary signal did not improve area")
    ax.text(1, 0.245, "curve mean: GRPO 14.5% · SEED 12.3%", fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "training_curve.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    save_final_split()
    save_training_curve()
