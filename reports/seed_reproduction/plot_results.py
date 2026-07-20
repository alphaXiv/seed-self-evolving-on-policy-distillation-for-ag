#!/usr/bin/env python3
"""Render the two figures in this report from committed measured CSV data."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
COLORS = {"Outcome-only GRPO": "#6b7280", "SEED": "#2563eb"}


def read_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "data" / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def learning_curve() -> None:
    rows = read_rows("learning_curves.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method in ("Outcome-only GRPO", "SEED"):
        method_rows = [row for row in rows if row["method"] == method]
        seeds = sorted({row["rl_seed"] for row in method_rows})
        if len(seeds) > 1:
            for seed in seeds:
                selected = sorted(
                    (row for row in method_rows if row["rl_seed"] == seed),
                    key=lambda row: int(row["update"]),
                )
                ax.plot(
                    [int(row["update"]) for row in selected],
                    [100 * float(row["category_macro_success"]) for row in selected],
                    linewidth=1.0,
                    alpha=0.22,
                    color=COLORS[method],
                )
        updates = sorted({int(row["update"]) for row in method_rows})
        values = [
            100
            * mean(
                float(row["category_macro_success"])
                for row in method_rows
                if int(row["update"]) == update
            )
            for update in updates
        ]
        ax.plot(
            updates,
            values,
            marker="o",
            linewidth=2.2,
            label=method,
            color=COLORS[method],
        )
    ax.set(xlabel="RL update", ylabel="Held ALFWorld category-macro success (%)")
    ax.set_xticks(sorted({int(row["update"]) for row in rows}))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "learning_curve.png", dpi=180)
    plt.close(fig)


def final_splits() -> None:
    rows = read_rows("final_splits.csv")
    splits = ["seen", "unseen"]
    methods = ["Outcome-only GRPO", "SEED"]
    x = range(len(splits))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for index, method in enumerate(methods):
        values = [
            100 * mean(
                float(row["category_macro_success"])
                for row in rows
                if row["method"] == method and row["split"] == split
            )
            for split in splits
        ]
        positions = [position + (index - 0.5) * width for position in x]
        bars = ax.bar(positions, values, width, label=method, color=COLORS[method])
        ax.bar_label(
            bars,
            fmt="%.1f",
            label_type="center",
            color="white",
            fontweight="bold",
        )
        for position, split in zip(positions, splits):
            seed_values = [
                100 * float(row["category_macro_success"])
                for row in rows
                if row["method"] == method and row["split"] == split
            ]
            if len(seed_values) > 1:
                offsets = [0.035 * (seed_index - (len(seed_values) - 1) / 2) for seed_index in range(len(seed_values))]
                ax.scatter(
                    [position + offset for offset in offsets],
                    seed_values,
                    s=25,
                    color="white",
                    edgecolor="#111827",
                    linewidth=0.7,
                    zorder=3,
                )
    ax.set_xticks(list(x), ["Seen subset", "Held-out subset"])
    ax.set_ylabel("Category-macro success (%)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "final_splits.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    learning_curve()
    final_splits()
