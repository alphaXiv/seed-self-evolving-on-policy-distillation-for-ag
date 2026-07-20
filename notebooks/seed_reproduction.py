# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.23.14",
#     "matplotlib==3.11.1",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _(mo):
    mo.md(r"""
    # SEED on ALFWorld: a bounded causal reproduction

    **Partially reproduced.** Across two matched training seeds,
    confidence-gated, skill-induced on-policy distillation improved mean fixed
    category-macro success from **13.0% to 15.0% on seen tasks** and from
    **16.4% to 16.7% on held-out tasks**. Both small effects reverse on the
    second seed, and early-training macro-AUC was lower than outcome-only GRPO.

    This notebook opens with already-produced Kubernetes evidence. It does not
    rerun expensive training and does not depend on repository-relative files.
    """)
    return


@app.cell
def _():
    final_results = [
        {"method": "Outcome-only GRPO", "seed": 260714777, "split": "Seen", "macro_success": 12.037},
        {"method": "SEED", "seed": 260714777, "split": "Seen", "macro_success": 18.056},
        {"method": "Outcome-only GRPO", "seed": 260714777, "split": "Held-out", "macro_success": 16.991},
        {"method": "SEED", "seed": 260714777, "split": "Held-out", "macro_success": 21.759},
        {"method": "Outcome-only GRPO", "seed": 260714778, "split": "Seen", "macro_success": 13.889},
        {"method": "SEED", "seed": 260714778, "split": "Seen", "macro_success": 12.037},
        {"method": "Outcome-only GRPO", "seed": 260714778, "split": "Held-out", "macro_success": 15.741},
        {"method": "SEED", "seed": 260714778, "split": "Held-out", "macro_success": 11.574},
    ]
    return (final_results,)


@app.cell
def _(final_results, plt):
    _colors = {"Outcome-only GRPO": "#6b7280", "SEED": "#2563eb"}
    _splits = ["Seen", "Held-out"]
    _methods = ["Outcome-only GRPO", "SEED"]
    _fig, _ax = plt.subplots(figsize=(7.2, 4.0))
    for _index, _method in enumerate(_methods):
        _seed_values = [
            [
                row["macro_success"]
                for row in final_results
                if row["split"] == split and row["method"] == _method
            ]
            for split in _splits
        ]
        _values = [sum(values) / len(values) for values in _seed_values]
        _positions = [_x + (_index - 0.5) * 0.34 for _x in range(2)]
        _bars = _ax.bar(
            _positions,
            _values,
            0.34,
            label=_method,
            color=_colors[_method],
        )
        _ax.bar_label(
            _bars,
            fmt="%.1f",
            label_type="center",
            color="white",
            fontweight="bold",
        )
        for _position, _points in zip(_positions, _seed_values):
            _ax.scatter(
                [_position - 0.025, _position + 0.025],
                _points,
                s=28,
                color="white",
                edgecolor="#111827",
                linewidth=0.7,
                zorder=3,
            )
    _ax.set_xticks(range(2), ["Seen subset", "Held-out subset"])
    _ax.set_ylabel("Category-macro success (%)")
    _ax.legend(frameon=False)
    _ax.grid(axis="y", alpha=0.2)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(final_results, mo):
    mo.vstack(
        [
            mo.md("## Measured fixed-split evidence"),
            mo.ui.table(final_results, selection=None),
            mo.callout(
                "Two training seeds and 36 tasks per split: dots are seeds and "
                "bars are means. No model-seed confidence interval is claimed.",
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _():
    updates = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    curves = {
        "Outcome-only GRPO": [
            [6.1, 9.7, 16.5, 17.9, 10.0, 11.0, 14.8, 13.3, 19.5],
            [19.7, 10.2, 22.8, 14.0, 14.8, 15.1, 10.6, 6.7, 19.8],
        ],
        "SEED": [
            [6.7, 4.6, 2.8, 16.7, 13.5, 8.8, 12.1, 15.0, 18.1],
            [18.8, 13.9, 16.1, 11.6, 12.3, 8.3, 8.3, 7.1, 8.9],
        ],
    }
    return curves, updates


@app.cell
def _(curves, plt, updates):
    _fig, _ax = plt.subplots(figsize=(7.2, 4.0))
    _colors = {"Outcome-only GRPO": "#6b7280", "SEED": "#2563eb"}
    for _method, _seed_curves in curves.items():
        for _curve in _seed_curves:
            _ax.plot(updates, _curve, linewidth=1.0, alpha=0.22, color=_colors[_method])
        _mean_curve = [sum(values) / len(values) for values in zip(*_seed_curves)]
        _ax.plot(updates, _mean_curve, marker="o", linewidth=2.2, color=_colors[_method], label=_method)
    _ax.set(xlabel="RL update", ylabel="Held validation macro success (%)")
    _ax.set_xticks(updates)
    _ax.grid(alpha=0.2)
    _ax.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What was tested

    Both agents start from the same deterministic 48-example trajectory-skill
    SFT initialization and use the same 16 ALFWorld training tasks, group size
    4, outcome reward, optimizer, and evaluations. The sole causal difference
    is the auxiliary loss:

    \[
    L_{\mathrm{SEED}} = L_{\mathrm{GRPO}} + \lambda L_{\mathrm{OPD}},
    \qquad \lambda=0.01.
    \]

    The sampled action is re-scored after adding a hindsight skill. A detached,
    beta-scaled log-probability shift gates dense token distillation. At update
    1, 93.1% of eligible tokens carried a signal and 33.9% passed the confidence
    gate, so the treatment demonstrably exercised OPD.

    The unavailable GLM-5.2 analyzer was replaced by a deterministic public
    task-family skill. Qwen3-1.7B, 40 updates, 16 training tasks, and two seeds
    are all smaller than the paper's Qwen2.5-3B, 150-update experiment.
    """)
    return


@app.cell
def _():
    claims = [
        {
            "claim": "Final seen improvement",
            "paper": "+16.8 points",
            "observed": "+2.1-point mean; seed effects +6.0, -1.9",
            "assessment": "Partially aligned; not seed-robust",
        },
        {
            "claim": "Early sample efficiency",
            "paper": "Positive",
            "observed": "Mean AUC -2.66 points; both seeds negative",
            "assessment": "Inconclusive under this setup",
        },
        {
            "claim": "Held-out improvement",
            "paper": "+15.3 points",
            "observed": "+0.3-point mean; seed effects +4.8, -4.2",
            "assessment": "Inconclusive; near-zero mean",
        },
    ]
    return (claims,)


@app.cell
def _(claims, mo):
    mo.vstack(
        [
            mo.md("## Claim-by-claim conclusion"),
            mo.ui.table(claims, selection=None),
            mo.md(r"""
            Two-seed normalized trapezoidal macro-AUC was **11.08% SEED vs
            13.75% GRPO** over updates 0–40, and **11.42% vs 14.55%** over
            updates 0–20. The small positive mean fixed endpoints coexist with
            negative learning-curve evidence and seed sign reversals; all are
            retained without checkpoint selection.
            """),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Evidence and compute

    Each matched run used 8 GPUs on Kubernetes, and paired runs filled all 16
    available **NVIDIA RTX PRO 6000 Blackwell** GPUs. The complete experiment
    campaign elapsed **7.480803 wall hours**. Every curve and table value is
    traceable to terminal logs for runs `a1bacf8d-0236-44dc-be84-0fd681ffbf3e`,
    `43c2689e-c47c-4f22-bad1-9abd7783ff35`,
    `0fbab25f-2c01-415f-a6f6-1e0d1160c6ba`, and
    `1a09c408-a0cd-457e-beb8-d4ad3fcb10d1`.

    - [Detailed report](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/reports/seed_reproduction/report.md)
    - [Measured data](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/main/reports/seed_reproduction/data)
    - [Paper](https://arxiv.org/abs/2607.14777)
    - [Authors' code](https://github.com/jinyangwu/SEED)
    """)
    return


if __name__ == "__main__":
    app.run()
