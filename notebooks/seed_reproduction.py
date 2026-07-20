# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "altair==5.5.0",
#   "marimo==0.23.0",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    return alt, mo


@app.cell
def _(mo):
    mo.md(r"""
    # SEED on bounded ALFWorld: measured evidence first

    **Partial reproduction of arXiv:2607.14777.** Across two matched
    40-update Qwen3-1.7B seeds, confidence-gated on-policy distillation moved
    pooled fixed seen success from **11/72 to 13/72**, while unseen success
    tied at **14/72**. The two-seed early-training curve moved in the opposite
    direction: its mean was 11.9% for SEED versus 15.0% for outcome-only GRPO.

    This notebook embeds the completed Kubernetes evidence. It does not ask
    you to rerun expensive training to see the result.
    """)
    return


@app.cell
def _():
    fixed_rows = [
        {"split": "Seen", "method": "Outcome-only GRPO", "success": 0.152778, "count": "11/72"},
        {"split": "Seen", "method": "SEED, public fallback", "success": 0.180556, "count": "13/72"},
        {"split": "Unseen", "method": "Outcome-only GRPO", "success": 0.194444, "count": "14/72"},
        {"split": "Unseen", "method": "SEED, public fallback", "success": 0.194444, "count": "14/72"},
    ]
    curve_rows = []
    updates = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    grpo = [0.0625, 0.1041, 0.1876, 0.1668, 0.1458, 0.1877, 0.1461, 0.1252, 0.2083]
    seed = [0.0521, 0.1146, 0.0834, 0.1458, 0.1459, 0.1251, 0.1146, 0.1253, 0.1457]
    for step, grpo_value, seed_value in zip(updates, grpo, seed):
        curve_rows.extend(
            [
                {"update": step, "method": "Outcome-only GRPO", "success": grpo_value},
                {"update": step, "method": "SEED, public fallback", "success": seed_value},
            ]
        )
    return curve_rows, fixed_rows, updates


@app.cell
def _(alt, fixed_rows, mo):
    final_chart = (
        alt.Chart(alt.Data(values=fixed_rows))
        .mark_bar()
        .encode(
            x=alt.X("method:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("success:Q", title="ALFWorld success", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.32])),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(domain=["Outcome-only GRPO", "SEED, public fallback"], range=["#65758b", "#ec5f45"]),
                legend=None,
            ),
            column=alt.Column("split:N", title=None),
            tooltip=["split:N", "method:N", alt.Tooltip("success:Q", format=".1%"), "count:N"],
        )
        .properties(width=240, height=300, title="Pooled two-seed fixed evaluations")
    )
    mo.vstack([final_chart, mo.md("Seen **+2.8 points** · unseen **0.0 points** for SEED.")])
    return


@app.cell
def _(alt, curve_rows, mo):
    curve_chart = (
        alt.Chart(alt.Data(values=curve_rows))
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("update:Q", title="RL update"),
            y=alt.Y("success:Q", title="48-task validation success", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.27])),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(domain=["Outcome-only GRPO", "SEED, public fallback"], range=["#65758b", "#ec5f45"]),
            ),
            tooltip=["update:Q", "method:N", alt.Tooltip("success:Q", format=".1%")],
        )
        .properties(width="container", height=330, title="Early-training curve")
    )
    mo.vstack(
        [
            mo.md("## The endpoint gain did not come with early sample efficiency"),
            curve_chart,
            mo.callout("Two-seed curve mean: GRPO 15.0%, SEED 11.9% (−3.1 points).", kind="warn"),
        ]
    )
    return


@app.cell
def _(mo, updates):
    selected_update = mo.ui.slider(start=0, stop=40, step=5, value=20, label="Inspect update")
    mo.vstack([mo.md("### Explore a matched validation checkpoint"), selected_update])
    return (selected_update,)


@app.cell
def _(curve_rows, mo, selected_update):
    selected_rows = [row for row in curve_rows if row["update"] == selected_update.value]
    selected_map = {row["method"]: row["success"] for row in selected_rows}
    difference = selected_map["SEED, public fallback"] - selected_map["Outcome-only GRPO"]
    mo.md(
        f"""
        At update **{selected_update.value}**, GRPO success was
        **{selected_map['Outcome-only GRPO']:.1%}** and SEED success was
        **{selected_map['SEED, public fallback']:.1%}**
        (SEED difference **{difference:+.1%}**).
        """
    )
    return


@app.cell
def _():
    claims = [
        {
            "claim": "Seen success",
            "paper": "75.0% GRPO → 91.8% SEED (+16.8 pt)",
            "observed": "pooled 15.3% → 18.1% (+2.8 pt)",
            "assessment": "weak direction; partial",
        },
        {
            "claim": "Early sample efficiency",
            "paper": "SEED higher across training fractions",
            "observed": "2-seed curve mean 15.0% GRPO vs 11.9% SEED",
            "assessment": "not aligned in this setup",
        },
        {
            "claim": "Unseen split",
            "paper": "70.9% GRPO → 86.2% SEED (+15.3 pt)",
            "observed": "pooled 19.4% → 19.4% (0.0 pt)",
            "assessment": "inconclusive under this setup",
        },
    ]
    return (claims,)


@app.cell
def _(claims, mo):
    mo.vstack([mo.md("## Claim-by-claim assessment"), mo.ui.table(claims, selection=None)])
    return


@app.cell
def _():
    sensitivities = [
        {
            "arm": "GRPO, 20 updates",
            "seen": "6/36 (16.7%)",
            "unseen": "6/36 (16.7%)",
            "curve mean": "7.5%",
        },
        {
            "arm": "SEED λ=0.001, 20 updates",
            "seen": "8/36 (22.2%)",
            "unseen": "7/36 (19.4%)",
            "curve mean": "8.8%",
        },
        {
            "arm": "SEED raw hindsight, 40 updates",
            "seen": "6/36 (16.7%)",
            "unseen": "6/36 (16.7%)",
            "curve mean": "9.7%",
        },
    ]
    return (sensitivities,)


@app.cell
def _(mo, sensitivities):
    mo.vstack(
        [
            mo.md(
                """
                ## Sensitivity to λ and analyzer content

                Lowering λ to 0.001 produced small positive differences over
                a separately run, matched 20-update control. Preserving raw
                unstructured Qwen3 prose at λ=0.01 was weaker than the primary
                concise task-family fallback. The auxiliary path was active in
                every SEED variant, so this is a content sensitivity—not an
                accidentally disabled loss.
                """
            ),
            mo.ui.table(sensitivities, selection=None),
        ]
    )
    return


@app.cell
def _():
    implementation = [
        {"stage": "Pinned source", "choice": "Authors' SEED commit 2cf2fad"},
        {"stage": "Stage-1 trajectories", "choice": "12 public ALFWorld rollouts across six task families"},
        {"stage": "Skill SFT", "choice": "48 balanced rows; four identical SFT steps in both arms"},
        {"stage": "Outcome-only arm", "choice": "GRPO, online analysis off, OPD coefficient 0"},
        {"stage": "SEED arm", "choice": "public Qwen3 analyzer + deterministic fallback, λ=0.01, β=5"},
        {"stage": "Final evidence", "choice": "merged checkpoints; fixed 36 seen + 36 unseen tasks"},
    ]
    return (implementation,)


@app.cell
def _(implementation, mo):
    mo.vstack(
        [
            mo.md(
                """
                ## How the bounded reproduction works

                The fixed command is `bash .orx/run.sh`. Both arms use the
                same SFT parquet hashes and task manifests. The SEED branch
                alone enables episode analysis and the released
                confidence-gated OPD loss.
                """
            ),
            mo.ui.table(implementation, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What the substitution changes

    Only 1 of 12 Stage-1 responses from the public Qwen3 analyzer satisfied
    the released JSON parser. A deterministic task-family fallback therefore
    supplied balanced skills for SFT and online training. The signal was
    active—at update 38, 61.0% of response tokens received OPD targets and
    33.1% passed the confidence gate—but these generic skills are not
    equivalent to the paper's unavailable GLM-5.2 hindsight.

    Other downscaling: Qwen3-1.7B instead of Qwen2.5-3B, 40 instead of 150
    updates, group size 4 instead of 8, 16 bounded training tasks, 36 tasks per
    final split per seed, and two full causal seeds.

    **Compute:** Kubernetes; NVIDIA RTX PRO 6000 Blackwell; 8 GPUs per arm;
    16 peak concurrent GPUs. The four full causal runs took 1.946944 h and
    1.969167 h for GRPO, and 2.145278 h and 2.178611 h for SEED. The measured
    observed Kubernetes campaign wall time was 7.480803 hours.

    **Sources:** [paper](https://arxiv.org/abs/2607.14777) ·
    [authors' code](https://github.com/jinyangwu/SEED) ·
    [detailed report](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/reports/seed_reproduction/report.md)
    """)
    return


if __name__ == "__main__":
    app.run()
