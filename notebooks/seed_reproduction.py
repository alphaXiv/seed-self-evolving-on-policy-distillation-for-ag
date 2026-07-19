import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # SEED ALFWorld checkpoint reproduction

    **Verdict: not reproduced.** This notebook is a self-contained,
    executable account of the evidence boundary for arXiv:2607.14777. A
    faithful evaluation protocol was implemented, but OpenResearch recorded
    no training or evaluation run. Paper values below are references—not
    reproduction measurements.
    """)
    return


@app.cell
def _():
    paper_results = [
        {"metric": "ALFWorld macro success", "GRPO": 75.0, "SEED": 91.8},
        {"metric": "Search-QA accuracy", "GRPO": 36.4, "SEED": 45.7},
        {"metric": "WebShop score", "GRPO": 79.8, "SEED": 88.5},
        {"metric": "WebShop success", "GRPO": 63.3, "SEED": 78.9},
    ]
    paper_results_with_gains = [
        {**row, "reported_gain": round(row["SEED"] - row["GRPO"], 1)}
        for row in paper_results
    ]
    return (paper_results_with_gains,)


@app.cell
def _(mo, paper_results_with_gains):
    mo.vstack(
        [
            mo.md("## Paper-reported Qwen2.5-3B reference values"),
            mo.ui.table(paper_results_with_gains, selection=None),
            mo.callout(
                "These values come from the paper. No observed reproduction "
                "metric exists in this project.",
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _():
    evidence = {
        "recorded_runs": 0,
        "maximum_concurrent_gpus": 0,
        "actual_wall_hours": 0.0,
        "observed_alfworld_score": None,
        "verdict": "not-reproduced",
    }
    return (evidence,)


@app.cell
def _(evidence, mo):
    observed = (
        "not measured"
        if evidence["observed_alfworld_score"] is None
        else str(evidence["observed_alfworld_score"])
    )
    mo.md(
        f"""
        ## Auditable result

        | Evidence item | Observed value |
        |---|---:|
        | OpenResearch runs | {evidence['recorded_runs']} |
        | Maximum concurrently allocated GPUs | {evidence['maximum_concurrent_gpus']} |
        | Actual compute wall time | {evidence['actual_wall_hours']:.1f} hours |
        | Reproduced ALFWorld score | {observed} |
        | Verdict | **{evidence['verdict']}** |

        A runnable harness is useful engineering work, but it cannot substitute
        for terminal run logs. With no observed score, agreement with the
        paper's 91.8 cannot be tested.
        """
    )
    return


@app.cell
def _():
    protocol = [
        {"setting": "Official code commit", "value": "2cf2fadca3c5aba28da68e8e1405182ba8d90e6c"},
        {"setting": "Evaluation split", "value": "eval_in_distribution"},
        {"setting": "Parallel environments", "value": "134"},
        {"setting": "Evaluation rounds", "value": "3"},
        {"setting": "Maximum steps", "value": "30"},
        {"setting": "History length", "value": "5"},
        {"setting": "Temperature", "value": "0.4"},
        {"setting": "Data parallel size", "value": "8"},
    ]
    return (protocol,)


@app.cell
def _(mo, protocol):
    mo.vstack(
        [
            mo.md(
                """
                ## Implemented protocol

                The fixed command `bash .orx/run.sh` pins the authors' code,
                installs the evaluator, downloads the selected checkpoint, runs
                ALFWorld, and emits the complete result JSON to stdout.
                """
            ),
            mo.ui.table(protocol, selection=None),
        ]
    )
    return


@app.cell
def _():
    experiment_nodes = [
        {
            "condition": "Baseline",
            "checkpoint": "Qwen/Qwen2.5-3B-Instruct",
            "experiment_id": "cb57b5fb-b1ef-48d0-b0cc-1f8985d89c79",
            "runs": 0,
        },
        {
            "condition": "Released SEED checkpoint",
            "checkpoint": "Jinyang23/Seed-AlfWorld-3B",
            "experiment_id": "872057f1-9e10-476c-a665-99d50c2cd54c",
            "runs": 0,
        },
    ]
    return (experiment_nodes,)


@app.cell
def _(experiment_nodes, mo):
    mo.vstack(
        [
            mo.md("## Prepared experiment tree"),
            mo.ui.table(experiment_nodes, selection=None),
            mo.md(
                """
                The child changes only `MODEL_ID` and `MODEL_LABEL`. The run
                command, evaluator, split, decoding settings, and evidence
                format remain fixed.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and next test

    The project established an evaluator-ready comparison, not an empirical
    confirmation. A future continuation should first run both prepared
    checkpoint evaluations on the same backend. If evaluator fidelity is
    established, the decisive causal test is a matched 160-update GRPO and
    SEED training comparison from the same hindsight-skill SFT checkpoint.

    **Sources**

    - [Paper](https://arxiv.org/abs/2607.14777)
    - [Authors' code](https://github.com/jinyangwu/SEED)
    - [Released checkpoint](https://huggingface.co/Jinyang23/Seed-AlfWorld-3B)
    - [Detailed report](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/reports/seed_reproduction/report.md)
    """)
    return


if __name__ == "__main__":
    app.run()
