# SEED reproduction

This repository contains a pinned, log-complete reproduction harness for
**SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement
Learning** (arXiv:2607.14777).

The first experiment round evaluates the authors' released ALFWorld checkpoint
against its Qwen2.5-3B-Instruct base model. Both conditions use the same
official evaluator, ALFWorld split, decoding parameters, task count, and three
evaluation rounds. The only experimental variable is the model checkpoint in
`.orx/reproduction.conf`.

The fixed OpenResearch command is:

```bash
bash .orx/run.sh
```

The harness pins the upstream SEED implementation commit, installs the runtime,
downloads ALFWorld assets and the configured Hugging Face checkpoint, runs the
official local-vLLM evaluator, and prints a final machine-readable evidence
block to stdout. This is required because local OpenResearch projects retain run
logs as their evidence channel.

This first round validates the released checkpoint's reported absolute
ALFWorld performance and quantifies its improvement over the untrained base
model. It does not by itself reproduce the paper's 160-update GRPO-versus-SEED
training comparison; that is a later, substantially more expensive stage.

## Publication artifacts

The reproduction closed without a recorded training or evaluation run, so the
public verdict is **not reproduced**. The package documents the implemented
protocol, the exact evidence boundary, and the work required for a future
empirical test.

- [Detailed reproduction report](reports/seed_reproduction/report.md)
- [Self-contained marimo notebook](notebooks/seed_reproduction.py)
- [Open the notebook on Molab](https://molab.marimo.io/github/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/notebooks/seed_reproduction.py)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/notebooks/seed_reproduction.py)
