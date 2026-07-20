# SEED on ALFWorld: bounded causal reproduction

**Assessment: partially reproduced.** This project tests the central ALFWorld claim from [SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.14777): whether confidence-gated, skill-induced on-policy distillation improves a matched outcome-only GRPO agent. Averaged over two training seeds and fixed 36-task evaluations, bounded SEED reached **15.0% vs 13.0% category-macro success on the seen subset (+2.1 points)** and **16.7% vs 16.4% on the held-out subset (+0.3 points)**. Those means have the paper's +16.8-point main and +15.3-point held-out directions, but the effects are much smaller and reverse on the second seed. The early-training claim did not appear: normalized macro-AUC was **11.08% vs 13.75%** over 40 updates.

This is a deliberately smaller causal test: Qwen3-1.7B instead of Qwen2.5-3B, 40 instead of 150 updates, 16 public training tasks, two training seeds, and a documented deterministic task-family analyzer fallback because the authors' GLM-5.2 prompts/annotations are unavailable. Both conditions use the same 48-example skill SFT initialization, rollout tasks, group size, optimizer settings, and evaluations. All formal runs used Kubernetes on **NVIDIA RTX PRO 6000 Blackwell** GPUs; peak concurrency was **16 GPUs** and the complete experiment campaign elapsed **7.480803 wall hours** from first submission to final terminal run.

- [Detailed claim-by-claim report](reports/seed_reproduction/report.md)
- [Self-contained tutorial notebook](notebooks/seed_reproduction.py)
- [Measured learning-curve](reports/seed_reproduction/data/learning_curves.csv) and [fixed-split CSV data](reports/seed_reproduction/data/final_splits.csv)
- [Open the notebook directly in Molab](https://molab.marimo.io/github/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/notebooks/seed_reproduction.py)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/notebooks/seed_reproduction.py)

## Experiment log

Every command below is copied verbatim from `orx exp status`. Branch links expose the exact committed code used by Kubernetes.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, figures, and metadata | Not run as an experiment (publication surface) | Presentation only | No experiment |
| [Base checkpoint evaluation](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/base-checkpoint-kubernetes-evaluation) | Official evaluator sanity check for Qwen2.5-3B base | `bash .orx/run.sh` | 15.8% macro success | Kubernetes, 8 GPUs, 8m46s |
| [Released SEED checkpoint](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/released-seed-checkpoint) | Official evaluator sanity check for authors' checkpoint | `bash .orx/run.sh` | 89.3% macro success; evaluator/checkpoint path validated | Kubernetes, 8 GPUs, 6m40s |
| [Outcome-only GRPO, seed 260714777](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/outcome-only-grpo-with-validated-sft-shard) | Matched 40-update control from common SFT initialization | `bash .orx/run.sh` | 12.0% seen / 17.0% held-out macro success | Kubernetes, 8 GPUs, 1h57m02s |
| [SEED, seed 260714777](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/seed-opd-with-deterministic-online-analyzer-fall) | Adds confidence-gated skill-induced OPD (`lambda=0.01`) | `bash .orx/run.sh` | +6.0 seen / +4.8 held-out macro vs matched control | Kubernetes, 8 GPUs, 2h08m47s |
| [Outcome-only GRPO, seed 260714778](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/outcome-only-grpo-seed-260714778) | Prespecified second-seed control | `bash .orx/run.sh` | 13.9% seen / 15.7% held-out macro success | Kubernetes, 8 GPUs, 1h58m09s active (3h04m33s incl. queue) |
| [SEED, seed 260714778](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/seed-deterministic-analyzer-seed-260714778) | Prespecified second-seed treatment | `bash .orx/run.sh` | −1.9 seen / −4.2 held-out macro vs matched control | Kubernetes, 8 GPUs, 2h10m43s active |
| [GRPO 20-update endpoint](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/outcome-only-grpo-20-update-endpoint) | Update-matched control for coefficient sensitivity | `bash .orx/run.sh` | 13.9% seen / 13.7% held-out macro | Kubernetes, 8 GPUs, 1h10m57s |
| [SEED λ=0.001 sensitivity](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/seed-opd-lambda-0-001-sensitivity) | Tenfold smaller OPD coefficient, 20 updates | `bash .orx/run.sh` | +5.8 seen / +4.2 held-out macro vs matched control | Kubernetes, 8 GPUs, 1h22m36s |
| [SEED raw-hindsight analyzer](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/seed-raw-hindsight-with-regex-import) | Preserve public Qwen3 non-JSON analysis as the online skill | `bash .orx/run.sh` | +1.9 seen / −3.6 held-out macro vs primary control | Kubernetes, 8 GPUs, 2h10m25s |

## Reproduction harness

The public `main` branch retains the causal harness under `.orx/`: it pins the authors' code, generates the documented skill-SFT set, runs the selected 8-GPU arm, merges the final FSDP checkpoint, and evaluates the fixed seen and held-out manifests. Experiment branches hold the exact configuration used for each row above; every formal launch used `bash .orx/run.sh` with `--backend k8s` supplied to OpenResearch at submission.

## Reproducing the figures

The plots are regenerated only from committed, run-ID-bearing CSV files:

```bash
python reports/seed_reproduction/plot_results.py
```

The expensive training is not required to inspect the evidence. The notebook embeds the measured arrays so it opens correctly in Molab without repository-relative data files.
