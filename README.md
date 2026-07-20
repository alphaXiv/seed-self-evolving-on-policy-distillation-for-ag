# SEED: bounded ALFWorld reproduction

This public repository reproduces the central ALFWorld claims from **SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning** ([arXiv:2607.14777](https://arxiv.org/abs/2607.14777)).

**Assessment: partially reproduced.** Across two matched seeds, confidence-gated skill-induced on-policy distillation raised pooled fixed seen success from **15.3% to 18.1% (+2.8 points)**, but pooled unseen success tied at **19.4%**, and the early-training curve favored GRPO (**15.0% versus 11.9%**). The paper reports +16.8 points seen, +15.3 points unseen, and better early efficiency. Seed 1 was positive on both endpoints; seed 2 reversed both effects.

![Fixed seen and unseen ALFWorld results](reports/seed_reproduction/images/final_split_success.png)

The two causal pairs used Qwen3-1.7B, 40 updates, group size 4, 16 bounded training tasks, and 36 fixed tasks per split per seed. The paper used Qwen2.5-3B, 150 updates, and group size 8. Because the exact GLM-5.2 analyzer prompts/annotations were unavailable, this implementation documents a public Qwen3 analyzer substitution plus deterministic task-family fallback.

- [Detailed technical report](reports/seed_reproduction/report.md)
- [Tutorial-style marimo notebook](notebooks/seed_reproduction.py)
- [Open the notebook directly on Molab](https://molab.marimo.io/github/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/notebooks/seed_reproduction.py)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/blob/main/notebooks/seed_reproduction.py)

## Experiment log

Every formal node uses the exact inherited command shown below. `main` is the publication surface and was not launched as an experiment.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Polished report, figures, notebook, and causal harness | Not run as an experiment (publication surface) | Publication artifact | — |
| [Base checkpoint evaluation](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/base-checkpoint-kubernetes-evaluation) | Official evaluator on Qwen2.5-3B-Instruct | `bash .orx/run.sh` | 15.8% category-macro success | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 8m46s |
| [Released SEED checkpoint](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/released-seed-checkpoint) | Evaluator-fidelity check on the authors' checkpoint | `bash .orx/run.sh` | 89.3% category-macro success; paper 91.8% | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 6m40s |
| [Matched outcome-only GRPO](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/outcome-only-grpo-with-validated-sft-shard) | Shared SFT and bounded tasks; online analysis and OPD disabled | `bash .orx/run.sh` | 13.9% seen; 19.4% unseen | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 1.946944 h |
| [Primary SEED](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/seed-opd-with-deterministic-online-analyzer-fall) | Add public analyzer fallback and confidence-gated OPD at λ=0.01 | `bash .orx/run.sh` | 22.2% seen; 25.0% unseen; endpoints positive, curve area negative | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 2.145278 h |
| [GRPO seed 260714778](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/outcome-only-grpo-seed-260714778) | Change only trainer/environment seed | `bash .orx/run.sh` | 16.7% seen; 19.4% unseen; 15.5% curve mean | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 1.969167 h |
| [SEED seed 260714778](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/seed-deterministic-analyzer-seed-260714778) | Change only trainer/environment seed | `bash .orx/run.sh` | 13.9% seen; 13.9% unseen; endpoint direction reversed | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 2.178611 h |
| [20-update outcome-only GRPO](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/outcome-only-grpo-20-update-endpoint) | Match the shortened λ-sensitivity budget | `bash .orx/run.sh` | 16.7% seen; 16.7% unseen; 7.5% curve mean | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 1.111944 h |
| [SEED λ=0.001 sensitivity](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/seed-opd-lambda-0-001-sensitivity) | Test whether weaker OPD better suits lower-fidelity fallback skills | `bash .orx/run.sh` | 22.2% seen; 19.4% unseen; positive versus matched 20-update control | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 1.213056 h |
| [Raw-hindsight SEED, corrected](https://github.com/alphaXiv/seed-self-evolving-on-policy-distillation-for-ag/tree/orx/seed-raw-hindsight-with-regex-import) | Preserve Qwen3's unstructured hindsight instead of task-family replacement | `bash .orx/run.sh` | 16.7% seen; 16.7% unseen; weaker than structured fallback | Kubernetes, 8 × NVIDIA RTX PRO 6000 Blackwell, 2.171667 h |

The raw-hindsight parent terminated before update 1 because its patch omitted `import re`; the linked corrected child changes only that import. Completed sensitivities show that λ=0.001 improved over its matched 20-update control, while unstructured raw hindsight was weaker than the primary concise fallback. Raw identifiers and full diagnostic histories are retained in OpenResearch experiment descriptions rather than this landing page.

Peak requested concurrency was **16 NVIDIA RTX PRO 6000 Blackwell GPUs on Kubernetes**. The queue runner measured **7.480803 hours of observed Kubernetes campaign wall time**. Complete fixed-split JSON and per-run elapsed times are present in terminal logs; plotted transcriptions live in [results.json](reports/seed_reproduction/data/results.json), and [plot_results.py](reports/seed_reproduction/plot_results.py) regenerates both figures.

## Reproduction harness

The fixed command is:

```bash
bash .orx/run.sh
```

The harness pins the authors' SEED repository at commit `2cf2fadca3c5aba28da68e8e1405182ba8d90e6c`, installs ALFWorld and the released training stack, creates the public skill-SFT set, runs four shared SFT steps, launches the selected 8-GPU RL arm, merges the final FSDP checkpoint, and evaluates fixed seen and unseen task manifests. Configuration is committed in `.orx/training.conf`; experiment branches vary code/configuration while keeping the command constant.

To inspect the notebook locally without training:

```bash
marimo edit notebooks/seed_reproduction.py
marimo run notebooks/seed_reproduction.py
```

The notebook embeds the measured evidence, so opening it does not trigger expensive computation.
