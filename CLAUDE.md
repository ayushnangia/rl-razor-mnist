# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Replication of "RL's Razor: Why Online Reinforcement Learning Forgets Less" (arXiv:2509.04259). Demonstrates that KL divergence from a base model predicts catastrophic forgetting in fine-tuning. Uses a 3-layer MLP on MNIST/FashionMNIST as a controlled testbed.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Pretrain (joint ParityMNIST + FashionMNIST)
python scripts/pretrain.py --epochs 50 --lr 1e-3 --n-samples 500 --scheduler cosine_with_warmup

# Fine-tune (methods: sft1, sft2, oracle, grpo, grpo_kl)
python scripts/finetune.py --pretrained-model path/to/model.pt --method grpo --lr 1e-4 --epochs 2

# W&B sweep
wandb sweep configs/sweep.yaml
wandb agent <sweep_id>

# Plot results (generates Figure 3 + Table 1)
python scripts/plot.py --sweep-dir path/to/sweep/results

# Lint and format
black --line-length 100 src/ scripts/
ruff check src/ scripts/

# Tests
pytest
```

## Full replication pipeline

Shell scripts in `runs/paper_replication/` run the complete pipeline: `pretrain.sh` -> `finetune_sweep.sh` -> `plot.sh` -> `analyze_drift_trajectory.sh`.

## Architecture

**Model** (`src/rl_razor/model.py`): MLP with 785->512->256->10. Input is flattened image (784) + task indicator (+1 for ParityMNIST, -1 for FashionMNIST).

**Datasets** (`src/rl_razor/data.py`): ParityMNIST is the new task (predict even/odd -- multiple correct labels exist, which is why different methods find different solutions). FashionMNIST is the old task (measures forgetting). Both use 500 training samples.

**Training methods** (`src/rl_razor/training/`):
- `sft.py`: Three SFT variants. SFT-1 uses deterministic labels (even->0, odd->1). SFT-2 uses limited diversity. Oracle uses KL-minimal soft targets (q* proportional to base policy over correct-parity labels).
- `grpo.py`: Group Relative Policy Optimization. Samples `group_size` actions per input, computes group-relative advantages, optional KL penalty (grpo_kl mode).
- `oracle.py`: Computes the KL-minimal label distribution used by oracle SFT.

**Metrics** (`src/rl_razor/metrics.py`): Computes ~10 divergence measures (forward/reverse KL, TV, L2), weight-level distances (L1, spectral norm, Fisher L2), and activation distances. All data-dependent metrics are computed on both datasets (suffixed `_new`/`_old`).

## Key design decisions

- The oracle SFT uses soft cross-entropy targets, not hard labels.
- GRPO loss: `-E[advantage * log_prob] + beta*KL + gamma*entropy`.
- Metrics are computed on checkpoint snapshots saved during fine-tuning, not just the final model.
- `configs/sweep.yaml` defines a grid of 5 methods x 15 LRs x 2 schedulers x 2 epoch counts.

## Code style

Black with 100-char line length. Ruff with E, F, W, I, UP rules.
