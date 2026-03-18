# procgen-augmented-rl

Reinforcement learning on **Procgen** (starting with **CoinRun**) focused on improving **generalization** with **data augmentation**. Supports training from **offline datasets** (e.g., **Gen-DGRL**) and (optionally) online fine-tuning, with evaluation on **unseen levels**.

## Why this repo?
Procgen environments are built to test **generalization**: you train on one set of procedural levels and evaluate on a disjoint set. This project studies whether **augmenting observations** (random shifts, color jitter, cutout, etc.) improves test-level performance.

---

## Project goals
- Train policies from an **offline dataset** (no environment interaction during training)
- Apply **data augmentation** to offline observations to improve generalization
- Evaluate learned policies by **rolling out in Procgen CoinRun** on **held-out levels**

---

## Setup

### Recommended environment
- Python **3.10** (Procgen wheels support 3.7–3.10)

### Install dependencies
```bash
pip install -U pip
pip install torch torchvision torchrl tensordict requests tqdm
```

## IQL step-sweep chart (6k / 10k / 20k)

Run IQL on the **20k L40 dataset** for multiple training budgets (baseline + top-K augmentations):

```bash
python scripts/run_iql_stepsweep_topk.py \
  --datasets-root data/stage1_datasets_L40 \
  --steps 6000 10000 20000 \
  --seeds 0 1 2 \
  --top-k 5
```

Plot the chart:

```bash
python scripts/plot_iql_stepsweep_topk.py \
  --glob 'runs/iql_stepsweep_L40_steps*_seed*/stage1_iql_train_eval_summary/results.json' \
  --steps 6000 10000 20000 \
  --out-dir results/plots/iql_stepsweep_L40
```
