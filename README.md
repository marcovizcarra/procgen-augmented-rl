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
pip install torch torchvision torchrl tensordict requests tqdm wandb
```

### NOTE
For Apple Silicon Macs, use the provided setup script (it creates an `osx-64`/Rosetta env so `procgen` can install):

- In `environment.yml`, comment out `numpy` and uncomment `- setuptools<58 - numpy<2`
- In `environment.yml`, comment out `gym==0.21.0` 

```bash
bash setup_env.sh
conda activate procgen-augmented-rl
```

## Troubleshooting (Apple Silicon)

- `ERROR: No matching distribution found for procgen==0.10.7`  
  Use `bash setup_env.sh` and verify `python -c "import platform; print(platform.machine())"` prints `x86_64`.
- `RuntimeError: Could not infer dtype of numpy.uint8` during dataset loading  
  Ensure NumPy is `<2`: `python -c "import numpy; print(numpy.__version__)"` (expected `1.26.x`).
- `gym==0.21.0` build failures  
  Re-run `bash setup_env.sh`; it pins compatible legacy packaging tools before installing Gym.

## W&B logging (BC + eval)

W&B logging is optional and only activates when `--wandb` is passed.

Example: BC training with W&B
```bash
python -B scripts/train_bc_min.py \
  --dataset-root data/stage1_datasets/baseline_none \
  --run-name stage1_datasets/bc_baseline_none \
  --steps 20000 \
  --batch-size 256 \
  --wandb \
  --wandb-project procgen-augmented-rl \
  --wandb-mode online \
  --wandb-tags bc,stage1
```

Example: evaluation with W&B
```bash
python -B scripts/eval_procgen.py \
  --ckpt runs/stage1_datasets/bc_baseline_none/bc_ckpt.pt \
  --episodes 200 \
  --wandb \
  --wandb-project procgen-augmented-rl \
  --wandb-mode online \
  --wandb-tags eval,stage1
```

Example: orchestrated Stage-1 train+eval with forwarded W&B args
```bash
python -B scripts/train_eval_stage1.py \
  --datasets-root data/stage1_datasets \
  --train-steps 20000 \
  --episodes 200 \
  --wandb \
  --wandb-project procgen-augmented-rl \
  --wandb-mode online \
  --wandb-tags stage1,ablation
```

Metric mapping for BC runs:
- `policy_loss` -> logged as `train/policy_loss` (BC cross-entropy)
- `entropy` -> logged as `train/entropy` (categorical entropy from policy logits)
- `reward_mean` -> logged as `eval/train_reward_mean` and `eval/test_reward_mean`
- `episode_mean_length` -> logged as `eval/train_episode_mean_length` and `eval/test_episode_mean_length`
- `value_loss` -> not defined for BC (logged as `NaN` with `value_loss_defined=false` in run config)
