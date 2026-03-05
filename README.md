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
