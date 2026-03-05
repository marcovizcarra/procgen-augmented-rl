#!/usr/bin/env python3
"""
Minimal Behavior Cloning (BC) training script for Gen-DGRL Procgen CoinRun.

What it does:
- Loads an offline batch stream from TorchRL GenDGRLExperienceReplay
- Trains a small CNN policy to predict dataset actions (Discrete(15))
- Saves a checkpoint to runs/bc_min/bc_ckpt.pt

Run:
  python scripts/train_bc_min.py

Optional environment variables:
  DATASET_ID=coinrun-level_1_E
  BATCH_SIZE=256
  STEPS=2000
  LR=3e-4
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data.datasets import GenDGRLExperienceReplay


def to_chw_float01(obs: torch.Tensor) -> torch.Tensor:
    """
    Convert observation tensor to float32 in [0,1] and CHW layout.
    Handles either:
      - (B, 64, 64, 3) HWC
      - (B, 3, 64, 64) CHW
    """
    if obs.ndim != 4:
        raise ValueError(f"Expected obs ndim=4, got {obs.shape}")

    # If last dim is 3 => HWC
    if obs.shape[-1] == 3:
        obs = obs.permute(0, 3, 1, 2)  # BHWC -> BCHW
    # If second dim is 3 => already CHW
    elif obs.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unrecognized obs layout: {obs.shape} (expected HWC or CHW)")

    return obs.float() / 255.0


class SmallCNN(nn.Module):
    def __init__(self, n_actions: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            feat_dim = self.net(dummy).shape[-1]
        self.head = nn.Linear(feat_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_id = os.environ.get("DATASET_ID", "coinrun-level_1_E")
    batch_size = int(os.environ.get("BATCH_SIZE", "256"))
    steps = int(os.environ.get("STEPS", "2000"))
    lr = float(os.environ.get("LR", "3e-4"))

    outdir = "runs/bc_min"
    os.makedirs(outdir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {dataset_id} | batch_size={batch_size} | steps={steps} | lr={lr}")

    ds = GenDGRLExperienceReplay(dataset_id, batch_size=batch_size, download=True)
    it = iter(ds)

    model = SmallCNN(n_actions=15).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    for step in range(1, steps + 1):
        batch = next(it)

        obs = batch["observation"]  # uint8 images
        act = batch["action"]       # (B,) or (B,1)

        obs = to_chw_float01(obs).to(device)
        act = act.to(device).long().view(-1)

        logits = model(obs)
        loss = F.cross_entropy(logits, act)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == act).float().mean().item()
            print(f"step {step:5d} | loss {loss.item():.4f} | acc {acc:.3f}")

    ckpt_path = os.path.join(outdir, "bc_ckpt.pt")
    torch.save(
        {
            "dataset_id": dataset_id,
            "model_state_dict": model.state_dict(),
            "steps": steps,
            "batch_size": batch_size,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path} (elapsed {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()