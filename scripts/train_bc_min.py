#!/usr/bin/env python3
"""
scripts/train_bc_min.py

Minimal Behavior Cloning (BC) trainer for Procgen CoinRun.

Supports TWO data sources:
1) Our saved Stage-1 datasets (NPZ shards + manifest.json):
     python scripts/train_bc_min.py --dataset-root data/stage1_datasets/baseline_none --run-name bc_baseline --steps 20000
   (or --manifest path/to/manifest.json)

2) Direct Gen-DGRL streaming via TorchRL (fallback):
     python scripts/train_bc_min.py --gendgrl-id coinrun-level_1_E --run-name bc_gendgrl --steps 20000

Saves:
  runs/<run-name>/bc_ckpt.pt
  runs/<run-name>/config.json

Checkpoint format includes multiple common keys for compatibility:
  {"model": state_dict, "state_dict": state_dict, "model_state_dict": state_dict, ...}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchrl.data.datasets import GenDGRLExperienceReplay
except Exception:
    GenDGRLExperienceReplay = None


# -----------------------------
# Model: small CNN policy
# -----------------------------
class CNNPolicy(nn.Module):
    def __init__(self, n_actions: int = 15):
        super().__init__()
        # Input: (B,3,64,64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)   # -> (B,32,15,15)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # -> (B,64,6,6)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # -> (B,64,4,4)
        self.fc = nn.Linear(64 * 4 * 4, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, obs_u8: torch.Tensor) -> torch.Tensor:
        # obs_u8: uint8 or float, (B,3,64,64)
        if obs_u8.dtype == torch.uint8:
            x = obs_u8.float() / 255.0
        else:
            x = obs_u8
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc(x))
        logits = self.head(x)
        return logits


# -----------------------------
# Data: NPZ shards
# -----------------------------
def load_manifest(manifest_path: Path) -> Dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_npz_batches(manifest_path: Path, batch_size: int, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields (obs_u8_bchw, action_int64) batches indefinitely by looping over shards.
    """
    rng = np.random.default_rng(seed)
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    shards = manifest["shards"]

    while True:
        # Shuffle shard order each epoch for a little variety
        shard_order = list(shards)
        rng.shuffle(shard_order)

        for shard_name in shard_order:
            shard_path = root / shard_name
            with np.load(shard_path) as d:
                obs = d["obs"]        # (N,3,64,64) uint8
                act = d["action"]     # (N,) int64
            n = obs.shape[0]
            idx = np.arange(n)
            rng.shuffle(idx)

            for start in range(0, n, batch_size):
                j = idx[start:start + batch_size]
                if j.size == 0:
                    continue
                yield obs[j], act[j]


# -----------------------------
# Data: Gen-DGRL streaming (fallback)
# -----------------------------
def iter_gendgrl_batches(dataset_id: str, batch_size: int, seed: int = 0, download: bool = True):
    if GenDGRLExperienceReplay is None:
        raise RuntimeError("TorchRL GenDGRLExperienceReplay not available. Install torchrl or use --dataset-root/--manifest.")
    torch.manual_seed(seed)
    ds = GenDGRLExperienceReplay(dataset_id, batch_size=batch_size, download=download)
    for batch in ds:
        obs = batch["observation"]
        act = batch["action"]
        # Normalize shapes/layout to (B,3,64,64) uint8, (B,) int64
        if obs.ndim == 4 and obs.shape[-1] == 3:  # BHWC
            obs = obs.permute(0, 3, 1, 2)
        obs = obs.cpu().numpy().astype(np.uint8)

        act = act.reshape(-1).cpu().numpy()
        # Sometimes actions come as float; cast safely.
        act = np.nan_to_num(act, nan=0.0).astype(np.int64)
        yield obs, act


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--dataset-root", type=str, default=None, help="Variant folder containing manifest.json (Stage-1 NPZ dataset).")
    src.add_argument("--manifest", type=str, default=None, help="Path to manifest.json for an NPZ dataset.")
    src.add_argument("--gendgrl-id", type=str, default=None, help="Gen-DGRL dataset id (TorchRL), e.g. coinrun-level_1_E.")

    p.add_argument("--run-name", type=str, default="bc_min")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--n-actions", type=int, default=15)
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve manifest path if using dataset-root
    manifest_path: Optional[Path] = None
    if args.dataset_root is not None:
        manifest_path = Path(args.dataset_root) / "manifest.json"
    elif args.manifest is not None:
        manifest_path = Path(args.manifest)

    # Select data iterator
    if manifest_path is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        data_iter = iter_npz_batches(manifest_path, batch_size=args.batch_size, seed=args.seed)
        data_desc = f"NPZ(manifest={manifest_path})"
    else:
        dataset_id = args.gendgrl_id or "coinrun-level_1_E"
        data_iter = iter_gendgrl_batches(dataset_id, batch_size=args.batch_size, seed=args.seed, download=True)
        data_desc = f"GenDGRL({dataset_id})"

    # Model/opt
    model = CNNPolicy(n_actions=args.n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train loop
    model.train()
    losses = []
    for step in range(1, args.steps + 1):
        obs_u8, act = next(data_iter)
        obs_t = torch.from_numpy(obs_u8).to(device)            # uint8
        act_t = torch.from_numpy(act).to(device).long()

        logits = model(obs_t)
        loss = F.cross_entropy(logits, act_t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        if step % args.log_every == 0:
            avg = sum(losses[-args.log_every:]) / args.log_every
            print(f"step {step:6d}/{args.steps}  loss={avg:.4f}  data={data_desc}  device={device}")

    # Save
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "bc_ckpt.pt"

    state = model.state_dict()
    ckpt = {
        "model": state,
        "state_dict": state,
        "model_state_dict": state,
        "n_actions": args.n_actions,
        "obs_shape": (3, 64, 64),
        "algo": "BC",
        "arch": "CNNPolicy_v1",
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "data_source": data_desc,
    }
    torch.save(ckpt, ckpt_path)

    cfg = {
        "run_name": args.run_name,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "n_actions": args.n_actions,
        "data_source": data_desc,
        "ckpt_path": str(ckpt_path),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
