# scripts/smoke_test_gendgrl.py

"""
Smoke test for the Gen-DGRL (Procgen) offline dataset loader.

What this script does:
1) Downloads + loads a small batch from a Gen-DGRL dataset (default: coinrun-level_1_E)
2) Prints the TensorDict structure (keys) and tensor shapes/dtypes
3) Prints a small "dataframe-like" preview for non-image fields (action/reward/done)
   plus simple image statistics (mean/std/min/max) to sanity-check pixel ranges
4) Visualizes a few observations (frames) so you can confirm the dataset looks correct

Why this matters:
- Confirms your dataset is available and TorchRL can read it
- Confirms observation format matches what your policy expects:
    CoinRun observations are typically uint8 RGB images with shape (B, 64, 64, 3)
    and actions are discrete integers in [0..14] (15 actions).
- Confirms next-state fields exist for offline RL (IQL/CQL) using TED format:
    observation/action at time t live at the root,
    next observation/reward/done live under ("next", ...)

How to run:
    python scripts/smoke_test_gendgrl.py

Optional environment variables:
    DATASET_ID : which Gen-DGRL dataset to load
        e.g. DATASET_ID=coinrun-1M_E
    BATCH_SIZE : batch size to sample (default 64)
    N_SHOW     : number of frames to visualize (default 8)

Example:
    DATASET_ID=coinrun-level_40_E BATCH_SIZE=128 N_SHOW=12 \\
      python scripts/smoke_test_gendgrl.py
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torchrl.data.datasets import GenDGRLExperienceReplay

def main():
    dataset_id = os.environ.get("DATASET_ID", "coinrun-level_1_E")
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    n_show = int(os.environ.get("N_SHOW", "8"))  # how many frames to visualize

    print(f"Loading Gen-DGRL dataset: {dataset_id} (batch_size={batch_size})")

    ds = GenDGRLExperienceReplay(dataset_id, batch_size=batch_size, download="force")
    batch = next(iter(ds))

    # --- Dataset characteristics (quick sanity + summary) ---
    print("\n=== Dataset characteristics ===")
    print("dataset_id:", dataset_id)

    # Where TorchRL cached it (best-effort)
    try:
        root = getattr(ds, "root", None) or getattr(ds, "_root", None)
        print("cache_root:", root)
    except Exception:
        pass

    # Observation format
    obs = batch["observation"]
    act = batch["action"]
    rew = batch["next", "reward"]
    done = batch["next", "done"]

    print("batch_size:", obs.shape[0])

    # Detect CHW vs HWC for single observation
    obs0 = obs[0]
    if obs0.ndim == 3 and obs0.shape[0] in (1, 3):
        obs_format = "CHW"
        H, W, C = int(obs0.shape[1]), int(obs0.shape[2]), int(obs0.shape[0])
    elif obs0.ndim == 3 and obs0.shape[-1] in (1, 3):
        obs_format = "HWC"
        H, W, C = int(obs0.shape[0]), int(obs0.shape[1]), int(obs0.shape[2])
    else:
        obs_format = f"unknown({tuple(obs0.shape)})"
        H = W = C = None

    print("observation_format:", obs_format)
    print("observation_shape (one):", tuple(obs0.shape), "dtype:", obs.dtype)

    # Pixel range check (uint8 should be 0..255)
    obs_min_all = int(obs.amin().item())
    obs_max_all = int(obs.amax().item())
    print("pixel_min/max:", obs_min_all, "/", obs_max_all)

    # Action characteristics
    act_flat = act.reshape(act.shape[0], -1)[:, 0]
    act_min = int(act_flat.min().item())
    act_max = int(act_flat.max().item())
    print("action_dtype:", act.dtype, "action_min/max (batch):", act_min, "/", act_max)
    print("expected_action_space:", "Discrete(15) (0..14)")

    # Reward characteristics
    rew_flat = rew.reshape(rew.shape[0], -1)[:, 0].to(torch.float32)
    print(
        "reward mean/std/min/max:",
        float(rew_flat.mean().item()),
        float(rew_flat.std().item()),
        float(rew_flat.min().item()),
        float(rew_flat.max().item()),
    )

    # Done rate
    done_flat = done.reshape(done.shape[0], -1)[:, 0].to(torch.float32)
    print("done_rate (batch):", float(done_flat.mean().item()))

    # Action histogram (batch)
    hist = torch.bincount(act_flat.to(torch.int64), minlength=15).cpu().numpy()
    print("action_hist (batch):", hist.tolist())

    # --- Filter out invalid / uninitialized rows (NaNs, all-zero obs) ---
    obs = batch["observation"]
    act = batch["action"]
    rew = batch["next", "reward"]

    act_flat = act.reshape(act.shape[0], -1)[:, 0].to(torch.float32)
    rew_flat = rew.reshape(rew.shape[0], -1)[:, 0].to(torch.float32)

    # valid if action/reward are finite AND observation isn't all zeros
    valid = torch.isfinite(act_flat) & torch.isfinite(rew_flat)
    valid &= (obs.view(obs.shape[0], -1).sum(dim=1) > 0)

    if valid.sum() == 0:
        raise RuntimeError(
            "No valid samples in this batch. Cache is likely corrupted.\n"
            "Try: download='force' OR delete ~/.cache/torchrl/gen_dgrl/<dataset_id> and rerun."
        )

    batch = batch[valid]

    # --- Print structure ---
    print("\n=== TensorDict keys (nested) ===")
    print(batch.keys(True))

    obs = batch["observation"]
    act = batch["action"]
    next_obs = batch["next", "observation"]
    next_reward = batch["next", "reward"]
    next_done = batch["next", "done"]

    print("\n=== Shapes / dtypes ===")
    print("observation      :", obs.shape, obs.dtype)        # expect (B,64,64,3) uint8
    print("action           :", act.shape, act.dtype)        # expect (B,) or (B,1) int
    print("next_observation :", next_obs.shape, next_obs.dtype)
    print("next_reward      :", next_reward.shape, next_reward.dtype)
    print("next_done        :", next_done.shape, next_done.dtype)

    # --- "Dataframe-like" preview (non-image columns + image stats) ---
    B = min(10, obs.shape[0])

    actions = act[:B].cpu().numpy().reshape(-1)
    rewards = next_reward[:B].cpu().numpy().reshape(-1)
    dones = next_done[:B].cpu().numpy().reshape(-1)

    obs_f = obs[:B].float()  # uint8 -> float for stats
    obs_mean = obs_f.mean(dim=(1, 2, 3)).cpu().numpy()
    obs_std = obs_f.std(dim=(1, 2, 3)).cpu().numpy()
    obs_min = obs_f.amin(dim=(1, 2, 3)).cpu().numpy()
    obs_max = obs_f.amax(dim=(1, 2, 3)).cpu().numpy()

    df = pd.DataFrame({
        "action": actions,
        "reward": rewards,
        "done": dones,
        "obs_mean": np.round(obs_mean, 2),
        "obs_std": np.round(obs_std, 2),
        "obs_min": obs_min.astype(int),
        "obs_max": obs_max.astype(int),
    })

    print("\n=== Preview table (first 10) ===")
    print(df.to_string(index=False))

    # --- Visualize a few frames ---
    n = min(n_show, obs.shape[0])
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(3.2 * cols, 3.2 * rows))
    for i in range(n):
        img = obs[i].cpu().numpy()

        # Handle both possible formats:
        # HWC: (64,64,3) OR CHW: (3,64,64)
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))

        a = int(act[i].cpu().numpy().reshape(-1)[0])
        r = float(next_reward[i].cpu().numpy().reshape(-1)[0])
        d = int(next_done[i].cpu().numpy().reshape(-1)[0])

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"i={i}  a={a}  r={r:.2f}  done={d}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()