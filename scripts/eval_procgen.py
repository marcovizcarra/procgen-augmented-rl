#!/usr/bin/env python3
"""
Evaluate a trained policy checkpoint in Procgen CoinRun.

Runs rollouts on:
- Train levels: start_level=0,   num_levels=500
- Test  levels: start_level=500, num_levels=500

Usage:
  python scripts/eval_procgen.py --ckpt runs/bc_min/bc_ckpt.pt --episodes 50
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

def to_chw_float01(obs: np.ndarray) -> torch.Tensor:
    """
    obs from procgen is usually HWC uint8 (64,64,3).
    Convert to BCHW float32 in [0,1].
    """
    x = torch.from_numpy(obs)
    if x.ndim == 3 and x.shape[-1] == 3:     # HWC
        x = x.permute(2, 0, 1)               # CHW
    elif x.ndim == 3 and x.shape[0] == 3:    # already CHW
        pass
    else:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")
    return x.unsqueeze(0).float() / 255.0    # BCHW


class SmallCNN(nn.Module):
    """
    Original eval architecture:
    - net = conv/relu stack + flatten
    - head maps flattened features -> actions
    State dict keys look like: net.0.weight, net.2.weight, ..., head.weight
    """
    def __init__(self, n_actions=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            feat_dim = self.net(dummy).shape[-1]  # typically 1024
        self.head = nn.Linear(feat_dim, n_actions)

    def forward(self, x):
        return self.head(self.net(x))


class CNNPolicy(nn.Module):
    """
    Matches scripts/train_bc_min.py checkpoints:
    conv1/conv2/conv3 + fc + head
    State dict keys look like: conv1.weight, conv2.weight, conv3.weight, fc.weight, head.weight
    """
    def __init__(self, n_actions=15, hidden=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)   # -> 15x15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # -> 6x6
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # -> 4x4 (64*4*4=1024)
        self.fc = nn.Linear(64 * 4 * 4, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        # accept uint8 or float
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc(x))
        return self.head(x)


def _extract_state_dict(ckpt: dict) -> dict:
    """
    Accept common checkpoint formats:
      ckpt["model_state_dict"], ckpt["state_dict"], ckpt["model"], or raw state_dict.
    """
    for k in ("model_state_dict", "state_dict", "model"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    # If the checkpoint itself looks like a raw state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError("Could not find a model state_dict in checkpoint.")


def _infer_n_actions(ckpt: dict, sd: dict) -> int:
    if "head.weight" in sd:
        return int(sd["head.weight"].shape[0])
    if "n_actions" in ckpt:
        return int(ckpt["n_actions"])
    return 15


def build_model_for_checkpoint(ckpt: dict, device: torch.device) -> nn.Module:
    sd = _extract_state_dict(ckpt)
    n_actions = _infer_n_actions(ckpt, sd)

    # Determine which architecture to instantiate based on key patterns.
    if any(k.startswith("net.") for k in sd.keys()):
        model = SmallCNN(n_actions=n_actions).to(device)
    elif "conv1.weight" in sd:
        # hidden dimension can be inferred from fc.weight rows if present
        hidden = int(sd["fc.weight"].shape[0]) if "fc.weight" in sd else 512
        model = CNNPolicy(n_actions=n_actions, hidden=hidden).to(device)
    else:
        first_keys = list(sd.keys())[:12]
        raise RuntimeError(f"Unknown checkpoint format; cannot infer model. First keys: {first_keys}")

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    # gymnasium returns 5 items; gym returns 4
    if len(out) == 5:
        obs, r, term, trunc, info = out
        done = bool(term or trunc)
    else:
        obs, r, done, info = out
        done = bool(done)
    return obs, float(r), done, info


def run_eval(env, model, device, episodes):
    returns = []
    obs = reset_env(env)
    ep_ret = 0.0

    while len(returns) < episodes:
        x = to_chw_float01(obs).to(device)
        with torch.no_grad():
            logits = model(x)
            action = int(torch.argmax(logits, dim=-1).item())

        obs, r, done, _ = step_env(env, action)
        ep_ret += r

        if done:
            returns.append(ep_ret)
            ep_ret = 0.0
            obs = reset_env(env)

    return float(np.mean(returns)), float(np.std(returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train_start", type=int, default=0)
    p.add_argument("--train_levels", type=int, default=500)
    p.add_argument("--test_start", type=int, default=500)
    p.add_argument("--test_levels", type=int, default=500)
    p.add_argument("--distribution_mode", default="hard", choices=["easy", "hard"])
    args = p.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model_for_checkpoint(ckpt, device=device)

    env_train = gym.make(
        "procgen:procgen-coinrun-v0",
        start_level=args.train_start,
        num_levels=args.train_levels,
        distribution_mode=args.distribution_mode,
    )
    env_test = gym.make(
        "procgen:procgen-coinrun-v0",
        start_level=args.test_start,
        num_levels=args.test_levels,
        distribution_mode=args.distribution_mode,
    )

    mean_tr, std_tr = run_eval(env_train, model, device, args.episodes)
    mean_te, std_te = run_eval(env_test, model, device, args.episodes)

    print(f"TRAIN levels: mean_return={mean_tr:.3f} std={std_tr:.3f} (episodes={args.episodes})")
    print(f"TEST  levels: mean_return={mean_te:.3f} std={std_te:.3f} (episodes={args.episodes})")

    env_train.close()
    env_test.close()


if __name__ == "__main__":
    main()