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
            feat_dim = self.net(dummy).shape[-1]
        self.head = nn.Linear(feat_dim, n_actions)

    def forward(self, x):
        return self.head(self.net(x))

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

    model = SmallCNN(n_actions=15).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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