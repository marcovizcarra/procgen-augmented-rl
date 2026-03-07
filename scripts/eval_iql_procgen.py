#!/usr/bin/env python3
"""
Evaluate IQL actor checkpoints in Procgen CoinRun.

Usage:
  python scripts/eval_iql_procgen.py --ckpt runs/iql_min/iql_ckpt.pt --episodes 50
"""

import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_env(env, seed: int):
    try:
        out = env.reset(seed=seed)
        return out[0] if isinstance(out, tuple) else out
    except TypeError:
        pass

    try:
        env.seed(seed)
    except Exception:
        pass

    try:
        env.action_space.seed(seed)
    except Exception:
        pass

    try:
        env.observation_space.seed(seed)
    except Exception:
        pass

    return None


def to_chw_float01(obs: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs)
    if x.ndim == 3:
        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.shape[0] == 3:
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected 3D obs shape: {tuple(obs.shape)}")
    elif x.ndim == 4:
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        elif x.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Unexpected 4D obs shape: {tuple(obs.shape)}")
    else:
        raise ValueError(f"Unexpected obs rank/shape: ndim={x.ndim}, shape={tuple(obs.shape)}")

    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
    return x


class ConvEncoder(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 4 * 4, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype == torch.uint8:
            x = obs.float() / 255.0
        else:
            x = obs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc(x))
        return x


class DiscreteActor(nn.Module):
    def __init__(self, n_actions: int = 15, hidden: int = 512):
        super().__init__()
        self.enc = ConvEncoder(hidden=hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(obs))


def _extract_actor_state_dict(ckpt: dict) -> dict:
    if "actor_state_dict" in ckpt and isinstance(ckpt["actor_state_dict"], dict):
        return ckpt["actor_state_dict"]
    if "actor" in ckpt and isinstance(ckpt["actor"], dict):
        return ckpt["actor"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
        if any(k.startswith("enc.") or k.startswith("head.") for k in sd.keys()):
            return sd
    raise KeyError("Could not find IQL actor state_dict in checkpoint.")


def build_actor_for_checkpoint(ckpt: dict, device: torch.device) -> nn.Module:
    sd = _extract_actor_state_dict(ckpt)
    n_actions = int(ckpt.get("n_actions", sd["head.weight"].shape[0] if "head.weight" in sd else 15))
    hidden = int(ckpt.get("hidden", sd["head.weight"].shape[1] if "head.weight" in sd else 512))

    model = DiscreteActor(n_actions=n_actions, hidden=hidden).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, r, term, trunc, info = out
        done = bool(term or trunc)
    else:
        obs, r, done, info = out
        done = bool(done)
    return obs, float(r), done, info


def run_eval(env, actor, device, episodes: int, desc: str = "", progress: bool = True, print_every: int = 25):
    returns = []
    obs = reset_env(env)
    ep_ret = 0.0
    total_steps = 0
    t0 = time.time()

    use_tqdm = progress and (tqdm is not None)
    pbar = tqdm(total=episodes, desc=desc or "eval", dynamic_ncols=True) if use_tqdm else None
    last_print = time.time()

    while len(returns) < episodes:
        x = to_chw_float01(obs).to(device)
        with torch.no_grad():
            logits = actor(x)
            action = int(torch.argmax(logits, dim=-1).item())

        obs, r, done, _ = step_env(env, action)
        ep_ret += r
        total_steps += 1

        if done:
            returns.append(ep_ret)
            if pbar is not None:
                elapsed = time.time() - t0
                mean_ret = float(np.mean(returns))
                sps = total_steps / max(1e-9, elapsed)
                pbar.update(1)
                pbar.set_postfix(mean=f"{mean_ret:.2f}", last=f"{ep_ret:.2f}", sps=f"{sps:.0f}")
            else:
                if progress:
                    now = time.time()
                    if (len(returns) % print_every == 0) or (now - last_print > 5.0):
                        elapsed = now - t0
                        mean_ret = float(np.mean(returns))
                        sps = total_steps / max(1e-9, elapsed)
                        print(f"[{desc}] episodes {len(returns)}/{episodes}  mean={mean_ret:.3f}  last={ep_ret:.3f}  sps={sps:.0f}", flush=True)
                        last_print = now

            ep_ret = 0.0
            obs = reset_env(env)

    if pbar is not None:
        pbar.close()

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
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()

    set_global_seeds(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    actor = build_actor_for_checkpoint(ckpt, device=device)

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

    try:
        seed_env(env_train, args.seed)
        seed_env(env_test, args.seed + 1)

        mean_tr, std_tr = run_eval(env_train, actor, device, args.episodes, desc="TRAIN", progress=(not args.no_progress))
        mean_te, std_te = run_eval(env_test, actor, device, args.episodes, desc="TEST ", progress=(not args.no_progress))

        print(f"TRAIN levels: mean_return={mean_tr:.3f} std={std_tr:.3f} (episodes={args.episodes})")
        print(f"TEST  levels: mean_return={mean_te:.3f} std={std_te:.3f} (episodes={args.episodes})")
    finally:
        env_train.close()
        env_test.close()


if __name__ == "__main__":
    main()
