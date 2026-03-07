#!/usr/bin/env python3
"""
Minimal offline CQL trainer for Procgen CoinRun datasets (discrete actions).

Separate from BC/IQL pipelines:
- Input: Stage-1 NPZ datasets (preferred) or Gen-DGRL stream (fallback)
- Output: runs/<run-name>/cql_ckpt.pt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import trange
except Exception:
    trange = None

try:
    from torchrl.data.datasets import GenDGRLExperienceReplay
except Exception:
    GenDGRLExperienceReplay = None


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


class DiscreteQ(nn.Module):
    def __init__(self, n_actions: int = 15, hidden: int = 512):
        super().__init__()
        self.enc = ConvEncoder(hidden=hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(obs))


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def load_manifest(manifest_path: Path) -> Dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_npz_batches(
    manifest_path: Path,
    batch_size: int,
    seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    shards = manifest["shards"]

    while True:
        shard_order = list(shards)
        rng.shuffle(shard_order)

        for shard_name in shard_order:
            shard_path = root / shard_name
            with np.load(shard_path) as d:
                obs = d["obs"]
                next_obs = d["next_obs"]
                act = d["action"]
                rew = d["reward"]
                done = d["done"]

            n = obs.shape[0]
            idx = np.arange(n)
            rng.shuffle(idx)

            for start in range(0, n, batch_size):
                j = idx[start:start + batch_size]
                if j.size == 0:
                    continue
                yield obs[j], next_obs[j], act[j], rew[j], done[j]


def iter_gendgrl_batches(dataset_id: str, batch_size: int, seed: int = 0, download: bool = True):
    if GenDGRLExperienceReplay is None:
        raise RuntimeError("TorchRL GenDGRLExperienceReplay not available. Install torchrl or use --dataset-root/--manifest.")
    torch.manual_seed(seed)
    ds = GenDGRLExperienceReplay(dataset_id, batch_size=batch_size, download=download)
    for batch in ds:
        obs = batch["observation"]
        next_obs = batch["next", "observation"]
        act = batch["action"].reshape(-1)
        rew = batch["next", "reward"].reshape(-1).to(torch.float32)
        done = batch["next", "done"].reshape(-1).to(torch.float32)

        if obs.ndim == 4 and obs.shape[-1] == 3:
            obs = obs.permute(0, 3, 1, 2)
        if next_obs.ndim == 4 and next_obs.shape[-1] == 3:
            next_obs = next_obs.permute(0, 3, 1, 2)

        yield (
            obs.cpu().numpy().astype(np.uint8),
            next_obs.cpu().numpy().astype(np.uint8),
            np.nan_to_num(act.cpu().numpy(), nan=0.0).astype(np.int64),
            np.nan_to_num(rew.cpu().numpy(), nan=0.0).astype(np.float32),
            np.nan_to_num(done.cpu().numpy(), nan=0.0).astype(np.float32),
        )


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--dataset-root", type=str, default=None)
    src.add_argument("--manifest", type=str, default=None)
    src.add_argument("--gendgrl-id", type=str, default=None)

    p.add_argument("--run-name", type=str, default="cql_min")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n-actions", type=int, default=15)
    p.add_argument("--hidden", type=int, default=512)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-tau", type=float, default=0.005)
    p.add_argument("--cql-alpha", type=float, default=1.0)

    p.add_argument("--lr-q", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest_path: Optional[Path] = None
    if args.dataset_root is not None:
        manifest_path = Path(args.dataset_root) / "manifest.json"
    elif args.manifest is not None:
        manifest_path = Path(args.manifest)

    if manifest_path is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        data_iter = iter_npz_batches(manifest_path, batch_size=args.batch_size, seed=args.seed)
        data_desc = f"NPZ(manifest={manifest_path})"
    else:
        dataset_id = args.gendgrl_id or "coinrun-level_1_E"
        data_iter = iter_gendgrl_batches(dataset_id, batch_size=args.batch_size, seed=args.seed, download=True)
        data_desc = f"GenDGRL({dataset_id})"

    q1 = DiscreteQ(n_actions=args.n_actions, hidden=args.hidden).to(device)
    q2 = DiscreteQ(n_actions=args.n_actions, hidden=args.hidden).to(device)
    q1_targ = DiscreteQ(n_actions=args.n_actions, hidden=args.hidden).to(device)
    q2_targ = DiscreteQ(n_actions=args.n_actions, hidden=args.hidden).to(device)
    q1_targ.load_state_dict(q1.state_dict())
    q2_targ.load_state_dict(q2.state_dict())

    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr_q)

    print(f"Device: {device}")
    print(f"Run: {args.run_name} | steps={args.steps} | batch_size={args.batch_size} | seed={args.seed}")
    print(f"Data: {data_desc}")

    start_time = time.time()
    if trange is not None:
        iterator = trange(1, args.steps + 1, desc=f"CQL[{args.run_name}]", dynamic_ncols=True)
    else:
        iterator = range(1, args.steps + 1)

    stats: Dict[str, List[float]] = {"bellman": [], "cql": [], "q": []}

    for step in iterator:
        obs_u8, next_obs_u8, act_np, rew_np, done_np = next(data_iter)

        obs = torch.from_numpy(obs_u8).to(device)
        next_obs = torch.from_numpy(next_obs_u8).to(device)
        act = torch.from_numpy(act_np).to(device).long()
        rew = torch.from_numpy(rew_np).to(device).float()
        done = torch.from_numpy(done_np).to(device).float()

        with torch.no_grad():
            q_next = torch.min(q1_targ(next_obs), q2_targ(next_obs))
            next_v = q_next.max(dim=1).values
            target = rew + args.gamma * (1.0 - done) * next_v

        q1_all = q1(obs)
        q2_all = q2(obs)
        q1_data = q1_all.gather(1, act.unsqueeze(1)).squeeze(1)
        q2_data = q2_all.gather(1, act.unsqueeze(1)).squeeze(1)

        bellman = F.mse_loss(q1_data, target) + F.mse_loss(q2_data, target)

        cql1 = (torch.logsumexp(q1_all, dim=1) - q1_data).mean()
        cql2 = (torch.logsumexp(q2_all, dim=1) - q2_data).mean()
        cql_pen = cql1 + cql2

        q_loss = bellman + args.cql_alpha * cql_pen

        opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        opt_q.step()

        soft_update(q1_targ, q1, args.target_tau)
        soft_update(q2_targ, q2, args.target_tau)

        stats["bellman"].append(float(bellman.item()))
        stats["cql"].append(float(cql_pen.item()))
        stats["q"].append(float(q_loss.item()))

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            k = min(args.log_every, len(stats["q"]))
            bellman_avg = float(np.mean(stats["bellman"][-k:]))
            cql_avg = float(np.mean(stats["cql"][-k:]))
            q_avg = float(np.mean(stats["q"][-k:]))

            elapsed = time.time() - start_time
            sps = step / max(1e-9, elapsed)
            eta = (args.steps - step) / max(1e-9, sps)

            if trange is not None:
                iterator.set_postfix(
                    bellman=f"{bellman_avg:.4f}",
                    cql=f"{cql_avg:.4f}",
                    q=f"{q_avg:.4f}",
                    sps=f"{sps:.1f}",
                    eta=_fmt_time(eta),
                )
            else:
                pct = 100.0 * step / args.steps
                print(
                    f"[{pct:6.2f}%] step {step:6d}/{args.steps} "
                    f"bellman={bellman_avg:.4f} cql={cql_avg:.4f} q={q_avg:.4f} "
                    f"sps={sps:.1f} eta={_fmt_time(eta)}",
                    flush=True,
                )

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "cql_ckpt.pt"

    ckpt = {
        "algo": "CQL",
        "arch": "CQL_Discrete_CNN_v1",
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "target_tau": args.target_tau,
        "cql_alpha": args.cql_alpha,
        "n_actions": args.n_actions,
        "hidden": args.hidden,
        "data_source": data_desc,
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
    }
    torch.save(ckpt, ckpt_path)

    cfg = {
        "run_name": args.run_name,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "n_actions": args.n_actions,
        "hidden": args.hidden,
        "gamma": args.gamma,
        "target_tau": args.target_tau,
        "cql_alpha": args.cql_alpha,
        "lr_q": args.lr_q,
        "data_source": data_desc,
        "ckpt_path": str(ckpt_path),
        "elapsed_sec": time.time() - start_time,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {ckpt_path} (elapsed {_fmt_time(cfg['elapsed_sec'])})")


if __name__ == "__main__":
    main()
