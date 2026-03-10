#!/usr/bin/env python3
"""
Minimal offline CQL trainer for Procgen CoinRun datasets (discrete actions).

- Input: Stage-1 NPZ datasets or Gen-DGRL stream (fallback)
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
import wandb
from tqdm.auto import trange

try:
    from torchrl.data.datasets import GenDGRLExperienceReplay
except Exception:
    GenDGRLExperienceReplay = None


class ConvEncoder(nn.Module):
    """
    Simple CNN for our environment. This is the same across all algorithms. 
    """
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 4 * 4, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype == torch.uint8:
            x = obs.float() / 255.0 # normalize pixel values to [0,1]
        else:
            x = obs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1) # flatten bc Linear layer expects 2D input
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
    """
    Pick device to use for training
    """
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
    """
    Update the 
    """
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def load_manifest(manifest_path: Path) -> Dict:
    """
    Helper function to load the dataset for training.
    """
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_npz_batches(
    manifest_path: Path,
    batch_size: int,
    seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """ 
    Helper function to iterate over the dataset in batches.
    """
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
    p.add_argument("--temp", type=float, default=1.0, help="Temperature for CQL logsumexp term.")
    p.add_argument(
        "--min-q-version",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="1: logsumexp (default), 2: mean-Q penalty, 3: max-Q penalty.",
    )
    p.add_argument("--with-lagrange", action="store_true", help="Enable Lagrange auto-tuning for conservative weight.")
    p.add_argument(
        "--lagrange-thresh",
        type=float,
        default=10.0,
        help="Target action-gap threshold for Lagrange CQL.",
    )

    p.add_argument("--lr-q", type=float, default=3e-4)
    p.add_argument("--lr-cql-alpha", type=float, default=1e-4, help="LR for Lagrange alpha (if enabled).")
    p.add_argument("--log-every", type=int, default=200)

    # optional Weights & Biases logging
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default="procgen-augmented-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", type=str, default="", help="Comma-separated tags, e.g. cql,stage1")
    args = p.parse_args()
    if args.temp <= 0:
        raise ValueError("--temp must be > 0")

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

    # use double q-learning strategy to avoid being overly optimistic about action values
    q1 = DiscreteQ(n_actions=args.n_actions, hidden=args.hidden).to(device)
    q2 = DiscreteQ(n_actions=args.n_actions, hidden=args.hidden).to(device)
    # init target networks 
    q1_targ = DiscreteQ(n_actions=args.n_actions, hidden=args.hidden).to(device)
    q2_targ = DiscreteQ(n_actions=args.n_actions, hidden=args.hidden).to(device)
    q1_targ.load_state_dict(q1.state_dict())
    q2_targ.load_state_dict(q2.state_dict())

    # use basic Adam optimizer 
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr_q)
    log_cql_alpha = None
    opt_cql_alpha = None

    # (optional) Lagrange CQL 
    if args.with_lagrange:
        log_cql_alpha = torch.zeros(1, device=device, requires_grad=True)
        opt_cql_alpha = torch.optim.Adam([log_cql_alpha], lr=args.lr_cql_alpha)

    print(f"Device: {device}")
    print(f"Run: {args.run_name} | steps={args.steps} | batch_size={args.batch_size} | seed={args.seed}")
    print(f"Data: {data_desc}")

    # W&B logging 
    wb_run = None
    if args.wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.run_name,
            mode=args.wandb_mode,
            tags=tags or None,
            config={
                "algo": "CQL",
                "run_name": args.run_name,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": str(device),
                "n_actions": args.n_actions,
                "hidden": args.hidden,
                "gamma": args.gamma,
                "target_tau": args.target_tau,
                "cql_alpha": args.cql_alpha,
                "temp": args.temp,
                "min_q_version": args.min_q_version,
                "with_lagrange": args.with_lagrange,
                "lagrange_thresh": args.lagrange_thresh,
                "lr_q": args.lr_q,
                "lr_cql_alpha": args.lr_cql_alpha,
                "data_source": data_desc,
            },
        )

    start_time = time.time()
    if trange is not None:
        iterator = trange(1, args.steps + 1, desc=f"CQL[{args.run_name}]", dynamic_ncols=True)
    else:
        iterator = range(1, args.steps + 1)

    stats: Dict[str, List[float]] = {"bellman": [], "cql": [], "q": [], "alpha": []}

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
            target = rew + args.gamma * (1.0 - done) * next_v # bellman target

        q1_all = q1(obs)
        q2_all = q2(obs)
        q1_data = q1_all.gather(1, act.unsqueeze(1)).squeeze(1)
        q2_data = q2_all.gather(1, act.unsqueeze(1)).squeeze(1)

        bellman = F.mse_loss(q1_data, target) + F.mse_loss(q2_data, target)

        # 3 different versions of CQL penalty
        if args.min_q_version == 1:
            # standard discrete CQL(H): temp * logsumexp(Q/temp) - Q(s,a_data)
            cql1 = (args.temp * torch.logsumexp(q1_all / args.temp, dim=1) - q1_data).mean()
            cql2 = (args.temp * torch.logsumexp(q2_all / args.temp, dim=1) - q2_data).mean()
        elif args.min_q_version == 2:
            # mean-Q variant (weaker conservative penalty)
            cql1 = (q1_all.mean(dim=1) - q1_data).mean()
            cql2 = (q2_all.mean(dim=1) - q2_data).mean()
        else:
            # max-Q variant (stronger penalty on largest competing action value)
            cql1 = (q1_all.max(dim=1).values - q1_data).mean()
            cql2 = (q2_all.max(dim=1).values - q2_data).mean()
        cql_pen = cql1 + cql2

        if args.with_lagrange:
            assert log_cql_alpha is not None and opt_cql_alpha is not None
            cql_alpha = torch.exp(log_cql_alpha).clamp(min=0.0, max=1e6)
            cql_term = cql_alpha.detach() * (cql_pen - args.lagrange_thresh)
            q_loss = bellman + cql_term
        else:
            cql_alpha = torch.tensor(args.cql_alpha, device=device)
            q_loss = bellman + args.cql_alpha * cql_pen

        # update Q-networks
        opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        opt_q.step()

        if args.with_lagrange:
            # Increase alpha when penalty exceeds threshold, decrease otherwise.
            assert log_cql_alpha is not None and opt_cql_alpha is not None
            alpha_loss = -(torch.exp(log_cql_alpha) * (cql_pen.detach() - args.lagrange_thresh))
            opt_cql_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            opt_cql_alpha.step()

        soft_update(q1_targ, q1, args.target_tau)
        soft_update(q2_targ, q2, args.target_tau)

        stats["bellman"].append(float(bellman.item()))
        stats["cql"].append(float(cql_pen.item()))
        stats["q"].append(float(q_loss.item()))
        stats["alpha"].append(float(cql_alpha.item()))

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            k = min(args.log_every, len(stats["q"]))
            bellman_avg = float(np.mean(stats["bellman"][-k:]))
            cql_avg = float(np.mean(stats["cql"][-k:]))
            q_avg = float(np.mean(stats["q"][-k:]))
            alpha_avg = float(np.mean(stats["alpha"][-k:]))

            elapsed = time.time() - start_time
            sps = step / max(1e-9, elapsed)
            eta = (args.steps - step) / max(1e-9, sps)

            if trange is not None:
                iterator.set_postfix(
                    bellman=f"{bellman_avg:.4f}",
                        cql=f"{cql_avg:.4f}",
                        q=f"{q_avg:.4f}",
                        alpha=f"{alpha_avg:.3f}",
                        sps=f"{sps:.1f}",
                        eta=_fmt_time(eta),
                    )
            else:
                pct = 100.0 * step / args.steps
                print(
                    f"[{pct:6.2f}%] step {step:6d}/{args.steps} "
                    f"bellman={bellman_avg:.4f} cql={cql_avg:.4f} q={q_avg:.4f} alpha={alpha_avg:.3f} "
                    f"sps={sps:.1f} eta={_fmt_time(eta)}",
                    flush=True,
                )
            if wb_run is not None:
                wandb.log(
                    {
                        "train/step": step,
                        "train/bellman_loss": bellman_avg,
                        "train/cql_penalty": cql_avg,
                        "train/q_loss": q_avg,
                        "train/cql_alpha": alpha_avg,
                        "train/steps_per_sec": float(sps),
                        "train/elapsed_sec": float(elapsed),
                    },
                    step=step,
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
        "temp": args.temp,
        "min_q_version": args.min_q_version,
        "with_lagrange": args.with_lagrange,
        "lagrange_thresh": args.lagrange_thresh,
        "learned_cql_alpha": float(np.mean(stats["alpha"][-min(args.log_every, len(stats["alpha"])):])),
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
        "temp": args.temp,
        "min_q_version": args.min_q_version,
        "with_lagrange": args.with_lagrange,
        "lagrange_thresh": args.lagrange_thresh,
        "lr_cql_alpha": args.lr_cql_alpha,
        "lr_q": args.lr_q,
        "data_source": data_desc,
        "ckpt_path": str(ckpt_path),
        "elapsed_sec": time.time() - start_time,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {ckpt_path} (elapsed {_fmt_time(cfg['elapsed_sec'])})")

    if wb_run is not None:
        wandb.log(
            {
                "train/final_step": args.steps,
                "train/total_elapsed_sec": float(cfg["elapsed_sec"]),
                "artifacts/ckpt_path": str(ckpt_path),
            },
            step=args.steps,
        )
        wb_run.finish()


if __name__ == "__main__":
    main()
