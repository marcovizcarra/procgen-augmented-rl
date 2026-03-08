#!/usr/bin/env python3
"""
scripts/train_bc_impala.py

Behavior Cloning (BC) trainer for Procgen CoinRun using an IMPALA-style CNN.

Designed as a near drop-in replacement for train_bc_min.py:
- same data sources:
  1) NPZ shards + manifest.json
  2) Direct Gen-DGRL streaming via TorchRL
- same output directory convention:
    runs/<run-name>/bc_ckpt.pt
    runs/<run-name>/config.json
- same checkpoint compatibility keys:
    {"model": state_dict, "state_dict": state_dict, "model_state_dict": state_dict, ...}
- same default observation shape assumption: (3, 64, 64)

Example usage:
  python scripts/train_bc_impala.py \
      --dataset-root data/stage1_datasets/baseline_none \
      --run-name bc_impala_baseline \
      --steps 20000

  python scripts/train_bc_impala.py \
      --gendgrl-id coinrun-level_1_E \
      --run-name bc_impala_gendgrl \
      --steps 20000

Notes on compatibility:
- Checkpoint keys remain compatible with downstream scripts that load state_dict-like entries.
- If a downstream script hardcodes the old CNNPolicy architecture, it must switch on ckpt['arch']
  and instantiate IMPALAPolicy for this checkpoint.
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
    from tqdm.auto import trange  # progress bar if installed
except Exception:
    trange = None

try:
    from torchrl.data.datasets import GenDGRLExperienceReplay
except Exception:
    GenDGRLExperienceReplay = None


# -----------------------------
# Model: IMPALA-style CNN policy
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(x, inplace=False)
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        return x + residual


class ImpalaBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_ch)
        self.res2 = ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class IMPALAPolicy(nn.Module):
    """
    IMPALA-style visual encoder + policy head.

    Input:
      obs: (B, 3, 64, 64), uint8 in [0,255] or float in [0,1]/[0,255]
    Output:
      logits: (B, n_actions)
    """

    def __init__(self, n_actions: int = 15, channels: Tuple[int, ...] = (16, 32, 32), hidden_dim: int = 256):
        super().__init__()
        c1, c2, c3 = channels
        self.block1 = ImpalaBlock(3, c1)
        self.block2 = ImpalaBlock(c1, c2)
        self.block3 = ImpalaBlock(c2, c3)
        self.fc = nn.Linear(c3 * 8 * 8, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_actions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _normalize_obs(obs: torch.Tensor) -> torch.Tensor:
        if obs.dtype == torch.uint8:
            return obs.float().div(255.0)
        obs = obs.float()
        if obs.numel() > 0:
            maxv = obs.detach().amax()
            if torch.isfinite(maxv) and maxv.item() > 1.0:
                obs = obs / 255.0
        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._normalize_obs(obs)
        x = self.block1(x)  # 64 -> 32
        x = self.block2(x)  # 32 -> 16
        x = self.block3(x)  # 16 -> 8
        x = F.relu(x, inplace=False)
        x = x.flatten(1)
        x = F.relu(self.fc(x), inplace=False)
        logits = self.head(x)
        return logits


# -----------------------------
# Data: NPZ shards
# -----------------------------
def load_manifest(manifest_path: Path) -> Dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_npz_batches(manifest_path: Path, batch_size: int, seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields (obs_bchw, action_int64) batches indefinitely by looping over shards.

    Expected shard arrays:
      obs: (N,3,64,64)
      action: (N,)
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
                act = d["action"]
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
        raise RuntimeError(
            "TorchRL GenDGRLExperienceReplay not available. Install torchrl or use --dataset-root/--manifest."
        )

    torch.manual_seed(seed)
    ds = GenDGRLExperienceReplay(dataset_id, batch_size=batch_size, download=download)
    for batch in ds:
        obs = batch["observation"]
        act = batch["action"]

        if obs.ndim == 4 and obs.shape[-1] == 3:  # BHWC -> BCHW
            obs = obs.permute(0, 3, 1, 2)

        obs = obs.cpu().numpy()
        act = act.reshape(-1).cpu().numpy()
        act = np.nan_to_num(act, nan=0.0).astype(np.int64)
        yield obs, act


# -----------------------------
# Utils
# -----------------------------
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


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--dataset-root", type=str, default=None, help="Variant folder containing manifest.json (Stage-1 NPZ dataset).")
    src.add_argument("--manifest", type=str, default=None, help="Path to manifest.json for an NPZ dataset.")
    src.add_argument("--gendgrl-id", type=str, default=None, help="Gen-DGRL dataset id (TorchRL), e.g. coinrun-level_1_E.")

    p.add_argument("--run-name", type=str, default="bc_impala")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--n-actions", type=int, default=15)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="Enable AMP on CUDA for faster training.")

    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default="procgen-augmented-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", type=str, default="", help="Comma-separated list, e.g. 'bc,impala,stage1'.")
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

    print(f"Device: {device}")
    print(
        f"Run: {args.run_name} | steps={args.steps} | batch_size={args.batch_size} | "
        f"lr={args.lr} | seed={args.seed}"
    )
    print(f"Data: {data_desc}")

    wb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise RuntimeError(
                "W&B logging requested (--wandb) but wandb is not installed. Install with: pip install wandb"
            ) from e

        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.run_name,
            mode=args.wandb_mode,
            tags=tags,
            config={
                "algo": "BC",
                "run_name": args.run_name,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "device": str(device),
                "n_actions": args.n_actions,
                "data_source": data_desc,
                "arch": "IMPALAPolicy_v1",
            },
        )

    model = IMPALAPolicy(n_actions=args.n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    losses_window: List[float] = []
    acc_window: List[float] = []
    ent_window: List[float] = []

    start_time = time.time()
    last_fallback_print = start_time

    if trange is not None:
        iterator = trange(1, args.steps + 1, desc=f"BC-IMPALA[{args.run_name}]", dynamic_ncols=True)
    else:
        iterator = range(1, args.steps + 1)

    try:
        for step in iterator:
            obs_np, act_np = next(data_iter)
            obs_t = torch.from_numpy(obs_np).to(device)
            act_t = torch.from_numpy(act_np).to(device).long()

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(obs_t)
                loss = F.cross_entropy(logits, act_t)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            losses_window.append(float(loss.item()))
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == act_t).float().mean().item()
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
            acc_window.append(float(acc))
            ent_window.append(float(entropy))

            if step % args.log_every == 0 or step == 1 or step == args.steps:
                k = min(args.log_every, len(losses_window))
                avg_loss = sum(losses_window[-k:]) / k
                avg_acc = sum(acc_window[-k:]) / k
                avg_ent = sum(ent_window[-k:]) / k
                elapsed = time.time() - start_time
                steps_per_s = step / max(1e-9, elapsed)
                eta = (args.steps - step) / max(1e-9, steps_per_s)

                if wb_run is not None:
                    wb_run.log(
                        {
                            "train/policy_loss": avg_loss,
                            "train/value_loss": float("nan"),
                            "train/entropy": avg_ent,
                            "train/acc": avg_acc,
                            "train/sps": steps_per_s,
                        },
                        step=step,
                    )

                if trange is not None:
                    iterator.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        acc=f"{avg_acc:.3f}",
                        sps=f"{steps_per_s:.1f}",
                        eta=_fmt_time(eta),
                        device=str(device),
                    )
                else:
                    now = time.time()
                    if (now - last_fallback_print) > 1.0 or step in (1, args.steps):
                        pct = 100.0 * step / args.steps
                        print(
                            f"[{pct:6.2f}%] step {step:6d}/{args.steps}  "
                            f"loss={avg_loss:.4f}  acc={avg_acc:.3f}  "
                            f"sps={steps_per_s:.1f}  eta={_fmt_time(eta)}",
                            flush=True,
                        )
                        last_fallback_print = now

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
            "arch": "IMPALAPolicy_v1",
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "data_source": data_desc,
            "channels": (16, 32, 32),
            "hidden_dim": 256,
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
            "arch": "IMPALAPolicy_v1",
            "ckpt_path": str(ckpt_path),
            "elapsed_sec": time.time() - start_time,
            "grad_clip": args.grad_clip,
            "amp": bool(use_amp),
        }
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"Saved checkpoint: {ckpt_path} (elapsed {_fmt_time(cfg['elapsed_sec'])})")
    finally:
        if wb_run is not None:
            wb_run.finish()


if __name__ == "__main__":
    main()
