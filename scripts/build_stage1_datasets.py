#!/usr/bin/env python3
"""
scripts/build_stage1_datasets.py

Build Stage-1 augmented offline datasets from Gen-DGRL (Procgen CoinRun).

Creates:
- 1 baseline dataset (no augmentation)
- 1 dataset per Stage-1 augmentation configuration (12 total including baseline)
- Saves one sample image per variant + a comparison grid

Output layout (default):
  data/stage1_datasets/<variant_name>/
    manifest.json
    shard_00000.npz
    shard_00001.npz
    ...
    sample_obs.png
  data/stage1_datasets/comparison_grid.png

Each shard contains:
  obs      uint8 (N,3,64,64)
  next_obs uint8 (N,3,64,64)
  action   int64 (N,)
  reward   float32 (N,)
  done     uint8 (N,)

Recommended (start small):
  python scripts/build_stage1_datasets.py --num-samples 20000 --shard-size 5000

Notes:
- This multiplies storage ~12x (one dataset per augmentation). Start small.
- Augmentations are applied to BOTH obs and next_obs for consistency.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchrl.data.datasets import GenDGRLExperienceReplay


# -----------------------------
# Layout utilities
# -----------------------------
def ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure a 4D image batch is in BCHW layout.

    Accepts:
      - BCHW: (B,C,H,W) where C in {1,3}
      - BHWC: (B,H,W,C) where C in {1,3}

    Returns:
      - BCHW tensor
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {x.shape}")
    if x.shape[1] in (1, 3):           # already BCHW
        return x
    if x.shape[-1] in (1, 3):          # BHWC -> BCHW
        return x.permute(0, 3, 1, 2)
    raise ValueError(f"Unrecognized image layout: {x.shape} (expected BCHW or BHWC)")


def to_chw_float01(obs_u8: torch.Tensor) -> torch.Tensor:
    """Convert uint8 observations to float32 [0,1] in BCHW layout."""
    obs_u8 = ensure_bchw(obs_u8)
    return obs_u8.float() / 255.0


def as_uint8_chw(x_float01: torch.Tensor) -> np.ndarray:
    """
    Convert float observations in [0,1] to uint8 in [0,255], forcing BCHW layout.

    Accepts BCHW or BHWC, returns BCHW uint8.
    """
    x_float01 = ensure_bchw(x_float01)
    x_u8 = (x_float01.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return x_u8.cpu().numpy()


def img_uint8_to_hwc(u8_img: np.ndarray) -> np.ndarray:
    """
    Convert a single uint8 image to HWC for saving/plotting.

    Accepts:
      - CHW: (3,H,W) or (1,H,W)
      - HWC: (H,W,3) or (H,W,1)
    """
    if u8_img.ndim != 3:
        raise ValueError(f"Expected 3D image, got {u8_img.shape}")

    # CHW -> HWC
    if u8_img.shape[0] in (1, 3) and u8_img.shape[1] > 1 and u8_img.shape[2] > 1:
        return np.transpose(u8_img, (1, 2, 0))

    # already HWC
    if u8_img.shape[-1] in (1, 3) and u8_img.shape[0] > 1 and u8_img.shape[1] > 1:
        return u8_img

    raise ValueError(f"Unrecognized CHW/HWC image shape: {u8_img.shape}")


# -----------------------------
# Augmentations (BCHW float01)
# -----------------------------
def random_shift(obs_bchw: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style random shift: replicate-pad then random crop (safe loop impl)."""
    obs_bchw = ensure_bchw(obs_bchw)
    if obs_bchw.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W), got {obs_bchw.shape}")
    b, c, h, w = obs_bchw.shape

    x = F.pad(obs_bchw, (pad, pad, pad, pad), mode="replicate")  # (B,C,H+2p,W+2p)
    out = torch.empty((b, c, h, w), device=obs_bchw.device, dtype=obs_bchw.dtype)

    # per-sample crop (simple + robust)
    for i in range(b):
        y0 = int(torch.randint(0, 2 * pad + 1, (1,), device=obs_bchw.device).item())
        x0 = int(torch.randint(0, 2 * pad + 1, (1,), device=obs_bchw.device).item())
        out[i] = x[i, :, y0:y0 + h, x0:x0 + w]
    return out


def color_jitter_bc(obs_bchw: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Simple brightness/contrast jitter without torchvision.
    strength=0.2 is mild, 0.4 is strong.
    """
    obs_bchw = ensure_bchw(obs_bchw)
    x = obs_bchw

    # brightness: add per-sample scalar in [-s, s]
    b = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * strength
    x = x + b

    # contrast: scale around per-sample mean
    mean = x.mean(dim=(2, 3), keepdim=True)
    c = 1.0 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1) * strength
    x = (x - mean) * c + mean

    return x.clamp(0.0, 1.0)


def cutout(obs_bchw: torch.Tensor, size: int) -> torch.Tensor:
    """Random square mask set to 0."""
    obs_bchw = ensure_bchw(obs_bchw)
    b, c, h, w = obs_bchw.shape
    x = obs_bchw.clone()

    if size <= 0 or size >= h or size >= w:
        return x

    ys = torch.randint(0, h - size + 1, (b,), device=x.device)
    xs = torch.randint(0, w - size + 1, (b,), device=x.device)
    for i in range(b):
        y0 = int(ys[i].item())
        x0 = int(xs[i].item())
        x[i, :, y0:y0 + size, x0:x0 + size] = 0.0
    return x


def gaussian_noise(obs_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
    obs_bchw = ensure_bchw(obs_bchw)
    return (obs_bchw + torch.randn_like(obs_bchw) * sigma).clamp(0.0, 1.0)


def blur_3x3(obs_bchw: torch.Tensor) -> torch.Tensor:
    """Light 3x3 blur using a fixed kernel. Depthwise conv (per-channel)."""
    obs_bchw = ensure_bchw(obs_bchw)
    _, c, _, _ = obs_bchw.shape

    kernel = torch.tensor(
        [[1.0, 2.0, 1.0],
         [2.0, 4.0, 2.0],
         [1.0, 2.0, 1.0]],
        device=obs_bchw.device,
        dtype=obs_bchw.dtype,
    )
    kernel = kernel / kernel.sum()
    weight = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)  # (C,1,3,3)

    x = F.pad(obs_bchw, (1, 1, 1, 1), mode="replicate")
    x = F.conv2d(x, weight=weight, bias=None, stride=1, padding=0, groups=c)
    return x


def scale_jitter(obs_bchw: torch.Tensor, scale_min: float, scale_max: float) -> torch.Tensor:
    """
    Random scale (zoom in/out) then crop/pad back to 64x64.
    Implemented per-sample for simplicity/reliability on 64x64.
    """
    obs_bchw = ensure_bchw(obs_bchw)
    b, c, h, w = obs_bchw.shape
    if h != 64 or w != 64:
        raise ValueError(f"scale_jitter expects 64x64 inputs, got {h}x{w}")

    out = torch.empty_like(obs_bchw)
    for i in range(b):
        s = float(torch.empty(1, device=obs_bchw.device).uniform_(scale_min, scale_max).item())
        new_hw = max(8, int(round(64 * s)))

        x = obs_bchw[i:i + 1]  # (1,C,64,64)
        x = F.interpolate(x, size=(new_hw, new_hw), mode="bilinear", align_corners=False)

        if new_hw > 64:
            y0 = int(torch.randint(0, new_hw - 64 + 1, (1,), device=obs_bchw.device).item())
            x0 = int(torch.randint(0, new_hw - 64 + 1, (1,), device=obs_bchw.device).item())
            x = x[:, :, y0:y0 + 64, x0:x0 + 64]
        elif new_hw < 64:
            pad_total = 64 - new_hw
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_left, pad_right), mode="replicate")

        out[i] = x[0]
    return out


# -----------------------------
# Stage-1 variants
# -----------------------------
@dataclass(frozen=True)
class Variant:
    name: str
    shift_pad: Optional[int] = None
    scale_range: Optional[Tuple[float, float]] = None
    jitter_strength: Optional[float] = None
    cutout_size: Optional[int] = None
    noise_sigma: Optional[float] = None
    blur: bool = False
    p_aug: float = 1.0  # probability of applying augmentation stack (only meaningful when shift_pad is set)


def get_stage1_variants() -> List[Variant]:
    return [
        Variant(name="baseline_none"),
        Variant(name="shift_pad4", shift_pad=4),
        Variant(name="shift_pad8", shift_pad=8),
        Variant(name="shift4_scale_0p9_1p1", shift_pad=4, scale_range=(0.9, 1.1)),
        Variant(name="shift4_scale_0p8_1p2", shift_pad=4, scale_range=(0.8, 1.2)),
        Variant(name="shift4_jitter_0p2", shift_pad=4, jitter_strength=0.2),
        Variant(name="shift4_jitter_0p4", shift_pad=4, jitter_strength=0.4),
        Variant(name="shift4_cutout16", shift_pad=4, cutout_size=16),
        Variant(name="shift4_cutout24", shift_pad=4, cutout_size=24),
        Variant(name="shift4_noise_0p02", shift_pad=4, noise_sigma=0.02),
        Variant(name="shift4_blur3x3", shift_pad=4, blur=True),
        Variant(name="shift4_paug_0p5", shift_pad=4, p_aug=0.5),
    ]


def apply_variant(x: torch.Tensor, v: Variant) -> torch.Tensor:
    """
    Apply augmentation stack to BCHW float01 tensor.
    Order:
      shift -> scale -> jitter -> cutout -> noise -> blur
    """
    x = ensure_bchw(x)

    if v.shift_pad is None:
        return x

    def _stack(z: torch.Tensor) -> torch.Tensor:
        z = random_shift(z, pad=v.shift_pad)
        if v.scale_range is not None:
            z = scale_jitter(z, *v.scale_range)
        if v.jitter_strength is not None:
            z = color_jitter_bc(z, v.jitter_strength)
        if v.cutout_size is not None:
            z = cutout(z, v.cutout_size)
        if v.noise_sigma is not None:
            z = gaussian_noise(z, v.noise_sigma)
        if v.blur:
            z = blur_3x3(z)
        return z

    # Optionally mix raw+aug
    if v.p_aug < 1.0:
        b = x.shape[0]
        mask = (torch.rand(b, device=x.device) < v.p_aug).view(b, 1, 1, 1)
        x_aug = _stack(x)
        return torch.where(mask, x_aug, x)

    return _stack(x)


# -----------------------------
# Sharded NPZ writer
# -----------------------------
class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.shard_idx = 0
        self.written = 0

        self._obs: List[np.ndarray] = []
        self._next_obs: List[np.ndarray] = []
        self._action: List[np.ndarray] = []
        self._reward: List[np.ndarray] = []
        self._done: List[np.ndarray] = []

        self.shards: List[str] = []

    def _buffer_len(self) -> int:
        return int(sum(x.shape[0] for x in self._action)) if self._action else 0

    def add_batch(self, obs_u8: np.ndarray, next_u8: np.ndarray,
                  action: np.ndarray, reward: np.ndarray, done: np.ndarray,
                  max_total: int):
        if self.written >= max_total:
            return

        remaining = max_total - self.written
        if obs_u8.shape[0] > remaining:
            obs_u8 = obs_u8[:remaining]
            next_u8 = next_u8[:remaining]
            action = action[:remaining]
            reward = reward[:remaining]
            done = done[:remaining]

        self._obs.append(obs_u8)
        self._next_obs.append(next_u8)
        self._action.append(action)
        self._reward.append(reward)
        self._done.append(done)

        while self._buffer_len() >= self.shard_size and self.written < max_total:
            self.flush_one(max_total=max_total)

    def flush_one(self, max_total: int):
        obs_cat = np.concatenate(self._obs, axis=0)
        next_cat = np.concatenate(self._next_obs, axis=0)
        act_cat = np.concatenate(self._action, axis=0)
        rew_cat = np.concatenate(self._reward, axis=0)
        done_cat = np.concatenate(self._done, axis=0)

        take = min(self.shard_size, obs_cat.shape[0], max_total - self.written)
        shard = {
            "obs": obs_cat[:take],
            "next_obs": next_cat[:take],
            "action": act_cat[:take],
            "reward": rew_cat[:take],
            "done": done_cat[:take],
        }
        out_path = self.out_dir / f"shard_{self.shard_idx:05d}.npz"
        np.savez_compressed(out_path, **shard)
        self.shards.append(out_path.name)

        self.shard_idx += 1
        self.written += take

        # leftovers
        obs_left = obs_cat[take:]
        next_left = next_cat[take:]
        act_left = act_cat[take:]
        rew_left = rew_cat[take:]
        done_left = done_cat[take:]

        self._obs = [obs_left] if obs_left.size else []
        self._next_obs = [next_left] if next_left.size else []
        self._action = [act_left] if act_left.size else []
        self._reward = [rew_left] if rew_left.size else []
        self._done = [done_left] if done_left.size else []

    def finalize(self, max_total: int):
        while self._buffer_len() > 0 and self.written < max_total:
            self.flush_one(max_total=max_total)


# -----------------------------
# Images
# -----------------------------
def save_sample_image(out_path: Path, obs_u8: np.ndarray, title: str):
    img = img_uint8_to_hwc(obs_u8)
    plt.figure(figsize=(2.5, 2.5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def build_comparison_grid(out_root: Path, variants: List[Variant]):
    imgs = []
    titles = []
    for v in variants:
        p = out_root / v.name / "sample_obs.png"
        if p.exists():
            imgs.append(plt.imread(p))
            titles.append(v.name)

    if not imgs:
        return

    n = len(imgs)
    cols = 4
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3.2 * cols, 3.2 * rows))
    for i, (img, t) in enumerate(zip(imgs, titles)):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    out_path = out_root / "comparison_grid.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved comparison grid: {out_path}")


# -----------------------------
# Main
# -----------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="coinrun-level_1_E")
    parser.add_argument("--out-root", default="data/stage1_datasets")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", default="true", choices=["true", "force"],
                        help="Use 'force' if you suspect a corrupted cache.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Device: {device}")
    print(f"Source dataset: {args.dataset_id}")
    print(f"Writing to: {out_root}")
    print(f"num_samples={args.num_samples}, batch_size={args.batch_size}, shard_size={args.shard_size}")

    variants = get_stage1_variants()
    writers: Dict[str, ShardWriter] = {v.name: ShardWriter(out_root / v.name, args.shard_size) for v in variants}
    saved_sample = {v.name: False for v in variants}

    download_arg = True if args.download == "true" else "force"
    ds = GenDGRLExperienceReplay(args.dataset_id, batch_size=args.batch_size, download=download_arg)
    it = iter(ds)

    with torch.no_grad():
        while not all(w.written >= args.num_samples for w in writers.values()):
            batch = next(it)

            obs = batch["observation"]
            next_obs = batch["next", "observation"]
            act = batch["action"].reshape(-1)
            rew = batch["next", "reward"].reshape(-1).to(torch.float32)
            done = batch["next", "done"].reshape(-1).to(torch.uint8)

            # filter invalid/empty rows
            valid = torch.isfinite(act.to(torch.float32)) & torch.isfinite(rew)
            valid &= (obs.view(obs.shape[0], -1).sum(dim=1) > 0)
            if valid.sum() == 0:
                continue

            obs = obs[valid]
            next_obs = next_obs[valid]
            act = act[valid]
            rew = rew[valid]
            done = done[valid]

            # float BCHW on device
            obs_f = to_chw_float01(obs).to(device)
            next_f = to_chw_float01(next_obs).to(device)

            # non-image arrays on CPU
            act_np = act.cpu().numpy().astype(np.int64)
            rew_np = rew.cpu().numpy().astype(np.float32)
            done_np = done.cpu().numpy().astype(np.uint8)

            for v in variants:
                w = writers[v.name]
                if w.written >= args.num_samples:
                    continue

                o = apply_variant(obs_f, v)
                n = apply_variant(next_f, v)

                o_u8 = as_uint8_chw(o)     # (B,3,64,64) uint8
                n_u8 = as_uint8_chw(n)

                w.add_batch(o_u8, n_u8, act_np, rew_np, done_np, max_total=args.num_samples)

                if (not saved_sample[v.name]) and o_u8.shape[0] > 0:
                    save_sample_image(out_root / v.name / "sample_obs.png", o_u8[0], title=v.name)
                    saved_sample[v.name] = True

            base_written = writers["baseline_none"].written
            if base_written > 0 and base_written % (args.shard_size * 2) == 0:
                print(f"Progress (baseline): {base_written}/{args.num_samples}")

    # finalize + manifests
    for v in variants:
        w = writers[v.name]
        w.finalize(max_total=args.num_samples)

        manifest = {
            "source": "GenDGRLExperienceReplay",
            "source_dataset_id": args.dataset_id,
            "variant": v.__dict__,
            "num_samples": args.num_samples,
            "shard_size": args.shard_size,
            "num_shards": len(w.shards),
            "shards": w.shards,
            "format": {
                "obs": "uint8 (N,3,64,64)",
                "next_obs": "uint8 (N,3,64,64)",
                "action": "int64 (N,)",
                "reward": "float32 (N,)",
                "done": "uint8 (N,)",
            },
            "seed": args.seed,
        }
        (out_root / v.name / "manifest.json").write_text(json.dumps(manifest, indent=2))

    build_comparison_grid(out_root, variants)
    print("Done.")


if __name__ == "__main__":
    main()