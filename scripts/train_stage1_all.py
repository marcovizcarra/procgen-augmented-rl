#!/usr/bin/env python3
"""
scripts/train_stage1_all.py  (updated: run-group support)

Train BC checkpoints for Stage-1 dataset variants (baseline + augmentations).

Update:
- By default, runs are stored under a subfolder based on the dataset used:
    runs/<run_group>/bc_<variant>/bc_ckpt.pt
  where run_group defaults to the *datasets-root folder name* (e.g. stage1_datasets_L40).

This keeps runs from different dataset builds separated automatically.

Usage:
  python -B scripts/train_stage1_all.py \
    --datasets-root data/stage1_datasets_L40 \
    --train-steps 20000 \
    --batch-size 256 \
    --seed 0 \
    --skip-existing

Then evaluate (matching run-group):
  python -B scripts/eval_stage1_all.py \
    --datasets-root data/stage1_datasets_L40 \
    --episodes 200
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


def discover_variants(datasets_root: Path) -> List[Path]:
    if not datasets_root.exists():
        return []
    out = []
    for p in datasets_root.iterdir():
        if p.is_dir() and (p / "manifest.json").exists():
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="data/stage1_datasets")
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run-group", default=None, help="Subfolder under runs-root (default: datasets-root folder name).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip training if checkpoint already exists.")
    ap.add_argument("--only", default=None, help="Comma-separated list of variants to train.")
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    variants = discover_variants(datasets_root)
    if not variants:
        raise RuntimeError(f"No variants found under {datasets_root} (expected folders with manifest.json).")

    if args.only:
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        variants = [v for v in variants if v.name in allow]
        if not variants:
            raise RuntimeError("After --only filter, no variants remain.")

    runs_root = Path(args.runs_root)
    run_group = args.run_group or datasets_root.name

    trained = []
    for vdir in variants:
        variant = vdir.name
        run_name = f"{run_group}/bc_{variant}"  # nested folder under runs/

        ckpt = runs_root / run_name / "bc_ckpt.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and ckpt.exists():
            print(f"[skip existing] {variant} -> {ckpt}")
            trained.append({"variant": variant, "run_name": run_name, "ckpt": str(ckpt), "skipped": True})
            continue

        cmd = [
            sys.executable, "scripts/train_bc_min.py",
            "--dataset-root", str(vdir),
            "--run-name", run_name,
            "--steps", str(args.train_steps),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--device", str(args.device),
        ]
        print("\n$ " + " ".join(cmd), flush=True)

        # Stream output instead of capturing; fixes "no output until end" issues
        ret = subprocess.call(cmd)
        if ret != 0:
            raise RuntimeError(f"Training failed for variant={variant} (exit code={ret}).")

        if not ckpt.exists():
            raise FileNotFoundError(f"Training finished but checkpoint not found: {ckpt}")

        trained.append({"variant": variant, "run_name": run_name, "ckpt": str(ckpt), "skipped": False})

        # incremental log per group
        log_path = runs_root / run_group / "stage1_train_log.json"
        log_path.write_text(json.dumps(trained, indent=2), encoding="utf-8")

    print(f"\nRun group: {run_group}")
    print(f"Wrote: {runs_root / run_group / 'stage1_train_log.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
