#!/usr/bin/env python3
"""
scripts/train_stage1_all.py

Train BC checkpoints for ALL Stage-1 dataset variants (baseline + augmentations).

Creates checkpoints that eval_stage1_all expects:
  runs/bc_<variant>/bc_ckpt.pt

Discovers variants under:
  <datasets-root>/<variant>/manifest.json

WHY THIS VERSION FIXES "NO OUTPUT":
- The old version used subprocess.check_output(), which buffers output.
- This version uses subprocess.Popen() and streams stdout+stderr live.
- It also runs train_bc_min.py with Python "-u" (unbuffered) so progress bars/logs show.

Usage:
  python -B scripts/train_stage1_all.py \
    --datasets-root data/stage1_datasets \
    --train-steps 20000 \
    --batch-size 256 \
    --seed 0 \
    --skip-existing \
    --only baseline_none,shift_pad4
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


def stream_subprocess(cmd: List[str], log_path: Path) -> None:
    """
    Run a subprocess and stream stdout+stderr live to terminal.
    Also tee everything to log_path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")  # live
            f.write(line)        # saved
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="data/stage1_datasets")
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs-root", default="runs")
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
            raise RuntimeError("After --only filter, no variants remain. Check names under datasets-root.")

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    trained = []
    for vdir in variants:
        variant = vdir.name
        run_name = f"bc_{variant}"
        run_dir = runs_root / run_name
        ckpt = run_dir / "bc_ckpt.pt"
        log_path = run_dir / "train.log"

        if args.skip_existing and ckpt.exists():
            print(f"[skip existing] {variant} -> {ckpt}")
            trained.append({"variant": variant, "run_name": run_name, "ckpt": str(ckpt), "skipped": True})
            continue

        cmd = [
            sys.executable, "-u", "scripts/train_bc_min.py",   # -u => unbuffered
            "--dataset-root", str(vdir),
            "--run-name", run_name,
            "--steps", str(args.train_steps),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--device", str(args.device),
        ]

        print("\n$ " + " ".join(cmd), flush=True)
        stream_subprocess(cmd, log_path)

        if not ckpt.exists():
            raise FileNotFoundError(f"Training finished but checkpoint not found: {ckpt}")

        trained.append({"variant": variant, "run_name": run_name, "ckpt": str(ckpt), "skipped": False})
        (runs_root / "stage1_train_log.json").write_text(json.dumps(trained, indent=2), encoding="utf-8")

    print(f"\nWrote: {runs_root / 'stage1_train_log.json'}")
    print("Done.")


if __name__ == "__main__":
    main()