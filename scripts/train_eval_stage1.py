#!/usr/bin/env python3
"""
scripts/train_eval_stage1.py

Train + evaluate Behavior Cloning (BC) policies for ALL Stage-1 dataset variants.

What this script does
---------------------
For each dataset root you pass (e.g., data/stage1_datasets_L40):
  1) Discovers variant folders containing manifest.json:
       <datasets_root>/<variant>/manifest.json
  2) Trains a BC model (scripts/train_bc_min.py) per variant:
       runs/<run_group>/bc_<variant>/bc_ckpt.pt
     where run_group defaults to the dataset root folder name
       datasets_root = data/stage1_datasets_L40  -> run_group = stage1_datasets_L40
  3) Evaluates each checkpoint (scripts/eval_procgen.py) and parses:
       TRAIN mean/std, TEST mean/std, gen_gap = test_mean - train_mean
  4) Writes per-variant summaries + an aggregate summary:
       runs/<run_group>/stage1_train_eval_summary/
         results.json
         results.csv
         results.md
         variants/
           <variant>.json
           <variant>.md
         logs/
           <variant>_train.log
           <variant>_eval.log

Stage-1 experiment set (baseline + augmentations)
------------------------------------------------
This script auto-discovers what exists under your datasets root.
Typical Stage-1 variants you already created include:

  baseline_none
  shift_pad4
  shift_pad8
  shift4_blur3x3
  shift4_cutout16
  shift4_cutout24
  shift4_jitter_0p2
  shift4_jitter_0p4
  shift4_noise_0p02
  shift4_paug_0p5
  shift4_scale_0p8_1p2
  shift4_scale_0p9_1p1

Recommended eval splits
-----------------------
For level_40 datasets (coinrun-level_40_E), a common split is:
  TRAIN: start=0,  num_levels=40
  TEST : start=40, num_levels=500

Usage examples
--------------
# (A) Run on one dataset root (auto run_group = folder name)
python -B scripts/train_eval_stage1.py \
  --datasets-root data/stage1_datasets_L40 \
  --train-steps 20000 \
  --batch-size 256 \
  --episodes 200 \
  --train_start 0 --train_levels 40 \
  --test_start 40 --test_levels 500 \
  --seed 0 \
  --skip-existing

# (B) Run on BOTH dataset roots you have (comma-separated)
python -B scripts/train_eval_stage1.py \
  --datasets-root data/stage1_datasets,data/stage1_datasets_L40 \
  --train-steps 20000 \
  --batch-size 256 \
  --episodes 200 \
  --seed 0 \
  --skip-existing

# (C) Only run a subset of variants
python -B scripts/train_eval_stage1.py \
  --datasets-root data/stage1_datasets_L40 \
  --only baseline_none,shift_pad4,shift4_scale_0p9_1p1 \
  --train-steps 20000 \
  --episodes 200 \
  --train_start 0 --train_levels 40 \
  --test_start 40 --test_levels 500
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TRAIN_RE = re.compile(
    r"TRAIN\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)
TEST_RE = re.compile(
    r"TEST\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)


def discover_variants(datasets_root: Path) -> List[Path]:
    if not datasets_root.exists():
        return []
    out: List[Path] = []
    for p in datasets_root.iterdir():
        if p.is_dir() and (p / "manifest.json").exists():
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def parse_eval(output: str) -> Dict[str, float]:
    m1 = TRAIN_RE.search(output)
    m2 = TEST_RE.search(output)
    if not m1 or not m2:
        raise RuntimeError(
            "Could not parse eval output.\n"
            "Expected lines like:\n"
            "  TRAIN levels: mean_return=... std=... (episodes=...)\n"
            "  TEST  levels: mean_return=... std=... (episodes=...)"
        )

    train_mean, train_std, train_eps = float(m1.group(1)), float(m1.group(2)), int(m1.group(3))
    test_mean, test_std, test_eps = float(m2.group(1)), float(m2.group(2)), int(m2.group(3))
    return {
        "train_mean": train_mean,
        "train_std": train_std,
        "train_episodes": train_eps,
        "test_mean": test_mean,
        "test_std": test_std,
        "test_episodes": test_eps,
        "gen_gap": test_mean - train_mean,
    }


def to_markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "# Stage-1 Train+Eval Summary\n\nNo results.\n"

    headers = ["variant", "train_mean", "train_std", "test_mean", "test_std", "gen_gap", "ckpt"]
    lines = []
    lines.append("# Stage-1 Train+Eval Summary\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("variant", "")),
                    f'{r.get("train_mean", float("nan")):.3f}',
                    f'{r.get("train_std", float("nan")):.3f}',
                    f'{r.get("test_mean", float("nan")):.3f}',
                    f'{r.get("test_std", float("nan")):.3f}',
                    f'{r.get("gen_gap", float("nan")):.3f}',
                    str(r.get("ckpt", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def run_training(
    vdir: Path,
    runs_root: Path,
    run_group: str,
    variant: str,
    train_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    skip_existing: bool,
    train_script: str = "scripts/train_bc_min.py",
    logs_dir: Optional[Path] = None,
    verbose: bool = False,
    wandb: bool = False,
    wandb_project: str = "procgen-augmented-rl",
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
    wandb_tags: str = "",
) -> Tuple[Path, bool, float]:
    """
    Returns: (ckpt_path, skipped, elapsed_sec)
    """
    run_name = f"{run_group}/bc_{variant}"
    ckpt = runs_root / run_name / "bc_ckpt.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and ckpt.exists():
        return ckpt, True, 0.0

    cmd = [
        sys.executable, train_script,
        "--dataset-root", str(vdir),
        "--run-name", run_name,
        "--steps", str(train_steps),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--seed", str(seed),
        "--device", str(device),
    ]
    if wandb:
        cmd += [
            "--wandb",
            "--wandb-project", str(wandb_project),
            "--wandb-mode", str(wandb_mode),
        ]
        if wandb_entity:
            cmd += ["--wandb-entity", str(wandb_entity)]
        if wandb_tags:
            cmd += ["--wandb-tags", str(wandb_tags)]

    start = time.time()
    if verbose:
        print("\n$ " + " ".join(cmd), flush=True)
        ret = subprocess.call(cmd)
        if ret != 0:
            raise RuntimeError(f"Training failed for variant={variant} (exit code={ret}).")
    else:
        assert logs_dir is not None
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{variant}_train.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n\n")
            ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        if ret != 0:
            raise RuntimeError(f"Training failed for variant={variant}. See log: {log_path}")

    elapsed = time.time() - start
    if not ckpt.exists():
        raise FileNotFoundError(f"Training finished but checkpoint not found: {ckpt}")
    return ckpt, False, elapsed


def run_evaluation(
    ckpt: Path,
    episodes: int,
    seed: int,
    train_start: int,
    train_levels: int,
    test_start: int,
    test_levels: int,
    distribution_mode: str,
    extra_eval_args: List[str],
    eval_script: str = "scripts/eval_procgen.py",
    logs_dir: Optional[Path] = None,
    verbose: bool = False,
    wandb: bool = False,
    wandb_project: str = "procgen-augmented-rl",
    wandb_entity: Optional[str] = None,
    wandb_mode: str = "online",
    wandb_tags: str = "",
) -> Tuple[Dict[str, float], float]:
    """
    Returns: (metrics_dict, elapsed_sec)
    """
    cmd = [
        sys.executable, eval_script,
        "--ckpt", str(ckpt),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--train_start", str(train_start),
        "--train_levels", str(train_levels),
        "--test_start", str(test_start),
        "--test_levels", str(test_levels),
        "--distribution_mode", str(distribution_mode),
    ] + list(extra_eval_args)
    if wandb:
        cmd += [
            "--wandb",
            "--wandb-project", str(wandb_project),
            "--wandb-mode", str(wandb_mode),
            "--wandb-run-name", f"eval_{ckpt.parent.name}",
        ]
        if wandb_entity:
            cmd += ["--wandb-entity", str(wandb_entity)]
        if wandb_tags:
            cmd += ["--wandb-tags", str(wandb_tags)]

    start = time.time()
    if verbose:
        print("\n$ " + " ".join(cmd), flush=True)
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        print(out, flush=True)
    else:
        assert logs_dir is not None
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{ckpt.parent.name}_eval.log"
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        log_path.write_text("$ " + " ".join(cmd) + "\n\n" + out, encoding="utf-8")

    elapsed = time.time() - start
    metrics = parse_eval(out)
    return metrics, elapsed


def write_variant_summary(variant_dir: Path, out_dir: Path, row: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{variant_dir.name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    # small human-readable md
    md = []
    md.append(f"# {variant_dir.name}\n")
    md.append(f"- dataset_root: `{row.get('datasets_root','')}`")
    md.append(f"- run_group: `{row.get('run_group','')}`")
    md.append(f"- ckpt: `{row.get('ckpt','')}`")
    md.append("")
    md.append("## Train")
    md.append(f"- steps: {row.get('train_steps')}")
    md.append(f"- batch_size: {row.get('batch_size')}")
    md.append(f"- lr: {row.get('lr')}")
    md.append(f"- seed: {row.get('seed')}")
    md.append(f"- elapsed_sec: {row.get('train_elapsed_sec'):.2f}")
    md.append(f"- skipped: {row.get('train_skipped')}")
    md.append("")
    md.append("## Eval")
    md.append(f"- episodes: {row.get('episodes')}")
    md.append(f"- split: train_start={row.get('train_start')}, train_levels={row.get('train_levels')}, "
              f"test_start={row.get('test_start')}, test_levels={row.get('test_levels')}")
    md.append(f"- distribution_mode: {row.get('distribution_mode')}")
    md.append(f"- train_mean/std: {row.get('train_mean'):.3f} / {row.get('train_std'):.3f}")
    md.append(f"- test_mean/std: {row.get('test_mean'):.3f} / {row.get('test_std'):.3f}")
    md.append(f"- gen_gap (test-train): {row.get('gen_gap'):.3f}")
    md.append(f"- eval_elapsed_sec: {row.get('eval_elapsed_sec'):.2f}")
    md.append("")
    (out_dir / f"{variant_dir.name}.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets-root",
        default="data/stage1_datasets",
        help="One or more dataset roots (comma-separated), each containing variant folders with manifest.json.",
    )
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument(
        "--run-group",
        default=None,
        help="Optional override group folder under runs-root. If multiple datasets-root are given, this applies to ALL.",
    )

    # training
    ap.add_argument("--train-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")

    # eval
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--train_start", type=int, default=0)
    ap.add_argument("--train_levels", type=int, default=500)
    ap.add_argument("--test_start", type=int, default=500)
    ap.add_argument("--test_levels", type=int, default=500)
    ap.add_argument("--distribution_mode", default="hard", choices=["easy", "hard"])
    ap.add_argument("--extra-eval-args", default="", help='Extra args forwarded to eval_procgen.py, e.g. "--device mps".')

    # selection / behavior
    ap.add_argument("--only", default=None, help="Comma-separated list of variants to run.")
    ap.add_argument("--skip-missing", action="store_true", help="Skip missing variants/ckpts instead of failing.")
    ap.add_argument("--verbose", action="store_true", help="Stream train/eval output to terminal instead of log files.")
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases for spawned train/eval scripts.")
    ap.add_argument("--wandb-project", default="procgen-augmented-rl")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-tags", default="", help="Comma-separated tags forwarded to train/eval scripts.")

    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    dataset_roots = [Path(x.strip()) for x in args.datasets_root.split(",") if x.strip()]
    if not dataset_roots:
        raise RuntimeError("No --datasets-root specified.")

    allow: Optional[set] = None
    if args.only:
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        if not allow:
            allow = None

    extra_eval_args = args.extra_eval_args.strip().split() if args.extra_eval_args.strip() else []

    for datasets_root in dataset_roots:
        if not datasets_root.exists():
            if args.skip_missing:
                print(f"[skip missing datasets-root] {datasets_root}")
                continue
            raise FileNotFoundError(f"datasets-root not found: {datasets_root}")

        run_group = args.run_group or datasets_root.name
        group_root = runs_root / run_group
        group_root.mkdir(parents=True, exist_ok=True)  # ✅ ensure runs/<run_group> exists

        # summary output (per dataset root)
        summary_root = group_root / "stage1_train_eval_summary"
        summary_root.mkdir(parents=True, exist_ok=True)
        variants_dir = summary_root / "variants"
        logs_dir = summary_root / "logs"

        variants = discover_variants(datasets_root)
        if allow is not None:
            variants = [v for v in variants if v.name in allow]

        if not variants:
            msg = f"No variants found under {datasets_root} (expected folders with manifest.json)."
            if args.skip_missing:
                print("[skip] " + msg)
                continue
            raise RuntimeError(msg)

        print(f"\n=== DATASETS ROOT: {datasets_root}  (run_group={run_group}) ===")
        print(f"Variants: {len(variants)} -> {[v.name for v in variants]}")
        print(f"Outputs: {summary_root}")

        rows: List[Dict] = []
        for i, vdir in enumerate(variants, start=1):
            variant = vdir.name
            print(f"\n[{i}/{len(variants)}] variant={variant}")

            # Train
            ckpt, skipped, train_elapsed = run_training(
                vdir=vdir,
                runs_root=runs_root,
                run_group=run_group,
                variant=variant,
                train_steps=args.train_steps,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
                skip_existing=args.skip_existing,
                logs_dir=None if args.verbose else logs_dir,
                verbose=args.verbose,
                wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_mode=args.wandb_mode,
                wandb_tags=args.wandb_tags,
            )

            # Eval
            metrics, eval_elapsed = run_evaluation(
                ckpt=ckpt,
                episodes=args.episodes,
                seed=args.seed,
                train_start=args.train_start,
                train_levels=args.train_levels,
                test_start=args.test_start,
                test_levels=args.test_levels,
                distribution_mode=args.distribution_mode,
                extra_eval_args=extra_eval_args,
                logs_dir=None if args.verbose else logs_dir,
                verbose=args.verbose,
                wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_mode=args.wandb_mode,
                wandb_tags=args.wandb_tags,
            )

            row = {
                "datasets_root": str(datasets_root),
                "run_group": run_group,
                "variant": variant,
                "ckpt": str(ckpt),

                "train_steps": args.train_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "device": args.device,
                "train_skipped": skipped,
                "train_elapsed_sec": float(train_elapsed),

                "episodes": args.episodes,
                "train_start": args.train_start,
                "train_levels": args.train_levels,
                "test_start": args.test_start,
                "test_levels": args.test_levels,
                "distribution_mode": args.distribution_mode,
                "eval_elapsed_sec": float(eval_elapsed),

                **metrics,
            }

            rows.append(row)

            # per-variant summary
            write_variant_summary(vdir, variants_dir, row)

            # incremental aggregate
            (summary_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

        # Aggregate outputs
        rows_sorted = sorted(rows, key=lambda r: r["test_mean"], reverse=True) if rows else []

        # CSV
        csv_path = summary_root / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows_sorted[0].keys()) if rows_sorted else ["variant"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)

        # Markdown table
        md_path = summary_root / "results.md"
        md_path.write_text(to_markdown_table(rows_sorted), encoding="utf-8")

        print(f"\nSaved JSON: {summary_root / 'results.json'}")
        print(f"Saved CSV : {csv_path}")
        print(f"Saved MD  : {md_path}")
        print(f"Per-variant summaries: {variants_dir}")
        if not args.verbose:
            print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()
