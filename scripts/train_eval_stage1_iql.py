#!/usr/bin/env python3
"""
Train + evaluate IQL policies for all Stage-1 dataset variants.

This is intentionally separate from BC orchestration.
Outputs are written to:
  runs/<run_group>/stage1_iql_train_eval_summary/
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
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
    out = []
    for p in datasets_root.iterdir():
        if p.is_dir() and (p / "manifest.json").exists():
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def parse_eval(output: str) -> Dict[str, float]:
    m1 = TRAIN_RE.search(output)
    m2 = TEST_RE.search(output)
    if not m1 or not m2:
        raise RuntimeError(
            "Could not parse eval output. Expected TRAIN/TEST lines from eval_iql_procgen.py"
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
        return "# Stage-1 IQL Train+Eval Summary\n\nNo results.\n"

    headers = ["variant", "train_mean", "train_std", "test_mean", "test_std", "gen_gap", "ckpt"]
    lines = ["# Stage-1 IQL Train+Eval Summary\n", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
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
    seed: int,
    device: str,
    skip_existing: bool,
    hidden: int,
    gamma: float,
    expectile: float,
    beta: float,
    exp_adv_max: float,
    target_tau: float,
    lr_actor: float,
    lr_q: float,
    lr_v: float,
    train_script: str = "scripts/train_iql.py",
    logs_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[Path, bool, float]:
    run_name = f"{run_group}/iql_{variant}"
    ckpt = runs_root / run_name / "iql_ckpt.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and ckpt.exists():
        return ckpt, True, 0.0

    cmd = [
        sys.executable,
        train_script,
        "--dataset-root",
        str(vdir),
        "--run-name",
        run_name,
        "--steps",
        str(train_steps),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--device",
        str(device),
        "--hidden",
        str(hidden),
        "--gamma",
        str(gamma),
        "--expectile",
        str(expectile),
        "--beta",
        str(beta),
        "--exp-adv-max",
        str(exp_adv_max),
        "--target-tau",
        str(target_tau),
        "--lr-actor",
        str(lr_actor),
        "--lr-q",
        str(lr_q),
        "--lr-v",
        str(lr_v),
    ]

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
    eval_script: str = "scripts/eval_iql_procgen.py",
    logs_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, float], float]:
    cmd = [
        sys.executable,
        eval_script,
        "--ckpt",
        str(ckpt),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--train_start",
        str(train_start),
        "--train_levels",
        str(train_levels),
        "--test_start",
        str(test_start),
        "--test_levels",
        str(test_levels),
        "--distribution_mode",
        str(distribution_mode),
    ] + list(extra_eval_args)

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
    return parse_eval(out), elapsed


def write_variant_summary(variant_dir: Path, out_dir: Path, row: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{variant_dir.name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="data/stage1_datasets")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run-group", default=None)

    ap.add_argument("--train-steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")

    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--exp-adv-max", type=float, default=100.0)
    ap.add_argument("--target-tau", type=float, default=0.005)
    ap.add_argument("--lr-actor", type=float, default=3e-4)
    ap.add_argument("--lr-q", type=float, default=3e-4)
    ap.add_argument("--lr-v", type=float, default=3e-4)

    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--train_start", type=int, default=0)
    ap.add_argument("--train_levels", type=int, default=500)
    ap.add_argument("--test_start", type=int, default=500)
    ap.add_argument("--test_levels", type=int, default=500)
    ap.add_argument("--distribution_mode", default="hard", choices=["easy", "hard"])
    ap.add_argument("--extra-eval-args", default="")

    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    datasets_root = Path(args.datasets_root)
    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets-root not found: {datasets_root}")

    run_group = args.run_group or datasets_root.name
    group_root = runs_root / run_group
    group_root.mkdir(parents=True, exist_ok=True)

    summary_root = group_root / "stage1_iql_train_eval_summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    variants_dir = summary_root / "variants"
    logs_dir = summary_root / "logs"

    variants = discover_variants(datasets_root)

    if args.only:
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        variants = [v for v in variants if v.name in allow]

    if not variants:
        raise RuntimeError(f"No variants found under {datasets_root} (expected folders with manifest.json).")

    extra_eval_args = args.extra_eval_args.strip().split() if args.extra_eval_args.strip() else []

    print(f"\n=== DATASETS ROOT: {datasets_root}  (run_group={run_group}) ===")
    print(f"Variants: {len(variants)} -> {[v.name for v in variants]}")
    print(f"Outputs: {summary_root}")

    rows: List[Dict] = []
    for i, vdir in enumerate(variants, start=1):
        variant = vdir.name
        print(f"\n[{i}/{len(variants)}] variant={variant}")

        ckpt, skipped, train_elapsed = run_training(
            vdir=vdir,
            runs_root=runs_root,
            run_group=run_group,
            variant=variant,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            skip_existing=args.skip_existing,
            hidden=args.hidden,
            gamma=args.gamma,
            expectile=args.expectile,
            beta=args.beta,
            exp_adv_max=args.exp_adv_max,
            target_tau=args.target_tau,
            lr_actor=args.lr_actor,
            lr_q=args.lr_q,
            lr_v=args.lr_v,
            logs_dir=None if args.verbose else logs_dir,
            verbose=args.verbose,
        )

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
        )

        row = {
            "datasets_root": str(datasets_root),
            "run_group": run_group,
            "variant": variant,
            "ckpt": str(ckpt),
            "algo": "IQL",
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "hidden": args.hidden,
            "gamma": args.gamma,
            "expectile": args.expectile,
            "beta": args.beta,
            "exp_adv_max": args.exp_adv_max,
            "target_tau": args.target_tau,
            "lr_actor": args.lr_actor,
            "lr_q": args.lr_q,
            "lr_v": args.lr_v,
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

        write_variant_summary(vdir, variants_dir, row)
        (summary_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    rows_sorted = sorted(rows, key=lambda r: r["test_mean"], reverse=True)

    csv_path = summary_root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_sorted[0].keys()) if rows_sorted else ["variant"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

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
