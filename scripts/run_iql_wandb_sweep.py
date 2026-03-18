#!/usr/bin/env python3
"""
W&B sweep agent entrypoint for IQL hyperparameter tuning.

Each agent run launches one train+eval job via train_eval_stage1_iql.py,
then logs summary metrics back to the sweep run.
"""

import csv
import subprocess
import sys
from pathlib import Path

import wandb


def main() -> None:
    run = wandb.init()
    cfg = wandb.config

    datasets_root = str(getattr(cfg, "datasets_root", "data/stage1_datasets_L40"))
    runs_root = str(getattr(cfg, "runs_root", "runs"))
    variant = str(getattr(cfg, "variant", "baseline_none"))
    train_steps = int(getattr(cfg, "train_steps", 20000))
    batch_size = int(getattr(cfg, "batch_size", 256))
    hidden = int(getattr(cfg, "hidden", 256))
    expectile = float(getattr(cfg, "expectile", 0.8))
    beta = float(getattr(cfg, "beta", 3.0))
    target_tau = float(getattr(cfg, "target_tau", 0.005))
    lr = float(getattr(cfg, "lr", 3e-4))
    seed = int(getattr(cfg, "seed", 0))
    episodes = int(getattr(cfg, "episodes", 200))
    train_start = int(getattr(cfg, "train_start", 0))
    train_levels = int(getattr(cfg, "train_levels", 40))
    test_start = int(getattr(cfg, "test_start", 40))
    test_levels = int(getattr(cfg, "test_levels", 500))
    distribution_mode = str(getattr(cfg, "distribution_mode", "hard"))

    run_group_prefix = str(getattr(cfg, "run_group_prefix", "wandb_iql_sweep"))
    run_group = f"{run_group_prefix}_{run.id}"

    cmd = [
        sys.executable,
        "-B",
        "scripts/train_eval_stage1_iql.py",
        "--datasets-root",
        datasets_root,
        "--runs-root",
        runs_root,
        "--run-group",
        run_group,
        "--only",
        variant,
        "--train-steps",
        str(train_steps),
        "--batch-size",
        str(batch_size),
        "--hidden",
        str(hidden),
        "--expectile",
        str(expectile),
        "--beta",
        str(beta),
        "--target-tau",
        str(target_tau),
        "--lr-actor",
        str(lr),
        "--lr-q",
        str(lr),
        "--lr-v",
        str(lr),
        "--episodes",
        str(episodes),
        "--train_start",
        str(train_start),
        "--train_levels",
        str(train_levels),
        "--test_start",
        str(test_start),
        "--test_levels",
        str(test_levels),
        "--distribution_mode",
        distribution_mode,
        "--seed",
        str(seed),
    ]

    subprocess.check_call(cmd)

    results_csv = Path(runs_root) / run_group / "stage1_iql_train_eval_summary" / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Expected results.csv not found: {results_csv}")

    with results_csv.open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    metrics = {
        "train_mean": float(row["train_mean"]),
        "train_std": float(row["train_std"]),
        "test_mean": float(row["test_mean"]),
        "test_std": float(row["test_std"]),
        "gen_gap": float(row["gen_gap"]),
    }
    wandb.log(metrics)
    run.summary["run_group"] = run_group
    run.summary["results_csv"] = str(results_csv)
    run.finish()


if __name__ == "__main__":
    main()
