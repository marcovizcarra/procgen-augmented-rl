#!/usr/bin/env python3
"""
W&B sweep wrapper for CQL: train + online eval, then log sweep metric.

We use this wrapper to do hyperparameter tuning for CQL.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

try:
    import wandb
except Exception:
    wandb = None


TEST_RE = re.compile(
    r"TEST\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)


def parse_eval_output(output: str) -> Dict[str, float]:
    m = TEST_RE.search(output)
    if not m:
        raise RuntimeError("Could not parse eval output from file")
    test_mean, test_std = float(m.group(1)), float(m.group(2))
    return {
        "test_mean": test_mean,
        "test_std": test_std,
    }


def sanitize_token(v: object) -> str:
    s = str(v)
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    s = s.replace("/", "_")
    s = s.replace(" ", "_")
    return s


def to_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True, default=None)
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--run-group", type=str, default="cql_wandb_sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--n-actions", type=int, default=15)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--target-tau", type=float, default=0.005)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--min-q-version", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--lr-cql-alpha", type=float, default=1e-4)

    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--test_start", type=int, default=40)
    p.add_argument("--test_levels", type=int, default=500)
    p.add_argument("--distribution_mode", type=str, default="hard", choices=["easy", "hard"])
    p.add_argument("--extra-eval-args", type=str, default="")

    p.add_argument("--wandb-project", type=str, default="procgen-augmented-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", type=str, default="cql,sweep")

    p.add_argument("--train-script", type=str, default="scripts/cql/train_cql.py")
    p.add_argument("--eval-script", type=str, default="scripts/cql/eval_cql_procgen.py")
    args = p.parse_args()

    default_cfg = {
        "cql_alpha": 5.0,
        "cql_lagrange": False,
        "lagrange_thresh": 10.0,
        "learning_rate": 3e-5,
        "batch_size": 256,
        "gamma": 0.99,
    }
    use_wandb = wandb is not None
    if not use_wandb and args.wandb_mode != "disabled":
        raise RuntimeError("wandb is not installed. Install it, or run with --wandb-mode disabled.")

    run: Optional[object] = None
    if use_wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            tags=tags or None,
            config=default_cfg,
        )
        assert run is not None
        cfg = wandb.config
        run_id = str(run.id)
    else:
        cfg = SimpleNamespace(**default_cfg)
        run_id = "no_wandb"

    start_all = time.time()
    try:
        cql_alpha = float(cfg.cql_alpha)
        with_lagrange = to_bool(cfg.cql_lagrange)
        lagrange_thresh = float(cfg.lagrange_thresh)
        lr_q = float(cfg.learning_rate)
        batch_size = int(cfg.batch_size)
        gamma = float(cfg.gamma)

        trial_name = (
            f"sweep_a{sanitize_token(cql_alpha)}"
            f"_lag{int(with_lagrange)}"
            f"_thr{sanitize_token(lagrange_thresh)}"
            f"_lr{sanitize_token(lr_q)}"
            f"_bs{batch_size}"
            f"_s{args.seed}_{run_id}"
        )
        run_name = f"{args.run_group}/{trial_name}"
        ckpt_path = Path(args.runs_root) / run_name / "cql_ckpt.pt"

        train_cmd: List[str] = [
            sys.executable,
            args.train_script,
            "--dataset-root",
            str(args.dataset_root),
            "--run-name",
            run_name,
            "--steps",
            str(args.train_steps),
            "--batch-size",
            str(batch_size),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--n-actions",
            str(args.n_actions),
            "--hidden",
            str(args.hidden),
            "--gamma",
            str(gamma),
            "--target-tau",
            str(args.target_tau),
            "--cql-alpha",
            str(cql_alpha),
            "--temp",
            str(args.temp),
            "--min-q-version",
            str(args.min_q_version),
            "--lagrange-thresh",
            str(lagrange_thresh),
            "--lr-q",
            str(lr_q),
            "--lr-cql-alpha",
            str(args.lr_cql_alpha),
        ]
        if with_lagrange:
            train_cmd.append("--with-lagrange")

        print("$ " + " ".join(train_cmd), flush=True)
        t0 = time.time()
        subprocess.check_call(train_cmd)
        train_elapsed = time.time() - t0

        extra_eval_args = args.extra_eval_args.strip().split() if args.extra_eval_args.strip() else []
        eval_cmd: List[str] = [
            sys.executable,
            args.eval_script,
            "--ckpt",
            str(ckpt_path),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--test_start",
            str(args.test_start),
            "--test_levels",
            str(args.test_levels),
            "--distribution_mode",
            str(args.distribution_mode),
        ] + extra_eval_args

        print("$ " + " ".join(eval_cmd), flush=True)
        t1 = time.time()
        out = subprocess.check_output(eval_cmd, stderr=subprocess.STDOUT, text=True)
        eval_elapsed = time.time() - t1
        print(out, flush=True)

        metrics = parse_eval_output(out)
        total_elapsed = time.time() - start_all
        payload = {
            "eval/average_return": float(metrics["test_mean"]),
            "eval/test_mean_return": float(metrics["test_mean"]),
            "eval/test_std_return": float(metrics["test_std"]),
            "train/elapsed_sec": float(train_elapsed),
            "eval/elapsed_sec": float(eval_elapsed),
            "trial/elapsed_sec": float(total_elapsed),
        }
        if use_wandb:
            wandb.log(payload)
            run.summary["run_name"] = run_name
            run.summary["ckpt_path"] = str(ckpt_path)
            run.summary["dataset_root"] = str(args.dataset_root)
        else:
            print("Smoke metrics:", payload, flush=True)
    except subprocess.CalledProcessError as e:
        if use_wandb:
            wandb.log({"error/subprocess_failed": 1})
        if e.output:
            print(e.output, flush=True)
        raise
    finally:
        if use_wandb and run is not None:
            run.finish()


if __name__ == "__main__":
    main()
