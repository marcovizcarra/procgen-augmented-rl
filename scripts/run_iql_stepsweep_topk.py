#!/usr/bin/env python3
"""
Run an IQL step-sweep for a small set of variants (baseline + top-K augmentations).

This is a thin orchestrator around `scripts/train_eval_stage1_iql.py` that creates one
run-group per (steps, seed) so plotting is straightforward.

Example:
  python scripts/run_iql_stepsweep_topk.py \
    --datasets-root data/stage1_datasets_L40 \
    --runs-root runs \
    --steps 6000 10000 20000 \
    --seeds 0 1 2 \
    --top-k 5 \
    --distribution-mode hard \
    --episodes 200
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def _load_results(path: Path) -> List[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise TypeError(f"Unexpected JSON in {path}: {type(data)}")
    if path.suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"Unsupported file: {path}")


def _discover_results_files(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        for m in glob.glob(pat):
            p = Path(m)
            if p.name in {"results.json", "results.csv"} and p.is_file():
                paths.append(p)
    return sorted({p.resolve() for p in paths})


def _rank_topk_variants(
    results_files: Iterable[Path],
    datasets_root_contains: str,
    rank_step: int,
    baseline_variant: str,
    top_k: int,
) -> List[str]:
    rows: List[dict] = []
    for p in results_files:
        for r in _load_results(p):
            if str(r.get("algo", "")).upper() != "IQL":
                continue
            if datasets_root_contains not in str(r.get("datasets_root", "")):
                continue
            if int(r.get("train_steps", -1)) != int(rank_step):
                continue
            if "variant" not in r or "test_mean" not in r:
                continue
            rows.append(r)

    if not rows:
        raise RuntimeError(
            f"No IQL rows found for ranking at train_steps={rank_step} (datasets_root_contains={datasets_root_contains!r})."
        )

    # Aggregate across seeds/runs by mean test_mean, then pick top-K (excluding baseline).
    from collections import defaultdict

    by_variant = defaultdict(list)
    for r in rows:
        by_variant[str(r["variant"])].append(float(r["test_mean"]))

    scored = []
    for v, vals in by_variant.items():
        if v == baseline_variant:
            continue
        scored.append((sum(vals) / max(1, len(vals)), v))
    scored.sort(reverse=True)

    top = [v for _, v in scored[: int(top_k)]]
    return [baseline_variant] + top


def _run_train_eval(
    *,
    datasets_root: Path,
    runs_root: Path,
    run_group: str,
    train_steps: int,
    seed: int,
    only_variants: List[str],
    episodes: int,
    train_start: int,
    train_levels: int,
    test_start: int,
    test_levels: int,
    distribution_mode: str,
    device: str,
    batch_size: int,
    hidden: int,
    gamma: float,
    expectile: float,
    beta: float,
    exp_adv_max: float,
    target_tau: float,
    lr_actor: float,
    lr_q: float,
    lr_v: float,
    skip_existing: bool,
    verbose: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/train_eval_stage1_iql.py",
        "--datasets-root",
        str(datasets_root),
        "--runs-root",
        str(runs_root),
        "--run-group",
        run_group,
        "--train-steps",
        str(train_steps),
        "--batch-size",
        str(batch_size),
        "--device",
        str(device),
        "--seed",
        str(seed),
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
        str(distribution_mode),
        "--only",
        ",".join(only_variants),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if verbose:
        cmd.append("--verbose")

    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="data/stage1_datasets_L40")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--steps", nargs="+", type=int, default=[6000, 10000, 20000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--run-group-prefix", default="iql_stepsweep_L40")

    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--train_start", type=int, default=0)
    ap.add_argument("--train_levels", type=int, default=40)
    ap.add_argument("--test_start", type=int, default=40)
    ap.add_argument("--test_levels", type=int, default=500)
    ap.add_argument("--distribution-mode", default="hard", choices=["easy", "hard"])

    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--exp-adv-max", type=float, default=100.0)
    ap.add_argument("--target-tau", type=float, default=0.005)
    ap.add_argument("--lr-actor", type=float, default=3e-4)
    ap.add_argument("--lr-q", type=float, default=3e-4)
    ap.add_argument("--lr-v", type=float, default=3e-4)

    ap.add_argument("--baseline-variant", default="baseline_none")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rank-step", type=int, default=None)
    ap.add_argument("--datasets-root-contains", default="stage1_datasets_L40")
    ap.add_argument(
        "--rank-glob",
        action="append",
        default=["runs/*/stage1_iql_train_eval_summary/results.json", "runs/*/stage1_iql_train_eval_summary/results.csv"],
        help="Glob(s) used to find prior results for ranking (repeatable).",
    )
    ap.add_argument("--variants", default=None, help="Comma-separated variants to run (overrides --top-k ranking).")

    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    runs_root = Path(args.runs_root)
    steps = [int(s) for s in args.steps]
    seeds = [int(s) for s in args.seeds]
    rank_step = int(args.rank_step) if args.rank_step is not None else max(steps)

    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        rank_files = _discover_results_files(args.rank_glob)
        variants = _rank_topk_variants(
            rank_files,
            datasets_root_contains=args.datasets_root_contains,
            rank_step=rank_step,
            baseline_variant=args.baseline_variant,
            top_k=args.top_k,
        )

    variants = list(dict.fromkeys(variants))
    print(f"Selected variants (baseline + topK): {variants}")

    for seed in seeds:
        for train_steps in steps:
            run_group = f"{args.run_group_prefix}_steps{train_steps}_seed{seed}"
            _run_train_eval(
                datasets_root=datasets_root,
                runs_root=runs_root,
                run_group=run_group,
                train_steps=train_steps,
                seed=seed,
                only_variants=variants,
                episodes=int(args.episodes),
                train_start=int(args.train_start),
                train_levels=int(args.train_levels),
                test_start=int(args.test_start),
                test_levels=int(args.test_levels),
                distribution_mode=str(args.distribution_mode),
                device=str(args.device),
                batch_size=int(args.batch_size),
                hidden=int(args.hidden),
                gamma=float(args.gamma),
                expectile=float(args.expectile),
                beta=float(args.beta),
                exp_adv_max=float(args.exp_adv_max),
                target_tau=float(args.target_tau),
                lr_actor=float(args.lr_actor),
                lr_q=float(args.lr_q),
                lr_v=float(args.lr_v),
                skip_existing=bool(args.skip_existing),
                verbose=bool(args.verbose),
            )


if __name__ == "__main__":
    main()

