#!/usr/bin/env python3
"""
Plot IQL performance vs training steps for the top-K augmentations (+ baseline).

This script is meant for step-sweep experiments where you have multiple run-groups
that differ only by `train_steps` (e.g., 6000/10000/20000), evaluated on the same
dataset root (e.g., `data/stage1_datasets_L40`).

Inputs supported:
  - `runs/<run_group>/stage1_iql_train_eval_summary/results.json`
  - `runs/<run_group>/stage1_iql_train_eval_summary/results.csv`

Example (after you have step-sweep run-groups on disk):
  python scripts/plot_iql_stepsweep_topk.py \
    --run-groups iql_L40_steps6000_seed0 iql_L40_steps10000_seed0 iql_L40_steps20000_seed0 \
    --steps 6000 10000 20000 \
    --top-k 5 \
    --out-dir results/plots/iql_stepsweep_L40_20kds
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _as_list(x: Optional[List[str]]) -> List[str]:
    return list(x) if x else []


def _discover_results_paths_from_run_groups(runs_root: Path, run_groups: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for rg in run_groups:
        summary = runs_root / rg / "stage1_iql_train_eval_summary"
        p_json = summary / "results.json"
        p_csv = summary / "results.csv"
        if p_json.exists():
            out.append(p_json)
        elif p_csv.exists():
            out.append(p_csv)
        else:
            raise FileNotFoundError(f"Could not find results.json/csv for run_group={rg} under {summary}")
    return out


def _discover_results_paths_from_globs(globs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in globs:
        for m in glob.glob(pat):
            p = Path(m)
            if p.name in {"results.json", "results.csv"}:
                paths.append(p)
    return paths


def _load_one_results_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise TypeError(f"Unexpected JSON shape in {path}: {type(data)}")
        return pd.DataFrame(data)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported results file: {path}")


def _normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["__source__"] = source

    if "algo" not in out.columns:
        out["algo"] = "IQL"

    if "seed" not in out.columns:
        if "train_seed" in out.columns:
            out["seed"] = out["train_seed"]
        else:
            out["seed"] = np.nan

    required = ["variant", "train_steps", "test_mean", "datasets_root", "algo"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")

    out["variant"] = out["variant"].astype(str)
    out["algo"] = out["algo"].astype(str)
    out["datasets_root"] = out["datasets_root"].astype(str)
    out["train_steps"] = out["train_steps"].astype(int)
    out["test_mean"] = out["test_mean"].astype(float)
    if "test_std" in out.columns:
        out["test_std"] = out["test_std"].astype(float)
    else:
        out["test_std"] = np.nan
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    return out


def _label_variant(v: str) -> str:
    # Keep names stable but slightly more readable in the legend.
    if v == "baseline_none":
        return "baseline"
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run-groups", nargs="*", default=None, help="Explicit run_groups under runs/.")
    ap.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob(s) that match results.json/results.csv (repeatable).",
    )
    ap.add_argument("--datasets-root-contains", default="stage1_datasets_L40")
    ap.add_argument("--steps", nargs="+", type=int, default=[6000, 10000, 20000])
    ap.add_argument("--rank-step", type=int, default=None, help="Step count used to rank top augmentations (default: max(steps)).")
    ap.add_argument("--top-k", type=int, default=5, help="Top-K augmentations to include (baseline added separately).")
    ap.add_argument("--baseline-variant", default="baseline_none")
    ap.add_argument("--variants", default=None, help="Comma-separated variant names to plot (overrides --top-k).")
    ap.add_argument("--out-dir", default="results/plots/iql_stepsweep")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    if args.run_groups:
        paths += _discover_results_paths_from_run_groups(runs_root, args.run_groups)
    paths += _discover_results_paths_from_globs(_as_list(args.glob))
    paths = sorted({p.resolve() for p in paths})
    if not paths:
        raise RuntimeError("No input results found. Provide --run-groups and/or --glob.")

    dfs = []
    for p in paths:
        df = _load_one_results_file(p)
        dfs.append(_normalize(df, source=str(p)))
    all_rows = pd.concat(dfs, ignore_index=True)

    all_rows = all_rows[all_rows["algo"].str.upper() == "IQL"]
    all_rows = all_rows[all_rows["datasets_root"].str.contains(args.datasets_root_contains)]
    if all_rows.empty:
        raise RuntimeError(
            f"No matching IQL rows after filtering datasets_root_contains={args.datasets_root_contains!r}."
        )

    steps = sorted(set(int(x) for x in args.steps))
    rank_step = int(args.rank_step) if args.rank_step is not None else max(steps)

    agg = (
        all_rows.groupby(["variant", "train_steps"], dropna=False)
        .agg(
            n=("test_mean", "count"),
            test_mean_mean=("test_mean", "mean"),
            test_mean_std_across_runs=("test_mean", "std"),
            test_std_mean=("test_std", "mean"),
        )
        .reset_index()
    )

    if args.variants:
        selected = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        rank_df = agg[agg["train_steps"] == rank_step].copy()
        if rank_df.empty:
            fallback = int(agg["train_steps"].max())
            rank_df = agg[agg["train_steps"] == fallback].copy()
            print(f"[warn] No rows at rank_step={rank_step}; ranking at train_steps={fallback} instead.")
            rank_step = fallback

        rank_df = rank_df[rank_df["variant"] != args.baseline_variant]
        rank_df = rank_df.sort_values("test_mean_mean", ascending=False).head(int(args.top_k))
        selected = [args.baseline_variant] + rank_df["variant"].tolist()

    selected = list(dict.fromkeys(selected))  # de-dupe, preserve order
    kept = agg[agg["variant"].isin(selected) & agg["train_steps"].isin(steps)].copy()

    # Write the aggregated table (useful for report tables).
    table_csv = out_dir / "iql_stepsweep_topk_table.csv"
    kept.sort_values(["train_steps", "test_mean_mean"], ascending=[True, False]).to_csv(table_csv, index=False)

    # Build plot arrays.
    plt.figure(figsize=(8.5, 5.0))
    for v in selected:
        ys = []
        yerr = []
        for s in steps:
            m = kept[(kept["variant"] == v) & (kept["train_steps"] == s)]
            if m.empty:
                ys.append(np.nan)
                yerr.append(np.nan)
            else:
                ys.append(float(m.iloc[0]["test_mean_mean"]))
                yerr.append(float(m.iloc[0]["test_mean_std_across_runs"]) if m.iloc[0]["n"] > 1 else np.nan)

        label = _label_variant(v)
        style = dict(marker="o", linewidth=2.0)
        if v == args.baseline_variant:
            style.update(dict(color="black", linestyle="--"))
        plt.plot(steps, ys, label=label, **style)

        # Optional: error bars across runs/seeds (if we have more than 1 point).
        if any(np.isfinite(y) and np.isfinite(e) and e > 0 for y, e in zip(ys, yerr)):
            plt.errorbar(steps, ys, yerr=yerr, fmt="none", ecolor=style.get("color", None), capsize=3, alpha=0.8)

    plt.xlabel("IQL training steps")
    plt.ylabel("Test mean return")
    if args.title:
        title = args.title
    else:
        title = f"IQL: top augmentations vs steps (rank@{rank_step})"
    plt.title(title)
    plt.xticks(steps, [f"{s//1000}k" if s % 1000 == 0 else str(s) for s in steps])
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_png = out_dir / "iql_stepsweep_topk.png"
    out_pdf = out_dir / "iql_stepsweep_topk.pdf"
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()

    missing_steps = sorted(set(steps) - set(int(x) for x in kept["train_steps"].unique()))
    if missing_steps:
        print(f"[warn] Missing any data for steps: {missing_steps} (plot will have gaps).")

    print(f"Wrote: {table_csv}")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
