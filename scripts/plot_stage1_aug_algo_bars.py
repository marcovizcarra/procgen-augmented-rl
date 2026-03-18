#!/usr/bin/env python3
"""
Grouped bar chart per augmentation variant comparing BC / IQL / CQL.

Input format: the Stage-1 orchestrator CSVs produced by:
  - scripts/train_eval_stage1.py (BC)
  - scripts/train_eval_stage1_iql.py (IQL)
  - scripts/train_eval_stage1_cql.py (CQL)

Each CSV is expected to contain at least:
  variant, test_mean, test_std

Example (current repo layout):
  MPLCONFIGDIR=.mplconfig /path/to/python scripts/plot_stage1_aug_algo_bars.py \
    --bc-csv runs/bc_run/results.csv \
    --iql-csv runs/iql_best_vs_bc_L40_top3_seed0_steps20k/stage1_iql_train_eval_summary/results.csv \
    --iql-csv runs/iql_best_vs_bc_L40_rest8_seed0_steps20k/stage1_iql_train_eval_summary/results.csv \
    --cql-csv runs/<your_cql_run_group>/stage1_cql_train_eval_summary/results.csv \
    --out-dir results/comparison/aug_bars \
    --exclude baseline_none
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Row:
    variant: str
    mean: float
    std: float
    source: str


def _to_float(x: object) -> float:
    s = "" if x is None else str(x).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def load_results_csv(path: Path, label: str) -> Dict[str, Row]:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    out: Dict[str, Row] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        dr = csv.DictReader(f)
        for r in dr:
            v = (r.get("variant") or "").strip()
            if not v:
                continue
            out[v] = Row(
                variant=v,
                mean=_to_float(r.get("test_mean")),
                std=_to_float(r.get("test_std")),
                source=label,
            )
    if not out:
        raise RuntimeError(f"{label} CSV had no rows with a 'variant' column: {path}")
    return out


def merge_variants(
    bc: Dict[str, Row],
    iql_maps: Sequence[Dict[str, Row]],
    cql_maps: Sequence[Dict[str, Row]],
    exclude: Sequence[str],
) -> Tuple[List[str], Dict[str, Dict[str, Row]]]:
    variants = set(bc.keys())
    for m in iql_maps:
        variants |= set(m.keys())
    for m in cql_maps:
        variants |= set(m.keys())
    for v in exclude:
        variants.discard(v)

    # Stable-ish ordering: sort by BC test_mean desc when available, else name.
    def sort_key(v: str) -> Tuple[int, float, str]:
        if v in bc and not math.isnan(bc[v].mean):
            return (0, -bc[v].mean, v)
        return (1, 0.0, v)

    ordered = sorted(variants, key=sort_key)

    merged: Dict[str, Dict[str, Row]] = {}
    for v in ordered:
        merged[v] = {}
        if v in bc:
            merged[v]["BC"] = bc[v]
        for m in iql_maps:
            if v in m:
                merged[v]["IQL"] = m[v]
        for m in cql_maps:
            if v in m:
                merged[v]["CQL"] = m[v]
    return ordered, merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-csv", required=True)
    ap.add_argument("--iql-csv", action="append", default=[], help="Repeatable; can pass multiple IQL results.csv files.")
    ap.add_argument("--cql-csv", action="append", default=[], help="Repeatable; can pass multiple CQL results.csv files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude", action="append", default=["baseline_none"], help="Variant(s) to exclude (repeatable).")
    ap.add_argument("--title", default="Stage-1: Algorithm Performance by Augmentation (Test Mean)")
    args = ap.parse_args()

    bc = load_results_csv(Path(args.bc_csv), "BC")

    if not args.iql_csv:
        raise RuntimeError("Need at least one --iql-csv (IQL results.csv path).")
    iql_maps = [load_results_csv(Path(p), f"IQL[{Path(p).parent.parent.name}]") for p in args.iql_csv]

    cql_maps: List[Dict[str, Row]] = []
    for p in args.cql_csv:
        cql_maps.append(load_results_csv(Path(p), f"CQL[{Path(p).parent.parent.name}]"))

    variants, merged = merge_variants(bc, iql_maps, cql_maps, exclude=args.exclude)

    # Write merged table for debugging/poster.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = out_dir / "aug_algo_merged.csv"
    with merged_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant",
            "bc_test_mean",
            "bc_test_std",
            "iql_test_mean",
            "iql_test_std",
            "cql_test_mean",
            "cql_test_std",
            "bc_source",
            "iql_source",
            "cql_source",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for v in variants:
            row = {"variant": v}
            for algo in ("BC", "IQL", "CQL"):
                r = merged[v].get(algo)
                prefix = algo.lower()
                row[f"{prefix}_test_mean"] = "" if r is None or math.isnan(r.mean) else f"{r.mean:.6g}"
                row[f"{prefix}_test_std"] = "" if r is None or math.isnan(r.std) else f"{r.std:.6g}"
                row[f"{prefix}_source"] = "" if r is None else r.source
            w.writerow(row)

    # Plot.
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to plot. Run this with your conda env python (procgen-rl-x64)."
        ) from e

    algos = ["BC", "IQL", "CQL"]
    present_algos = [a for a in algos if any(a in merged[v] for v in variants)]

    if "CQL" not in present_algos:
        print("WARNING: No CQL results provided/found; plotting only BC + IQL.", flush=True)

    import numpy as np  # type: ignore

    x = np.arange(len(variants))
    width = 0.25 if len(present_algos) == 3 else 0.35
    offsets = {
        1: [0.0],
        2: [-width / 2, width / 2],
        3: [-width, 0.0, width],
    }[len(present_algos)]

    plt.figure(figsize=(max(12, 0.9 * len(variants)), 6))
    for algo, off in zip(present_algos, offsets):
        means = []
        stds = []
        for v in variants:
            r = merged[v].get(algo)
            means.append(float("nan") if r is None else r.mean)
            stds.append(float("nan") if r is None else r.std)
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        plt.bar(x + off, means_arr, width=width, yerr=stds_arr, capsize=2, label=algo)

    plt.xticks(x, variants, rotation=45, ha="right")
    plt.ylabel("Return (test_mean)")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / "aug_algo_bars.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Wrote: {merged_csv}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()

