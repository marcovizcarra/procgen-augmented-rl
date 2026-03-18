#!/usr/bin/env python3
"""
Plot test-mean curves for the same variants at multiple budgets (e.g., 2k / 6k / 20k).

Each `--step-config` entry should look like `6000=runs/…/results.json`.
The script averages over all rows for a variant at a given budget before plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))


def load_results(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def parse_step_config(entry: str) -> Tuple[int, Path]:
    if "=" not in entry:
        raise ValueError(f"Expected <steps>=<path>, got {entry!r}")
    step_str, path = entry.split("=", 1)
    return int(step_str), Path(path)


def normalize_df(df: pd.DataFrame, step_label: int) -> pd.DataFrame:
    out = df.copy()
    out["step_label"] = int(step_label)
    if "test_mean" not in out.columns:
        raise ValueError("Results missing test_mean column.")
    if "variant" not in out.columns:
        raise ValueError("Results missing variant column.")
    out["variant"] = out["variant"].astype(str)
    out["test_mean"] = out["test_mean"].astype(float)
    if "test_std" in out.columns:
        out["test_std"] = out["test_std"].astype(float)
    else:
        out["test_std"] = np.nan
    return out[["step_label", "variant", "test_mean", "test_std"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--step-config",
        action="append",
        required=True,
        help="Budget label (e.g., 2000) and path to results.json/csv: 2000=runs/.../results.json",
    )
    ap.add_argument(
        "--variants",
        default="baseline_none,shift4_noise_0p02,shift4_scale_0p8_1p2,shift4_scale_0p9_1p1",
        help="Comma-separated variants to plot.",
    )
    ap.add_argument("--out-dir", default="results/plots/iql_budget_comparison")
    ap.add_argument("--title", default=None)
    ap.add_argument("--log-scale", action="store_true", help="Use log scale on y-axis.")
    args = ap.parse_args()

    configs: List[Tuple[int, Path]] = [parse_step_config(e) for e in args.step_config]
    dfs: List[pd.DataFrame] = []
    for step, path in configs:
        if not path.exists():
            raise FileNotFoundError(f"Results path missing: {path}")
        df = load_results(path)
        dfs.append(normalize_df(df, step))
    combined = pd.concat(dfs, ignore_index=True)

    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    combined = combined[combined["variant"].isin(wanted)]
    if combined.empty:
        raise RuntimeError(f"No rows matched variants={wanted}.")

    agg = (
        combined.groupby(["step_label", "variant"], sort=True, as_index=False)
        .agg(test_mean=("test_mean", "mean"), test_std=("test_std", "mean"), n=("test_mean", "count"))
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "iql_budget_comparison_table.csv"
    agg.to_csv(csv_path, index=False)

    plt.figure(figsize=(8.5, 5.0))
    steps_sorted = sorted(agg["step_label"].unique())
    for variant in wanted:
        row = agg[agg["variant"] == variant].set_index("step_label")
        means = []
        stds = []
        for s in steps_sorted:
            if s not in row.index:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(row.loc[s, "test_mean"]))
                stds.append(float(row.loc[s, "test_std"]))
        plt.plot(
            steps_sorted,
            means,
            marker="o",
            linewidth=2,
            label=variant.replace("shift4_", "").replace("_", " "),
        )
        if not all(np.isnan(stds)):
            plt.fill_between(
                steps_sorted,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.2,
            )

    plt.xlabel("Training steps")
    plt.ylabel("Test-return mean")
    plt.title(args.title or "IQL variants across budgets")
    if args.log_scale:
        plt.yscale("log")
    plt.xticks(steps_sorted, [f"{int(s):,}" for s in steps_sorted])
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    png_path = out_dir / "iql_budget_comparison.png"
    pdf_path = out_dir / "iql_budget_comparison.pdf"
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Wrote table: {csv_path}")
    print(f"Wrote PNG: {png_path}")
    print(f"Wrote PDF: {pdf_path}")


if __name__ == "__main__":
    main()
