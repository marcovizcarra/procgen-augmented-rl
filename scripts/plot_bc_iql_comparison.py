#!/usr/bin/env python3
"""
Plot BC vs IQL comparison from two Stage-1 results CSV files.

Expected columns in both files:
- variant
- train_mean
- test_mean
- gen_gap

Example:
  python3 scripts/plot_bc_iql_comparison.py \
    --bc-csv runs/bc_run/results.csv \
    --iql-csv runs/compare_bc_iql_L40_seed0_steps20k_ep200/stage1_iql_train_eval_summary/results.csv \
    --out-dir runs/compare_bc_iql_L40_seed0_steps20k_ep200/comparison_plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED = ["variant", "train_mean", "test_mean", "gen_gap"]


def load_results(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV missing columns {missing}: {path}")
    return df[REQUIRED].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-csv", required=True)
    ap.add_argument("--iql-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sort-by", default="bc_test_mean", choices=["variant", "bc_test_mean", "iql_test_mean", "delta_test"])
    args = ap.parse_args()

    bc_path = Path(args.bc_csv)
    iql_path = Path(args.iql_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bc = load_results(bc_path, "BC").rename(
        columns={"train_mean": "bc_train_mean", "test_mean": "bc_test_mean", "gen_gap": "bc_gen_gap"}
    )
    iql = load_results(iql_path, "IQL").rename(
        columns={"train_mean": "iql_train_mean", "test_mean": "iql_test_mean", "gen_gap": "iql_gen_gap"}
    )

    merged = bc.merge(iql, on="variant", how="outer", indicator=True)

    common = merged[merged["_merge"] == "both"].copy()
    only_bc = merged[merged["_merge"] == "left_only"]["variant"].tolist()
    only_iql = merged[merged["_merge"] == "right_only"]["variant"].tolist()

    if common.empty:
        raise RuntimeError("No overlapping variants between BC and IQL CSVs.")

    common["delta_test"] = common["iql_test_mean"] - common["bc_test_mean"]
    common["delta_train"] = common["iql_train_mean"] - common["bc_train_mean"]
    common["delta_gen_gap"] = common["iql_gen_gap"] - common["bc_gen_gap"]

    if args.sort_by != "variant":
        common = common.sort_values(args.sort_by, ascending=False)
    else:
        common = common.sort_values("variant")

    merged_csv = out_dir / "bc_iql_merged.csv"
    common.to_csv(merged_csv, index=False)

    variants = common["variant"].tolist()
    x = range(len(variants))
    width = 0.38

    # Plot 1: test mean comparison
    plt.figure(figsize=(max(10, len(variants) * 0.9), 6))
    plt.bar([i - width / 2 for i in x], common["bc_test_mean"], width=width, label="BC test_mean")
    plt.bar([i + width / 2 for i in x], common["iql_test_mean"], width=width, label="IQL test_mean")
    plt.xticks(list(x), variants, rotation=45, ha="right")
    plt.ylabel("Return")
    plt.title("BC vs IQL: Test Mean Return")
    plt.legend()
    plt.tight_layout()
    test_plot = out_dir / "bc_vs_iql_test_mean.png"
    plt.savefig(test_plot, dpi=180)
    plt.close()

    # Plot 2: deltas
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(variants) * 1.0), 5), constrained_layout=True)

    axes[0].bar(list(x), common["delta_test"])
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(list(x), variants, rotation=45, ha="right")
    axes[0].set_title("Delta Test (IQL - BC)")
    axes[0].set_ylabel("Return")

    axes[1].bar(list(x), common["delta_gen_gap"])
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(list(x), variants, rotation=45, ha="right")
    axes[1].set_title("Delta Gen Gap (IQL - BC)")
    axes[1].set_ylabel("Gap")

    delta_plot = out_dir / "bc_vs_iql_deltas.png"
    fig.savefig(delta_plot, dpi=180)
    plt.close(fig)

    summary_txt = out_dir / "summary.txt"
    lines = []
    lines.append(f"Overlapping variants: {len(common)}")
    lines.append(f"BC-only variants: {only_bc}")
    lines.append(f"IQL-only variants: {only_iql}")
    lines.append(f"Mean delta_test (IQL-BC): {common['delta_test'].mean():.4f}")
    lines.append(f"Best delta_test variant: {common.loc[common['delta_test'].idxmax(), 'variant']}")
    lines.append(f"Worst delta_test variant: {common.loc[common['delta_test'].idxmin(), 'variant']}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {merged_csv}")
    print(f"Wrote: {test_plot}")
    print(f"Wrote: {delta_plot}")
    print(f"Wrote: {summary_txt}")


if __name__ == "__main__":
    main()
