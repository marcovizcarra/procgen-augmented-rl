#!/usr/bin/env python3
"""Analyze BC vs IQL result CSVs with additional statistics and plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ci95(std: pd.Series, n: pd.Series) -> pd.Series:
    n = n.clip(lower=1)
    return 1.96 * (std / np.sqrt(n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-csv", required=True)
    ap.add_argument("--iql-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    bc_path = Path(args.bc_csv)
    iql_path = Path(args.iql_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bc = pd.read_csv(bc_path)
    iql = pd.read_csv(iql_path)

    needed = ["variant", "train_mean", "train_std", "train_episodes", "test_mean", "test_std", "test_episodes", "gen_gap"]
    for name, df in [("BC", bc), ("IQL", iql)]:
        miss = [c for c in needed if c not in df.columns]
        if miss:
            raise ValueError(f"{name} CSV missing columns: {miss}")

    bc = bc[needed].copy().rename(columns={
        "train_mean": "bc_train_mean",
        "train_std": "bc_train_std",
        "train_episodes": "bc_train_n",
        "test_mean": "bc_test_mean",
        "test_std": "bc_test_std",
        "test_episodes": "bc_test_n",
        "gen_gap": "bc_gen_gap",
    })
    iql = iql[needed].copy().rename(columns={
        "train_mean": "iql_train_mean",
        "train_std": "iql_train_std",
        "train_episodes": "iql_train_n",
        "test_mean": "iql_test_mean",
        "test_std": "iql_test_std",
        "test_episodes": "iql_test_n",
        "gen_gap": "iql_gen_gap",
    })

    m = bc.merge(iql, on="variant", how="inner")
    if m.empty:
        raise RuntimeError("No overlapping variants between BC and IQL CSVs.")

    m["delta_test"] = m["iql_test_mean"] - m["bc_test_mean"]
    m["delta_train"] = m["iql_train_mean"] - m["bc_train_mean"]
    m["delta_gen_gap"] = m["iql_gen_gap"] - m["bc_gen_gap"]

    m["bc_test_ci95"] = ci95(m["bc_test_std"], m["bc_test_n"])
    m["iql_test_ci95"] = ci95(m["iql_test_std"], m["iql_test_n"])
    m["bc_train_ci95"] = ci95(m["bc_train_std"], m["bc_train_n"])
    m["iql_train_ci95"] = ci95(m["iql_train_std"], m["iql_train_n"])

    # Approx z-score for difference in means (assuming independent episodes)
    se_diff_test = np.sqrt((m["bc_test_std"] ** 2) / m["bc_test_n"].clip(lower=1) + (m["iql_test_std"] ** 2) / m["iql_test_n"].clip(lower=1))
    m["z_test_diff"] = m["delta_test"] / se_diff_test.replace(0, np.nan)

    # Save table sorted by BC test performance
    out_csv = out_dir / "bc_iql_stats.csv"
    m.sort_values("bc_test_mean", ascending=False).to_csv(out_csv, index=False)

    # Plot: test means with CI
    p = m.sort_values("bc_test_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(p))

    plt.figure(figsize=(max(8, len(p) * 1.2), 5.5))
    plt.errorbar(x - 0.08, p["bc_test_mean"], yerr=p["bc_test_ci95"], fmt="o", capsize=4, label="BC test_mean ±95% CI")
    plt.errorbar(x + 0.08, p["iql_test_mean"], yerr=p["iql_test_ci95"], fmt="o", capsize=4, label="IQL test_mean ±95% CI")
    plt.xticks(x, p["variant"], rotation=35, ha="right")
    plt.ylabel("Test return")
    plt.title("BC vs IQL Test Performance (with CI)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "test_mean_ci95.png", dpi=180)
    plt.close()

    # Plot: train/test/gen-gap deltas
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(p) * 1.5), 4.5), constrained_layout=True)
    axes[0].bar(x, p["delta_train"])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Delta Train (IQL-BC)")
    axes[0].set_xticks(x, p["variant"], rotation=35, ha="right")

    axes[1].bar(x, p["delta_test"])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Delta Test (IQL-BC)")
    axes[1].set_xticks(x, p["variant"], rotation=35, ha="right")

    axes[2].bar(x, p["delta_gen_gap"])
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("Delta Gen Gap (IQL-BC)")
    axes[2].set_xticks(x, p["variant"], rotation=35, ha="right")

    fig.savefig(out_dir / "delta_train_test_gap.png", dpi=180)
    plt.close(fig)

    summary = []
    summary.append(f"Overlapping variants: {len(m)}")
    summary.append(f"Mean delta_test (IQL-BC): {m['delta_test'].mean():.4f}")
    summary.append(f"Mean delta_train (IQL-BC): {m['delta_train'].mean():.4f}")
    summary.append(f"Mean delta_gen_gap (IQL-BC): {m['delta_gen_gap'].mean():.4f}")
    best = m.loc[m['delta_test'].idxmax(), 'variant']
    worst = m.loc[m['delta_test'].idxmin(), 'variant']
    summary.append(f"Best delta_test variant: {best}")
    summary.append(f"Worst delta_test variant: {worst}")
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_dir / 'test_mean_ci95.png'}")
    print(f"Wrote: {out_dir / 'delta_train_test_gap.png'}")
    print(f"Wrote: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
