#!/usr/bin/env python3
"""
Plot comparison across multiple IQL run-group result CSVs.

Example:
  python3 scripts/plot_iql_sweep_comparison.py \
    --glob "runs/wandb_iql_quick_align_*/stage1_iql_train_eval_summary/results.csv" \
    --out-dir results/comparison/iql_sweep \
    --bc-csv runs/bc_run/results.csv
"""

import argparse
import glob
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def load_iql_results(paths: List[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        r = df.iloc[0]
        rows.append(
            {
                "run_group": str(r.get("run_group", Path(p).parts[-4])),
                "variant": str(r.get("variant", "")),
                "test_mean": float(r.get("test_mean")),
                "test_std": float(r.get("test_std")),
                "train_mean": float(r.get("train_mean")),
                "train_std": float(r.get("train_std")),
                "gen_gap": float(r.get("gen_gap")),
                "lr": float(r.get("lr_actor")),
                "expectile": float(r.get("expectile")),
                "beta": float(r.get("beta")),
                "batch_size": int(float(r.get("batch_size"))),
                "hidden": int(float(r.get("hidden"))),
                "path": p,
            }
        )
    if not rows:
        raise RuntimeError("No valid IQL results found.")
    out = pd.DataFrame(rows).sort_values("test_mean", ascending=False).reset_index(drop=True)
    return out


def load_bc_baseline(bc_csv: Optional[str]) -> Optional[float]:
    if not bc_csv:
        return None
    p = Path(bc_csv)
    if not p.exists():
        raise FileNotFoundError(f"BC CSV not found: {p}")
    df = pd.read_csv(p)
    if "variant" in df.columns and "test_mean" in df.columns:
        m = df[df["variant"] == "baseline_none"]
        if not m.empty:
            return float(m.iloc[0]["test_mean"])
    if "test_mean" not in df.columns:
        raise ValueError(f"BC CSV missing test_mean column: {p}")
    return float(df.iloc[0]["test_mean"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="Glob for IQL results.csv files.")
    ap.add_argument("--out-dir", required=True, help="Output directory for plots and summary.")
    ap.add_argument("--bc-csv", default=None, help="Optional BC results CSV for baseline reference line.")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise RuntimeError(f"No files matched glob: {args.glob}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iql = load_iql_results(paths)
    bc_baseline = load_bc_baseline(args.bc_csv)

    # Save ranked stats table.
    stats_csv = out_dir / "iql_sweep_ranked.csv"
    iql.to_csv(stats_csv, index=False)

    labels = [f"{rg}\n(lr={lr:g}, ex={ex:g})" for rg, lr, ex in zip(iql["run_group"], iql["lr"], iql["expectile"])]

    # Plot 1: test_mean bar plot.
    plt.figure(figsize=(max(10, 1.2 * len(iql)), 6))
    plt.bar(range(len(iql)), iql["test_mean"], yerr=iql["test_std"], capsize=3)
    if bc_baseline is not None:
        plt.axhline(bc_baseline, linestyle="--", linewidth=1.5, label=f"BC baseline_none = {bc_baseline:.2f}")
        plt.legend()
    plt.xticks(range(len(iql)), labels, rotation=35, ha="right")
    plt.ylabel("Return")
    plt.title("IQL Sweep: Test Mean (with Test Std error bars)")
    plt.tight_layout()
    bar_png = out_dir / "iql_sweep_test_mean.png"
    plt.savefig(bar_png, dpi=180)
    plt.close()

    # Plot 2: hyperparam scatter (expectile/lr colored by test mean).
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(iql["expectile"], iql["lr"], c=iql["test_mean"], s=130, cmap="viridis")
    for _, r in iql.iterrows():
        plt.text(r["expectile"] + 0.004, r["lr"], f"{r['test_mean']:.2f}", fontsize=8)
    plt.xlabel("Expectile")
    plt.ylabel("Learning Rate")
    plt.title("IQL Sweep Hyperparams vs Test Mean")
    plt.colorbar(sc, label="test_mean")
    plt.tight_layout()
    scatter_png = out_dir / "iql_sweep_hparam_scatter.png"
    plt.savefig(scatter_png, dpi=180)
    plt.close()

    # Markdown summary.
    best = iql.iloc[0]
    md = []
    md.append("# IQL Sweep Summary")
    md.append("")
    md.append(f"- Runs: {len(iql)}")
    md.append(f"- Best run_group: `{best['run_group']}`")
    md.append(f"- Best test_mean: {best['test_mean']:.3f}")
    md.append(f"- Best config: lr={best['lr']}, expectile={best['expectile']}, beta={best['beta']}, batch_size={best['batch_size']}, hidden={best['hidden']}")
    if bc_baseline is not None:
        md.append(f"- BC baseline_none test_mean: {bc_baseline:.3f}")
        md.append(f"- Best IQL - BC delta: {best['test_mean'] - bc_baseline:+.3f}")
    md_path = out_dir / "iql_sweep_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {stats_csv}")
    print(f"Wrote: {bar_png}")
    print(f"Wrote: {scatter_png}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
