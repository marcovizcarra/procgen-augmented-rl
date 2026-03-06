#!/usr/bin/env python3
"""
scripts/eval_stage1_all.py  (updated: run-group / dataset-folder support)

Evaluate ALL Stage-1 dataset variants (baseline + augmentation variants) by running
the existing evaluator: scripts/eval_procgen.py, and save a summary table.

NEW: Run-group (dataset-folder) support
- If you are organizing runs by dataset folder (recommended), e.g.:
    runs/stage1_datasets_L40/bc_<variant>/bc_ckpt.pt

  then this script will find checkpoints under:
    runs/<run_group>/bc_<variant>/bc_ckpt.pt

  where:
    run_group defaults to the datasets-root folder name, i.e.
      datasets-root = data/stage1_datasets_L40  -> run_group = stage1_datasets_L40

- Legacy layout is still supported (fallback):
    runs/bc_<variant>/bc_ckpt.pt

It will:
- discover variants under --datasets-root (folders with manifest.json)
- find each variant's checkpoint (grouped, then legacy)
- run: python scripts/eval_procgen.py --ckpt <ckpt> --episodes <N> [extra args...]
- parse TRAIN/TEST mean/std from stdout
- save:
    <out-dir>/results.csv
    <out-dir>/results.json
    <out-dir>/results.md   (nice markdown table)

Usage (grouped, recommended):
  python -B scripts/eval_stage1_all.py \
    --datasets-root data/stage1_datasets_L40 \
    --runs-root runs \
    --episodes 200

This writes by default to:
  runs/stage1_datasets_L40/stage1_eval_summary

Override output location:
  --out-dir runs/stage1_datasets_L40/stage1_eval_summary_seed0

Optional:
- evaluate only some variants:
    --only baseline_none,shift_pad4,shift4_jitter_0p2
- allow missing ckpts (skip instead of error):
    --skip-missing

Notes:
- This relies on eval_procgen.py printing lines like:
    TRAIN levels: mean_return=2.600 std=4.386 (episodes=50)
    TEST  levels: mean_return=3.600 std=4.800 (episodes=50)
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


TRAIN_RE = re.compile(
    r"TRAIN\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)
TEST_RE = re.compile(
    r"TEST\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)


def run_eval(ckpt: Path, episodes: int, extra_args: List[str]) -> str:
    cmd = [sys.executable, "scripts/eval_procgen.py", "--ckpt", str(ckpt), "--episodes", str(episodes)] + extra_args
    print("\n$ " + " ".join(cmd), flush=True)
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    print(out, flush=True)
    return out


def parse_eval(output: str) -> Dict[str, float]:
    m1 = TRAIN_RE.search(output)
    m2 = TEST_RE.search(output)
    if not m1 or not m2:
        raise RuntimeError(
            "Could not parse eval output.\n"
            "Make sure eval_procgen.py prints TRAIN/TEST lines like:\n"
            "  TRAIN levels: mean_return=... std=... (episodes=...)\n"
            "  TEST  levels: mean_return=... std=... (episodes=...)"
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


def discover_variants(datasets_root: Path) -> List[str]:
    if not datasets_root.exists():
        return []
    vars_ = []
    for p in datasets_root.iterdir():
        if p.is_dir() and (p / "manifest.json").exists():
            vars_.append(p.name)
    return sorted(vars_)


def find_ckpt(runs_root: Path, run_group: str, variant: str) -> Optional[Path]:
    """
    Try a few common patterns.

    NEW grouped (preferred):
      runs/<run_group>/bc_<variant>/bc_ckpt.pt
      runs/<run_group>/bc_<variant>/ckpt.pt
      runs/<run_group>/<variant>/bc_ckpt.pt

    LEGACY (fallback):
      runs/bc_<variant>/bc_ckpt.pt
      runs/bc_<variant>/ckpt.pt
      runs/<variant>/bc_ckpt.pt
    """
    group_root = runs_root / run_group

    candidates = [
        # grouped
        group_root / f"bc_{variant}" / "bc_ckpt.pt",
        group_root / f"bc_{variant}" / "ckpt.pt",
        group_root / variant / "bc_ckpt.pt",
        # legacy
        runs_root / f"bc_{variant}" / "bc_ckpt.pt",
        runs_root / f"bc_{variant}" / "ckpt.pt",
        runs_root / variant / "bc_ckpt.pt",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: search for directories containing variant (grouped then legacy)
    for root in [group_root, runs_root]:
        if root.exists():
            for d in root.iterdir():
                if d.is_dir() and variant in d.name:
                    c = d / "bc_ckpt.pt"
                    if c.exists():
                        return c
    return None


def to_markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "# Stage-1 Eval Summary\n\nNo results.\n"

    headers = ["variant", "train_mean", "train_std", "test_mean", "test_std", "gen_gap", "ckpt"]
    lines = []
    lines.append("# Stage-1 Eval Summary\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="data/stage1_datasets", help="Folder containing variant subfolders with manifest.json.")
    ap.add_argument("--runs-root", default="runs", help="Base folder containing run subfolders with checkpoints.")
    ap.add_argument(
        "--run-group",
        default=None,
        help="Subfolder under --runs-root to use (default: datasets-root folder name, e.g. stage1_datasets_L40).",
    )
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Where to write results (default: runs/<run_group>/stage1_eval_summary).",
    )
    ap.add_argument("--only", default=None, help="Comma-separated list of variants to evaluate.")
    ap.add_argument("--skip-missing", action="store_true", help="Skip variants with missing checkpoints instead of failing.")
    ap.add_argument("--extra-eval-args", default="", help='Extra args forwarded to eval_procgen.py, e.g. "--deterministic".')
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    runs_root = Path(args.runs_root)

    run_group = args.run_group or datasets_root.name

    # Ensure the group folder exists (so outputs go where you expect)
    group_root = runs_root / run_group
    group_root.mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir) if args.out_dir else (group_root / "stage1_eval_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = discover_variants(datasets_root)
    if not variants:
        raise RuntimeError(f"No variants found under {datasets_root} (expected folders with manifest.json).")

    if args.only:
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        variants = [v for v in variants if v in allow]
        if not variants:
            raise RuntimeError("After --only filter, no variants remain.")

    extra_args = args.extra_eval_args.strip().split() if args.extra_eval_args.strip() else []

    rows: List[Dict] = []
    for v in variants:
        ckpt = find_ckpt(runs_root, run_group, v)
        if ckpt is None:
            msg = (
                f"[missing ckpt] variant={v}\n"
                f"  expected (grouped): {runs_root}/{run_group}/bc_{v}/bc_ckpt.pt\n"
                f"  expected (legacy) : {runs_root}/bc_{v}/bc_ckpt.pt"
            )
            if args.skip_missing:
                print(msg)
                continue
            raise FileNotFoundError(msg)

        try:
            out = run_eval(ckpt, args.episodes, extra_args=extra_args)
            metrics = parse_eval(out)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"eval_procgen.py failed for {v}:\n{e.output}") from e

        row = {"variant": v, "ckpt": str(ckpt), **metrics}
        rows.append(row)

        # incremental save
        (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    rows_sorted = sorted(rows, key=lambda r: r["test_mean"], reverse=True) if rows else []

    # CSV
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()) if rows_sorted else ["variant"])
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    # Markdown
    md_path = out_dir / "results.md"
    md_path.write_text(to_markdown_table(rows_sorted), encoding="utf-8")

    print(f"\nRun group: {run_group}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {out_dir / 'results.json'}")
    print(f"Saved MD : {md_path}")


if __name__ == "__main__":
    main()
