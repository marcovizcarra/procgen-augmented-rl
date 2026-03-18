#!/usr/bin/env python3
"""
scripts/run_top3_train_eval_matrix.py

Run the "tonight" replication matrix you described:

A) Training runs (12 total)
- Variants (4):
  1) baseline_none
  2) shift4_scale_0p8_1p2
  3) shift4_scale_0p9_1p1
  4) shift4_noise_0p02
- Train seeds: 0,1,2
- Train steps: 100000
- Batch: 256
- LR: 3e-4
- Run-name pattern:
    <run_group>/bc_<variant>_tr<train_seed>
  which saves checkpoints to:
    runs/<run_group>/bc_<variant>_tr<train_seed>/bc_ckpt.pt

B) Evaluation runs (60 total)
- For each trained checkpoint above:
  - Eval seeds: 0..4
  - Episodes: 200
  - Split: Train(0–39) / Test(40–539)
    --train_start 0 --train_levels 40
    --test_start 40 --test_levels 500
- Outputs (per dataset root / run_group):
    runs/<run_group>/top3_matrix_summary/
      eval_rows.csv        (all 60 eval rows)
      eval_rows.json
      eval_rows.md
      agg_by_trainseed.csv (mean across eval seeds, per (variant, train_seed))
      agg_by_variant.csv   (mean across train+eval seeds, per variant)

The script streams progress to the terminal AND writes per-run logs:
  runs/<run_group>/top3_matrix_summary/logs/

Usage
-----
# Standard run (exactly your plan):
python -B scripts/run_top3_train_eval_matrix.py \
  --datasets-root data/stage1_datasets_L40 \
  --run-group stage1_datasets_L40 \
  --skip-existing

# If you want to change budgets quickly:
python -B scripts/run_top3_train_eval_matrix.py \
  --datasets-root data/stage1_datasets_L40 \
  --train-steps 20000 \
  --episodes 100 \
  --skip-existing
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


TRAIN_RE = re.compile(
    r"TRAIN\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)
TEST_RE = re.compile(
    r"TEST\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)


VARIANTS_DEFAULT = [
    "baseline_none",
    "shift4_scale_0p8_1p2",
    "shift4_scale_0p9_1p1",
    "shift4_noise_0p02",
]


def run_and_tee(cmd: List[str], log_path: Path, prefix: str = "") -> str:
    """
    Run a command, stream stdout+stderr live to terminal, and tee to log_path.
    Returns the full captured output (useful for parsing eval output).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(cmd), flush=True)

    captured: List[str] = []

    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert p.stdout is not None
        for line in p.stdout:
            captured.append(line)
            f.write(line)
            if prefix:
                print(prefix + line, end="", flush=True)
            else:
                print(line, end="", flush=True)

        rc = p.wait()

    if rc != 0:
        tail = "".join(captured[-80:])
        raise RuntimeError(
            f"Command failed with exit code {rc}: {' '.join(cmd)}\n\n--- last ~80 lines ---\n{tail}"
        )

    return "".join(captured)


def parse_eval(output: str) -> Dict[str, float]:
    m1 = TRAIN_RE.search(output)
    m2 = TEST_RE.search(output)
    if not m1 or not m2:
        raise RuntimeError(
            "Could not parse eval output. Expected lines like:\n"
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
        "gen_gap": test_mean - train_mean,  # (test - train)
    }


def ensure_variant_exists(datasets_root: Path, variant: str) -> Path:
    vdir = datasets_root / variant
    if not (vdir / "manifest.json").exists():
        raise FileNotFoundError(f"Missing variant dataset (manifest.json not found): {vdir}")
    return vdir


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md_table(path: Path, rows: List[Dict], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo results.\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [f"# {title}\n", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="data/stage1_datasets_L40")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run-group", default=None, help="Default: datasets-root folder name (e.g., stage1_datasets_L40).")
    ap.add_argument("--variants", default=",".join(VARIANTS_DEFAULT), help="Comma-separated variants to run.")
    ap.add_argument("--train-seeds", default="0,1,2", help="Comma-separated training seeds.")
    ap.add_argument("--eval-seeds", default="0,1,2,3,4", help="Comma-separated eval seeds.")
    ap.add_argument("--train-steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--train_start", type=int, default=0)
    ap.add_argument("--train_levels", type=int, default=40)
    ap.add_argument("--test_start", type=int, default=40)
    ap.add_argument("--test_levels", type=int, default=500)
    ap.add_argument("--distribution_mode", default="hard", choices=["easy", "hard"])
    ap.add_argument("--skip-existing", action="store_true", help="Skip training if ckpt exists; skip eval if eval log exists.")
    ap.add_argument("--train-script", default="scripts/train_bc_min.py")
    ap.add_argument("--eval-script", default="scripts/eval_procgen.py")
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    runs_root = Path(args.runs_root)
    run_group = args.run_group or datasets_root.name

    # Ensure group folder exists
    group_root = runs_root / run_group
    group_root.mkdir(parents=True, exist_ok=True)

    summary_root = group_root / "top3_matrix_summary"
    logs_dir = summary_root / "logs"
    summary_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    train_seeds = [int(x.strip()) for x in args.train_seeds.split(",") if x.strip()]
    eval_seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()]

    # Validate datasets exist
    vdirs = {v: ensure_variant_exists(datasets_root, v) for v in variants}

    print("\n=== TOP3 MATRIX RUN ===")
    print(f"datasets_root   : {datasets_root}")
    print(f"runs_root       : {runs_root}")
    print(f"run_group       : {run_group}")
    print(f"variants        : {variants}")
    print(f"train_seeds     : {train_seeds}")
    print(f"eval_seeds      : {eval_seeds}")
    print(f"train_steps     : {args.train_steps}")
    print(f"batch_size      : {args.batch_size}")
    print(f"lr              : {args.lr}")
    print(f"device          : {args.device}")
    print(f"episodes        : {args.episodes}")
    print(f"split           : train({args.train_start}..{args.train_start+args.train_levels-1}) "
          f"test({args.test_start}..{args.test_start+args.test_levels-1})")
    print(f"summary_root    : {summary_root}\n")

    # -----------------------
    # A) Training
    # -----------------------
    ckpts: Dict[Tuple[str, int], Path] = {}
    total_train = len(variants) * len(train_seeds)
    train_idx = 0

    for variant in variants:
        for tr in train_seeds:
            train_idx += 1
            run_name = f"{run_group}/bc_{variant}_tr{tr}"
            ckpt = runs_root / run_name / "bc_ckpt.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpts[(variant, tr)] = ckpt

            print(f"\n[TRAIN {train_idx}/{total_train}] variant={variant} train_seed={tr}")

            if args.skip_existing and ckpt.exists():
                print(f"[TRAIN] skip existing ckpt: {ckpt}")
                continue

            cmd = [
                sys.executable, "-u", args.train_script,
                "--dataset-root", str(vdirs[variant]),
                "--run-name", run_name,
                "--steps", str(args.train_steps),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(tr),
                "--device", str(args.device),
            ]
            log_path = logs_dir / f"train_{variant}_tr{tr}.log"
            run_and_tee(cmd, log_path, prefix=f"[train:{variant}:tr{tr}] ")

            if not ckpt.exists():
                raise FileNotFoundError(f"Training finished but checkpoint not found: {ckpt}")

    # -----------------------
    # B) Evaluation
    # -----------------------
    eval_rows: List[Dict] = []
    total_eval = len(variants) * len(train_seeds) * len(eval_seeds)
    eval_idx = 0

    for variant in variants:
        for tr in train_seeds:
            ckpt = ckpts[(variant, tr)]
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint for eval: {ckpt}")

            for es in eval_seeds:
                eval_idx += 1
                print(f"\n[EVAL {eval_idx}/{total_eval}] variant={variant} train_seed={tr} eval_seed={es}")

                eval_log_path = logs_dir / f"eval_{variant}_tr{tr}_es{es}.log"
                if args.skip_existing and eval_log_path.exists():
                    txt = eval_log_path.read_text(encoding="utf-8", errors="ignore")
                    metrics = parse_eval(txt)
                else:
                    cmd = [
                        sys.executable, "-u", args.eval_script,
                        "--ckpt", str(ckpt),
                        "--episodes", str(args.episodes),
                        "--seed", str(es),
                        "--train_start", str(args.train_start),
                        "--train_levels", str(args.train_levels),
                        "--test_start", str(args.test_start),
                        "--test_levels", str(args.test_levels),
                        "--distribution_mode", str(args.distribution_mode),
                    ]
                    out = run_and_tee(cmd, eval_log_path, prefix=f"[eval:{variant}:tr{tr}:es{es}] ")
                    metrics = parse_eval(out)

                row = {
                    "variant": variant,
                    "train_seed": tr,
                    "eval_seed": es,
                    "train_mean": round(metrics["train_mean"], 6),
                    "train_std": round(metrics["train_std"], 6),
                    "test_mean": round(metrics["test_mean"], 6),
                    "test_std": round(metrics["test_std"], 6),
                    "gen_gap": round(metrics["gen_gap"], 6),
                    "ckpt": str(ckpt),
                }
                eval_rows.append(row)

                (summary_root / "eval_rows.json").write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")

    # Save all eval rows
    eval_rows_sorted = sorted(eval_rows, key=lambda r: (r["variant"], r["train_seed"], r["eval_seed"]))
    write_csv(summary_root / "eval_rows.csv", eval_rows_sorted)
    write_md_table(summary_root / "eval_rows.md", eval_rows_sorted, title="Top3 Matrix — All Eval Rows (60)")

    # Aggregations
    agg_ts: List[Dict] = []
    for variant in variants:
        for tr in train_seeds:
            rows = [r for r in eval_rows if r["variant"] == variant and r["train_seed"] == tr]
            agg_ts.append({
                "variant": variant,
                "train_seed": tr,
                "n_eval_seeds": len(rows),
                "train_mean_avg": round(mean([r["train_mean"] for r in rows]), 6),
                "test_mean_avg": round(mean([r["test_mean"] for r in rows]), 6),
                "gen_gap_avg": round(mean([r["gen_gap"] for r in rows]), 6),
            })
    agg_ts_sorted = sorted(agg_ts, key=lambda r: (r["variant"], r["train_seed"]))
    write_csv(summary_root / "agg_by_trainseed.csv", agg_ts_sorted)
    write_md_table(summary_root / "agg_by_trainseed.md", agg_ts_sorted, title="Top3 Matrix — Aggregated by Train Seed")

    agg_v: List[Dict] = []
    for variant in variants:
        rows = [r for r in agg_ts if r["variant"] == variant]
        agg_v.append({
            "variant": variant,
            "n_train_seeds": len(rows),
            "train_mean_avg": round(mean([r["train_mean_avg"] for r in rows]), 6),
            "test_mean_avg": round(mean([r["test_mean_avg"] for r in rows]), 6),
            "gen_gap_avg": round(mean([r["gen_gap_avg"] for r in rows]), 6),
        })
    agg_v_sorted = sorted(agg_v, key=lambda r: r["test_mean_avg"], reverse=True)
    write_csv(summary_root / "agg_by_variant.csv", agg_v_sorted)
    write_md_table(summary_root / "agg_by_variant.md", agg_v_sorted, title="Top3 Matrix — Aggregated by Variant")

    print("\n=== DONE ===")
    print(f"Summary folder: {summary_root}")
    print(f"- eval_rows.csv / eval_rows.json / eval_rows.md")
    print(f"- agg_by_trainseed.csv / agg_by_trainseed.md")
    print(f"- agg_by_variant.csv / agg_by_variant.md")
    print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()
