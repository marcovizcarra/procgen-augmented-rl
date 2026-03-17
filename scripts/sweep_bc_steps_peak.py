#!/usr/bin/env python3
"""
scripts/sweep_bc_steps_peak.py

Lock down the "BC generalization peak" by sweeping training steps and
replicating across training seeds and evaluation seeds.

Default experiment (recommended)
--------------------------------
Variants:
  - baseline_none
  - shift4_scale_0p8_1p2

Training:
  - train_steps: 10000, 20000, 50000, 100000
  - train seeds: 0,1,2
  - batch_size: 256
  - lr: 3e-4
  - device: auto

Evaluation:
  - eval seeds: 0..4
  - episodes: 200
  - split (L40):
      train_start=0 train_levels=40
      test_start=40  test_levels=500

Outputs
-------
Runs are stored under:
  runs/<run_group>/
where run_group defaults to:
  <datasets_root name>_stepsweep

Summary files written under:
  runs/<run_group>/bc_stepsweep_summary/
    eval_rows.csv / .json / .md          (all eval rows)
    agg_by_steps_trainseed.csv / .md     (mean over eval seeds per (variant, steps, train_seed))
    agg_by_steps.csv / .md               (mean over train seeds per (variant, steps))
    best_by_variant.csv / .md            (best step count per variant by TEST mean)

Logs:
  runs/<run_group>/bc_stepsweep_summary/logs/

Usage
-----
# Standard L40 sweep:
python -B scripts/sweep_bc_steps_peak.py \
  --datasets-root data/stage1_datasets_L40 \
  --runs-root runs \
  --skip-existing

# Faster debug run:
python -B scripts/sweep_bc_steps_peak.py \
  --datasets-root data/stage1_datasets_L40 \
  --steps 2000,5000 \
  --train-seeds 0 \
  --eval-seeds 0 \
  --episodes 20
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


def run_and_tee(cmd: List[str], log_path: Path, prefix: str = "") -> str:
    """Run a command, stream output live to terminal, tee to log file, and return captured output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(cmd), flush=True)

    captured: List[str] = []
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
        "gen_gap": test_mean - train_mean,
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
    ap.add_argument("--run-group", default=None, help="Default: <datasets_root name>_stepsweep")
    ap.add_argument("--variants", default="baseline_none,shift4_scale_0p8_1p2")
    ap.add_argument("--steps", default="10000,20000,50000,100000", help="Comma-separated train steps to sweep.")
    ap.add_argument("--train-seeds", default="0,1,2")
    ap.add_argument("--eval-seeds", default="0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--train_start", type=int, default=40)
    ap.add_argument("--train_levels", type=int, default=1)
    ap.add_argument("--test_start", type=int, default=41)
    ap.add_argument("--test_levels", type=int, default=500)
    ap.add_argument("--distribution_mode", default="hard", choices=["easy", "hard"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--train-script", default="scripts/train_bc_min.py")
    ap.add_argument("--eval-script", default="scripts/eval_procgen.py")
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    runs_root = Path(args.runs_root)
    run_group = args.run_group or (datasets_root.name + "_stepsweep")

    group_root = runs_root / run_group
    group_root.mkdir(parents=True, exist_ok=True)

    summary_root = group_root / "bc_stepsweep_summary"
    logs_dir = summary_root / "logs"
    summary_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    steps_list = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    train_seeds = [int(x.strip()) for x in args.train_seeds.split(",") if x.strip()]
    eval_seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()]

    vdirs = {v: ensure_variant_exists(datasets_root, v) for v in variants}

    print("\n=== BC STEP SWEEP (GENERALIZATION PEAK) ===")
    print(f"datasets_root : {datasets_root}")
    print(f"run_group     : {run_group}")
    print(f"variants      : {variants}")
    print(f"steps         : {steps_list}")
    print(f"train_seeds   : {train_seeds}")
    print(f"eval_seeds    : {eval_seeds}")
    print(f"episodes      : {args.episodes}")
    print(f"split         : train({args.train_start}..{args.train_start+args.train_levels-1}) "
          f"test({args.test_start}..{args.test_start+args.test_levels-1})")
    print(f"summary_root  : {summary_root}\n")

    # Train all (variant, steps, train_seed)
    ckpts: Dict[Tuple[str, int, int], Path] = {}
    total_train = len(variants) * len(steps_list) * len(train_seeds)
    train_idx = 0

    for variant in variants:
        for steps in steps_list:
            for tr in train_seeds:
                train_idx += 1
                run_name = f"{run_group}/bc_{variant}_steps{steps}_tr{tr}"
                ckpt = runs_root / run_name / "bc_ckpt.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                ckpts[(variant, steps, tr)] = ckpt

                print(f"\n[TRAIN {train_idx}/{total_train}] variant={variant} steps={steps} train_seed={tr}")

                if args.skip_existing and ckpt.exists():
                    print(f"[TRAIN] skip existing ckpt: {ckpt}")
                    continue

                cmd = [
                    sys.executable, "-u", args.train_script,
                    "--dataset-root", str(vdirs[variant]),
                    "--run-name", run_name,
                    "--steps", str(steps),
                    "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--seed", str(tr),
                    "--device", str(args.device),
                ]
                log_path = logs_dir / f"train_{variant}_steps{steps}_tr{tr}.log"
                run_and_tee(cmd, log_path, prefix=f"[train:{variant}:steps{steps}:tr{tr}] ")

                if not ckpt.exists():
                    raise FileNotFoundError(f"Training finished but checkpoint not found: {ckpt}")

    # Evaluate all (variant, steps, train_seed, eval_seed)
    eval_rows: List[Dict] = []
    total_eval = len(variants) * len(steps_list) * len(train_seeds) * len(eval_seeds)
    eval_idx = 0

    for variant in variants:
        for steps in steps_list:
            for tr in train_seeds:
                ckpt = ckpts[(variant, steps, tr)]
                if not ckpt.exists():
                    raise FileNotFoundError(f"Missing checkpoint for eval: {ckpt}")

                for es in eval_seeds:
                    eval_idx += 1
                    print(f"\n[EVAL {eval_idx}/{total_eval}] variant={variant} steps={steps} tr={tr} es={es}")

                    eval_log_path = logs_dir / f"eval_{variant}_steps{steps}_tr{tr}_es{es}.log"
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
                        out = run_and_tee(cmd, eval_log_path, prefix=f"[eval:{variant}:steps{steps}:tr{tr}:es{es}] ")
                        metrics = parse_eval(out)

                    row = {
                        "variant": variant,
                        "train_steps": steps,
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

    # Save raw eval rows
    eval_rows_sorted = sorted(eval_rows, key=lambda r: (r["variant"], r["train_steps"], r["train_seed"], r["eval_seed"]))
    write_csv(summary_root / "eval_rows.csv", eval_rows_sorted)
    write_md_table(summary_root / "eval_rows.md", eval_rows_sorted, title="BC Step Sweep — All Eval Rows")

    # Aggregation 1: mean over eval_seeds per (variant, steps, train_seed)
    agg_vst: List[Dict] = []
    for variant in variants:
        for steps in steps_list:
            for tr in train_seeds:
                rows = [r for r in eval_rows if r["variant"] == variant and r["train_steps"] == steps and r["train_seed"] == tr]
                agg_vst.append({
                    "variant": variant,
                    "train_steps": steps,
                    "train_seed": tr,
                    "n_eval_seeds": len(rows),
                    "train_mean_avg": round(mean([r["train_mean"] for r in rows]), 6),
                    "test_mean_avg": round(mean([r["test_mean"] for r in rows]), 6),
                    "gen_gap_avg": round(mean([r["gen_gap"] for r in rows]), 6),
                })
    agg_vst_sorted = sorted(agg_vst, key=lambda r: (r["variant"], r["train_steps"], r["train_seed"]))
    write_csv(summary_root / "agg_by_steps_trainseed.csv", agg_vst_sorted)
    write_md_table(summary_root / "agg_by_steps_trainseed.md", agg_vst_sorted, title="BC Step Sweep — Aggregated by (Steps, Train Seed)")

    # Aggregation 2: mean over train_seeds (using the per-train-seed means)
    agg_vs: List[Dict] = []
    for variant in variants:
        for steps in steps_list:
            rows = [r for r in agg_vst if r["variant"] == variant and r["train_steps"] == steps]
            agg_vs.append({
                "variant": variant,
                "train_steps": steps,
                "n_train_seeds": len(rows),
                "test_mean_avg": round(mean([r["test_mean_avg"] for r in rows]), 6),
                "train_mean_avg": round(mean([r["train_mean_avg"] for r in rows]), 6),
                "gen_gap_avg": round(mean([r["gen_gap_avg"] for r in rows]), 6),
            })
    agg_vs_sorted = sorted(agg_vs, key=lambda r: (r["variant"], r["train_steps"]))
    write_csv(summary_root / "agg_by_steps.csv", agg_vs_sorted)
    write_md_table(summary_root / "agg_by_steps.md", agg_vs_sorted, title="BC Step Sweep — Aggregated by Steps (mean over train seeds)")

    # Best steps per variant (by TEST mean)
    best_rows: List[Dict] = []
    for variant in variants:
        rows = [r for r in agg_vs if r["variant"] == variant]
        rows_sorted = sorted(rows, key=lambda r: r["test_mean_avg"], reverse=True)
        if rows_sorted:
            best = rows_sorted[0]
            best_rows.append({
                "variant": variant,
                "best_train_steps": best["train_steps"],
                "best_test_mean_avg": best["test_mean_avg"],
                "best_train_mean_avg": best["train_mean_avg"],
                "best_gen_gap_avg": best["gen_gap_avg"],
            })
    best_rows_sorted = sorted(best_rows, key=lambda r: r["best_test_mean_avg"], reverse=True)
    write_csv(summary_root / "best_by_variant.csv", best_rows_sorted)
    write_md_table(summary_root / "best_by_variant.md", best_rows_sorted, title="BC Step Sweep — Best Steps per Variant")

    print("\n=== DONE ===")
    print(f"Summary folder: {summary_root}")
    print(f"- eval_rows.csv/.json/.md")
    print(f"- agg_by_steps_trainseed.csv/.md")
    print(f"- agg_by_steps.csv/.md")
    print(f"- best_by_variant.csv/.md")
    print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()
