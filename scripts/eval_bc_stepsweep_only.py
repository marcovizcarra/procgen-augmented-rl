#!/usr/bin/env python3
"""
scripts/eval_bc_stepsweep_only.py

EVAL-ONLY utility for your BC "generalization peak" steps sweep.

This script does NOT train. It only:
- locates existing checkpoints under runs/<run_group>/bc_<variant>_steps<STEPS>_tr<TRAINSEED>/bc_ckpt.pt
- runs scripts/eval_procgen.py with your requested split + distribution_mode
- saves raw eval rows + aggregations to a summary folder

Typical use case
----------------
You already trained with distribution_mode=hard (or anything), and now you want to
re-evaluate the SAME checkpoints with distribution_mode=easy (or different seeds/episodes)
WITHOUT retraining.

Outputs
-------
runs/<run_group>/stepsweep_eval_<tag>/
  eval_rows.csv / eval_rows.json / eval_rows.md
  agg_by_steps_trainseed.csv / .md
  agg_by_steps.csv / .md
  best_by_variant.csv / .md
  logs/

Example
-------
python -B scripts/eval_bc_stepsweep_only.py \
  --run-group stage1_datasets_L40_stepsweep \
  --variants baseline_none,shift4_scale_0p8_1p2 \
  --steps 10000,20000,50000,75000,100000 \
  --train-seeds 0,1,2 \
  --eval-seeds 0,1,2 \
  --episodes 200 \
  --train_start 40 --train_levels 1 \
  --test_start 41 --test_levels 500 \
  --distribution_mode easy \
  --tag easy
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


TRAIN_RE = re.compile(
    r"TRAIN\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)
TEST_RE = re.compile(
    r"TEST\s+levels:\s+mean_return=([0-9\.\-eE]+)\s+std=([0-9\.\-eE]+)\s+\(episodes=([0-9]+)\)"
)


def run_and_tee(cmd: List[str], log_path: Path, prefix: str = "") -> str:
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
            print((prefix + line) if prefix else line, end="", flush=True)
        rc = p.wait()
    if rc != 0:
        tail = "".join(captured[-80:])
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}\n\n--- tail ---\n{tail}")
    return "".join(captured)


def parse_eval(output: str) -> Dict[str, float]:
    m1 = TRAIN_RE.search(output)
    m2 = TEST_RE.search(output)
    if not m1 or not m2:
        raise RuntimeError("Could not parse eval output (missing TRAIN/TEST lines).")
    train_mean, train_std = float(m1.group(1)), float(m1.group(2))
    test_mean, test_std = float(m2.group(1)), float(m2.group(2))
    return {
        "train_mean": train_mean,
        "train_std": train_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "gen_gap": test_mean - train_mean,
    }


def mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run-group", required=True, help="Folder under runs-root that contains bc_* checkpoints.")
    ap.add_argument("--variants", default="baseline_none,shift4_scale_0p8_1p2")
    ap.add_argument("--steps", default="10000,20000,50000,75000,100000")
    ap.add_argument("--train-seeds", default="0,1,2")
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--train_start", type=int, default=40)
    ap.add_argument("--train_levels", type=int, default=1)
    ap.add_argument("--test_start", type=int, default=41)
    ap.add_argument("--test_levels", type=int, default=500)
    ap.add_argument("--distribution_mode", default="easy", choices=["easy", "hard"])
    ap.add_argument("--eval-script", default="scripts/eval_procgen.py")
    ap.add_argument("--tag", default="easy", help="Label appended to output folder name.")
    ap.add_argument("--skip-missing-ckpt", action="store_true")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    group_root = runs_root / args.run_group
    if not group_root.exists():
        raise FileNotFoundError(f"run-group folder not found: {group_root}")

    out_dir = group_root / f"stepsweep_eval_{args.tag}"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    steps_list = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    train_seeds = [int(x.strip()) for x in args.train_seeds.split(",") if x.strip()]
    eval_seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()]

    print("\n=== EVAL-ONLY STEP SWEEP ===")
    print(f"group_root : {group_root}")
    print(f"out_dir    : {out_dir}")
    print(f"mode       : {args.distribution_mode}")
    print(f"variants   : {variants}")
    print(f"steps      : {steps_list}")
    print(f"train_seeds: {train_seeds}")
    print(f"eval_seeds : {eval_seeds}")
    print(f"episodes   : {args.episodes}")
    print(f"split      : train_start={args.train_start}, train_levels={args.train_levels}, "
          f"test_start={args.test_start}, test_levels={args.test_levels}\n")

    eval_rows: List[Dict] = []
    total = len(variants) * len(steps_list) * len(train_seeds) * len(eval_seeds)
    k = 0

    for variant in variants:
        for steps in steps_list:
            for tr in train_seeds:
                ckpt = group_root / f"bc_{variant}_steps{steps}_tr{tr}" / "bc_ckpt.pt"
                if not ckpt.exists():
                    msg = f"[missing ckpt] {ckpt}"
                    if args.skip_missing_ckpt:
                        print(msg)
                        continue
                    raise FileNotFoundError(msg)

                for es in eval_seeds:
                    k += 1
                    print(f"\n[{k}/{total}] variant={variant} steps={steps} tr={tr} es={es}")

                    log_path = out_dir / "logs" / f"eval_{variant}_steps{steps}_tr{tr}_es{es}.log"
                    if log_path.exists():
                        txt = log_path.read_text(encoding="utf-8", errors="ignore")
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
                        out = run_and_tee(cmd, log_path, prefix=f"[eval:{variant}:steps{steps}:tr{tr}:es{es}] ")
                        metrics = parse_eval(out)

                    eval_rows.append({
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
                    })

                    (out_dir / "eval_rows.json").write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")

    # Save raw rows
    eval_rows_sorted = sorted(eval_rows, key=lambda r: (r["variant"], r["train_steps"], r["train_seed"], r["eval_seed"]))
    write_csv(out_dir / "eval_rows.csv", eval_rows_sorted)
    write_md_table(out_dir / "eval_rows.md", eval_rows_sorted, "Eval-only Stepsweep — All Rows")

    # Aggregate over eval seeds per (variant, steps, train_seed)
    agg_vst: List[Dict] = []
    for variant in variants:
        for steps in steps_list:
            for tr in train_seeds:
                rows = [r for r in eval_rows if r["variant"] == variant and r["train_steps"] == steps and r["train_seed"] == tr]
                if not rows:
                    continue
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
    write_csv(out_dir / "agg_by_steps_trainseed.csv", agg_vst_sorted)
    write_md_table(out_dir / "agg_by_steps_trainseed.md", agg_vst_sorted, "Eval-only Stepsweep — Aggregated by (Steps, Train Seed)")

    # Aggregate over train seeds (using per-train-seed means)
    agg_vs: List[Dict] = []
    for variant in variants:
        for steps in steps_list:
            rows = [r for r in agg_vst if r["variant"] == variant and r["train_steps"] == steps]
            if not rows:
                continue
            agg_vs.append({
                "variant": variant,
                "train_steps": steps,
                "n_train_seeds": len(rows),
                "test_mean_avg": round(mean([r["test_mean_avg"] for r in rows]), 6),
                "train_mean_avg": round(mean([r["train_mean_avg"] for r in rows]), 6),
                "gen_gap_avg": round(mean([r["gen_gap_avg"] for r in rows]), 6),
            })
    agg_vs_sorted = sorted(agg_vs, key=lambda r: (r["variant"], r["train_steps"]))
    write_csv(out_dir / "agg_by_steps.csv", agg_vs_sorted)
    write_md_table(out_dir / "agg_by_steps.md", agg_vs_sorted, "Eval-only Stepsweep — Aggregated by Steps")

    # Best steps per variant (by TEST mean)
    best_rows: List[Dict] = []
    for variant in variants:
        rows = [r for r in agg_vs if r["variant"] == variant]
        if not rows:
            continue
        best = sorted(rows, key=lambda r: r["test_mean_avg"], reverse=True)[0]
        best_rows.append({
            "variant": variant,
            "best_train_steps": best["train_steps"],
            "best_test_mean_avg": best["test_mean_avg"],
            "best_train_mean_avg": best["train_mean_avg"],
            "best_gen_gap_avg": best["gen_gap_avg"],
        })
    best_rows_sorted = sorted(best_rows, key=lambda r: r["best_test_mean_avg"], reverse=True)
    write_csv(out_dir / "best_by_variant.csv", best_rows_sorted)
    write_md_table(out_dir / "best_by_variant.md", best_rows_sorted, "Eval-only Stepsweep — Best Steps per Variant")

    print("\n=== DONE ===")
    print(f"Wrote: {out_dir}")
    print("Key files:")
    print(f"- {out_dir/'best_by_variant.csv'}")
    print(f"- {out_dir/'agg_by_steps.csv'}")
    print(f"- {out_dir/'eval_rows.csv'}")


if __name__ == "__main__":
    main()
