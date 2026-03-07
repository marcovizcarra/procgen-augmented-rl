#!/usr/bin/env python3
"""
Plot IQL training metrics from a train log.

Example:
  python3 scripts/plot_iql_train_metrics.py \
    --log runs/stage1_iql_L40_seed0/stage1_iql_train_eval_summary/logs/baseline_none_train.log
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_metrics(text: str):
    # Matches lines with: v=... q=... pi=... adv=...
    pat = re.compile(
        r"v=([0-9eE+\-.]+).*?q=([0-9eE+\-.]+).*?pi=([0-9eE+\-.]+).*?adv=([0-9eE+\-.]+)"
    )
    vals = [tuple(map(float, m)) for m in pat.findall(text)]
    if vals:
        arr = np.asarray(vals, dtype=np.float64)
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # Fallback for logs that may contain named losses separately.
    v = [float(x) for x in re.findall(r"v_loss[:=]\s*([0-9eE+\-.]+)", text)]
    q = [float(x) for x in re.findall(r"q_loss[:=]\s*([0-9eE+\-.]+)", text)]
    pi = [float(x) for x in re.findall(r"pi_loss[:=]\s*([0-9eE+\-.]+)", text)]
    adv = [float(x) for x in re.findall(r"adv(?:_mean)?[:=]\s*([0-9eE+\-.]+)", text)]

    k = min(len(v), len(q), len(pi), len(adv))
    if k == 0:
        return None
    return np.asarray(v[:k]), np.asarray(q[:k]), np.asarray(pi[:k]), np.asarray(adv[:k])


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[w:] - c[:-w]) / float(w)
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, y])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to IQL training log file")
    ap.add_argument("--out", default=None, help="Output PNG path (default: <log_dir>/train_metrics.png)")
    ap.add_argument("--smooth", type=int, default=1, help="Moving-average window for curves (default: 1 = no smoothing)")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"log not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_metrics(text)
    if parsed is None:
        raise RuntimeError("No parsable metrics found in log. Expected tokens like 'v=... q=... pi=... adv=...'.")

    v, q, pi, adv = parsed
    n = len(v)
    x = np.arange(1, n + 1)

    if args.smooth > 1:
        v = moving_average(v, args.smooth)
        q = moving_average(q, args.smooth)
        pi = moving_average(pi, args.smooth)
        adv = moving_average(adv, args.smooth)

    out_path = Path(args.out) if args.out else (log_path.parent / "train_metrics.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(x, v, label="v_loss")
    plt.plot(x, q, label="q_loss")
    plt.plot(x, pi, label="pi_loss")
    plt.plot(x, adv, label="adv_mean")
    plt.xlabel("Log index")
    plt.ylabel("Value")
    plt.title("IQL Training Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
