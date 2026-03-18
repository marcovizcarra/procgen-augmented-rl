import argparse
import csv
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_data_from_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_data(
    rows: list[dict[str, str]],
    title: str,
    out_path: str,
    x_col: str,
    y_col: str,
    group_cols: list[str],
    x_divisor: float,
    x_label: str,
    y_min: Optional[float],
    y_max: Optional[float],
) -> None:
    plt.figure(figsize=(10, 6))
    grouped_points: dict[str, list[tuple[float, float]]] = {}

    for row in rows:
        x_raw = (row.get(x_col) or "").strip()
        y_raw = (row.get(y_col) or "").strip()
        if x_raw == "" or y_raw == "":
            continue

        try:
            x = float(x_raw) / x_divisor
            y = float(y_raw)
        except ValueError:
            continue

        label_parts = [(row.get(col) or "").strip() for col in group_cols]
        if all(part == "" for part in label_parts):
            group_label = "unknown"
        else:
            group_label = " + ".join([part for part in label_parts if part != ""])

        grouped_points.setdefault(group_label, []).append((x, y))

    cmap = plt.colormaps.get_cmap("tab10")
    for i, (group_label, points) in enumerate(grouped_points.items()):
        points = sorted(points, key=lambda point: point[0])
        x_vals = [point[0] for point in points]
        y_vals = [point[1] for point in points]
        color = cmap(i % cmap.N)

        plt.plot(
            x_vals,
            y_vals,
            marker="o",
            linestyle="-",
            label=group_label,
            color=color,
        )

    x_ticks = sorted({point[0] for points in grouped_points.values() for point in points})
    if x_ticks:
        if x_divisor != 1:
            tick_labels = [f"{tick:g}k" for tick in x_ticks]
            plt.xticks(x_ticks, tick_labels)
        else:
            plt.xticks(x_ticks)
    plt.xlabel(x_label)
    plt.ylabel("Reward")
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True, help="Path to input csv file")
    ap.add_argument("--out-path", default="reward_vs_datasize.jpg", help="Path to output plot")
    ap.add_argument("--title", default="Reward vs Steps", help="Plot title")
    ap.add_argument("--x-col", default="data_size", help="CSV column for x-axis")
    ap.add_argument("--y-col", default="reward_mean", help="CSV column for y-axis")
    ap.add_argument("--group-cols", nargs="+", default=["algorithm"], help="CSV columns to combine as series label")
    ap.add_argument("--x-divisor", type=float, default=1000.0, help="Divide x values before plotting")
    ap.add_argument("--x-label", default="Steps", help="X-axis label")
    ap.add_argument("--y-min", type=float, default=-0.5, help="Optional lower y-axis bound")
    ap.add_argument("--y-max", type=float, default=4.0, help="Optional upper y-axis bound")
    args = ap.parse_args()

    rows = load_data_from_csv(args.input_path)
    plot_data(
        rows=rows,
        title=args.title,
        out_path=args.out_path,
        x_col=args.x_col,
        y_col=args.y_col,
        group_cols=args.group_cols,
        x_divisor=args.x_divisor,
        x_label=args.x_label,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()