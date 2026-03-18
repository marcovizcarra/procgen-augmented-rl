import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def load_algo_data_augmentation(path: str) -> np.ndarray:
    """
    Load in data from csv file with columns: algorithm, data_augmentation, reward_mean, reward_std
    """
    with open(path, "r") as f:
        return np.array([line.split(",") for line in f.readlines()[1:]])

def plot_algo_data_aug(data: np.ndarray, title: str, out_path: str) -> None:
    """
    Plot bar chart groups with x-axis as data augmentations and y-axis as reward mean with error bars as reward std.

    Each group will have a different color for the bars.
    """
    plt.figure(figsize=(12, 6))

    cleaned_rows = []
    for row in data:
        data_aug, algo, reward_mean, reward_std, _data_size = [x.strip() for x in row]
        if reward_mean == "":
            continue
        cleaned_rows.append((data_aug, algo, float(reward_mean), reward_std))

    if not cleaned_rows:
        plt.savefig(out_path)
        plt.close()
        return

    algos = sorted({row[1] for row in cleaned_rows})
    cmap = plt.colormaps.get_cmap("tab10")
    algo_to_color = {algo: cmap(i / max(1, len(algos) - 1)) for i, algo in enumerate(algos)}

    # Draw dashed baseline_none ref per algorithm 
    baseline_by_algo = {}
    for data_aug, algo, reward_mean, _reward_std in cleaned_rows:
        if data_aug == "baseline_none":
            baseline_by_algo[algo] = reward_mean

    # Group bars 
    non_baseline_rows = [row for row in cleaned_rows if row[0] != "baseline_none"]
    x_labels = list(dict.fromkeys(row[0] for row in non_baseline_rows))
    x_index = {label: i for i, label in enumerate(x_labels)}
    x_base = np.arange(len(x_labels))

    total_width = 0.8
    bar_width = total_width / len(algos)
    offsets = (np.arange(len(algos)) - (len(algos) - 1) / 2.0) * bar_width

    for i, algo in enumerate(algos):
        if algo in baseline_by_algo:
            plt.axhline(
                baseline_by_algo[algo],
                color=algo_to_color[algo],
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label=f"{algo} baseline",
            )

        if len(x_labels) == 0:
            continue

        heights = np.full(len(x_labels), np.nan)
        for data_aug, row_algo, reward_mean, _reward_std in non_baseline_rows:
            if row_algo == algo:
                heights[x_index[data_aug]] = reward_mean

        plt.bar(
            x_base + offsets[i],
            heights,
            width=bar_width,
            label=algo,
            color=algo_to_color[algo],
        )

    plt.xticks(x_base, x_labels, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel("Reward Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def filter_by_data_size(data: np.ndarray, data_size: str) -> np.ndarray:
    return np.array([row for row in data if row[4].strip() == data_size], dtype=object)

def main() -> None:     
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True, help="Path to input csv file")
    # get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out-path-20k", default=os.path.join(script_dir, "analysis_data/algo_data_augmentation_20k.jpg"), help="Path to 20k output plot")
    ap.add_argument("--out-path-100k", default=os.path.join(script_dir, "analysis_data/algo_data_augmentation_100k.jpg"), help="Path to 100k output plot")
    args = ap.parse_args()

    data = load_algo_data_augmentation(args.input_path)
    data_20k = filter_by_data_size(data, "20000")
    data_100k = filter_by_data_size(data, "100000")

    plot_algo_data_aug(data_20k, "Reward vs Data Augmentation (20k samples)", args.out_path_20k)
    plot_algo_data_aug(data_100k, "Reward vs Data Augmentation (100k samples)", args.out_path_100k)

if __name__ == "__main__":
    main()