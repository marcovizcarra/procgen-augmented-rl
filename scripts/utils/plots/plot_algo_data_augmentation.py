import matplotlib.pyplot as plt
import numpy as np
import json
import argparse


def load_algo_data_augmentation(path: str) -> np.ndarray:
    """
    Load in data from csv file with columns: algorithm, data_augmentation, reward_mean, reward_std
    """
    with open(path, "r") as f:
        return np.array([line.split(",") for line in f])

def plot_algo_data_aug(algo_data_augmentation: np.ndarray, title: str, out_path: str) -> None:
    """
    Plot bar chart groups with x-axis as data augmentations and y-axis as reward mean with error bars as reward std.

    Each group will have a different color for the bars.
    """
    plt.figure(figsize=(10, 6))
    # plot will have a different color line for each algorithm  
    colors = plt.cm.tab10.colors
    for i, row in enumerate(algo_data_augmentation):
        algorithm, data_augmentation, reward_mean, reward_std = row
        plt.bar(data_augmentation, reward_mean, yerr=reward_std, label=algorithm, color=colors[i])
    plt.savefig(out_path)
    plt.close()

def main() -> None:     
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="Path to input csv file")
    ap.add_argument("--out-path", default="algo_data_augmentation.jpg", help="Path to output plot")
    args = ap.parse_args()

    algo_data_augmentation = load_algo_data_augmentation(args.algo_data_augmentation)
    plot_algo_data_aug(algo_data_augmentation, args.out_path)

if __name__ == "__main__":
    main()