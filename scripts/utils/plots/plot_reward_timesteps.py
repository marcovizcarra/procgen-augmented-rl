import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_data_from_csv(path: str) -> np.ndarray:
    # csv format rows: [algorithm, reward_mean, reward_std, timesteps]
    with open(path, "r") as f:
        # convert to numpy array
        return np.array([line.split(",") for line in f.readlines()[1:]])

def plot_data(data: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    # build timeseries per algorithm: x=data size, y=reward mean.
    grouped_points: dict[str, list[tuple[float, float, float]]] = {}
    for row in data:
        algorithm, reward_mean, reward_std, data_size = row[:4]

        # skip rows without reward mean
        if reward_mean == "":
            continue

        x = int(data_size) / 1000  # show x-axis in thousands
        y = float(reward_mean)
        grouped_points.setdefault(algorithm, []).append((x, y))

    cmap = plt.colormaps.get_cmap("tab10")
    for i, (algorithm, points) in enumerate(grouped_points.items()):
        points = sorted(points, key=lambda point: point[0])
        x_vals = [point[0] for point in points]
        y_vals = [point[1] for point in points]
        color = cmap(i % cmap.N)
        plt.errorbar(
            x_vals,
            y_vals,
            marker="o",
            linestyle="-",
            label=algorithm,
            color=color,
        )

    # put x ticks at all observed data sizes.
    x_ticks = sorted({point[0] for points in grouped_points.values() for point in points})
    if x_ticks:
        plt.xticks(x_ticks)
    plt.xlabel("Data Size (10e3)")
    plt.ylabel("Reward")
    plt.ylim(0, 4)
    plt.title(title)
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True, help="Path to input csv file")
    ap.add_argument("--out-path", default="reward_vs_datasize.jpg", help="Path to output plot")
    args = ap.parse_args()

    data = load_data_from_csv(args.input_path)
    title = "Reward vs Data Size"
    plot_data(data, title, args.out_path)

if __name__ == "__main__":
    main()