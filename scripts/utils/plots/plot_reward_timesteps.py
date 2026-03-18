import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_rewards(path: str) -> np.ndarray:
    # csv format rows: [algorithm, reward, timesteps]
    with open(path, "r") as f:
        # convert to numpy array
        return np.array([line.split(",") for line in f])

def plot_reward_timesteps(reward_timesteps: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    # plot will have a different color line for each algorithm  
    colors = plt.cm.tab10.colors
    for i, row in enumerate(reward_timesteps):
        algorithm, reward, timesteps = row
        timesteps = int(timesteps) / 1000.0 # for better labeling 
        plt.plot(timesteps, float(reward), label=algorithm, color=colors[i])
    # add error bars to denote reward std
    reward_std = reward_timesteps[:, 2]
    plt.errorbar(timesteps, float(reward), yerr=float(reward_std), label=algorithm, color=colors[i])
    plt.xticks(np.arange(0, max(timesteps) + 1, 1))
    plt.xlabel("Timesteps (10e3)")
    plt.ylabel("Reward")
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-path", required=True, help="Path to input csv file")
    ap.add_argument("--out-path", default="algo_reward_timesteps.jpg", help="Path to output plot")
    args = ap.parse_args()

    reward_timesteps = load_rewards(args.reward_timesteps)
    plot_reward_timesteps(reward_timesteps, args.out_path)

if __name__ == "__main__":
    main()