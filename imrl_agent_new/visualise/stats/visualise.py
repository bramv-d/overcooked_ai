import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_IMG_PATH = "fitness_plot.png"


def load_records(path):
    """Load ExperimentRecord list from .pkl file"""
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------ smoothing helpers ------------------------

def moving_average(values, window_size):
    """Simple moving average. Keeps line smooth."""
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')


# ------------------ plot function ----------------------------

def plot_fitness(records, output_path, smoothing_window=5):
    """
    Line plot of FITNESS and LEARNING-PROGRESS over time, one line per goal space.
    Fitness  → solid line (left y-axis)
    LP       → dashed line (right y-axis)
    """
    fitness_by_space = defaultdict(list)
    lp_by_space = defaultdict(list)

    for rec in records:
        key = getattr(rec, "goal_space", str(tuple(rec.goal)))
        fitness_by_space[key].append(rec.fitness)
        lp_by_space[key].append(rec.learning_progress)

    if not fitness_by_space:
        print("❌ No fitness data found.")
        return

    plt.figure(figsize=(11, 6))

    ax_f = plt.gca()  # left axis  (fitness)
    ax_lp = ax_f.twinx()  # right axis (learning progress)

    colors = plt.cm.tab10.colors  # up to 10 distinct colours

    for idx, goal_space in enumerate(fitness_by_space):
        # ----- fitness (solid) -----
        fit_vals = moving_average(fitness_by_space[goal_space], smoothing_window)
        x = range(len(fit_vals))
        ax_f.plot(x, fit_vals, color=colors[idx % 10], label=f"{goal_space} fitness")

        lp_vals = moving_average(lp_by_space[goal_space], smoothing_window)
        ax_lp.plot(x, lp_vals, color=colors[idx % 10], linestyle="--",
                   label=f"{goal_space} LP")

    # ----- axis cosmetics -----
    ax_f.set_xlabel("Roll-out index")
    ax_f.set_ylabel("Fitness (smoothed)")
    ax_lp.set_ylabel("Learning progress (smoothed)")
    ax_f.set_ylim(0, 1.05)  # fitness is in [0, 1]
    ax_f.set_title("Fitness & Learning-Progress per goal space")
    ax_f.grid(True, which="both", linestyle=":")

    # combine legends from both axes
    lines, labels = ax_f.get_legend_handles_labels()
    lines2, labels2 = ax_lp.get_legend_handles_labels()
    ax_f.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved line plot to {output_path}")


# ------------------ load & run -------------------------------
def make_graphs():
    for ag in range(2):
        path = f"kb/buffer_rollouts{ag}.pkl"
        if os.path.exists(path):
            print(f"✅ Found {path}")
            records = load_records(path)
            plot_fitness(records, f"visualise/stats/fitness_plot_{ag}.png", 150)
        else:
            print(f"❌ Could not find {path}")
