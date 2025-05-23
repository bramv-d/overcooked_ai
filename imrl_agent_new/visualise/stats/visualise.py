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

def moving_average(values, window_size=1):
    """Simple moving average. Keeps line smooth."""
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')


# ------------------ plot function ----------------------------

def plot_fitness(records, output_path, smoothing_window=1):
    """
    Line plot of fitness over time, one line per goal space.
    Applies moving average smoothing.
    """
    fitness_by_space = defaultdict(list)

    for rec in records:
        key = getattr(rec, "goal_space", str(tuple(rec.goal)))
        fitness_by_space[key].append(rec.fitness)

    if not fitness_by_space:
        print("❌ No fitness data found.")
        return

    plt.figure(figsize=(10, 6))

    for goal_space, fitness_vals in fitness_by_space.items():
        smoothed = moving_average(fitness_vals, smoothing_window)
        x_vals = list(range(len(smoothed)))
        plt.plot(x_vals, smoothed, label=goal_space)

    plt.xlabel("Roll-out index")
    plt.ylabel("Fitness (smoothed)")
    plt.title("Fitness per goal space over time")
    plt.legend()
    plt.grid(True)
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
            plot_fitness(records, f"visualise/stats/fitness_plot_{ag}.png")
        else:
            print(f"❌ Could not find {path}")
