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

def plot_fitness(records, output_path, smoothing_window):
    fitness_by_space = defaultdict(list)
    ir_by_space = defaultdict(list)

    for rec in records:
        if rec.exploit:
            goal_space = getattr(rec, "goal_space", None)
            # goal = getattr(rec, "goal", str(rec.goal))
            label = goal_space
            fitness_by_space[label].append((rec.rollout_idx, rec.fitness))
            ir_by_space[label].append((rec.rollout_idx, rec.intrinsic_reward))

    for label, records in fitness_by_space.items():
        rollout_idxs = sorted(idx for idx, _ in records)
        print(f"{label} starts at rollout {rollout_idxs[0]}")

    if not fitness_by_space:
        print("❌ No fitness data found.")
        return

    plt.figure(figsize=(11, 6))
    ax_f = plt.gca()
    ax_ir = ax_f.twinx()

    colors = plt.cm.tab10.colors

    for idx, label in enumerate(fitness_by_space):
        # sort by rollout index to ensure lines are smooth
        fitness_sorted = sorted(fitness_by_space[label])
        ir_sorted = sorted(ir_by_space[label])

        x_fit, y_fit = zip(*fitness_sorted)
        x_ir, y_ir = zip(*ir_sorted)

        # apply smoothing
        y_fit_smooth = moving_average(y_fit, smoothing_window)
        y_ir_smooth = moving_average(y_ir, smoothing_window)

        # adjust x accordingly
        x_fit = x_fit[len(x_fit) - len(y_fit_smooth):]
        x_ir = x_ir[len(x_ir) - len(y_ir_smooth):]

        ax_f.plot(x_fit, y_fit_smooth, color=colors[idx % 10], label=f"{label} fitness")
        ax_ir.plot(x_ir, y_ir_smooth, color=colors[idx % 10], linestyle=":", label=f"{label} IR")

    ax_f.set_xlabel("Roll-out index")
    ax_f.set_ylabel("Fitness (smoothed)")
    ax_ir.set_ylabel("Intrinsic reward (smoothed)")
    ax_f.set_ylim(0, 1.05)
    ax_f.set_title("Fitness & Learning-Progress per goal space")
    ax_f.grid(True, which="both", linestyle=":")

    lines, labels = ax_f.get_legend_handles_labels()
    lines2, labels2 = ax_ir.get_legend_handles_labels()
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
            plot_fitness(records, f"visualise/stats/fitness_plot_{ag}.png", 30)
        else:
            print(f"❌ Could not find {path}")
