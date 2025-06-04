import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from imrl_agent_new.overcooked.outcome import int_to_item

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
            goal = getattr(rec, "goal", str(rec.goal))
            label = goal_space + " " + int_to_item(goal)
            fitness_by_space[label].append((rec.rollout_idx, rec.fitness))
            ir_by_space[label].append((rec.rollout_idx, rec.intrinsic_reward))

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


def plot_competence(records, output_path):
    """
    Line plot of COMPETENCE over time, one line per goal space.
    """
    competence_by_space = defaultdict(list)

    for rec in records:
        if rec.exploit:
            key = getattr(rec, "goal_space", str(tuple(rec.goal)))
            if rec.rollout_idx not in competence_by_space[key]:
                competence_by_space[key].append(rec.fitness)

    if not competence_by_space:
        print("❌ No competence data found.")
        return

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    colors = plt.cm.tab10.colors  # up to 10 distinct colours

    for idx, goal_space in enumerate(competence_by_space):
        comp_vals = moving_average(competence_by_space[goal_space], 1)
        x = range(len(comp_vals))
        ax.plot(x, comp_vals, color=colors[idx % 10], label=f"{goal_space} competence")

    # ----- axis cosmetics -----
    ax.set_xlabel("Roll-out index")
    ax.set_ylabel("Competence (smoothed)")
    ax.set_ylim(0, 1.05)  # competence is in [0, 1]
    ax.set_title("Competence per goal space")
    ax.grid(True, which="both", linestyle=":")

    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved competence plot to {output_path}")


# ------------------ load & run -------------------------------
def make_graphs():
    for ag in range(2):
        path = f"kb/buffer_rollouts{ag}.pkl"
        if os.path.exists(path):
            print(f"✅ Found {path}")
            records = load_records(path)
            plot_fitness(records, f"visualise/stats/fitness_plot_{ag}.png", 50)
        else:
            print(f"❌ Could not find {path}")
