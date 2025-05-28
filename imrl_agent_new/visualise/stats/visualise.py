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
    """
    Line plot of FITNESS and LEARNING-PROGRESS over time, one line per goal space.
    Fitness  → solid line (left y-axis)
    LP       → dashed line (right y-axis)
    """
    fitness_by_space = defaultdict(list)
    ir_by_space = defaultdict(list)

    for rec in records:
        if rec.exploit:
            key = getattr(rec, "goal", str(tuple(rec.goal)))[0]
            fitness_by_space[key].append(rec.fitness)
            ir_by_space[key].append(rec.intrinsic_reward)

    if not fitness_by_space:
        print("❌ No fitness data found.")
        return

    plt.figure(figsize=(11, 6))

    ax_f = plt.gca()  # left axis  (fitness)
    ax_ir = ax_f.twinx()  # right axis (intrinsic reward)

    colors = plt.cm.tab10.colors  # up to 10 distinct colours

    for idx, goal_space in enumerate(fitness_by_space):
        # ----- fitness (solid) -----
        fit_vals = moving_average(fitness_by_space[goal_space], smoothing_window)
        x = range(len(fit_vals))
        ax_f.plot(x, fit_vals, color=colors[idx % 10], label=f"{goal_space} fitness")

        ir_vals = moving_average(ir_by_space[goal_space], smoothing_window)
        ax_ir.plot(x, ir_vals, color=colors[idx % 10], linestyle=":",
                   label=f"{goal_space} IR")

    # ----- axis cosmetics -----
    ax_f.set_xlabel("Roll-out index")
    ax_f.set_ylabel("Fitness (smoothed)")
    ax_ir.set_ylabel("Intrinsic reward (smoothed)")

    ax_f.set_ylim(0, 1.05)  # fitness is in [0, 1]
    ax_f.set_title("Fitness & Learning-Progress per goal space")
    ax_f.grid(True, which="both", linestyle=":")

    # combine legends from both axes
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
            competence_by_space[key].append(rec.fitness)

    if not competence_by_space:
        print("❌ No competence data found.")
        return

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    colors = plt.cm.tab10.colors  # up to 10 distinct colours

    for idx, goal_space in enumerate(competence_by_space):
        comp_vals = moving_average(competence_by_space[goal_space], 20)
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
            plot_fitness(records, f"visualise/stats/fitness_plot_{ag}.png", 150)
        else:
            print(f"❌ Could not find {path}")
