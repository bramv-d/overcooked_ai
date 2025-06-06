import os
import pickle
from collections import Counter

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


from collections import defaultdict


def plot_goalspace_distribution(records, output_path, smoothing_window):
    """Plot the % of rollouts per goal space over time."""
    goal_counts = defaultdict(list)

    # STEP 0: deduplicate rollout indices
    seen = {}
    for rec in records:
        if rec.exploit:
            seen[rec.rollout_idx] = rec  # keeps the *last* per rollout_idx

    # STEP 1: Tally rollout counts
    per_rollout = defaultdict(lambda: defaultdict(int))
    all_idxs = set()
    for rec in seen.values():
        rollout_idx = rec.rollout_idx
        space = rec.goal_space
        per_rollout[rollout_idx][space] += 1
        all_idxs.add(rollout_idx)

    all_idxs = sorted(all_idxs)
    all_spaces = sorted({space for counts in per_rollout.values() for space in counts})

    # 2. For each goal space, collect a % per rollout index
    for space in all_spaces:
        percentages = []
        for i in all_idxs:
            total = sum(per_rollout[i].values())
            pct = per_rollout[i][space] / total if total > 0 else 0
            percentages.append(pct)
        # 3. Smooth and store
        goal_counts[space] = moving_average(percentages, smoothing_window)

    # 4. Plotting
    plt.figure(figsize=(11, 5))
    for idx, (space, y_vals) in enumerate(goal_counts.items()):
        x_vals = all_idxs[len(all_idxs) - len(y_vals):]
        plt.plot(x_vals, y_vals, label=space, color=plt.cm.tab10(idx))

    plt.title("Goal-space rollout distribution over time")
    plt.xlabel("Roll-out index")
    plt.ylabel("Proportion (smoothed)")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=":")
    plt.legend(title="Goal Space", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved distribution plot to {output_path}")


# ------------------ load & run -------------------------------
def make_graphs():
    for ag in range(2):
        path = f"kb/buffer_rollouts{ag}.pkl"
        if os.path.exists(path):
            print(f"✅ Found {path}")
            records = load_records(path)
            plot_fitness(records, f"visualise/stats/fitness_plot_{ag}.png", 20)
            plot_goalspace_distribution(records, f"visualise/stats/goalspace_dist_{ag}.png", 20)
            get_statistics(path)
        else:
            print(f"❌ Could not find {path}")


def get_statistics(pickle_path):
    """Load data from a pickle file and compute rollout-level statistics."""
    with open(pickle_path, "rb") as f:
        records = pickle.load(f)

    # STEP 1: Keep only one record per rollout index (e.g. latest)
    rollout_records = {}
    for rec in records:
        if getattr(rec, "exploit", False):
            rollout_records[rec.rollout_idx] = rec  # keeps latest per index

    filtered = list(rollout_records.values())

    # STEP 2: Compute stats
    total = len(filtered)
    goal_space_counts = Counter(rec.goal_space for rec in filtered)
    rollout_indices = [rec.rollout_idx for rec in filtered]
    min_rollout = min(rollout_indices) if rollout_indices else 0
    max_rollout = max(rollout_indices) if rollout_indices else 0

    # Compute averages per goal space
    fitness_by_space = defaultdict(list)
    ir_by_space = defaultdict(list)
    for rec in filtered:
        fitness_by_space[rec.goal_space].append(rec.fitness)
        ir_by_space[rec.goal_space].append(rec.intrinsic_reward)

    avg_fitness_by_space = {space: np.mean(values) for space, values in fitness_by_space.items()}
    avg_ir_by_space = {space: np.mean(values) for space, values in ir_by_space.items()}

    # STEP 3: Print summary
    print("\n📊 Rollout Statistics")
    print(f"Total rollouts (deduplicated): {total}")
    print(f"Goal space counts:")
    for space, count in goal_space_counts.items():
        print(f"  • {space:<15} : {count}")
    print(f"Rollout index range : {min_rollout} to {max_rollout}")
    print("Average fitness per goal space:")
    for space, avg in avg_fitness_by_space.items():
        print(f"  • {space:<15} : {avg:.4f}")
    print("Average intrinsic reward per goal space:")
    for space, avg in avg_ir_by_space.items():
        print(f"  • {space:<15} : {avg:.4f}")
