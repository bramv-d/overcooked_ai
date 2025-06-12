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

def plot_fitness(records, output_path, smoothing_window, goal_space_colors):
    fitness_by_space = defaultdict(list)
    ir_by_space = defaultdict(list)

    for rec in records:
        if rec.exploit:
            goal_space = getattr(rec, "goal_space", None)
            label = goal_space
            fitness_by_space[label].append((rec.rollout_idx, rec.fitness))
            ir_by_space[label].append((rec.rollout_idx, rec.intrinsic_reward))

    if not fitness_by_space:
        print("❌ No fitness data found.")
        return

    plt.figure(figsize=(11, 6))
    ax_f = plt.gca()
    ax_ir = ax_f.twinx()

    for label in sorted(fitness_by_space):
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

        color = goal_space_colors[label]

        ax_f.plot(x_fit, y_fit_smooth, color=color, label=f"{label} fitness")
        ax_ir.plot(x_ir, y_ir_smooth, color=color, linestyle=":", label=f"{label} IR")

    ax_f.set_xlabel("Roll-out index")
    ax_f.set_ylabel("Fitness (smoothed)")
    ax_ir.set_ylabel("Intrinsic reward (smoothed)")
    ax_f.set_title("Fitness & Learning-Progress per goal space")
    ax_f.grid(True, which="both", linestyle=":")

    lines, labels = ax_f.get_legend_handles_labels()
    lines2, labels2 = ax_ir.get_legend_handles_labels()
    ax_f.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved line plot to {output_path}")


from collections import defaultdict

import math


def plot_goalspace_distribution(records, output_path, num_bins, goal_space_colors):
    goal_counts_per_bin = defaultdict(list)

    # STEP 0: deduplicate rollout indices
    seen = {}
    for rec in records:
        seen[rec.rollout_idx] = rec

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

    # STEP 2: Compute bin size
    total_rollouts = len(all_idxs)
    bin_size = math.ceil(total_rollouts / num_bins)

    # STEP 3: Bin the rollouts
    bins = []
    current_bin = []
    for idx in all_idxs:
        current_bin.append(idx)
        if len(current_bin) == bin_size:
            bins.append(current_bin)
            current_bin = []
    if current_bin:
        bins.append(current_bin)

    # STEP 4: For each bin, compute % of each goal space
    for bin_idxs in bins:
        total_counts = defaultdict(int)
        total_in_bin = 0
        for idx in bin_idxs:
            for space in all_spaces:
                total_counts[space] += per_rollout[idx][space]
            total_in_bin += sum(per_rollout[idx].values())

        for space in all_spaces:
            pct = total_counts[space] / total_in_bin if total_in_bin > 0 else 0
            goal_counts_per_bin[space].append(pct)

    # STEP 5: Plot stacked bar chart
    x_vals = np.arange(len(bins))
    bottom_vals = np.zeros(len(bins))

    plt.figure(figsize=(12, 6))
    for space in sorted(all_spaces):
        y_vals = goal_counts_per_bin[space]
        color = goal_space_colors[space]
        plt.bar(x_vals, y_vals, bottom=bottom_vals, label=space, color=color)
        bottom_vals += y_vals

    plt.title(f"Goal-space rollout distribution ({num_bins} bins)")
    plt.xlabel(f"Bins (variable size: ~{bin_size} rollouts per bin)")
    plt.ylabel("Proportion per bin")
    plt.xticks(ticks=x_vals, labels=[f"{bin_idxs[0]}-{bin_idxs[-1]}" for bin_idxs in bins], rotation=45, ha='right')
    plt.grid(axis='y', linestyle=":")
    plt.legend(title="Goal Space", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved stacked bar distribution plot to {output_path}")


# ------------------ load & run -------------------------------
def make_graphs():
    for ag in range(2):
        path = f"kb/buffer_rollouts{ag}.pkl"
        if os.path.exists(path):
            print(f"✅ Found {path}")
            records = load_records(path)

            # 1️⃣ Collect all goal spaces
            all_spaces = sorted({rec.goal_space for rec in records if rec.exploit})

            # 2️⃣ Build consistent color map
            goal_space_colors = get_goal_space_colors(all_spaces)

            # 3️⃣ Plot with consistent colors
            plot_fitness(records, f"visualise/stats/fitness_plot_{ag}.png", 5, goal_space_colors)
            plot_goalspace_distribution(records, f"visualise/stats/goalspace_dist_{ag}.png", 5, goal_space_colors)
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


def get_goal_space_colors(all_spaces):
    """Assign consistent colors to goal spaces."""
    colors = plt.cm.tab10.colors
    color_map = {}
    for idx, space in enumerate(sorted(all_spaces)):
        color_map[space] = colors[idx % len(colors)]
    return color_map
