import math
import os
import pickle
from collections import Counter, defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from HiMAGE.agent.goals.goal_spaces import GoalSpaceEnum
from HiMAGE.agent.knowledge_base import RolloutRecord

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


# ------------------ plot functions ----------------------------

def plot_fitness(records, output_path, smoothing_window, goal_space_colors):
    fitness_by_space = defaultdict(list)
    ir_by_space = defaultdict(list)

    for rec in records:
        if rec.exploit:
            label = GoalSpaceEnum.get_goal_space_name(rec.goal_id)
            fitness_by_space[label].append((rec.rollout_idx, rec.fitness))
            ir_by_space[label].append((rec.rollout_idx, rec.intrinsic_reward))

    if not fitness_by_space:
        print("❌ No fitness data found.")
        return

    plt.figure(figsize=(11, 6))
    ax_f = plt.gca()
    ax_ir = ax_f.twinx()

    for label in sorted(fitness_by_space):
        fitness_sorted = sorted(fitness_by_space[label])
        ir_sorted = sorted(ir_by_space[label])

        x_fit, y_fit = zip(*fitness_sorted)
        x_ir, y_ir = zip(*ir_sorted)

        y_fit_smooth = moving_average(y_fit, smoothing_window)
        y_ir_smooth = moving_average(y_ir, smoothing_window)

        x_fit = x_fit[-len(y_fit_smooth):]
        x_ir = x_ir[-len(y_ir_smooth):]

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


def plot_shared_episode_reward(records, output_path, smoothing_window, goal_space_colors):
    shared_reward_by_space = defaultdict(list)

    for rec in records:
        if rec.exploit:
            label = rec.goal_id
            shared_reward = rec.shared_episode_reward
            shared_reward_by_space[label].append((rec.rollout_idx, shared_reward))

    if not shared_reward_by_space:
        print("❌ No shared_episode_reward data found.")
        return

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    for label in sorted(shared_reward_by_space):
        reward_sorted = sorted(shared_reward_by_space[label])
        x_vals, y_vals = zip(*reward_sorted)

        y_smooth = moving_average(y_vals, smoothing_window)
        x_vals = x_vals[-len(y_smooth):]

        color = goal_space_colors[label]
        ax.plot(x_vals, y_smooth, color=color, label=f"{label} shared reward")

    ax.set_xlabel("Roll-out index")
    ax.set_ylabel("Shared Episode Reward (smoothed)")
    ax.set_title("Shared Episode Reward per goal space")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved shared_episode_reward plot to {output_path}")


def plot_goalspace_distribution(records, output_path, num_bins, goal_space_colors):
    """
    Plot a stacked bar chart of goal-space distribution per bin.
    Uses string labels for goal spaces to match color mapping.
    """
    # STEP 1: Deduplicate and filter only exploit records
    seen = {}
    for rec in records:
        if rec.exploit:
            seen[rec.rollout_idx] = rec

    # STEP 2: Tally counts per rollout index, using string labels
    per_rollout = defaultdict(lambda: defaultdict(int))
    all_idxs = set()
    for rec in seen.values():
        idx = rec.rollout_idx
        label = GoalSpaceEnum.get_goal_space_name(rec.goal_id)
        per_rollout[idx][label] += 1
        all_idxs.add(idx)

    all_idxs = sorted(all_idxs)
    all_spaces = sorted({label for counts in per_rollout.values() for label in counts})

    # STEP 3: Create bins
    total_rollouts = len(all_idxs)
    bin_size = math.ceil(total_rollouts / num_bins)
    bins = []
    current = []
    for i in all_idxs:
        current.append(i)
        if len(current) == bin_size:
            bins.append(current)
            current = []
    if current:
        bins.append(current)

    # STEP 4: Compute percentages per bin
    goal_counts_per_bin = defaultdict(list)
    for bin_idxs in bins:
        counts = defaultdict(int)
        total = 0
        for i in bin_idxs:
            for label in all_spaces:
                counts[label] += per_rollout[i][label]
            total += sum(per_rollout[i].values())
        for label in all_spaces:
            pct = counts[label] / total if total > 0 else 0
            goal_counts_per_bin[label].append(pct)

    # STEP 5: Plot
    x = np.arange(len(bins))
    bottom = np.zeros(len(bins))
    plt.figure(figsize=(12, 6))
    for label in all_spaces:
        y = goal_counts_per_bin[label]
        plt.bar(x, y, bottom=bottom, label=label, color=goal_space_colors[label])
        bottom += y

    plt.title(f"Goal-space rollout distribution ({num_bins} bins)")
    plt.xlabel(f"Bins (~{bin_size} rollouts each)")
    plt.ylabel("Proportion per bin")
    plt.xticks(x, [f"{b[0]}-{b[-1]}" for b in bins], rotation=45, ha='right')
    plt.grid(axis='y', linestyle=":")
    plt.legend(title="Goal Space", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved stacked bar distribution plot to {output_path}")


def make_graphs():
    for ag in range(2):
        path = f"agent/kb/buffer_rollouts{ag}.pkl"
        if not os.path.exists(path):
            print(f"❌ Could not find {path}")
            continue

        print(f"✅ Found {path}")
        records = load_records(path)

        all_spaces = sorted({rec.goal_id for rec in records})
        goal_space_colors = get_goal_space_colors(all_spaces)

        plot_fitness(records,
                     f"visualise/stats/fitness_plot_{ag}.png",
                     smoothing_window=5,
                     goal_space_colors=goal_space_colors)

        plot_goalspace_distribution(records,
                                    f"visualise/stats/goalspace_dist_{ag}.png",
                                    num_bins=5,
                                    goal_space_colors=goal_space_colors)

        # plot_shared_episode_reward(records,
        #                            f"visualise/stats/shared_reward_plot_{ag}.png",
        #                            smoothing_window=5,
        #                            goal_space_colors=goal_space_colors)

        get_statistics(path)


def get_statistics(pickle_path):
    with open(pickle_path, "rb") as f:
        records: List[RolloutRecord] = pickle.load(f)

    rollout_records = {rec.rollout_idx: rec for rec in records if rec.exploit}
    filtered = list(rollout_records.values())

    total = len(filtered)
    goal_space_counts = Counter(rec.goal_id for rec in filtered)
    idxs = [rec.rollout_idx for rec in filtered]
    print("\n📊 Rollout Statistics")
    print(f"Total rollouts: {total}")
    for space, count in goal_space_counts.items():
        print(f"  • {space:<15}: {count}")
    if idxs:
        print(f"Rollout index range: {min(idxs)} to {max(idxs)}")

    fitness_by = defaultdict(list)
    ir_by = defaultdict(list)
    for rec in filtered:
        fitness_by[rec.goal_id].append(rec.fitness)
        ir_by[rec.goal_id].append(rec.intrinsic_reward)

    avg_fit = {s: np.mean(v) for s, v in fitness_by.items()}
    avg_ir = {s: np.mean(v) for s, v in ir_by.items()}
    print("Average fitness per goal space:")
    for s, a in avg_fit.items(): print(f"  • {s:<15}: {a:.4f}")
    print("Average intrinsic reward per goal space:")
    for s, a in avg_ir.items(): print(f"  • {s:<15}: {a:.4f}")


def get_goal_space_colors(all_spaces):
    colors = plt.cm.tab10.colors
    all_spaces = [GoalSpaceEnum.get_goal_space_name(s) for s in all_spaces]
    return {space: colors[idx % len(colors)]
            for idx, space in enumerate(sorted(all_spaces, key=str))}
