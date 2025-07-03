# make_graphs.py
# ---------------------------------------------------------------------------
# Faster plotting & statistics for IMGEP roll-out buffers
# ---------------------------------------------------------------------------
import json
import os
from collections import Counter
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from base_dir.shared_files.goal_spaces import GoalSpaceEnum


# ------------------- I/O ----------------------------------------------------


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    """Zero-copy load of the new compressed buffer."""
    arr = np.load(path, mmap_mode='r', allow_pickle=False)
    return {
        "goal_id": arr["goal_id"].astype(np.int32, copy=False),
        "theta": arr["theta"],  # NOT used for plots
        "fitness": arr["fitness"].astype(np.float32, copy=False),
        "intr_reward": arr["intr_reward"].astype(np.float32, copy=False),
        "exploit": arr["exploit"].astype(bool, copy=False),
        "rollout_idx": arr["rollout_idx"].astype(np.int32, copy=False),
        "size": int(arr["size"]),
    }


def load_buffer(path: str) -> Dict[str, np.ndarray]:
    """Return a dict of parallel NumPy arrays plus 'size'."""
    return _load_npz(path)


# ------------------- utilities ---------------------------------------------


def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or y.size < k:
        return y
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(y, kernel, mode="valid")


def _unique_sorted(arr: np.ndarray) -> np.ndarray:
    """Return unique values, sorted (NumPy does this in C)."""
    return np.unique(arr)


def _goal_name_array(goal_ids: np.ndarray) -> np.ndarray:
    """Vectorised mapping goal_id -> human-readable name."""
    # NumPy cannot broadcast Python calls, we build a lookup once:
    unique_ids = _unique_sorted(goal_ids)
    lut = np.array([GoalSpaceEnum.get_goal_space_name(g) for g in unique_ids])
    # Use searchsorted to vectorise the mapping
    idx = np.searchsorted(unique_ids, goal_ids)
    return lut[idx]


def get_goal_space_colors(goal_names: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    palette = plt.cm.tab10.colors
    unique_names = _unique_sorted(goal_names)
    return {name: palette[i % len(palette)] for i, name in enumerate(unique_names)}


# ------------------- plotting ----------------------------------------------


def plot_fitness(data, out_path: str, smooth_k: int, colors):
    gid = data["goal_id"]
    fit = data["fitness"]
    ir = data["intr_reward"]
    idx = data["rollout_idx"]
    mask = data["exploit"]

    if not mask.any():
        print("❌ No exploit episodes → no fitness curves.")
        return

    gid_e, fit_e, ir_e, idx_e = gid[mask], fit[mask], ir[mask], idx[mask]
    names = _goal_name_array(gid_e)

    plt.figure(figsize=(11, 6))
    ax_f, ax_ir = plt.gca(), plt.gca().twinx()

    for name in _unique_sorted(names):
        sel = names == name
        x = idx_e[sel]
        order = x.argsort()
        x = x[order]
        y_fit = fit_e[sel][order]
        y_ir = ir_e[sel][order]

        ax_f.plot(x[-len(moving_average(y_fit, smooth_k)):],
                  moving_average(y_fit, smooth_k),
                  color=colors[name], label=f"{name} fitness")
        ax_ir.plot(x[-len(moving_average(y_ir, smooth_k)):],
                   moving_average(y_ir, smooth_k),
                   color=colors[name], linestyle=":", label=f"{name} IR")

    ax_f.set_xlabel("Roll-out index")
    ax_f.set_ylabel("Fitness (smoothed)")
    ax_ir.set_ylabel("Intrinsic reward (smoothed)")
    ax_f.set_title("Fitness & Learning-Progress per goal space")
    ax_f.grid(True, which="both", linestyle=":")

    lines, labels = ax_f.get_legend_handles_labels()
    lines2, labels2 = ax_ir.get_legend_handles_labels()
    ax_f.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Saved line plot → {out_path}")


def plot_goalspace_distribution(data, out_path: str, num_bins: int, colors):
    gid = data["goal_id"]
    idx = data["rollout_idx"]
    mask = data["exploit"]

    if not mask.any():
        print("❌ No exploit episodes → no distribution plot.")
        return

    gid_e, idx_e = gid[mask], idx[mask]
    names = _goal_name_array(gid_e)
    unique_names = _unique_sorted(names)

    # 1) sort by rollout index once
    order = idx_e.argsort()
    idx_sorted = idx_e[order]
    names_sorted = names[order]

    # 2) make bins
    total = idx_sorted.size
    bin_edges = np.linspace(0, total, num_bins + 1, dtype=int)

    counts = np.zeros((num_bins, unique_names.size), dtype=int)
    name_to_col = {n: i for i, n in enumerate(unique_names)}

    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        slice_names = names_sorted[lo:hi]
        if slice_names.size == 0:
            continue
        col_idx, freq = np.unique(slice_names, return_counts=True)
        col_idx = [name_to_col[n] for n in col_idx]
        counts[b, col_idx] = freq

    # 3) convert to proportions
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1  # avoid division by 0
    props = counts / totals

    # 4) stack-bar plot
    x = np.arange(num_bins)
    bottom = np.zeros(num_bins)
    plt.figure(figsize=(12, 6))
    for j, name in enumerate(unique_names):
        plt.bar(x, props[:, j], bottom=bottom, label=name,
                color=colors[name])
        bottom += props[:, j]

    bin_labels = [f"{idx_sorted[bin_edges[b]]}-{idx_sorted[bin_edges[b + 1] - 1]}"
                  for b in range(num_bins)]
    plt.title(f"Goal-space rollout distribution ({num_bins} bins)")
    plt.xlabel("Roll-out bins")
    plt.ylabel("Proportion")
    plt.xticks(x, bin_labels, rotation=45, ha='right')
    plt.grid(axis='y', linestyle=":")
    plt.legend(title="Goal Space", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Saved stacked bar plot → {out_path}")


# ------------------- statistics --------------------------------------------


def print_statistics(buffer_path: str, data, ag):
    mask = data["exploit"]
    if not mask.any():
        print("\n📊 Rollout Statistics: no exploit data.")
        return

    gid = data["goal_id"][mask]
    fit = data["fitness"][mask]
    ir = data["intr_reward"][mask]
    idx = data["rollout_idx"][mask]

    total = gid.size
    print("\n📊 Rollout Statistics")
    print(f"Total exploit rollouts: {total}")
    counts = Counter(gid)
    for g, c in counts.items():
        print(f"  • {GoalSpaceEnum.get_goal_space_name(g):<15}: {c}")

    print(f"Rollout index range: {idx.min()}–{idx.max()}")

    # averages
    for g in _unique_sorted(gid):
        sel = gid == g
        print(f"{GoalSpaceEnum.get_goal_space_name(g):<15}: "
              f"avg fitness = {fit[sel].mean():7.4f}   "
              f"avg IR = {ir[sel].mean():7.4f}")

    total = gid.size
    counts = Counter(gid)

    # Build new stats entry
    new_stats = {
        "total_rollouts": int(total),
        "rollout_index_range": [int(idx.min()), int(idx.max())],
        "goals": {}
    }

    for g in _unique_sorted(gid):
        sel = gid == g
        goal_name = GoalSpaceEnum.get_goal_space_name(g)
        new_stats["goals"][goal_name] = {
            "count": int(counts[g]),
            "avg_fitness": float(np.mean(fit[sel])),
            "avg_ir": float(np.mean(ir[sel]))
        }

    # Append to existing JSON file
    output_file = f"rollout_stats{ag}.json"

    # Load existing data if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                all_stats = json.load(f)
                if not isinstance(all_stats, list):
                    all_stats = [all_stats]
            except json.JSONDecodeError:
                all_stats = []
    else:
        all_stats = []

    # Add the new stats entry
    all_stats.append(new_stats)

    # Write updated list back to file
    with open(output_file, "w") as f:
        json.dump(all_stats, f, indent=4)

    print(f"✅ Added new rollout statistics to {output_file}")
# ------------------- main ---------------------------------------------------


def make_graphs():
    os.makedirs("visualise/stats", exist_ok=True)

    for ag in range(2):
        path = f"../kb/buffer_rollouts{ag}.npz"
        if not os.path.exists(path):
            print(f"❌ {path} not found — skipping.")
            continue

        print(f"✅ Loading {path}")
        data = load_buffer(path)

        goal_names = _goal_name_array(data["goal_id"])
        colors = get_goal_space_colors(goal_names)

        plot_fitness(data,
                     f"visualise/stats/fitness_plot_{ag}.png",
                     smooth_k=5,
                     colors=colors)

        plot_goalspace_distribution(data,
                                    f"visualise/stats/goalspace_dist_{ag}.png",
                                    num_bins=5,
                                    colors=colors)

        print_statistics(path, data, ag)


if __name__ == "__main__":
    make_graphs()
