# make_all_graphs.py
# ---------------------------------------------------------------------------
# Run fitness graph generation for each prototype
# Output goes to: base_dir/stats/PrototypeX/
# ---------------------------------------------------------------------------

import os

import matplotlib.pyplot as plt
import numpy as np

from base_dir.shared_files.goal_spaces import GoalSpaceEnum

# ----------- CONFIG --------------------------------------------------------

PROTOTYPES = ["proto1", "proto2", "proto3", "proto4"]
LAYOUTS = ["layout20", "layout22"]
AGENT_IDS = [0, 1]
NUM_FILES = 50
SMOOTH_K = 10


# ----------- I/O -----------------------------------------------------------

def load_buffer(path: str) -> dict:
    """Read one compressed buffer file and return a dict of ndarray views.
    Tries to read an 'ir' (intrinsic-reward) field, falling back to zeros if
    the field is missing so the rest of the pipeline keeps working."""
    arr = np.load(path, mmap_mode='r', allow_pickle=False)

    out = {
        "goal_id": arr["goal_id"].astype(np.int32, copy=False),
        "fitness": arr["fitness"].astype(np.float32, copy=False),
        "exploit": arr["exploit"].astype(bool, copy=False),
        "rollout_idx": arr["rollout_idx"].astype(np.int32, copy=False),
        "ir": arr["intr_reward"].astype(np.float32, copy=False)
    }
    return out


def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or y.size < k:
        return y
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(y, kernel, mode="valid")


# ----------- Plotting ------------------------------------------------------

def plot_fitness_per_goalspace(prototype: str, agent_id: int, root_dir: str, layout: str):
    print(f"\n📊 Generating plots for {prototype} – Agent {agent_id}")
    used_goal_ids = set()

    for i in range(1, NUM_FILES + 1):
        path = os.path.join(root_dir, prototype, layout, "buffer_rollouts", str(i), f"buffer_agent{agent_id}.npz")
        if os.path.exists(path):
            data = load_buffer(path)
            used_goal_ids.update(np.unique(data["goal_id"]))

    used_goal_ids = sorted(used_goal_ids)
    output_dir = os.path.join(root_dir, "stats", layout, prototype)
    os.makedirs(output_dir, exist_ok=True)

    for goal_id in used_goal_ids:
        goal_name = GoalSpaceEnum.get_goal_space_name(goal_id)
        plt.figure(figsize=(12, 6))
        has_data = False

        all_x = []
        all_y = []

        for i in range(1, NUM_FILES + 1):
            path = os.path.join(root_dir, prototype, layout, "buffer_rollouts", str(i), f"buffer_agent{agent_id}.npz")
            if not os.path.exists(path):
                continue

            data = load_buffer(path)
            mask = (data["goal_id"] == goal_id) & data["exploit"]
            if not mask.any():
                continue

            has_data = True
            fitness = data["fitness"][mask]
            rollout_idx = data["rollout_idx"][mask]

            order = rollout_idx.argsort()
            fitness = fitness[order]
            rollout_idx = rollout_idx[order]

            smoothed = moving_average(fitness, SMOOTH_K)
            x = rollout_idx[:len(smoothed)]

            all_x.append(x)
            all_y.append(smoothed)

            plt.plot(x, smoothed, linewidth=0.5, color="lightgrey", alpha=0.6)

        if not has_data:
            print(f"⚠️  No data for {prototype}, Agent {agent_id}, Goal '{goal_name}' — skipping.")
            plt.close()
            continue

        x_common = np.arange(5000)
        y_interp = []
        for x_vals, y_vals in zip(all_x, all_y):
            if len(x_vals) < 2:
                continue
            y_interp.append(np.interp(x_common, x_vals, y_vals))

        if y_interp:
            y_mean = np.mean(y_interp, axis=0)
            plt.plot(x_common, y_mean, color="black", linewidth=2, label="Average")

        plt.title(f"{prototype} – Agent {agent_id} – Goal: {goal_name}")
        plt.xlabel("Rollout Index (0–4999)")
        plt.ylabel("Fitness (smoothed)")
        plt.ylim(0, 1)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()

        filename = f"{prototype}_agent{agent_id}_goal_{goal_name.replace(' ', '_')}.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {out_path}")


# ----------- NEW: cross-prototype comparison --------------------------------
from itertools import cycle


def plot_compare_prototypes(agent_id: int, root_dir: str, layout):
    """Create one figure per goal-space that overlays the average curves from
    proto1–4.  Only the prototype averages are shown (no individual runs)."""
    # 1. collect the list of *all* goal_ids that appear in any prototype
    goal_ids_all = set()
    for proto in PROTOTYPES:
        for i in range(1, NUM_FILES + 1):
            p = os.path.join(root_dir, proto, layout, "buffer_rollouts", str(i),
                             f"buffer_agent{agent_id}.npz")
            if os.path.exists(p):
                goal_ids_all.update(np.unique(load_buffer(p)["goal_id"]))
    goal_ids_all = sorted(goal_ids_all)

    # 2. output directory
    out_dir = os.path.join(root_dir, "stats", layout, "comparison", f"agent{agent_id}")
    os.makedirs(out_dir, exist_ok=True)

    # 3. a simple colour cycle (user explicitly asked for different colours)
    colours = cycle(["tab:blue", "tab:orange", "tab:green", "tab:red"])

    # 4. per-goal plot
    for goal_id in goal_ids_all:
        plt.figure(figsize=(12, 6))
        plotted_any = False
        for proto, c in zip(PROTOTYPES, colours):
            # gather all smoothed curves for this (proto, agent, goal)
            all_interp = []
            for i in range(1, NUM_FILES + 1):
                p = os.path.join(root_dir, proto, layout, "buffer_rollouts", str(i),
                                 f"buffer_agent{agent_id}.npz")
                if not os.path.exists(p):
                    continue
                d = load_buffer(p)
                m = (d["goal_id"] == goal_id) & d["exploit"]
                if not m.any():
                    continue
                f = d["fitness"][m]
                r = d["rollout_idx"][m]
                order = r.argsort()
                f, r = f[order], r[order]
                s = moving_average(f, SMOOTH_K)
                if len(s) < 2:
                    continue
                x_common = np.arange(5000)
                all_interp.append(np.interp(x_common, r[:len(s)], s))
            if not all_interp:
                continue  # nothing for this prototype/goal
            y_mean = np.mean(all_interp, axis=0)
            plt.plot(x_common, y_mean, label=proto, color=c, linewidth=2)
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        goal_name = GoalSpaceEnum.get_goal_space_name(goal_id)
        plt.title(f"Agent {agent_id} – Goal: {goal_name}\nPrototype comparison")
        plt.xlabel("Rollout Index (0–4999)")
        plt.ylabel("Fitness (smoothed average)")
        plt.ylim(0, 1)
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()

        fname = f"compare_agent{agent_id}_goal_{goal_name.replace(' ', '_')}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()
        print(f"📈   Saved comparison: {os.path.join(out_dir, fname)}")


# ----------- 2. LaTeX table generation -------------------------------------
import pandas as pd

TABLE_FMT = "llrrrr"  # matches your example
FLOAT = lambda x: f"{x:.2f}"


def build_stats_table(root_dir: str, layout: str, prototype: str) -> None:
    """Scan all 50 rollout-buffers, compute summary statistics, and dump a
    LaTeX table (one per layout-prototype)."""
    rows = []  # collects dict-rows
    for agent_id in AGENT_IDS:
        # --- aggregate by goal name -------------------------------------------------
        fit_by_goal, ir_by_goal = {}, {}
        for i in range(1, NUM_FILES + 1):
            path = os.path.join(root_dir, prototype, layout,
                                "buffer_rollouts", str(i),
                                f"buffer_agent{agent_id}.npz")
            if not os.path.exists(path):
                continue
            dat = load_buffer(path)
            mask = dat["exploit"]  # only evaluated rollouts
            gids, fit, ir = dat["goal_id"][mask], dat["fitness"][mask], dat["ir"][mask]

            for g, f, r in zip(gids, fit, ir):
                name = GoalSpaceEnum.get_goal_space_name(g)
                fit_by_goal.setdefault(name, []).append(f)
                ir_by_goal.setdefault(name, []).append(r)

            # TOTAL row -----------------------------------------------------
            fit_by_goal.setdefault("TOTAL", []).extend(fit)
            ir_by_goal.setdefault("TOTAL", []).extend(ir)

        # --- append one row per goal ---------------------------------------
        for goal in sorted(fit_by_goal):  # deterministic ordering
            f_arr, r_arr = np.asarray(fit_by_goal[goal]), np.asarray(ir_by_goal[goal])
            rows.append({
                "Agent": agent_id,
                "Goal": goal,
                "Mean Fitness": FLOAT(f_arr.mean() if f_arr.size else 0.0),
                "Std Fitness": FLOAT(f_arr.std(ddof=0) if f_arr.size else 0.0),
                "Mean IR": FLOAT(r_arr.mean() if r_arr.size else 0.0),
                "Std IR": FLOAT(r_arr.std(ddof=0) if r_arr.size else 0.0),
            })

    # --- DataFrame ➜ LaTeX ---------------------------------------------------
    df = pd.DataFrame(rows,
                      columns=["Agent", "Goal",
                               "Mean Fitness", "Std Fitness",
                               "Mean IR", "Std IR"])
    tex = df.to_latex(index=False,
                      column_format=TABLE_FMT,
                      escape=False,
                      na_rep="0.00")
    # ensure top/bottom rules match your snippet
    tex = tex.replace("\\midrule", "\\midrule\n")  # nicer spacing
    # write to file
    out_dir = os.path.join(root_dir, "stats", layout, prototype)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.tex"), "w") as fh:
        fh.write(tex)
    print(f"📄  LaTeX table saved to: {os.path.join(out_dir, 'summary.tex')}")


# ----------- Main Loop -----------------------------------------------------

def make_all_graphs():
    root_dir = os.path.dirname(os.path.abspath(__file__))  # base_dir
    for layout in LAYOUTS:
        for prototype in PROTOTYPES:
            for agent_id in AGENT_IDS:
                plot_fitness_per_goalspace(prototype, agent_id, root_dir, layout)
            build_stats_table(root_dir, layout, prototype)

        # --- new: cross-prototype plots ---
        for agent_id in AGENT_IDS:
            plot_compare_prototypes(agent_id, root_dir, layout)


if __name__ == "__main__":
    make_all_graphs()
