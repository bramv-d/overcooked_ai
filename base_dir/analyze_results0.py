import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ====== CONFIGURATION ======
BASE_DIR = "proto1/results1/layout20"  # Change this to your desired directory path
# ===========================

layout_name = os.path.basename(os.path.normpath(BASE_DIR))


def load_fitness_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    records = []
    for entry in data:
        goals = entry.get("goals", {})
        for goal, goal_data in goals.items():
            records.append({
                "goal": goal,
                "fitness": goal_data.get("avg_fitness", 0.0)
            })
    return pd.DataFrame(records)


# Load data
agent0_path = os.path.join(BASE_DIR, "rollout_stats0.json")
agent1_path = os.path.join(BASE_DIR, "rollout_stats1.json")

df0 = load_fitness_data(agent0_path)
df1 = load_fitness_data(agent1_path)

df0["agent"] = "agent0"
df1["agent"] = "agent1"

df = pd.concat([df0, df1], ignore_index=True)

# Compute mean and std per goal and total
summary = df.groupby(["agent", "goal"])["fitness"].agg(["mean", "std"]).reset_index()
summary_total = df.groupby("agent")["fitness"].agg(["mean", "std"]).reset_index()
summary_total["goal"] = "TOTAL"
summary = pd.concat([summary, summary_total], ignore_index=True)

# Generate output filenames based on layout name
latex_filename = f"{layout_name}_results_summary.tex"
boxplot_filename = f"{layout_name}_fitness_boxplot.png"

latex_path = os.path.join(BASE_DIR, latex_filename)
boxplot_path = os.path.join(BASE_DIR, boxplot_filename)

summary["goal"] = summary["goal"].str.replace("_", "\\_", regex=False)

# Save LaTeX table
summary.to_latex(latex_path, index=False, float_format="%.4f")

# Create and save boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="goal", y="fitness", hue="agent")
plt.title(f"Fitness Distribution per Goal - {layout_name}")
plt.ylabel("Fitness")
plt.xlabel("Goal")
plt.tight_layout()
plt.savefig(boxplot_path)
plt.close()
