import json
import numpy as np
from collections import defaultdict

# Load data
with open("rollout_stats.json", "r") as f:
    data = json.load(f)

# Track stats by goal
goal_stats = defaultdict(lambda: {
    "count": 0,
    "avg_fitness_vals": [],
    "avg_ir_vals": []
})

# Aggregate across all records
for record in data:
    goals = record.get("goals", {})
    for goal_name, metrics in goals.items():
        goal_stats[goal_name]["count"] += metrics["count"]
        goal_stats[goal_name]["avg_fitness_vals"].append(metrics["avg_fitness"])
        goal_stats[goal_name]["avg_ir_vals"].append(metrics["avg_ir"])

# Compute and print summary
print("\n📊 Aggregated Goal Statistics")
print(f"Total records analyzed: {len(data)}\n")

for goal_name, stats in goal_stats.items():
    fit_arr = np.array(stats["avg_fitness_vals"])
    ir_arr = np.array(stats["avg_ir_vals"])

    print(f"🔹 {goal_name}")
    print(f"   Total count: {stats['count']}")
    print(f"   Avg Fitness     → mean = {fit_arr.mean():.4f}, std = {fit_arr.std():.4f}")
    print(f"   Avg Intrinsic R → mean = {ir_arr.mean():.4f}, std = {ir_arr.std():.4f}")

    if np.allclose(fit_arr, 0) and np.allclose(ir_arr, 0):
        print(f"   ⚠️ No learning progress observed for this goal.")

print("\n✅ Done.")
