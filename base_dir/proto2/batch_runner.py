# batch_runner.py
import subprocess

NUM_RUNS = 50

for i in range(NUM_RUNS):
    print(f"\n🚀 Starting run {i + 1}/{NUM_RUNS}...\n")
    subprocess.run(["/opt/homebrew/bin/python3.11", "run_rollout.py"])  # Replace with your filename
