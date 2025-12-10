Experimental Workflow

The overall workflow consists of four main steps.

First, the global hyperparameters must be configured according to the layout being evaluated. In particular, the
LAYOUT_ID and USE_COUNTERS settings are layout-dependent and must be set correctly in hyper_parameters.py. If these
constraints are not respected, no meaningful results will be produced.

Next, batch experiments are executed using the batch_runner.py script for each prototype (proto1–proto4). Each batch
runner performs a large number of rollouts for two agents and saves the resulting trajectories and fitness data to disk.
Experiments are run separately for layout 20 and layout 22. All prototypes can be executed in parallel if sufficient
computational resources are available.

The rollout data is stored as buffer files (.npz), organised by prototype, layout, batch index, and agent. These buffers
represent the complete experimental output and are required for all downstream analysis.

Finally, visualisations are generated using plot_agent_fitness_debug.py. This script processes the stored rollout
buffers and produces fitness and behaviour visualisations for every prototype, for both layouts and both agents. All
generated figures are saved to the base_dir/stats directory.

In addition to these aggregate visualisations, it is also possible to generate detailed trajectory-level visualisations
to better understand agent behaviour. During experimentation, this proved especially useful for diagnosing failure modes
and understanding what the agents were doing wrong. An example implementation can be found in
base_dir/proto1/run_rollout.py (around line 65), where the trajectory of one of the final agents is rendered
step-by-step and saved both as individual images and as an animated GIF.
EXPERIMENTAL WORKFLOW

1. Configure hyperparameters

----------------------------
Edit:
base_dir/hyper_parameters.py

Required layout settings:

Layout 20:
LAYOUT_ID = 20
USE_COUNTERS = False

Layout 22:
LAYOUT_ID = 22
USE_COUNTERS = True

IMPORTANT:
Any other configuration will produce invalid results.
There are many additional hyperparameters available—see the file for details.

2. Run batch experiments

-----------------------
Run batch_runner.py for each prototype (repeat once per layout):

python base_dir/proto1/batch_runner.py
python base_dir/proto2/batch_runner.py
python base_dir/proto3/batch_runner.py
python base_dir/proto4/batch_runner.py

Experiment setup:

- 50 batches
- 5,000 rollouts per batch
- Horizon: 50
- 2 agents per run

All prototypes can be executed in parallel if compute allows.

3. Rollout buffer output

-----------------------
Each run saves rollout buffers per agent, for example:

base_dir/proto1/layout20/buffer_rollouts/1/buffer_agent0.npz
base_dir/proto1/layout20/buffer_rollouts/1/buffer_agent1.npz

These buffers contain the complete experimental output and are required
for all downstream analysis and visualisation.

4. Generate visualisations

-------------------------
After all batch runs are complete, generate visualisations using:

python base_dir/plot_agent_fitness_debug.py

This creates plots for:

- All prototypes
- Both layouts (20 and 22)
- Both agents

All figures are saved to:
base_dir/stats/

5. Optional: trajectory-level visualisation

-------------------------------------------
For detailed debugging, trajectory-level renderings can be generated.
This was especially useful for understanding failure modes.

Example implementation:
base_dir/proto1/run_rollout.py  (around line 65)

This code renders the trajectory of one of the final agents and saves:

- Individual images per timestep
- An animated GIF of the full trajectory
