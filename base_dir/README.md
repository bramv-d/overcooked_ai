### Experimental Workflow

The overall workflow consists of four main steps. First, the global hyperparameters must be configured according to the
layout being evaluated. In particular, the `LAYOUT_ID` and `USE_COUNTERS` settings are layout-dependent and must be set
correctly in `hyper_parameters.py`. If these constraints are not respected, no meaningful results will be produced.

Next, batch experiments are executed using the `batch_runner.py` script for each prototype (`proto1`–`proto4`). Each
batch runner performs a large number of rollouts for two agents and saves the resulting trajectories and fitness data to
disk. Experiments are run separately for layout 20 and layout 22. All prototypes can be executed in parallel if
sufficient computational resources are available.

The rollout data is stored as buffer files (`.npz`), organised by prototype, layout, batch index, and agent. These
buffers represent the complete experimental output and are required for all downstream analysis.

Finally, visualisations are generated using `plot_agent_fitness_debug.py`. This script processes the stored rollout
buffers and produces fitness and behaviour visualisations for every prototype, for both layouts and both agents. All
generated figures are saved to the `base_dir/stats` directory.

In addition to these aggregate visualisations, it is also possible to generate detailed trajectory-level visualisations
to better understand agent behaviour. During experimentation, this proved especially useful for diagnosing failure modes
and understanding what the agents were doing wrong. An example implementation can be found in
`base_dir/proto1/run_rollout.py` (around line 65), where the trajectory of one of the final agents is rendered
step-by-step and saved both as individual images and as an animated GIF.

# ------------------------------------------------------------

# 1. Configure hyperparameters

# ------------------------------------------------------------

# In base_dir/hyper_parameters.py

# For layout 20:

# LAYOUT_ID = 20

# USE_COUNTERS = False

#

# For layout 22:

# LAYOUT_ID = 22

# USE_COUNTERS = True

#

# IMPORTANT: Any other configuration will produce invalid results

# There are a lot of other parameters to tweak, checkout what they do!

# ------------------------------------------------------------

# 2. Run batch experiments

# ------------------------------------------------------------

# Run 50 batches of 5000 rollouts (horizon = 50) per agent

# Repeat once for each layout (20 & 22)

python base_dir/proto1/batch_runner.py
python base_dir/proto2/batch_runner.py
python base_dir/proto3/batch_runner.py
python base_dir/proto4/batch_runner.py

# ------------------------------------------------------------

# 3. Stored rollout buffers (example)

# ------------------------------------------------------------

# Each run saves buffers per agent, e.g.:

# base_dir/proto1/layout20/buffer_rollouts/1/buffer_agent0.npz

# base_dir/proto1/layout20/buffer_rollouts/1/buffer_agent1.npz

# ------------------------------------------------------------

# 4. Generate visualisations

# ------------------------------------------------------------

# This creates plots for all protos, both layouts, and both agents

# Outputs are saved to base_dir/stats/

python base_dir/plot_agent_fitness_debug.py
