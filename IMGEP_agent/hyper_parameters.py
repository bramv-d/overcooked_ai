# ROLLOUT PARAMETERS
HORIZON = 100  # number of steps to run per rollout
ROLLOUTS = 4000  # number of rollouts to run
LAYOUT_ID = 21  # layout index to use from the layouts list in overcooked_ai_py
LOAD_KB = False  # whether to load the knowledge base from a file or not

# AGENT PARAMETERS

### NEURAL NETWORK PARAMETERS
NEURO_POLICY_HIDDEN_DIM = 64  # hidden layer size for the neural network policy

### GOAL SELECTION PARAMETERS
GOAL_EMA_K = 20  # How many past exploit records to include in the EMA
GOAL_EMA_ALPHA = 0.5  # Smoothing factor for the EMA (0 < ALPHA <= 1)
GOAL_GREEDY_PROB = 0.8
