# ROLLOUT PARAMETERS
HORIZON = 100  # number of steps to run per rollout
ROLLOUTS = 2000  # number of rollouts to run
LAYOUT_ID = 21  # layout index to use from the layouts list in overcooked_ai_py
LOAD_KB = False  # whether to load the knowledge base from a file or not

# AGENT PARAMETERS
EXPLOIT_PROB = 0.2  # probability of exploiting the current policy

### NEURAL NETWORK PARAMETERS
NEURO_POLICY_HIDDEN_DIM = 64  # hidden layer size for the neural network policy
ADAPTIVE_NOISE_STD = 0.1  # standard deviation for the adaptive noise added to the policy outputs
PARENT_POLICY_RECENT_RECORDS = 50  # number of recent records to consider when selecting a parent policy

### GOAL SELECTION PARAMETERS
GOAL_EMA_K = 20  # How many past exploit records to include in the EMA
GOAL_EMA_ALPHA = 0.6  # Smoothing factor for the EMA (0 < ALPHA <= 1)
GOAL_GREEDY_PROB = 0.8  # Probability of selecting the greedy goal space
