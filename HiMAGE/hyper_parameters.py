# ROLLOUT PARAMETERS
HORIZON = 100  # number of steps to run per rollout
ROLLOUTS = 5000  # number of rollouts to run
LAYOUT_ID = 22  # layout index to use from the layouts list in overcooked_ai_py
LOAD_KB = False  # whether to load the knowledge base from a file or not

# AGENT PARAMETERS
EXPLOIT_PROB = 0.2  # probability of exploiting the current policy

### ACTION POLICY
ACTION_POLICY_HIDDEN_DIM = 20  # hidden layer size for the action policy neural network
ACTION_POLICY_ADAPTIVE_NOISE_STD = 0.1  # standard deviation for the adaptive noise added to the action policy outputs
ACTION_POLICY_PARENT_RECENT_RECORDS = 50  # number of recent records to consider when selecting a parent action policy
ACTION_POLICY_NEUROEVOLUTION_MULTIPLIER = 10  # multiplier for the number of action policies to generate during neuroevolution

### GOAL POLICY
GOAL_POLICY_HIDDEN_DIM = 64  # hidden layer size for the goal selection neural network
GOAL_POLICY_ADAPTIVE_NOISE_STD = 0.1  # standard deviation for the adaptive noise in goal selection
GOAL_POLICY_PARENT_RECENT_RECORDS = 20  # number of recent records to consider when selecting a parent goal policy
