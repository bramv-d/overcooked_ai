# ROLLOUT PARAMETERS
HORIZON = 15  # number of steps to run per rollout
ROLLOUTS = 10000  # number of rollouts to run
LAYOUT_ID = 22  # layout index to use from the layouts list in overcooked_ai_py
LOAD_KB = True  # whether to load the knowledge base from a file or not

# ---- INTRINSIC REWARD ---
IR_BONUS_CAP: float = 0.20
IR_BONUS_SLOPE: float = 0.01
# config.py
from dataclasses import dataclass


@dataclass
class AgentConfig:
    # -------- AGENT ----------
    exploit_prob: float = 0.2

    # ------ NEURO POLICY -----
    neuro_policy_hidden_dim: int = 100
    adaptive_noise_std: float = 0.1
    parent_policy_recent: int = 100
    mutate_records: int = 50
    neuro_evolution_multiplier: int = 10

    # ----- GOAL SELECTION ----
    n_recent: int = 75

    # Selectin of sub policies
    minimum_goal_fitness: float = 0.52

    ir_avg_prev_records: int = 40
