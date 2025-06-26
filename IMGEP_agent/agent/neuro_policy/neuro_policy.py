# neuro_policy.py
import random

import numpy as np

from IMGEP_agent.agent.neuro_policy.high_level_actions import HighLevelActions
from IMGEP_agent.agent.neuro_policy.neuro_policy_input_vector import POLICY_INPUT_VECTOR_SIZE
from IMGEP_agent.hyper_parameters import AgentConfig
from IMGEP_agent.shared_agent.goal_spaces import Goal
from IMGEP_agent.shared_agent.knowledge_base import KnowledgeBase


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Kaiming-He init for a linear layer (ReLU)."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std


class NeuroPolicy:
    def __init__(self, config: AgentConfig, theta: np.ndarray | None = None):

        self.inp_dim = POLICY_INPUT_VECTOR_SIZE
        self.hidden_dim = config.neuro_policy_hidden_dim
        self.num_tokens = len(HighLevelActions)

        if theta is None:  # fresh initialization
            W1 = he_init(self.inp_dim, self.hidden_dim)
            b1 = np.zeros(self.hidden_dim, dtype=np.float32)
            W2 = he_init(self.hidden_dim, self.num_tokens)
            b2 = np.zeros(self.num_tokens, dtype=np.float32)
            self.theta = self._pack(W1, b1, W2, b2)
        else:  # copy provided weights
            self.theta = theta.astype(np.float32)

    # ──────────────────────────── public ──────────────────────────────
    def select_token(self,
                     obs_vec: np.ndarray,
                     greedy: bool) -> int:

        x = np.concatenate([obs_vec]).astype(np.float32)

        W1, b1, W2, b2 = self._unpack()
        h = np.maximum(0.0, x @ W1 + b1)  # ReLU
        z = h @ W2 + b2  # logits  (num_tokens,)

        if greedy:
            return int(np.argmax(z))
        else:
            p = np.exp(z - z.max(), dtype=np.float32)
            p /= p.sum()

            return int(np.random.choice(self.num_tokens, p=p))

    # ────────────────────────── helpers ───────────────────────────────
    def _pack(self, W1, b1, W2, b2) -> np.ndarray:
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def _unpack(self):
        """Recover weight matrices from flat θ."""
        D, H, A = self.inp_dim, self.hidden_dim, self.num_tokens
        i = 0
        W1 = self.theta[i:i + D * H].reshape(D, H)
        i += D * H
        b1 = self.theta[i:i + H]
        i += H
        W2 = self.theta[i:i + H * A].reshape(H, A)
        i += H * A
        b2 = self.theta[i:i + A]
        return W1, b1, W2, b2


class GoalSpaceNeuroPolicy:
    """
    A NeuroPolicy that is specialized for a specific goal space.
    """

    def __init__(self, goal: Goal, neuro_policy: NeuroPolicy, exploit: bool):
        self.goal = goal
        self.neuro_policy = neuro_policy
        self.exploit = exploit


def get_neuro_policy(selected_goal: Goal, kb: KnowledgeBase, exploit: bool,
                     config: AgentConfig, agent_id) -> NeuroPolicy:
    def get_theta(record):
        # agent_id selects which theta to use
        return getattr(record, f"theta{agent_id}")

    if not kb.nearest(goal=selected_goal, k=1):
        return NeuroPolicy(config)  # no records for this goal, return a fresh policy

    if exploit:
        rec = kb.nearest(goal=selected_goal, k=config.parent_policy_recent)
        # exploit: use the best policy from the knowledge base
        best_idx = np.argmax([r.fitness for r in rec])
        return NeuroPolicy(theta=get_theta(rec[best_idx]), config=config)

    rec = kb.nearest(goal=selected_goal, k=config.mutate_records)
    # explore: pick a random parent policy to mutate from
    parent_policy = random.choice(rec)
    adaptive_noise = config.adaptive_noise_std / (
            max(0, parent_policy.intrinsic_reward) + config.adaptive_noise_std)  # more noise when progress is low
    parent_theta = get_theta(parent_policy)
    child_theta = parent_theta + np.random.normal(0, adaptive_noise, parent_theta.shape)
    return NeuroPolicy(theta=child_theta, config=config)
