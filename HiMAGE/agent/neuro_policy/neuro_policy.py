# neuro_policy.py
import random
from typing import List

import numpy as np

from HiMAGE.agent.goals.goal_spaces import Goal, GoalSpaceEnum
from HiMAGE.agent.helpers.obs_to_vect import GOAL_POLICY_INPUT_VECTOR_SIZE
from HiMAGE.agent.knowledge_base import KnowledgeBase
from HiMAGE.agent.neuro_policy.high_level_actions import HighLevelActions
from HiMAGE.hyper_parameters import ACTION_POLICY_ADAPTIVE_NOISE_STD, ACTION_POLICY_HIDDEN_DIM, \
    ACTION_POLICY_PARENT_RECENT_RECORDS, \
    GOAL_POLICY_ADAPTIVE_NOISE_STD, GOAL_POLICY_HIDDEN_DIM, \
    GOAL_POLICY_PARENT_RECENT_RECORDS


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Kaiming-He init for a linear layer (ReLU)."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std


class NeuroPolicy:
    def __init__(self, input_dimension, hidden_dim, output_token, theta: np.ndarray | None = None):
        self.inp_dim = input_dimension
        self.hidden_dim = hidden_dim
        self.output_token = output_token

        if theta is None:  # fresh initialization
            W1 = he_init(self.inp_dim, self.hidden_dim)
            b1 = np.zeros(self.hidden_dim, dtype=np.float32)
            W2 = he_init(self.hidden_dim, self.output_token)
            b2 = np.zeros(self.output_token, dtype=np.float32)
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

            return int(np.random.choice(self.output_token, p=p))

    # ────────────────────────── helpers ───────────────────────────────
    def _pack(self, W1, b1, W2, b2) -> np.ndarray:
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def _unpack(self):
        """Recover weight matrices from flat θ."""
        D, H, A = self.inp_dim, self.hidden_dim, self.output_token
        i = 0
        W1 = self.theta[i:i + D * H].reshape(D, H)
        i += D * H
        b1 = self.theta[i:i + H]
        i += H
        W2 = self.theta[i:i + H * A].reshape(H, A)
        i += H * A
        b2 = self.theta[i:i + A]
        return W1, b1, W2, b2


def get_neuro_policy(input_dimension, hidden_dim, output_token, selected_goal_id: int, kb: KnowledgeBase, exploit: bool,
                     parent_policy_recent_records, adaptive_noise_std) -> NeuroPolicy:
    if not kb.nearest(selected_goal_id, 1):
        return NeuroPolicy(input_dimension, hidden_dim, output_token)
    rec = kb.nearest(selected_goal_id, parent_policy_recent_records)
    if exploit:
        # exploit: use the best policy from the knowledge base
        best_idx = np.argmax([r.fitness for r in rec])
        return NeuroPolicy(input_dimension, hidden_dim, output_token, theta=rec[best_idx].theta)

    # explore: pick a random parent policy to mutate from
    parent_policy = random.choice(rec)
    adaptive_noise = adaptive_noise_std / (
            parent_policy.intrinsic_reward + adaptive_noise_std)  # more noise when progress is low
    child_theta = parent_policy.theta + np.random.normal(0, adaptive_noise, parent_policy.theta.shape)
    return NeuroPolicy(input_dimension, hidden_dim, output_token, theta=child_theta)


def get_action_policy(kb: KnowledgeBase, exploit: bool, selected_goal_id: int) -> NeuroPolicy:
    return get_neuro_policy(GOAL_POLICY_INPUT_VECTOR_SIZE,
                            ACTION_POLICY_HIDDEN_DIM,
                            len(HighLevelActions),
                            selected_goal_id=selected_goal_id,
                            kb=kb,
                            exploit=exploit,
                            parent_policy_recent_records=ACTION_POLICY_PARENT_RECENT_RECORDS,
                            adaptive_noise_std=ACTION_POLICY_ADAPTIVE_NOISE_STD)


def get_goal_policy(kb: KnowledgeBase, goal_spaces: List[Goal], exploit: bool) -> NeuroPolicy:
    return get_neuro_policy(GOAL_POLICY_INPUT_VECTOR_SIZE,
                            GOAL_POLICY_HIDDEN_DIM,
                            len(goal_spaces),
                            selected_goal_id=GoalSpaceEnum.OVERALL_GOAL,
                            kb=kb,
                            exploit=exploit,
                            parent_policy_recent_records=GOAL_POLICY_PARENT_RECENT_RECORDS,
                            adaptive_noise_std=GOAL_POLICY_ADAPTIVE_NOISE_STD)


class ActionPolicy:
    def __init__(self, neuro_policy: NeuroPolicy, goal: Goal):
        self.neuro_policy = neuro_policy
        self.goal = goal
