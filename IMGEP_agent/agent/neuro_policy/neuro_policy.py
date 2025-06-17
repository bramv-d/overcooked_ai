# neuro_policy.py
import random
from typing import List

import numpy as np

from IMGEP_agent.agent.goals.goal_spaces import Goal
from IMGEP_agent.agent.helpers.obs_to_vect import OBS_VEC_SIZE
from IMGEP_agent.agent.knowledge_base import KnowledgeBase
from IMGEP_agent.agent.neuro_policy.high_level_actions import HighLevelActions
from IMGEP_agent.hyper_parameters import ADAPTIVE_NOISE_STD, NEURO_POLICY_HIDDEN_DIM, PARENT_POLICY_RECENT_RECORDS


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Kaiming-He init for a linear layer (ReLU)."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std


class NeuroPolicy:
    def __init__(self, goal_spaces: List[Goal], theta: np.ndarray | None = None):

        self.inp_dim = OBS_VEC_SIZE  # from obs_to_vec()
        self.hidden_dim = NEURO_POLICY_HIDDEN_DIM
        self.num_tokens = len(HighLevelActions) + len(goal_spaces)

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


def get_neuro_policy(selected_goal: Goal, kb: KnowledgeBase, goal_spaces: List[Goal], exploit: bool) -> NeuroPolicy:
    # TODO include the shared episode reward into the policy selection.
    # TODO this should probably be done on reset. The shared episode reward should be a scalar from 0 tot 1 depending on the 50 recent episodes shared reward

    if not kb.nearest(selected_goal, 1):
        return NeuroPolicy(goal_spaces)
    rec = kb.nearest(selected_goal, PARENT_POLICY_RECENT_RECORDS)
    if exploit:
        # exploit: use the best policy from the knowledge base
        best_idx = np.argmax([r.fitness for r in rec])
        return NeuroPolicy(theta=rec[best_idx].theta, goal_spaces=goal_spaces)

    # explore: pick a random parent policy to mutate from
    parent_policy = random.choice(rec)
    adaptive_noise = ADAPTIVE_NOISE_STD / (
            parent_policy.intrinsic_reward + ADAPTIVE_NOISE_STD)  # more noise when progress is low
    child_theta = parent_policy.theta + np.random.normal(0, adaptive_noise, parent_policy.theta.shape)
    return NeuroPolicy(theta=child_theta, goal_spaces=goal_spaces)
