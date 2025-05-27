# core/population_explorer.py
from typing import Any, Type

import numpy as np

from imrl_agent_new.core.neuro_policy import NeuroPolicy


class PopulationExplorer:
    """
    Handles EXPLORATION controllers (Πε):

    1.  If the KB is empty        → return a brand-new random policy
    2.  Otherwise
        • pick a *parent* θ close to the current goal
        • clone & mutate it
    """

    def __init__(self, kb, PolicyClass: Type[NeuroPolicy], obs_dim: int, goal_enc_dim: int = 3,
                 mut_std: float = 0.05):
        self.kb = kb
        self.PolicyClass = PolicyClass  # e.g. NeuroPolicy
        self.obs_dim = obs_dim
        self.goal_dim = goal_enc_dim
        self.mut_std = mut_std

    # ----------------------------------------------------------------------
    # core/population_explorer.py
    def sample_or_mutate(self, goal_vec) -> Any:
        if len(self.kb) == 0:
            return self.PolicyClass(self.obs_dim, self.goal_dim)  # brand-new random

        # 1) Get the previous policy θ
        parent_theta = self.kb.buffer[-1].theta.copy()

        child_theta = parent_theta + np.random.normal(0, 0.01, parent_theta.shape)
        return self.PolicyClass(self.obs_dim, self.goal_dim, theta=child_theta)
