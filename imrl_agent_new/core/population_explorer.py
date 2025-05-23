# core/population_explorer.py
import random
from typing import Any

import numpy as np


class PopulationExplorer:
    """
    Handles EXPLORATION controllers (Πε):

    1.  If the KB is empty        → return a brand-new random policy
    2.  Otherwise
        • pick a *parent* θ close to the current goal
        • clone & mutate it
    """

    def __init__(self, kb, PolicyClass, obs_dim: int, goal_enc_dim: int = 3,
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

        # 1) choose parent **by fitness**
        parent_theta = self._fittest_theta()

        # 2) 10 % chance: evaluate the parent unchanged  (elitism)
        if random.random() < 0.10:
            return self.PolicyClass(self.obs_dim, self.goal_dim, theta=parent_theta)

        # 3) otherwise mutate with smaller σ if parent is good
        best_fit = max(r.fitness for r in self.kb.buffer)
        sigma = 0.02 if best_fit > 0 else 0.05
        child_theta = parent_theta + np.random.normal(0, sigma, parent_theta.shape)
        return self.PolicyClass(self.obs_dim, self.goal_dim, theta=child_theta)

    def _fittest_theta(self):
        idx = np.argmax([r.fitness for r in self.kb.buffer])
        return self.kb.buffer[idx].theta

    # ----------------------------------------------------------------------
    def _nearest_theta(self):
        """
        Finds the experiment record whose outcome vector is closest
        to *any* dummy query (context ignored → [0]).
        """
        dummy_out = np.zeros(self.kb.buffer[0].outcome.shape)
        idx, _ = self.kb.nearest(np.array([0]), dummy_out)
        return self.kb.buffer[idx].theta
