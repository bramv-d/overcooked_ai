# core/population_explorer.py
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
    def sample_or_mutate(self, goal_vec) -> Any:
        """
        Returns a *new instance* of PolicyClass ready for a fresh roll-out.
        """
        if len(self.kb) == 0:
            # 1) brand-new random controller
            return self.PolicyClass(obs_dim=self.obs_dim, goal_enc_dim=self.goal_dim)

        # 2) choose parent θ
        parent_theta = self._nearest_theta()  # 1-NN in outcome space
        # small Gaussian mutation in parameter space
        child_theta = parent_theta + np.random.normal(0, self.mut_std, parent_theta.shape)
        return self.PolicyClass(obs_dim=self.obs_dim,
                                goal_enc_dim=self.goal_dim,
                                theta=child_theta)

    # ----------------------------------------------------------------------
    def _nearest_theta(self):
        """
        Finds the experiment record whose outcome vector is closest
        to *any* dummy query (context ignored → [0]).
        """
        dummy_out = np.zeros(self.kb.buffer[0].outcome.shape)
        idx, _ = self.kb.nearest(np.array([0]), dummy_out)
        return self.kb.buffer[idx].theta
