# population_explorer.py
import numpy as np

class PopulationExplorer:
    """
    Handles exploration controllers: pick a parent θ, mutate, and act.
    """
    def __init__(self, kb, PolicyClass, inp_dim):
        self.kb = kb
        self.PolicyClass = PolicyClass  # expected NeuroPolicy
        self.inp_dim = inp_dim
        self.current = None

    # ---------------------------------------------------------------- act
    def act(self, obs_vec, goal_vec):
        if self.current is None:
            self.current = self._sample_or_mutate()
        return self.current.act(obs_vec, goal_vec)

    # ---------------------------------------------------------------- utils
    def current_theta(self):
        return self.current.theta

    def _sample_or_mutate(self):
        """Return a NeuroPolicy: new random if KB empty, else mutate nearest."""
        if len(self.kb) == 0:
            return self.PolicyClass(obs_dim=self.inp_dim)

        # nearest based on outcome only (context ignored here)
        idx, _ = self.kb.nearest(np.array([0]), np.zeros(self.kb.buffer[0].outcome.shape))
        parent = self.kb.buffer[idx].theta
        return self.PolicyClass(obs_dim=self.inp_dim, theta=parent).mutate()
