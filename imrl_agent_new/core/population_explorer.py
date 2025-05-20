# population_explorer.py   ← only the first import & constructor change
import numpy as np

from imrl_agent_new.core.rbf_policy import RBFPolicy  # NEW


class PopulationExplorer:
    def __init__(self, kb):
        self.kb = kb
        self.PolicyC = RBFPolicy
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
        if len(self.kb) == 0:
            return self.PolicyC()  # random 20-dim θ
        idx, _ = self.kb.nearest(np.array([0]), np.zeros(self.kb.buffer[0].outcome.shape))
        parent = self.kb.buffer[idx].theta
        return self.PolicyC(parent).mutate()
