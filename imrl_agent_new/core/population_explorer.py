import numpy as np

from imrl_agent_new.core.neuro_policy import NeuroPolicy


class PopulationExplorer:
    def __init__(self, kb, PolicyClass, inp_dim):
        self.kb = kb
        self.PolicyClass = PolicyClass
        self.inp_dim = inp_dim
        self.current = None

    def act(self, obs_vec, goal_vec, t):
        if len(self.kb) == 0:
            self.current = self.PolicyClass(self.inp_dim)
        else:
            idx, _ = self.kb.nearest(np.array([0]), obs_vec[:3])  # crude key
            parent = self.kb.buffer[idx].theta
            self.current = self.PolicyClass(self.inp_dim, parent).mutate()
        return self.current.act(obs_vec, goal_vec)

    def current_theta(self):
        return self.current.theta

    def sample_or_mutate(self, goal_vec):
        if len(self.kb) == 0:
            return NeuroPolicy(self.inp_dim)  # new random net
        idx, _ = self.kb.nearest(np.array([0]), np.zeros(self.kb.outcome_dim))
        parent = self.kb.buffer[idx].theta
        return NeuroPolicy(obs_dim=self.inp_dim, parent).mutate()
