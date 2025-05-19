# neuro_policy.py
import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction


def he_init(in_dim, out_dim):
    std = np.sqrt(2.0 / in_dim)
    return np.random.randn(in_dim, out_dim).astype(np.float32) * std


class NeuroPolicy:
    """
    1-hidden-layer MLP:   (obs ⊕ goal) → 64 ReLU → 3 tanh
    θ is packed into one flat array so we can mutate it easily.
    """

    def __init__(self,
                 obs_dim: int,
                 theta: np.ndarray | None = None,
                 goal_enc_dim: int = 3,
                 sigma_mut: float = 0.05):
        self.obs_dim = obs_dim
        self.goal_dim = goal_enc_dim
        self.inp_dim = obs_dim + goal_enc_dim
        self.hidden_dim = 64
        self.out_dim = 3
        self.sigma_mut = sigma_mut

        if theta is None:
            W1 = he_init(self.inp_dim, self.hidden_dim)
            b1 = np.zeros(self.hidden_dim, dtype=np.float32)
            W2 = he_init(self.hidden_dim, self.out_dim)
            b2 = np.zeros(self.out_dim, dtype=np.float32)
            self.theta = self._pack(W1, b1, W2, b2)
        else:
            self.theta = theta.astype(np.float32)

    # ---------------------------------------------------------------- act
    def act(self, obs_vec: np.ndarray, g_enc: np.ndarray):
        x = np.concatenate([obs_vec, g_enc], dtype=np.float32)  # shape inp_dim
        W1, b1, W2, b2 = self._unpack()

        h = np.maximum(0, x @ W1 + b1)  # ReLU
        v = np.tanh(h @ W2 + b2)  # tanh → (vx, vy, interact_bias)

        vx, vy, bias = v
        # ------------- primitive selection ---------------------------------
        if bias > 0.0:
            return Action.INTERACT
        if abs(vx) < 0.15 and abs(vy) < 0.15:
            return Action.STAY
        if abs(vx) >= abs(vy):
            return Direction.EAST if vx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if vy > 0 else Direction.NORTH

    # ---------------------------------------------------------------- mutate
    def mutate(self):
        new_theta = self.theta + np.random.normal(0, self.sigma_mut, self.theta.shape)
        return NeuroPolicy(self.obs_dim, theta=new_theta,
                           goal_enc_dim=self.goal_dim, sigma_mut=self.sigma_mut)

    # ---------------------------------------------------------------- helpers
    def _pack(self, W1, b1, W2, b2):
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def _unpack(self):
        s0 = self.obs_dim + self.goal_dim
        s0 *= self.hidden_dim
        s1 = s0 + self.hidden_dim
        s2 = s1 + self.hidden_dim * self.out_dim
        W1 = self.theta[:s0].reshape(self.inp_dim, self.hidden_dim)
        b1 = self.theta[s0:s1]
        W2 = self.theta[s1:s2].reshape(self.hidden_dim, self.out_dim)
        b2 = self.theta[s2:]
        return W1, b1, W2, b2
