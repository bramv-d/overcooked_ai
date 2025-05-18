# neuro_policy.py
import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction


def he_init(in_dim, out_dim):
    std = np.sqrt(2.0 / in_dim)
    return np.random.randn(in_dim, out_dim).astype(np.float32) * std


class NeuroPolicy:
    """
    One-hidden-layer MLP whose parameters are packed in a flat vector θ.
        input  -> 64 ReLU -> 3 tanh  (raw vector v)
        v is mapped to the 6 primitives.

    Parameter count:
        W1  :  inp × 64
        b1  :  64
        W2  :  64 × 3
        b2  :  3
    """

    def __init__(self, obs_dim=None, goal_enc_dim=3, theta: np.ndarray | None = None, sigma_mut=0.05):
        self.inp_dim = obs_dim + goal_enc_dim
        self.hidden_dim = 64
        self.out_dim = 3  # raw continuous vec (vx, vy, interact_bias)
        self.sigma_mut = sigma_mut

        if theta is None:
            # He init
            W1 = he_init(self.inp_dim, self.hidden_dim)
            b1 = np.zeros(self.hidden_dim, dtype=np.float32)
            W2 = he_init(self.hidden_dim, self.out_dim)
            b2 = np.zeros(self.out_dim, dtype=np.float32)
            self.theta = self._pack(W1, b1, W2, b2)
        else:
            self.theta = theta.astype(np.float32)

    # ---------- forward -----------------------------------------------------
    def act(self, obs_vec, g_enc):
        x = np.concatenate([obs_vec, g_enc], dtype=np.float32)
        W1, b1, W2, b2 = self._unpack()

        h = np.maximum(0, x @ W1 + b1)  # ReLU
        v = np.tanh(h @ W2 + b2)  # tanh → (vx, vy, bias)

        vx, vy, bias = v
        # decide primitive
        if bias > 0.6:  # high bias ⇒ interact
            return Action.INTERACT
        if abs(vx) < 0.15 and abs(vy) < 0.15:
            return Action.STAY
        if abs(vx) >= abs(vy):
            return Direction.EAST if vx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if vy > 0 else Direction.NORTH

    # ---------- mutation ----------------------------------------------------
    def mutate(self):
        new_theta = self.theta + np.random.normal(0, self.sigma_mut, self.theta.shape)
        return NeuroPolicy(self.inp_dim, new_theta, self.sigma_mut)

    # ---------- packing helpers --------------------------------------------
    def _pack(self, W1, b1, W2, b2):
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def _unpack(self):
        s0 = self.inp_dim * self.hidden_dim
        s1 = s0 + self.hidden_dim
        s2 = s1 + self.hidden_dim * self.out_dim
        W1 = self.theta[:s0].reshape(self.inp_dim, self.hidden_dim)
        b1 = self.theta[s0:s1]
        W2 = self.theta[s1:s2].reshape(self.hidden_dim, self.out_dim)
        b2 = self.theta[s2:]
        return W1, b1, W2, b2
