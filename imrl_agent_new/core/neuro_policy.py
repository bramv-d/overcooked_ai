# neuro_policy.py
import numpy as np


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Kaiming-He init for a linear layer (ReLU)."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std


class NeuroPolicy:
    def __init__(self,
                 obs_dim: int,
                 goal_enc_dim: int = 0,
                 num_tokens: int = 9,
                 hidden_dim: int = 64,
                 sigma_mut: float = 0.01,
                 theta: np.ndarray | None = None):

        self.obs_dim = obs_dim
        self.goal_dim = goal_enc_dim
        self.inp_dim = obs_dim + goal_enc_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.sigma_mut = sigma_mut

        if theta is None:  # fresh initialisation
            W1 = he_init(self.inp_dim, hidden_dim)
            b1 = np.zeros(hidden_dim, dtype=np.float32)
            W2 = he_init(hidden_dim, num_tokens)
            b2 = np.zeros(num_tokens, dtype=np.float32)
            self.theta = self._pack(W1, b1, W2, b2)
        else:  # copy provided weights
            self.theta = theta.astype(np.float32)

    # ──────────────────────────── public ──────────────────────────────
    def select_token(self,
                     obs_vec: np.ndarray,
                     goal_enc: np.ndarray | None = None,
                     greedy: bool = True) -> int:
        if goal_enc is None:
            goal_enc = np.empty(0, dtype=np.float32)

        x = np.concatenate([obs_vec, goal_enc]).astype(np.float32)

        W1, b1, W2, b2 = self._unpack()
        h = np.maximum(0.0, x @ W1 + b1)  # ReLU
        z = h @ W2 + b2  # logits  (num_tokens,)

        if greedy:
            return int(np.argmax(z))
        else:
            p = np.exp(z - z.max(), dtype=np.float32)
            p /= p.sum()
            return int(np.random.choice(self.num_tokens, p=p))

    def mutate(self) -> "NeuroPolicy":
        """Return *new* policy with θ ← θ + 𝒩(0, σ²)."""
        new_theta = self.theta + np.random.normal(
            0.0, self.sigma_mut, size=self.theta.shape).astype(np.float32)
        return NeuroPolicy(self.obs_dim,
                           goal_enc_dim=self.goal_dim,
                           num_tokens=self.num_tokens,
                           hidden_dim=self.hidden_dim,
                           sigma_mut=self.sigma_mut,
                           theta=new_theta)

    # ────────────────────────── helpers ───────────────────────────────
    def _pack(self, W1, b1, W2, b2) -> np.ndarray:
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def _unpack(self):
        """Recover weight matrices from flat θ."""
        D, H, A = self.inp_dim, self.hidden_dim, self.num_tokens
        i = 0
        W1 = self.theta[i:i + D * H].reshape(D, H);
        i += D * H
        b1 = self.theta[i:i + H];
        i += H
        W2 = self.theta[i:i + H * A].reshape(H, A);
        i += H * A
        b2 = self.theta[i:i + A]
        return W1, b1, W2, b2
