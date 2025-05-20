# rbf_policy.py
import numpy as np

from imrl_agent_new.VARIABLES import HORIZON
from overcooked_ai_py.mdp.actions import Action, Direction


class RBFPolicy:
    """
    5 fixed Gaussians per channel, total θ length = 20
        • 5 weights → desired vx(t)
        • 5 weights → desired vy(t)
        • 5 weights → interact bias(t)
        • 5 weights → stay bias(t)      (optional, helps idle)

    Steps per roll-out (H) = 50       σ = 5
    """

    H = HORIZON
    N_BASIS = 5
    centers = np.linspace(0, H - 1, N_BASIS)  # 0 12.5 25 37.5 50
    sigma = 5.0

    def __init__(self, theta: np.ndarray | None = None, mut_std=0.1):
        if theta is None:
            theta = np.random.uniform(-1, 1, 20)
        self.theta = theta.astype(np.float32)
        self.mut_std = mut_std  # mutation σ

    # ----------------------------------------------------------------- act
    def act(self, t_step: int):
        g = np.exp(-0.5 * ((t_step - RBFPolicy.centers) / RBFPolicy.sigma) ** 2)
        vx = np.dot(self.theta[0: 5], g)  # weights 0-4
        vy = np.dot(self.theta[5:10], g)  # weights 5-9
        bias_int = np.dot(self.theta[10:15], g)  # weights 10-14
        bias_stay = np.dot(self.theta[15:20], g)  # weights 15-19

        # --- primitive selection ------------------------------------------
        if bias_int > 0.2:
            return Action.INTERACT
        if bias_stay > 0.2:
            return Action.STAY
        if abs(vx) >= abs(vy):
            return Direction.EAST if vx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if vy > 0 else Direction.NORTH

    # ----------------------------------------------------------------- mutate
    def mutate(self):
        return RBFPolicy(self.theta + np.random.normal(0, self.mut_std, 20),
                         mut_std=self.mut_std)
